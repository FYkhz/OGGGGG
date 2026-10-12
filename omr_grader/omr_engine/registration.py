"""Page detection, registration-marker detection and perspective correction."""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import cv2
import numpy as np

from document_scanner import detect_page_corners

from .template import CORNER_NAMES, MARKER_SIZE, PAGE_HEIGHT, PAGE_WIDTH, marker_centers

DETECT_MAX_SIDE = 1600

_MARKER_SPAN_X = float(marker_centers()[1, 0] - marker_centers()[0, 0])  # 1480
_MARKER_SPAN_Y = float(marker_centers()[3, 1] - marker_centers()[0, 1])  # 1980


@dataclass
class MarkerBlob:
    center: np.ndarray  # (2,) in detection-image pixels
    area: float
    side: float
    fill: float


@dataclass
class Registration:
    """Maps the input image to the canonical page frame."""

    homographies: list[np.ndarray]  # candidate image->canonical transforms, one per plausible orientation
    method: str  # "markers", "markers-3", "page-edge", "none"
    quad: np.ndarray | None  # ordered image-space quad used (full-resolution pixels)
    missing_corner_slot: int | None = None  # index in the quad of an inferred (missing) marker
    marker_side_px: float = 0.0
    corner_angles: tuple[float, ...] = ()
    messages: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        return {"markers": 1.0, "markers-3": 0.85, "page-edge": 0.6}.get(self.method, 0.0)


def _flatten(gray: np.ndarray) -> np.ndarray:
    """Divide out slow illumination changes (shadows, colored or uneven light)."""
    h, w = gray.shape
    small = cv2.resize(gray, (max(8, w // 4), max(8, h // 4)), interpolation=cv2.INTER_AREA)
    k = max(15, int(min(small.shape) / 7) | 1)
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 4.0)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, np.maximum(bg, 1), scale=255)


def _marker_blobs(gray_small: np.ndarray) -> list[MarkerBlob]:
    flat = _flatten(cv2.GaussianBlur(gray_small, (3, 3), 0))
    binary = (flat < 120).astype(np.uint8) * 255
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    h, w = gray_small.shape
    min_side = max(8.0, 0.010 * max(h, w))
    max_side = 0.14 * min(h, w)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs: list[MarkerBlob] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_side**2 or area > max_side**2:
            continue
        (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
        if rw < 1 or rh < 1:
            continue
        aspect = min(rw, rh) / max(rw, rh)
        if aspect < 0.55:
            continue
        if area / (rw * rh) < 0.86:  # squares fill their bounding rectangle; filled circles (~0.79) do not
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area <= 0 or area / hull_area < 0.9:
            continue
        approx = cv2.approxPolyDP(contour, 0.06 * cv2.arcLength(contour, True), True)
        if len(approx) != 4:
            continue
        x, y, bw, bh = cv2.boundingRect(contour)
        mask = np.zeros((bh, bw), np.uint8)
        cv2.drawContours(mask, [contour - [x, y]], -1, 255, -1)
        inside = binary[y : y + bh, x : x + bw][mask > 0]
        fill = float(np.mean(inside > 0)) if inside.size else 0.0
        if fill < 0.88:  # rejects hollow squares such as QR finder patterns
            continue
        m = cv2.moments(contour)
        if m["m00"] <= 0:
            continue
        blobs.append(MarkerBlob(np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], np.float32), area, float(np.sqrt(area)), fill))
    blobs.sort(key=lambda b: b.area, reverse=True)
    return blobs[:16]


def _order_clockwise(points: np.ndarray) -> np.ndarray:
    """Order four points clockwise starting from the one nearest the image top-left."""
    pts = np.asarray(points, np.float32).reshape(4, 2)
    center = pts.mean(axis=0)
    angles = np.arctan2(pts[:, 1] - center[1], pts[:, 0] - center[0])
    pts = pts[np.argsort(angles)]  # clockwise in image coords (y down)
    start = int(np.argmin(pts.sum(axis=1)))
    return np.roll(pts, -start, axis=0)


def _corner_angles(quad: np.ndarray) -> tuple[float, ...]:
    angles = []
    for i in range(4):
        a, b, c = quad[i - 1], quad[i], quad[(i + 1) % 4]
        v1, v2 = a - b, c - b
        cosang = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9))
        angles.append(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))))
    return tuple(angles)


def _quad_score(quad: np.ndarray, sides: np.ndarray, largest_side: float | None = None) -> float | None:
    """Lower is better. None when the four points cannot be the registration markers."""
    if not cv2.isContourConvex(quad.reshape(4, 1, 2).astype(np.float32)):
        return None
    edges = np.array([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)])
    m = float(np.median(sides))
    if m <= 0 or sides.max() / max(sides.min(), 1e-6) > 2.2:
        return None
    pair_a = (edges[0] + edges[2]) / 2.0
    pair_b = (edges[1] + edges[3]) / 2.0
    short, long_ = sorted((pair_a, pair_b))
    exp_short = _MARKER_SPAN_X / MARKER_SIZE
    exp_long = _MARKER_SPAN_Y / MARKER_SIZE
    r_short = short / m / exp_short
    r_long = long_ / m / exp_long
    if not (0.55 <= r_short <= 1.7 and 0.55 <= r_long <= 1.7):
        return None
    angles = _corner_angles(quad)
    if min(angles) < 35 or max(angles) > 145:
        return None
    aspect = long_ / short
    # The registration squares are the largest solid squares on the page.
    size_penalty = 0.5 * np.log(largest_side / m) if largest_side else 0.0
    return float(
        np.log(r_short) ** 2
        + np.log(r_long) ** 2
        + 0.5 * np.log(aspect / (exp_long / exp_short)) ** 2
        + 0.2 * np.log(sides.max() / sides.min())
        + size_penalty
    )


def _find_quad(blobs: list[MarkerBlob]) -> tuple[np.ndarray, float, int | None] | None:
    """Return (ordered quad, marker side, inferred slot) from 4 markers, or 3 markers + inference."""
    largest = max(b.side for b in blobs)
    best: tuple[float, np.ndarray, float] | None = None
    for combo in itertools.combinations(range(len(blobs)), 4):
        pts = np.array([blobs[i].center for i in combo])
        sides = np.array([blobs[i].side for i in combo])
        quad = _order_clockwise(pts)
        score = _quad_score(quad, sides, largest)
        if score is not None and (best is None or score < best[0]):
            best = (score, quad, float(np.median(sides)))
    if best is not None:
        return best[1], best[2], None

    # Graceful degradation: three markers forming an L, the fourth completed as a parallelogram.
    best3: tuple[float, np.ndarray, float, int] | None = None
    for combo in itertools.combinations(range(len(blobs)), 3):
        sides = np.array([blobs[i].side for i in combo])
        pts = [blobs[i].center for i in combo]
        for k in range(3):
            b = pts[k]
            a, c = pts[(k + 1) % 3], pts[(k + 2) % 3]
            d = a + c - b
            quad = _order_clockwise(np.array([a, b, c, d]))
            score = _quad_score(quad, np.append(sides, np.median(sides)), largest)
            if score is None:
                continue
            slot = int(np.argmin(np.linalg.norm(quad - d, axis=1)))
            if best3 is None or score < best3[0]:
                best3 = (score, quad, float(np.median(sides)), slot)
    if best3 is not None:
        return best3[1], best3[2], best3[3]
    return None


def _orientation_homographies(quad: np.ndarray, targets: np.ndarray) -> list[np.ndarray]:
    """Homographies for the cyclic assignments of quad points to canonical corners that keep a portrait page."""
    edges = np.array([np.linalg.norm(quad[(i + 1) % 4] - quad[i]) for i in range(4)])
    horizontal_first = (edges[0] + edges[2]) <= (edges[1] + edges[3])
    ambiguous = abs((edges[0] + edges[2]) - (edges[1] + edges[3])) / edges.sum() < 0.04
    result: list[np.ndarray] = []
    for k in range(4):
        # Rotation k maps quad[(i + k) % 4] -> canonical corner i; odd k swaps which edges are horizontal.
        if not ambiguous and (k % 2 == 0) != horizontal_first:
            continue
        src = np.roll(quad, -k, axis=0).astype(np.float32)
        result.append(cv2.getPerspectiveTransform(src, targets.astype(np.float32)))
    return result


def register(image_bgr: np.ndarray) -> Registration:
    h, w = image_bgr.shape[:2]
    scale = min(1.0, DETECT_MAX_SIDE / float(max(h, w)))
    small = image_bgr if scale == 1.0 else cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    gray_small = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    blobs = _marker_blobs(gray_small)
    found = _find_quad(blobs) if len(blobs) >= 3 else None
    if found is not None:
        quad_small, side_small, slot = found
        quad = quad_small / scale
        homographies = _orientation_homographies(quad, marker_centers())
        reg = Registration(
            homographies=homographies,
            method="markers" if slot is None else "markers-3",
            quad=quad,
            missing_corner_slot=slot,
            marker_side_px=side_small / scale,
            corner_angles=_corner_angles(quad),
        )
        return reg

    corners = detect_page_corners(image_bgr)
    if corners is not None:
        quad = _order_clockwise(corners)
        page = np.array([[0, 0], [PAGE_WIDTH - 1, 0], [PAGE_WIDTH - 1, PAGE_HEIGHT - 1], [0, PAGE_HEIGHT - 1]], np.float32)
        return Registration(
            homographies=_orientation_homographies(quad, page),
            method="page-edge",
            quad=quad,
            corner_angles=_corner_angles(quad),
            messages=["The black corner squares could not be found; the paper edge was used for alignment instead."],
        )
    return Registration(
        homographies=[],
        method="none",
        quad=None,
        messages=["Neither the four black corner squares nor the paper edge could be found. Retake the photo with the whole sheet visible."],
    )


def warp_to_canonical(image_bgr: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        image_bgr,
        homography,
        (PAGE_WIDTH, PAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def missing_corner_name(reg: Registration, homography: np.ndarray) -> str | None:
    """Name the canonical corner whose marker was inferred, given the chosen orientation."""
    if reg.missing_corner_slot is None or reg.quad is None:
        return None
    point = reg.quad[reg.missing_corner_slot].reshape(1, 1, 2).astype(np.float32)
    mapped = cv2.perspectiveTransform(point, homography).reshape(2)
    index = int(np.argmin(np.linalg.norm(marker_centers() - mapped, axis=1)))
    return CORNER_NAMES[index]
