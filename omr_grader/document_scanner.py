from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class DocumentScan:
    image_bgr: np.ndarray
    page_found: bool
    message: str | None = None


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def _resize_for_detection(image_bgr: np.ndarray, max_side: int = 1400) -> tuple[np.ndarray, float]:
    h, w = image_bgr.shape[:2]
    largest = max(h, w)
    if largest <= max_side:
        return image_bgr.copy(), 1.0
    scale = max_side / float(largest)
    resized = cv2.resize(image_bgr, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    return resized, scale


def _quad_candidates(binary: np.ndarray, image_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    candidates: list[np.ndarray] = []
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:30]:
        area = cv2.contourArea(contour)
        if area < image_area * 0.18:
            continue
        hull = cv2.convexHull(contour)
        perimeter = cv2.arcLength(hull, True)
        for epsilon in (0.012, 0.018, 0.025, 0.035):
            approx = cv2.approxPolyDP(hull, epsilon * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                candidates.append(approx.reshape(4, 2).astype(np.float32))
                break
    return candidates


def _score_quad(quad: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    area = abs(cv2.contourArea(quad)) / float(h * w)
    ordered = _order_points(quad)
    tl, tr, br, bl = ordered
    top = np.linalg.norm(tr - tl)
    bottom = np.linalg.norm(br - bl)
    left = np.linalg.norm(bl - tl)
    right = np.linalg.norm(br - tr)
    width = max(top, bottom)
    height = max(left, right)
    if min(width, height) < 1:
        return -1.0

    # A sheet can be strongly perspective-skewed, so aspect is only a soft preference.
    observed_ratio = min(width, height) / max(width, height)
    letter_ratio = 8.5 / 11.0
    ratio_penalty = min(abs(observed_ratio - letter_ratio), 0.45)

    # Prefer large quadrilaterals, especially ones reasonably close to the frame edges.
    centroid = ordered.mean(axis=0)
    center = np.array([w / 2.0, h / 2.0], dtype=np.float32)
    center_penalty = np.linalg.norm(centroid - center) / max(w, h)
    return area * 10.0 - ratio_penalty * 1.5 - center_penalty * 0.5


def detect_page_corners(image_bgr: np.ndarray) -> np.ndarray | None:
    small, scale = _resize_for_detection(image_bgr)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    image_area = float(h * w)

    candidates: list[np.ndarray] = []

    # Edge-based path: best when the paper border contrasts with the table/background.
    median = float(np.median(gray))
    lower = int(max(20, 0.55 * median))
    upper = int(min(240, max(lower + 30, 1.35 * median)))
    edges = cv2.Canny(gray, lower, upper)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8), iterations=2)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    candidates.extend(_quad_candidates(edges, image_area))

    # Bright-paper path: helps when the outer paper edge is weak but the page is lighter than its surroundings.
    adaptive = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        51,
        7,
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8), iterations=2)
    candidates.extend(_quad_candidates(adaptive, image_area))

    if not candidates:
        return None

    best = max(candidates, key=lambda q: _score_quad(q, (h, w)))
    if _score_quad(best, (h, w)) < 1.2:
        return None
    return _order_points(best / scale)


def _warp_page(image_bgr: np.ndarray, corners: np.ndarray, output_size: tuple[int, int]) -> np.ndarray:
    width, height = output_size
    destination = np.array(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(_order_points(corners), destination)
    return cv2.warpPerspective(
        image_bgr,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def _normalize_document(image_bgr: np.ndarray) -> np.ndarray:
    """Aggressive CamScanner-style cleanup with a near-white paper background.

    The important part for OMR is to remove slow gray/shadow gradients without
    washing out pencil fills, printed bubbles, or the black registration squares.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # 1) Remove broad lighting/shadow gradients. A large morphological close is
    # robust to desk shadows and gray paper while ignoring normal printed marks.
    h, w = gray.shape
    kernel_size = int(round(min(h, w) / 18.0))
    kernel_size = max(51, min(kernel_size, 151))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=max(9.0, kernel_size / 5.0))

    # Divide by the estimated background: gray/shadowed paper becomes close to
    # white, while marks that are darker than their local background remain dark.
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)

    # 2) Gentle local contrast to recover pencil marks after illumination flattening.
    clahe = cv2.createCLAHE(clipLimit=1.35, tileGridSize=(12, 12))
    flat = clahe.apply(flat)

    # 3) Aggressive soft white-point curve. Pixels that already look like paper
    # are pushed hard toward pure white; dark/mid-tone ink is largely preserved.
    lut = np.arange(256, dtype=np.float32)
    white_start = 168.0
    paper = lut >= white_start
    t = (lut[paper] - white_start) / (255.0 - white_start)
    # Ease-out curve: rapidly cleans light gray while avoiding a hard binary page.
    lut[paper] = white_start + (255.0 - white_start) * (1.0 - (1.0 - t) ** 2.8)
    # Give darker marks a small contrast boost without crushing faint pencil.
    dark = lut < white_start
    lut[dark] = np.clip((lut[dark] - 8.0) * 1.04 + 8.0, 0, 255)
    flat = cv2.LUT(flat, lut.astype(np.uint8))

    # 4) Any residual near-white texture is scanner noise, not useful OMR data.
    flat[flat >= 238] = 255

    # Small denoise only; stronger filtering can remove lightly shaded bubbles.
    flat = cv2.bilateralFilter(flat, 5, 16, 16)
    flat[flat >= 240] = 255

    # Return a neutral grayscale scan. This prevents colored lighting/camera tint
    # from reintroducing a gray cast in the preview or marker detection.
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


def scan_document(
    image_bgr: np.ndarray,
    *,
    output_size: tuple[int, int],
) -> DocumentScan:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image supplied to document scanner.")

    corners = detect_page_corners(image_bgr)
    if corners is not None:
        warped = _warp_page(image_bgr, corners, output_size)
        cleaned = _normalize_document(warped)
        return DocumentScan(cleaned, True, "Document scanner found and flattened the page.")

    # Graceful fallback: improve lighting without making up a crop. Registration markers can still align it later.
    cleaned = _normalize_document(image_bgr)
    return DocumentScan(
        cleaned,
        False,
        "Document scanner could not confidently find the paper edge. The app kept the full photo and will try the four black registration squares.",
    )
