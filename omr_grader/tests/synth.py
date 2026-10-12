"""Synthetic sheets with realistic student marks and camera/printer degradations."""

from __future__ import annotations

import cv2
import numpy as np

from omr_engine.sheet_generator import render_page
from omr_engine.template import MARKER_MARGIN, MARKER_SIZE, Template

RNG = np.random.default_rng(7)


def _pencil(img: np.ndarray, mask: np.ndarray, tone: float, texture: float = 18.0) -> None:
    noise = RNG.normal(0, texture, mask.shape)
    value = np.clip(tone + noise, 0, 255)
    for ch in range(3):
        channel = img[:, :, ch].astype(np.float32)
        channel[mask] = np.minimum(channel[mask], value[mask])
        img[:, :, ch] = channel.astype(np.uint8)


def draw_mark(img: np.ndarray, center, radius: float, kind: str) -> None:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cx, cy = int(round(center[0])), int(round(center[1]))
    R = radius
    tone = 70.0
    if kind == "solid":
        cv2.circle(mask, (cx, cy), int(R * 0.82), 255, -1)
    elif kind == "heavy":
        cv2.circle(mask, (cx, cy), int(R * 1.12), 255, -1)
        tone = 35.0
    elif kind == "light":
        cv2.circle(mask, (cx, cy), int(R * 0.8), 255, -1)
        tone = 165.0
    elif kind == "x":
        d = int(R * 0.62)
        cv2.line(mask, (cx - d, cy - d), (cx + d, cy + d), 255, max(3, int(R / 6)))
        cv2.line(mask, (cx - d, cy + d), (cx + d, cy - d), 255, max(3, int(R / 6)))
    elif kind == "check":
        pts = np.array([[cx - R * 0.55, cy], [cx - R * 0.1, cy + R * 0.5], [cx + R * 0.6, cy - R * 0.6]], np.int32)
        cv2.polylines(mask, [pts], False, 255, max(3, int(R / 6)))
    elif kind == "dot":
        cv2.circle(mask, (cx, cy), max(2, int(R * 0.18)), 255, -1)
    elif kind == "erased":
        cv2.circle(mask, (cx, cy), int(R * 0.78), 255, -1)
        tone = 205.0
    elif kind == "half":
        cv2.ellipse(mask, (cx, cy), (int(R * 0.8), int(R * 0.8)), 0, 0, 200, 255, -1)
    elif kind == "line":  # stray stroke passing through the bubble and beyond
        cv2.line(mask, (int(cx - R * 2.5), int(cy - R * 0.2)), (int(cx + R * 2.5), int(cy + R * 0.3)), 255, 3)
    else:
        raise ValueError(kind)
    _pencil(img, mask > 0, tone, 10.0 if kind in ("light", "erased") else 18.0)


def make_sheet(template: Template, page: int, answers: dict[int, list[tuple[str, str]]], *, title: str | None = None) -> np.ndarray:
    """answers: question -> list of (choice letter, mark kind)."""
    img = render_page(template, page, title=title)
    layout = template.page(page)
    for q, marks in answers.items():
        if q not in layout.questions:
            continue
        r = layout.questions.index(q)
        for letter, kind in marks:
            c = template.choices.index(letter)
            draw_mark(img, layout.centers[r, c], template.bubble_radius, kind)
    return img


# --------------------------- degradations ---------------------------------


def photo(
    img: np.ndarray,
    *,
    rotate_deg: float = 0.0,
    tilt: float = 0.0,
    canvas: tuple[int, int] = (2600, 3200),
    page_frac: float = 0.82,
    background=(95, 105, 115),
) -> np.ndarray:
    """Place the page in a larger frame with rotation and perspective (trapezoid)."""
    h, w = img.shape[:2]
    cw, ch = canvas
    s = page_frac * min(cw / w, ch / h)
    pw, ph = w * s, h * s
    cx, cy = cw / 2, ch / 2
    corners = np.array([[-pw / 2, -ph / 2], [pw / 2, -ph / 2], [pw / 2, ph / 2], [-pw / 2, ph / 2]], np.float32)
    # perspective: top edge narrower
    corners[0, 0] += tilt * pw
    corners[1, 0] -= tilt * pw
    a = np.radians(rotate_deg)
    rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    corners = corners @ rot.T + [cx, cy]
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, corners.astype(np.float32))
    return cv2.warpPerspective(img, M, (cw, ch), flags=cv2.INTER_AREA, borderValue=background)


def lighting(img: np.ndarray, *, gain=(1.0, 1.0, 1.0), shadow: float = 0.0, gray: float = 1.0) -> np.ndarray:
    h, w = img.shape[:2]
    f = img.astype(np.float32) * gray
    f *= np.array(gain, np.float32)[None, None, :]
    if shadow:
        xx = np.linspace(0, 1, w)[None, :]
        yy = np.linspace(0, 1, h)[:, None]
        field = 1.0 - shadow * np.clip(1.3 - (xx + yy), 0, 1)
        f *= field[:, :, None]
    return np.clip(f, 0, 255).astype(np.uint8)


def blur(img: np.ndarray, sigma: float) -> np.ndarray:
    return cv2.GaussianBlur(img, (0, 0), sigma)


def noise(img: np.ndarray, sigma: float) -> np.ndarray:
    return np.clip(img.astype(np.float32) + RNG.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def jpeg(img: np.ndarray, quality: int = 60) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def printer_scale(img: np.ndarray, factor: float, shift=(0, 0)) -> np.ndarray:
    h, w = img.shape[:2]
    M = np.array([[factor, 0, (1 - factor) * w / 2 + shift[0]], [0, factor, (1 - factor) * h / 2 + shift[1]]], np.float32)
    return cv2.warpAffine(img, M, (w, h), borderValue=(255, 255, 255))


def remove_marker(img: np.ndarray, corner: int) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    xs = [MARKER_MARGIN, w - MARKER_MARGIN - MARKER_SIZE]
    ys = [MARKER_MARGIN, h - MARKER_MARGIN - MARKER_SIZE]
    x, y = [(xs[0], ys[0]), (xs[1], ys[0]), (xs[1], ys[1]), (xs[0], ys[1])][corner]
    out[y - 5 : y + MARKER_SIZE + 5, x - 5 : x + MARKER_SIZE + 5] = 255
    return out


def curl(img: np.ndarray, amplitude: float = 6.0) -> np.ndarray:
    h, w = img.shape[:2]
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    dy = amplitude * np.sin(np.pi * xx / w)
    dx = amplitude * 0.5 * np.sin(np.pi * yy / h)
    return cv2.remap(img, xx + dx, yy + dy, cv2.INTER_LINEAR, borderValue=(255, 255, 255))


def crop(img: np.ndarray, frac_right: float) -> np.ndarray:
    w = img.shape[1]
    return img[:, : int(w * (1 - frac_right))].copy()
