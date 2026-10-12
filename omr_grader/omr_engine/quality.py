"""Image-quality checks that run before any answer is trusted."""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .features import NormalizedPage


@dataclass
class QualityReport:
    score: float  # 0..1
    reject: bool
    problems: list[str] = field(default_factory=list)  # reasons the page cannot be graded
    warnings: list[str] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)


def check_input(image_bgr: np.ndarray) -> QualityReport:
    h, w = image_bgr.shape[:2]
    report = QualityReport(score=1.0, reject=False, metrics={"input_width": w, "input_height": h})
    if min(h, w) < 500:
        report.reject = True
        report.score = 0.0
        report.problems.append(f"Image resolution is too low ({w}×{h}). Use a photo or scan at least 1000 pixels wide.")
    return report


def sharpness(norm: NormalizedPage, centers: np.ndarray, radius: float) -> float:
    """Edge steepness of the printed bubble outlines, independent of their darkness.

    Sharp print has outline edges that rise over ~1-2 px; blur spreads them out.
    Returns roughly the reciprocal edge width in pixels (sharp ~0.3+, blurry < 0.12).
    """
    d = norm.darkness
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3) / 8.0
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3) / 8.0
    grad = np.hypot(gx, gy)
    half = int(np.ceil(radius * 1.3))
    values = []
    step = max(1, len(centers) // 60)
    for cx, cy in centers[::step]:
        x1, y1 = int(cx) - half, int(cy) - half
        x2, y2 = int(cx) + half + 1, int(cy) + half + 1
        if x1 < 0 or y1 < 0 or y2 > d.shape[0] or x2 > d.shape[1]:
            continue
        pd = d[y1:y2, x1:x2]
        pg = grad[y1:y2, x1:x2]
        contrast = float(np.percentile(pd, 99) - np.percentile(pd, 30))
        if contrast < 0.15:
            continue
        values.append(float(np.percentile(pg, 99)) / contrast)
    return float(np.median(values)) if values else 0.0


def page_sharpness(canonical_bgr: np.ndarray) -> float:
    """Template-free version of :func:`sharpness`, computed over high-contrast tiles of the page."""
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    d = 1.0 - gray
    grad = np.hypot(cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)) / 8.0
    values = []
    tile = 64
    for y in range(0, d.shape[0] - tile, tile):
        for x in range(0, d.shape[1] - tile, tile):
            pd = d[y : y + tile, x : x + tile]
            contrast = float(np.percentile(pd, 99) - np.percentile(pd, 30))
            if contrast < 0.3:
                continue
            values.append(float(np.percentile(grad[y : y + tile, x : x + tile], 99)) / contrast)
    # Not enough contrast anywhere: washed out rather than blurry; callers report it separately.
    return float(np.median(values)) if len(values) >= 8 else float("nan")


BLUR_REJECT = 0.125
BLUR_WARN = 0.19


def check_page(
    norm: NormalizedPage,
    centers: np.ndarray,
    radius: float,
    *,
    corner_angles: tuple[float, ...],
    bubble_radius_in_photo: float,
    paper_noise: float,
    ring_baseline: float,
) -> QualityReport:
    report = QualityReport(score=1.0, reject=False)
    m = report.metrics

    m["bubble_radius_in_photo_px"] = round(bubble_radius_in_photo, 1)
    if bubble_radius_in_photo and bubble_radius_in_photo < 5.0:
        report.problems.append("The sheet is too small in the photo to read the bubbles. Move the camera closer so the page fills the frame.")
    elif bubble_radius_in_photo and bubble_radius_in_photo < 8.0:
        report.warnings.append("The sheet is fairly small in the photo; a closer photo will be more reliable.")

    if corner_angles:
        worst = max(abs(a - 90.0) for a in corner_angles)
        m["max_corner_angle_error_deg"] = round(worst, 1)
        if worst > 40:
            report.problems.append("The photo is taken at too steep an angle. Hold the camera directly above the sheet.")
        elif worst > 25:
            report.warnings.append("Strong camera tilt detected; it was corrected, but a straighter photo is more reliable.")

    x1 = int(max(0, centers[:, 0].min() - 2 * radius))
    x2 = int(min(norm.gray.shape[1], centers[:, 0].max() + 2 * radius))
    y1 = int(max(0, centers[:, 1].min() - 2 * radius))
    y2 = int(min(norm.gray.shape[0], centers[:, 1].max() + 2 * radius))
    bg = norm.background[y1:y2, x1:x2]
    paper = float(np.median(bg))
    lo, hi = np.percentile(bg, [2, 98])
    shadow_ratio = float(hi / max(lo, 1.0))
    m["paper_brightness"] = round(paper, 1)
    m["shadow_ratio"] = round(shadow_ratio, 2)
    if paper < 45:
        report.problems.append("The image is too dark to grade reliably. Add light or move away from shadows.")
    elif paper < 85:
        report.warnings.append("The image is quite dark; results were normalized but may be less reliable.")
    if shadow_ratio > 4.0:
        report.problems.append("A strong shadow covers part of the answer area. Retake the photo with even lighting.")
    elif shadow_ratio > 1.8:
        report.warnings.append("Uneven lighting or a shadow was detected and corrected.")

    m["paper_noise"] = round(paper_noise, 4)
    if paper_noise > 0.10:
        report.problems.append("The image is very noisy (low light or heavy compression). Retake it in better light.")
    elif paper_noise > 0.05:
        report.warnings.append("The image is noisy; faint marks may be flagged for review.")

    m["ring_contrast"] = round(ring_baseline, 3)
    if ring_baseline < 0.07:
        report.problems.append("The printed bubbles are barely visible (overexposure, glare, or a very faint print).")

    page_sharp = page_sharpness(cv2.cvtColor(norm.gray, cv2.COLOR_GRAY2BGR))
    sharp = sharpness(norm, centers, radius)
    if not np.isnan(page_sharp):
        sharp = min(sharp, page_sharp) if sharp else page_sharp
    m["sharpness"] = round(sharp, 3)
    if sharp and sharp < BLUR_REJECT:
        report.problems.append("Page is too blurry to grade reliably. Hold the camera steady and let it focus.")
    elif sharp and sharp < BLUR_WARN:
        report.warnings.append("The image is slightly blurry.")

    penalties = [
        0.0 if not report.problems else 1.0,
        0.15 * len(report.warnings),
    ]
    report.score = float(np.clip(1.0 - sum(penalties), 0.0, 1.0))
    report.reject = bool(report.problems)
    return report
