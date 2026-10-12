"""Illumination normalization, per-bubble features and whole-sheet calibration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class NormalizedPage:
    gray: np.ndarray  # uint8 luminance of the canonical page
    flat: np.ndarray  # uint8, illumination-flattened (paper ~255)
    background: np.ndarray  # uint8 estimated paper brightness per pixel
    darkness: np.ndarray  # float32 0..1, 1 - flat/255


def normalize_page(canonical_bgr: np.ndarray, radius: float) -> NormalizedPage:
    """Estimate the local paper brightness and divide it out.

    A morphological close with a kernel wider than a filled bubble removes all
    printed and pencil marks, leaving only the slowly varying paper/lighting
    level (gray paper, colored light, shadows, vignetting).
    """
    gray = cv2.cvtColor(canonical_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    f = 4
    small = cv2.resize(gray, (w // f, h // f), interpolation=cv2.INTER_AREA)
    k = int(max(9, round(4.4 * radius / f))) | 1
    bg = cv2.morphologyEx(small, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    bg = cv2.GaussianBlur(bg, (0, 0), k / 3.0)
    background = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR)
    flat = cv2.divide(gray, np.maximum(background, 1), scale=255)
    darkness = 1.0 - flat.astype(np.float32) / 255.0
    return NormalizedPage(gray=gray, flat=flat, background=background, darkness=darkness)


SOLID_INK = 0.42  # relative darkness of an ordinary pencil fill

FEATURE_NAMES = (
    "inner_fill",
    "inner_soft",
    "center_soft",
    "center_fill",
    "inner_mean",
    "center_mean",
    "ring_mean",
    "local_bg",
    "spill",
    "extent",
    "quadrant_balance",
    "components",
    "ink_intensity",
    "max_component",
    "score",
)
F = {name: i for i, name in enumerate(FEATURE_NAMES)}


@dataclass
class Masks:
    size: int
    inner: np.ndarray
    center: np.ndarray
    ring: np.ndarray
    annulus: np.ndarray
    quadrants: list[np.ndarray]


def build_masks(radius: float) -> Masks:
    half = int(np.ceil(radius * 1.6))
    size = 2 * half + 1
    yy, xx = np.mgrid[:size, :size].astype(np.float32) - half
    r = np.hypot(xx, yy)
    inner = r <= 0.66 * radius
    center = r <= 0.36 * radius
    ring = (r >= 0.85 * radius) & (r <= 1.15 * radius)
    annulus = (r >= 1.25 * radius) & (r <= 1.5 * radius)
    quadrants = [inner & (xx >= 0) & (yy < 0), inner & (xx < 0) & (yy < 0), inner & (xx < 0) & (yy >= 0), inner & (xx >= 0) & (yy >= 0)]
    return Masks(size=size, inner=inner, center=center, ring=ring, annulus=annulus, quadrants=quadrants)


def _patch(image: np.ndarray, center: np.ndarray, size: int) -> np.ndarray:
    return cv2.getRectSubPix(image, (size, size), (float(center[0]), float(center[1])))


def paper_noise(norm: NormalizedPage, centers: np.ndarray, masks: Masks) -> float:
    """Robust noise level of paper pixels around bubbles (darkness units)."""
    samples = []
    step = max(1, len(centers) // 120)
    for c in centers[::step]:
        p = _patch(norm.darkness, c, masks.size)[masks.annulus]
        samples.append(p)
    if not samples:
        return 0.02
    values = np.concatenate(samples)
    med = np.median(values)
    mad = np.median(np.abs(values - med)) * 1.4826
    return float(max(mad, 0.004))


def extract_features(norm: NormalizedPage, centers: np.ndarray, radius: float) -> tuple[np.ndarray, dict]:
    """Return (M, len(FEATURE_NAMES)) features for every expected bubble plus calibration info.

    All darkness values are relative to the bubble's own surrounding paper, so a
    shadowed corner and a brightly lit centre are judged on the same scale.
    """
    masks = build_masks(radius)
    noise = paper_noise(norm, centers, masks)
    # Pixel is "inked" when it is clearly darker than local paper. Light pencil is ~0.2-0.35, solid ~0.5+.
    pixel_thr = float(np.clip(6.0 * noise, 0.12, 0.22))
    n = len(centers)
    feats = np.zeros((n, len(FEATURE_NAMES)), np.float32)
    inner_area = float(masks.inner.sum())
    inner_diameter = 2 * 0.66 * radius

    for i, c in enumerate(centers):
        p = _patch(norm.darkness, c, masks.size)
        local_bg = float(np.median(p[masks.annulus]))
        d = np.clip(p - local_bg, 0.0, 1.0)
        inked = d > pixel_thr
        inner_ink = inked & masks.inner
        # Soft ink: faint residue (erasures) counts a little, real pencil counts fully.
        soft = np.clip((d - pixel_thr) / max(SOLID_INK - pixel_thr, 0.05), 0.0, 1.0)
        inner_soft = float(soft[masks.inner].mean())
        center_soft = float(soft[masks.center].mean())
        inner_fill = float(inner_ink.sum()) / inner_area
        center_fill = float((inked & masks.center).sum()) / float(masks.center.sum())
        inner_mean = float(d[masks.inner].mean())
        ink_intensity = float(d[inner_ink].mean()) if inner_ink.any() else 0.0
        center_mean = float(d[masks.center].mean())
        ring_mean = float(d[masks.ring].mean())
        # Ink just outside the bubble relative to ink inside it: high for strokes that pass through.
        spill = min(3.0, float((inked & masks.annulus).sum()) / max(float(inner_ink.sum()), 20.0))
        quads = [float((inked & q).sum()) / max(1.0, float(q.sum())) for q in masks.quadrants]
        balance = min(quads) / max(max(quads), 1e-6) if max(quads) > 0 else 0.0
        extent = 0.0
        components = 0
        max_component = 0.0
        if inner_ink.any():
            num, labels, st, _ = cv2.connectedComponentsWithStats(inner_ink.astype(np.uint8), connectivity=8)
            if num > 1:
                areas = st[1:, cv2.CC_STAT_AREA]
                components = int(np.sum(areas >= max(3, 0.01 * inner_area)))
                biggest = 1 + int(np.argmax(areas))
                max_component = float(areas.max()) / inner_area
                extent = float(max(st[biggest, cv2.CC_STAT_WIDTH], st[biggest, cv2.CC_STAT_HEIGHT])) / inner_diameter
        feats[i] = (inner_fill, inner_soft, center_soft, center_fill, inner_mean, center_mean, ring_mean, local_bg, spill, min(extent, 1.2), balance, components, ink_intensity, max_component, 0.0)

    return feats, {"paper_noise": noise, "pixel_threshold": pixel_thr}


def bubble_scores(feats: np.ndarray, ring_baseline: float) -> np.ndarray:
    """Weighted mark evidence: interior dominates, the printed outline barely counts."""
    inner_fill = feats[:, F["inner_fill"]]
    inner_soft = feats[:, F["inner_soft"]]
    center_soft = feats[:, F["center_soft"]]
    inner_mean = feats[:, F["inner_mean"]]
    ring_mean = feats[:, F["ring_mean"]]
    spill = feats[:, F["spill"]]
    score = (
        0.55 * inner_soft
        + 0.25 * center_soft
        + 0.15 * np.clip(inner_mean / 0.45, 0.0, 1.0)
        + 0.05 * np.clip((ring_mean - ring_baseline) / 0.4, 0.0, 1.0)
    )
    # A stroke that runs through the bubble and well beyond it (a line, handwriting) is weaker evidence
    # than ink that stays inside the bubble.
    stray = (spill > 0.3) & (inner_fill < 0.55)
    score = np.where(stray, score * 0.6, score)
    return score.astype(np.float32)


@dataclass
class Calibration:
    empty_level: float
    empty_spread: float
    filled_level: float
    ring_baseline: float
    paper_noise: float
    pixel_threshold: float
    n_filled_reference: int

    def as_dict(self) -> dict:
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in self.__dict__.items()}


def calibrate(feats: np.ndarray, n_choices: int, info: dict) -> tuple[Calibration, np.ndarray]:
    """Learn what empty and filled bubbles look like on this particular sheet."""
    rows = feats.reshape(-1, n_choices, feats.shape[1])
    ring = rows[:, :, F["ring_mean"]]
    # Most bubbles are empty: the lower choices of each row describe the printed-but-empty bubble.
    ring_baseline = float(np.median(np.sort(ring, axis=1)[:, : max(1, n_choices - 1)]))
    scores = bubble_scores(feats, ring_baseline).reshape(-1, n_choices)
    lower = np.sort(scores, axis=1)[:, : max(1, n_choices - 1)].ravel()
    empty_level = float(np.median(lower))
    spread = float(max(np.median(np.abs(lower - empty_level)) * 1.4826, 0.012))

    best = scores.max(axis=1)
    strong = best[best > empty_level + max(0.3, 12 * spread)]
    default_filled = 0.85
    if len(strong) >= 3:
        filled = float(np.median(strong))
        # Blend toward the default when only a few references exist.
        w = min(1.0, len(strong) / 8.0)
        filled = w * filled + (1 - w) * default_filled
    else:
        filled = default_filled
    filled = max(filled, empty_level + 0.3)
    cal = Calibration(
        empty_level=empty_level,
        empty_spread=spread,
        filled_level=filled,
        ring_baseline=ring_baseline,
        paper_noise=float(info["paper_noise"]),
        pixel_threshold=float(info["pixel_threshold"]),
        n_filled_reference=int(len(strong)),
    )
    return cal, scores.ravel()
