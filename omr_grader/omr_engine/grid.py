"""Bubble-pattern detection and grid reconstruction.

The registration markers give a first, global page alignment. This module uses
the repeated bubble pattern itself as a second, much denser alignment system:

1. detect printed bubble candidates with two independent detectors
   (contour shape analysis and ring template matching);
2. vote for the translation between the template grid and the candidates;
3. match candidates to expected bubbles and fit a global affine/homography
   with RANSAC, so stray circles (letters, pencil loops) cannot pull the grid;
4. estimate smooth local offsets from neighbouring inliers (robust median), so
   paper curvature and printer distortion are absorbed while a single abnormal
   detection has no influence;
5. validate rows/columns and produce a grid confidence.

Final sampling coordinates come from the fitted grid, never from a single
detection, so a heavily filled bubble whose outline vanished is still sampled
at the right place.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GridFit:
    ok: bool
    centers: np.ndarray  # (M, 2) corrected sampling centres in the canonical page
    predicted: np.ndarray  # (M, 2) global-model centres before local refinement
    expected: np.ndarray  # (M, 2) template centres
    candidates: np.ndarray  # (N, 2) detected bubble candidates
    inlier: np.ndarray  # (M,) bool: expected bubble confirmed by a consistent detection
    observed: np.ndarray  # (M, 2) matched detection (nan when none)
    local_offset: np.ndarray  # (M, 2)
    model: np.ndarray  # 3x3 template->page transform
    model_kind: str
    radius_scale: float
    stats: dict = field(default_factory=dict)
    confidence: float = 0.0
    messages: list[str] = field(default_factory=list)


def _contour_candidates(flat: np.ndarray, radius: float, roi: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = roi
    region = flat[y1:y2, x1:x2]
    binary = (region < 175).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    target_area = np.pi * radius**2
    out: list[tuple[float, float]] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not (1.25 * radius <= w <= 2.9 * radius and 1.25 * radius <= h <= 2.9 * radius):
            continue
        if min(w, h) / max(w, h) < 0.72:
            continue
        area = cv2.contourArea(contour)
        if not 0.4 * target_area <= area <= 1.8 * target_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0 or 4 * np.pi * area / perimeter**2 < 0.62:
            continue
        (cx, cy), _ = cv2.minEnclosingCircle(contour)
        out.append((cx + x1, cy + y1))
    return out


def _ring_template(radius: float, width: float) -> np.ndarray:
    size = int(np.ceil(radius * 2.6)) | 1
    c = size // 2
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - c, yy - c)
    ring = np.clip(1.0 - np.abs(r - radius) / max(width * 0.75, 1.0), 0, 1)
    return ring.astype(np.float32)


def _template_candidates(flat: np.ndarray, radius: float, ring_width: int, roi: tuple[int, int, int, int]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = roi
    region = 255.0 - flat[y1:y2, x1:x2].astype(np.float32)
    tmpl = _ring_template(radius, ring_width)
    if region.shape[0] < tmpl.shape[0] or region.shape[1] < tmpl.shape[1]:
        return []
    score = cv2.matchTemplate(region, tmpl, cv2.TM_CCOEFF_NORMED)
    peaks = (score > 0.45).astype(np.uint8)
    dil = cv2.dilate(score, np.ones((int(radius) | 1, int(radius) | 1), np.uint8))
    peaks &= (score >= dil - 1e-6).astype(np.uint8)
    ys, xs = np.nonzero(peaks)
    off = tmpl.shape[0] // 2
    return [(float(x + off + x1), float(y + off + y1)) for x, y in zip(xs, ys)]


def _merge(points: list[tuple[float, float]], tol: float) -> np.ndarray:
    if not points:
        return np.zeros((0, 2), np.float32)
    pts = np.array(points, np.float32)
    used = np.zeros(len(pts), bool)
    merged = []
    for i in range(len(pts)):
        if used[i]:
            continue
        close = np.linalg.norm(pts - pts[i], axis=1) < tol
        close &= ~used
        used |= close
        merged.append(pts[close].mean(axis=0))
    return np.array(merged, np.float32)


def detect_candidates(flat: np.ndarray, expected: np.ndarray, radius: float, ring_width: int, pitch: float) -> np.ndarray:
    h, w = flat.shape
    pad = int(pitch * 1.5 + radius * 2)
    x1 = max(0, int(expected[:, 0].min()) - pad)
    y1 = max(0, int(expected[:, 1].min()) - pad)
    x2 = min(w, int(expected[:, 0].max()) + pad)
    y2 = min(h, int(expected[:, 1].max()) + pad)
    roi = (x1, y1, x2, y2)
    points = _contour_candidates(flat, radius, roi) + _template_candidates(flat, radius, ring_width, roi)
    return _merge(points, tol=radius * 0.45)


def _vote_translation(expected: np.ndarray, candidates: np.ndarray, pitch: float, max_shift: float) -> np.ndarray:
    diff = (candidates[:, None, :] - expected[None, :, :]).reshape(-1, 2)
    keep = np.abs(diff).max(axis=1) < max_shift
    diff = diff[keep]
    if len(diff) == 0:
        return np.zeros(2, np.float32)
    bin_size = 3.0
    n_bins = int(np.ceil(2 * max_shift / bin_size)) + 1
    idx = np.floor((diff + max_shift) / bin_size).astype(int)
    idx = np.clip(idx, 0, n_bins - 1)
    weights = np.exp(-np.sum(diff**2, axis=1) / (2.0 * (pitch * 0.9) ** 2))
    hist = np.zeros((n_bins, n_bins), np.float32)
    np.add.at(hist, (idx[:, 1], idx[:, 0]), weights)
    hist = cv2.GaussianBlur(hist, (5, 5), 1.0)
    by, bx = np.unravel_index(int(np.argmax(hist)), hist.shape)
    peak = np.array([bx, by], np.float32) * bin_size - max_shift + bin_size / 2
    near = np.linalg.norm(diff - peak, axis=1) < bin_size * 2
    return diff[near].mean(axis=0) if np.any(near) else peak


def _apply(model: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), model).reshape(-1, 2).astype(np.float32)


def _match(pred: np.ndarray, candidates: np.ndarray, tol: float) -> np.ndarray:
    """For each predicted bubble, index of a uniquely matched candidate within tol, else -1."""
    match = -np.ones(len(pred), int)
    if len(candidates) == 0:
        return match
    dist = np.linalg.norm(pred[:, None, :] - candidates[None, :, :], axis=2)
    nearest = np.argmin(dist, axis=1)
    best = dist[np.arange(len(pred)), nearest]
    order = np.argsort(best)
    taken: set[int] = set()
    for i in order:
        if best[i] > tol:
            break
        j = int(nearest[i])
        if j in taken:
            continue
        taken.add(j)
        match[i] = j
    return match


def _affine_params(model: np.ndarray) -> dict:
    a = model[:2, :2] / model[2, 2]
    sx = float(np.hypot(a[0, 0], a[1, 0]))
    sy = float(np.hypot(a[0, 1], a[1, 1]))
    rotation = float(np.degrees(np.arctan2(a[1, 0], a[0, 0])))
    skew = float(np.degrees(np.arctan2(a[0, 1], a[1, 1])) + rotation)
    return {
        "scale_x": sx,
        "scale_y": sy,
        "rotation_deg": rotation,
        "skew_deg": skew,
        "offset_x": float(model[0, 2] / model[2, 2]),
        "offset_y": float(model[1, 2] / model[2, 2]),
        "perspective": float(np.hypot(model[2, 0], model[2, 1]) / model[2, 2]),
    }


def fit_grid(
    flat: np.ndarray,
    expected: np.ndarray,
    *,
    radius: float,
    ring_width: int,
    choice_pitch: float,
    row_pitch: float,
    row_index: np.ndarray,
    col_key: np.ndarray,
) -> GridFit:
    """Fit the template grid to the bubble pattern found in the canonical page image.

    ``row_index`` gives the question row of each expected bubble and ``col_key`` a
    column identifier (block * n_choices + choice) used for row/column validation.
    """
    pitch = float(min(choice_pitch, row_pitch))
    candidates = detect_candidates(flat, expected, radius, ring_width, pitch)
    M = len(expected)
    empty = GridFit(
        ok=False,
        centers=expected.copy(),
        predicted=expected.copy(),
        expected=expected,
        candidates=candidates,
        inlier=np.zeros(M, bool),
        observed=np.full((M, 2), np.nan, np.float32),
        local_offset=np.zeros((M, 2), np.float32),
        model=np.eye(3),
        model_kind="identity",
        radius_scale=1.0,
    )
    if len(candidates) < max(6, 0.15 * M):
        empty.messages.append("Too few answer bubbles were visible to confirm the grid. The photo may be blurry, cropped, or of a different sheet.")
        empty.stats = {"candidates": int(len(candidates)), "expected": M, "inlier_fraction": 0.0}
        return empty

    shift = _vote_translation(expected, candidates, pitch, max_shift=min(1.6 * pitch, 140.0))
    model = np.array([[1, 0, shift[0]], [0, 1, shift[1]], [0, 0, 1]], np.float64)
    kind = "translation"
    thr = max(2.5, 0.14 * radius)
    inlier_mask = np.zeros(M, bool)
    match = -np.ones(M, int)

    for tol in (0.45 * pitch, 0.32 * pitch, 0.22 * pitch):
        pred = _apply(model, expected)
        match = _match(pred, candidates, tol)
        idx = np.nonzero(match >= 0)[0]
        if len(idx) < 6:
            break
        src = expected[idx].astype(np.float32)
        dst = candidates[match[idx]].astype(np.float32)
        affine, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=thr * 1.6, maxIters=3000, confidence=0.995)
        if affine is None:
            break
        inl = inl.ravel().astype(bool)
        # Least-squares refinement on RANSAC inliers.
        refined, _ = cv2.estimateAffine2D(src[inl], dst[inl], method=cv2.LMEDS)
        if refined is not None:
            affine = refined
        model = np.vstack([affine, [0, 0, 1]]).astype(np.float64)
        kind = "affine"
        if inl.sum() >= 24:
            hom, hinl = cv2.findHomography(src, dst, cv2.RANSAC, thr * 1.6, maxIters=3000, confidence=0.995)
            if hom is not None:
                res_a = np.linalg.norm(_apply(model, src) - dst, axis=1)
                res_h = np.linalg.norm(_apply(hom, src) - dst, axis=1)
                ia = res_a < thr * 1.6
                ih = res_h < thr * 1.6
                if ih.sum() >= ia.sum() and np.sqrt(np.mean(res_h[ih] ** 2)) < 0.85 * np.sqrt(np.mean(res_a[ia] ** 2)):
                    model, kind = hom.astype(np.float64), "homography"
        pred = _apply(model, expected)
        res = np.full(M, np.inf)
        res[idx] = np.linalg.norm(pred[idx] - candidates[match[idx]], axis=1)
        inlier_mask = res < thr * 1.6

    pred = _apply(model, expected)
    observed = np.full((M, 2), np.nan, np.float32)
    has = match >= 0
    observed[has] = candidates[match[has]]
    residual = np.where(inlier_mask[:, None], observed - pred, 0.0).astype(np.float32)

    # Local refinement: robust neighbourhood median of inlier residuals, shrunk toward zero and clamped.
    local = np.zeros((M, 2), np.float32)
    inl_idx = np.nonzero(inlier_mask)[0]
    if len(inl_idx) >= 6:
        radius_nb = 2.6 * max(choice_pitch, row_pitch)
        d = np.linalg.norm(expected[:, None, :] - expected[None, inl_idx, :], axis=2)
        max_off = 0.3 * radius
        for i in range(M):
            nb = inl_idx[d[i] < radius_nb]
            if len(nb) < 4:
                continue
            med = np.median(residual[nb], axis=0)
            shrink = len(nb) / (len(nb) + 3.0)
            local[i] = np.clip(med * shrink, -max_off, max_off)
    centers = pred + local

    params = _affine_params(model)
    scale = float(np.sqrt(abs(np.linalg.det(model[:2, :2] / model[2, 2]))))
    after = np.where(inlier_mask[:, None], observed - centers, 0.0)
    rms = float(np.sqrt(np.mean(np.sum(after[inlier_mask] ** 2, axis=1)))) if inlier_mask.any() else float("inf")
    inlier_fraction = float(inlier_mask.mean())

    # Row validation: every question row should have its bubbles confirmed on a straight line.
    rows = np.unique(row_index)
    row_support = np.array([inlier_mask[row_index == r].mean() for r in rows])
    rows_unsupported = int(np.sum(row_support == 0))
    # Column validation: per printed column, confirmed fraction and drift.
    cols = np.unique(col_key)
    col_support = np.array([inlier_mask[col_key == c].mean() for c in cols])
    drift = float(np.max(np.linalg.norm(local, axis=1))) if M else 0.0

    # Template-mismatch evidence: many consistent unmatched bubbles inside the grid area.
    used = set(match[inlier_mask].tolist())
    unmatched = np.array([j for j in range(len(candidates)) if j not in used], int)
    stray_near_grid = 0
    if len(unmatched):
        d_un = np.min(np.linalg.norm(candidates[unmatched][:, None, :] - centers[None, :, :], axis=2), axis=1)
        stray_near_grid = int(np.sum((d_un > 0.5 * pitch) & (d_un < 1.6 * pitch)))

    messages: list[str] = []
    conf_inliers = float(np.clip((inlier_fraction - 0.35) / 0.45, 0.0, 1.0))
    conf_rms = float(np.clip(1.0 - max(0.0, rms - 1.0) / (0.3 * radius), 0.0, 1.0))
    conf_scale = float(np.clip(1.0 - max(0.0, abs(scale - 1.0) - 0.06) / 0.1, 0.0, 1.0))
    conf_cols = float(np.clip((col_support.min() - 0.1) / 0.4, 0.0, 1.0)) if len(col_support) else 0.0
    conf_rows = float(np.clip(1.0 - rows_unsupported / max(1, len(rows)) * 4, 0.0, 1.0))
    conf_stray = float(np.clip(1.0 - (stray_near_grid / max(1.0, 0.12 * M) - 1.0), 0.0, 1.0))
    confidence = min(conf_inliers, conf_rms, conf_scale, max(conf_cols, 0.0), conf_rows, conf_stray)

    if abs(scale - 1.0) > 0.06:
        messages.append(f"The sheet appears to be printed at {scale * 100:.0f}% scale; the grid was rescaled to compensate.")
    if len(col_support) and col_support.min() < 0.2 and np.median(col_support) > 0.6:
        messages.append("One column of bubbles expected by this template was not found. The uploaded sheet appears to use a different exam template.")
    if stray_near_grid > 0.12 * M:
        messages.append("More bubbles were found than this template expects. The uploaded sheet appears to use a different exam template.")
    if inlier_fraction < 0.35:
        messages.append("The bubble pattern could not be matched to the template reliably.")

    return GridFit(
        ok=inlier_fraction >= 0.35 and len(inl_idx) >= 6,
        centers=centers.astype(np.float32),
        predicted=pred.astype(np.float32),
        expected=expected,
        candidates=candidates,
        inlier=inlier_mask,
        observed=observed,
        local_offset=local,
        model=model,
        model_kind=kind,
        radius_scale=scale,
        stats={
            "candidates": int(len(candidates)),
            "expected": M,
            "inliers": int(inlier_mask.sum()),
            "inlier_fraction": inlier_fraction,
            "rms_px": rms,
            "max_local_offset_px": drift,
            "rows_without_support": rows_unsupported,
            "min_column_support": float(col_support.min()) if len(col_support) else 0.0,
            "stray_bubbles_near_grid": stray_near_grid,
            "scale": scale,
            **params,
            "confidence_parts": {
                "inliers": conf_inliers,
                "residual": conf_rms,
                "scale": conf_scale,
                "columns": conf_cols,
                "rows": conf_rows,
                "template": conf_stray,
            },
        },
        confidence=float(confidence),
        messages=messages,
    )
