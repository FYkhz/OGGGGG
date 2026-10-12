"""End-to-end recognition pipeline for one page and for a multi-page exam."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import cv2
import numpy as np

from document_scanner import scan_document

from .classifier import BLANK, CLEAR, MULTIPLE, REVIEW, classify_row
from .features import F, FEATURE_NAMES, Calibration, NormalizedPage, bubble_scores, calibrate, extract_features, normalize_page
from .grid import GridFit, fit_grid
from .quality import BLUR_REJECT, QualityReport, check_input, check_page, page_sharpness
from .registration import Registration, missing_corner_name, register, warp_to_canonical
from .template import PAGE_HEIGHT, PAGE_WIDTH, Template, parse_qr_payload, template_from_qr

MIN_PAGE_CONFIDENCE = 0.45
MARKERS_MISSING = (
    "Fewer than three of the four black corner squares were found. Make sure the whole sheet, "
    "including all four corners, is in the photo."
)
WASHED_OUT = "The image is washed out (overexposure or glare); the printed sheet is barely visible."


@dataclass
class BubbleDetection:
    choice: str
    center: tuple[float, float]
    score: float
    level: float
    features: dict


@dataclass
class QuestionResult:
    question: int
    page: int
    status: str  # clear / blank / multiple / review
    selected: str | None  # the single answer for CLEAR rows
    marked: tuple[str, ...]
    confidence: float  # 0..100
    reason: str
    notes: list[str]
    suggestion: str | None
    choices: tuple[str, ...]
    bubbles: list[BubbleDetection]

    @property
    def densities(self) -> tuple[float, ...]:
        """Per-choice fill score (0..1), kept for the original UI and tests."""
        return tuple(b.score for b in self.bubbles)

    @property
    def answer(self) -> str | None:
        """Machine answer: a letter, 'A+C' for multiple, None for blank/unresolved."""
        if self.status == CLEAR:
            return self.selected
        if self.status == MULTIPLE:
            return "+".join(self.marked)
        return None

    @property
    def label(self) -> str:
        if self.status == CLEAR:
            return str(self.selected)
        if self.status == MULTIPLE:
            return "MULTIPLE " + "/".join(self.marked)
        return self.status.upper()


@dataclass
class PageResult:
    ok: bool
    template: Template | None
    page_number: int | None
    questions: list[QuestionResult] = field(default_factory=list)
    canonical_bgr: np.ndarray | None = None
    scanner_bgr: np.ndarray | None = None
    normalized: NormalizedPage | None = None
    grid: GridFit | None = None
    registration: Registration | None = None
    calibration: Calibration | None = None
    quality: QualityReport | None = None
    qr_found: bool = False
    confidence_parts: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    homography: np.ndarray | None = None
    source_index: int = 0

    @property
    def confidence(self) -> float:
        return float(min(self.confidence_parts.values())) if self.confidence_parts else 0.0


def _qr_attempts(image_bgr: np.ndarray):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    yield gray
    up = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    yield up
    sharp = cv2.addWeighted(up, 1.8, cv2.GaussianBlur(up, (0, 0), 3), -0.8, 0)
    yield sharp
    yield cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]


def _decode_qr(canonical_bgr: np.ndarray, template_box: tuple[int, int, int, int]) -> tuple[str | None, bool]:
    """Return (payload, located_in_expected_box). Tries several enhancements before giving up."""
    detectors = [cv2.QRCodeDetector()]
    if hasattr(cv2, "QRCodeDetectorAruco"):
        detectors.append(cv2.QRCodeDetectorAruco())
    x1, y1, x2, y2 = template_box
    pad = 70
    crop = canonical_bgr[max(0, y1 - pad) : y2 + pad, max(0, x1 - pad) : x2 + pad]
    for image in _qr_attempts(crop):
        for detector in detectors:
            try:
                text, _, _ = detector.detectAndDecode(image)
            except cv2.error:
                continue
            if text:
                return text, True
    # Upside-down page: the code sits in the opposite corner.
    h, w = canonical_bgr.shape[:2]
    opposite = canonical_bgr[max(0, h - y2 - pad) : h - y1 + pad, max(0, w - x2 - pad) : w - x1 + pad]
    for detector in detectors:
        try:
            text, _, _ = detector.detectAndDecode(cv2.cvtColor(opposite, cv2.COLOR_BGR2GRAY))
        except cv2.error:
            continue
        if text:
            return text, False
    return None, False


def _expected_arrays(template: Template, page_number: int):
    layout = template.page(page_number)
    n_c = template.n_choices
    expected = layout.centers.reshape(-1, 2).astype(np.float32)
    row_index = np.repeat(np.arange(layout.n_rows), n_c)
    col_key = np.array([layout.block_index[r] * n_c + c for r in range(layout.n_rows) for c in range(n_c)])
    return layout, expected, row_index, col_key


def _fit(template: Template, page_number: int, norm: NormalizedPage) -> GridFit:
    _, expected, row_index, col_key = _expected_arrays(template, page_number)
    return fit_grid(
        norm.flat,
        expected,
        radius=template.bubble_radius,
        ring_width=template.ring_width,
        choice_pitch=template.choice_pitch,
        row_pitch=template.row_pitch,
        row_index=row_index,
        col_key=col_key,
    )


def _reject(result: PageResult, *messages: str) -> PageResult:
    result.ok = False
    for m in messages:
        if m and m not in result.problems:
            result.problems.append(m)
    return result


def process_page(
    image_bgr: np.ndarray,
    *,
    template_hint: Template | None = None,
    enforce_template: bool = True,
    answer_key_mode: bool = False,
    scanner_mode: bool = False,
    source_index: int = 0,
) -> PageResult:
    """Read one page.

    ``template_hint`` is the expected template. With ``enforce_template`` a sheet whose QR code names a
    different template is rejected; without it the hint is only a fallback for sheets without a readable QR.
    """
    result = PageResult(ok=False, template=None, page_number=None, source_index=source_index)

    input_quality = check_input(image_bgr)
    if input_quality.reject:
        result.quality = input_quality
        return _reject(result, *input_quality.problems)

    if scanner_mode:
        # Keep the original camera "document scanner" preview; registration still runs on the raw photo.
        result.scanner_bgr = scan_document(image_bgr, output_size=(PAGE_WIDTH, PAGE_HEIGHT)).image_bgr

    reg = register(image_bgr)
    if not reg.homographies and result.scanner_bgr is not None:
        reg2 = register(result.scanner_bgr)
        if reg2.homographies:
            image_bgr, reg = result.scanner_bgr, reg2
    result.registration = reg
    if not reg.homographies:
        return _reject(result, *reg.messages)

    # Orientation + template identification. QR position disambiguates 0/180 degrees.
    candidates: list[tuple[np.ndarray, np.ndarray, str | None, bool]] = []
    default_box = template_hint.qr_box if template_hint is not None else (1415, 175, 1565, 325)
    for H in reg.homographies:
        canonical = warp_to_canonical(image_bgr, H)
        payload, in_box = _decode_qr(canonical, default_box)
        candidates.append((H, canonical, payload, in_box))
        if payload and in_box:
            break

    chosen = next((c for c in candidates if c[2] and c[3]), None)
    template = template_hint
    page_number = 1
    if chosen is not None:
        info = parse_qr_payload(chosen[2])
        if info is None:
            chosen = None
        else:
            result.qr_found = True
            qr_template = template_from_qr(info)
            if qr_template is None:
                return _reject(result, "The sheet's QR code is not from a supported template version.")
            if template_hint is not None and enforce_template and qr_template.template_id != template_hint.template_id:
                return _reject(
                    result,
                    f"The uploaded sheet appears to use a different exam template (sheet {qr_template.template_id}, "
                    f"expected {template_hint.template_id}).",
                )
            if template_hint is not None and qr_template.template_id == template_hint.template_id:
                qr_template = template_hint  # keep the teacher's title and metadata
            template = qr_template
            page_number = info.page_number
            if template.n_pages != info.n_pages:
                return _reject(result, "The QR code page count does not match the template.")

    if template is None:
        reasons = []
        if reg.method not in ("markers", "markers-3"):
            reasons.append(MARKERS_MISSING)
        sharp = page_sharpness(candidates[0][1]) if candidates else float("nan")
        if np.isnan(sharp):
            reasons.append(WASHED_OUT)
        elif sharp < BLUR_REJECT:
            reasons.append("Page is too blurry to grade reliably (even the QR code is unreadable). Hold the camera steady and let it focus.")
        reasons.append("Could not read the sheet's QR code to identify the template. Choose the template manually and try again.")
        return _reject(result, *reasons)

    result.template = template
    result.page_number = page_number
    if page_number > template.n_pages:
        return _reject(result, f"Page {page_number} does not exist in template {template.template_id}.")

    norm: NormalizedPage | None = None
    grid: GridFit | None = None
    if chosen is not None:
        H, canonical = chosen[0], chosen[1]
        norm = normalize_page(canonical, template.bubble_radius)
        grid = _fit(template, page_number, norm)
    else:
        # No QR (legacy sheet or unreadable code): choose the orientation whose bubble pattern fits best.
        best = None
        for H_c, canonical_c, _, _ in candidates:
            norm_c = normalize_page(canonical_c, template.bubble_radius)
            grid_c = _fit(template, page_number, norm_c)
            key = (grid_c.ok, grid_c.confidence, grid_c.stats.get("inlier_fraction", 0.0))
            if best is None or key > best[0]:
                best = (key, H_c, canonical_c, norm_c, grid_c)
        _, H, canonical, norm, grid = best
        if template.has_qr:
            result.warnings.append("The QR code could not be read; the selected template was used and confirmed by the bubble pattern.")

    result.homography = H
    result.canonical_bgr = canonical
    result.normalized = norm
    result.grid = grid
    result.warnings.extend(reg.messages)
    missing = missing_corner_name(reg, H)
    if missing:
        result.warnings.append(f"The {missing} registration marker is missing or damaged; alignment was recovered from the other three markers and the bubble pattern.")
    result.warnings.extend(m for m in grid.messages if "different exam template" not in m)

    template_conf = 1.0 if result.qr_found else 0.85
    mismatch = any("different exam template" in m for m in grid.messages)
    if mismatch and reg.method not in ("markers", "markers-3"):
        # With weak registration a bad grid fit is more likely a cropped or distorted photo than a wrong template.
        return _reject(result, MARKERS_MISSING, "The answer grid could not be matched to the template. Please retake the photo.")
    if mismatch:
        return _reject(result, *[m for m in grid.messages if "different exam template" in m])
    if not grid.ok:
        extra = [MARKERS_MISSING] if reg.method not in ("markers", "markers-3") else []
        return _reject(result, *extra, *(grid.messages or ["The answer grid could not be located reliably. Please retake the photo."]))

    radius = template.bubble_radius * grid.radius_scale
    feats, info = extract_features(norm, grid.centers, radius)
    n_c = template.n_choices
    cal, scores = calibrate(feats, n_c, info)
    feats[:, -1] = scores
    result.calibration = cal

    # Bubble size in the original photo, for the "too far away" check.
    photo_radius = 0.0
    if reg.marker_side_px:
        photo_radius = template.bubble_radius * reg.marker_side_px / 70.0
    quality = check_page(
        norm,
        grid.centers,
        radius,
        corner_angles=reg.corner_angles,
        bubble_radius_in_photo=photo_radius,
        paper_noise=cal.paper_noise,
        ring_baseline=cal.ring_baseline,
    )
    result.quality = quality
    result.warnings.extend(quality.warnings)

    angles = reg.corner_angles
    perspective_conf = float(np.clip(1.0 - max(0.0, max(abs(a - 90) for a in angles) - 15) / 40.0, 0.0, 1.0)) if angles else 0.7
    result.confidence_parts = {
        "page_detection": reg.confidence,
        "perspective": perspective_conf,
        "template_match": template_conf,
        "bubble_grid": grid.confidence,
        "image_quality": quality.score,
    }
    if quality.reject:
        return _reject(result, *quality.problems)
    if grid.confidence < MIN_PAGE_CONFIDENCE:
        if reg.method not in ("markers", "markers-3"):
            result.problems.append(MARKERS_MISSING)
        return _reject(result, "The bubble grid could not be confirmed with enough confidence to grade this page. Please retake the photo.")

    layout = template.page(page_number)
    page_conf = float(np.clip(min(reg.confidence, perspective_conf, grid.confidence, quality.score + 0.3), 0.0, 1.0))
    rows = feats.reshape(layout.n_rows, n_c, -1)
    centers = grid.centers.reshape(layout.n_rows, n_c, 2)
    inlier_rows = grid.inlier.reshape(layout.n_rows, n_c)
    span = max(cal.filled_level - cal.empty_level, 0.25)
    for r, q in enumerate(layout.questions):
        row_feats = rows[r]
        row_scores = row_feats[:, -1]
        # Rows far from any confirmed bubble, or washed out by glare, get less alignment trust.
        support = float(inlier_rows[r].mean())
        washed = bool(np.any(row_feats[:, F["ring_mean"]] < 0.35 * cal.ring_baseline) and np.all(row_scores < cal.empty_level + 0.3 * span))
        row_quality = page_conf * (0.85 + 0.15 * support) * (0.6 if washed else 1.0)
        decision = classify_row(row_scores, row_feats, template.choices, cal, answer_key_mode=answer_key_mode, alignment_quality=row_quality)
        notes = list(decision.notes)
        if washed:
            notes.append("Glare or overexposure on this row.")
        bubbles = [
            BubbleDetection(
                choice=template.choices[c],
                center=(float(centers[r, c, 0]), float(centers[r, c, 1])),
                score=float(row_scores[c]),
                level=float((row_scores[c] - cal.empty_level) / span),
                features={name: float(row_feats[c, k]) for k, name in enumerate(FEATURE_NAMES)},
            )
            for c in range(n_c)
        ]
        result.questions.append(
            QuestionResult(
                question=q,
                page=page_number,
                status=decision.status,
                selected=decision.selected,
                marked=decision.marked,
                confidence=decision.confidence,
                reason=decision.reason,
                notes=notes,
                suggestion=decision.suggestion,
                choices=template.choices,
                bubbles=bubbles,
            )
        )
    result.ok = True
    return result


@dataclass
class ExamScan:
    template: Template | None
    pages: list[PageResult]
    questions: list[QuestionResult]
    problems: list[str]
    warnings: list[str]
    missing_pages: list[int]

    @property
    def ok(self) -> bool:
        return self.template is not None and not self.problems and not self.missing_pages

    @property
    def detections(self) -> list[QuestionResult]:
        return self.questions


def process_exam(
    images: Sequence[tuple[np.ndarray, bool]],
    *,
    template_hint: Template | None = None,
    enforce_template: bool = True,
    answer_key_mode: bool = False,
) -> ExamScan:
    """Process every uploaded page and merge them into one exam result.

    ``images`` holds (image, scanner_mode) pairs; pages may arrive in any order.
    """
    pages: list[PageResult] = []
    template = template_hint
    enforce = enforce_template
    for index, (image, scanner_mode) in enumerate(images):
        page = process_page(
            image,
            template_hint=template,
            enforce_template=enforce,
            answer_key_mode=answer_key_mode,
            scanner_mode=scanner_mode,
            source_index=index,
        )
        pages.append(page)
        if page.ok and (not enforce or template is None):
            # The first page read decides the template; every other page must match it.
            template, enforce = page.template, True

    problems: list[str] = []
    warnings: list[str] = []
    by_number: dict[int, PageResult] = {}
    for page in pages:
        tag = f"Upload {page.source_index + 1}" + (f" (page {page.page_number})" if page.page_number else "")
        if not page.ok:
            problems.extend(f"{tag}: {p}" for p in page.problems)
            continue
        warnings.extend(f"{tag}: {w}" for w in page.warnings)
        if page.page_number in by_number:
            warnings.append(f"Page {page.page_number} was uploaded more than once; the clearer scan was kept.")
            if page.confidence <= by_number[page.page_number].confidence:
                continue
        by_number[page.page_number] = page

    missing: list[int] = []
    questions: list[QuestionResult] = []
    if template is not None:
        for n in range(1, template.n_pages + 1):
            if n not in by_number:
                missing.append(n)
                problems.append(f"Page {n} of the exam is missing.")
        for n in sorted(by_number):
            questions.extend(by_number[n].questions)
        questions.sort(key=lambda q: q.question)
    return ExamScan(template=template, pages=pages, questions=questions, problems=problems, warnings=warnings, missing_pages=missing)


__all__ = [
    "BLANK",
    "CLEAR",
    "MULTIPLE",
    "REVIEW",
    "BubbleDetection",
    "QuestionResult",
    "PageResult",
    "ExamScan",
    "process_page",
    "process_exam",
]
