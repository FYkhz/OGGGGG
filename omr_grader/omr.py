"""Original fixed-layout reader for the 20-question prototype sheet.

The app now uses the template-driven engine in ``omr_engine`` (which also reads
this original sheet through ``omr_engine.template.legacy_template``). This module
is kept unchanged so existing scripts and ``test_mvp.py`` keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import cv2
import numpy as np

from document_scanner import scan_document


PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
MARKER_SIZE = 70
MARKER_MARGIN = 75
MARKER_CENTERS = np.array(
    [
        [MARKER_MARGIN + MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, MARKER_MARGIN + MARKER_SIZE / 2],
        [PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
        [MARKER_MARGIN + MARKER_SIZE / 2, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE / 2],
    ],
    dtype=np.float32,
)

QUESTION_COUNT = 20
CHOICES = ("A", "B", "C", "D")
BUBBLE_X = np.array([690, 900, 1110, 1320], dtype=np.int32)
QUESTION_START_Y = 430
QUESTION_GAP_Y = 78
BUBBLE_RADIUS = 28
INNER_RADIUS = 17


@dataclass(frozen=True)
class QuestionDetection:
    question: int
    densities: tuple[float, float, float, float]
    selected: str | None
    status: str
    confidence: float


@dataclass(frozen=True)
class ScanResult:
    aligned_bgr: np.ndarray
    threshold: np.ndarray
    detections: tuple[QuestionDetection, ...]
    marker_detection_used: bool
    warning: str | None = None
    scanner_bgr: np.ndarray | None = None
    scanner_page_found: bool = False


class OMRProcessingError(RuntimeError):
    pass


def decode_image_bytes(data: bytes) -> np.ndarray:
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise OMRProcessingError("The uploaded file could not be decoded as an image.")
    return image


def _order_points(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]  # top-left
    ordered[2] = pts[np.argmax(sums)]  # bottom-right
    ordered[1] = pts[np.argmin(diffs)]  # top-right
    ordered[3] = pts[np.argmax(diffs)]  # bottom-left
    return ordered


def _find_registration_markers(image_bgr: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape
    image_area = float(h * w)
    candidates: list[tuple[float, float, float]] = []

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.00035 or area > image_area * 0.05:
            continue
        x, y, cw, ch = cv2.boundingRect(contour)
        if cw == 0 or ch == 0:
            continue
        aspect = cw / float(ch)
        extent = area / float(cw * ch)
        if not 0.70 <= aspect <= 1.30:
            continue
        if extent < 0.72:
            continue
        cx = x + cw / 2.0
        cy = y + ch / 2.0
        candidates.append((cx, cy, area))

    if len(candidates) < 4:
        return None

    corner_targets = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    chosen: list[tuple[float, float]] = []
    used: set[int] = set()

    for target in corner_targets:
        scored: list[tuple[float, int]] = []
        for index, (cx, cy, area) in enumerate(candidates):
            if index in used:
                continue
            dist = float(np.linalg.norm(np.array([cx, cy]) - target))
            area_bonus = np.sqrt(area) * 0.20
            scored.append((dist - area_bonus, index))
        if not scored:
            return None
        _, best_index = min(scored)
        used.add(best_index)
        cx, cy, _ = candidates[best_index]
        chosen.append((cx, cy))

    ordered = _order_points(np.array(chosen, dtype=np.float32))

    # Reject markers that are clustered or clearly not near page corners.
    tl, tr, br, bl = ordered
    horizontal = min(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    vertical = min(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    if horizontal < w * 0.45 or vertical < h * 0.45:
        return None

    return ordered


def _fallback_page_warp(image_bgr: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    edges = cv2.dilate(edges, np.ones((5, 5), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:12]

    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) != 4:
            continue
        if cv2.contourArea(approx) < image_bgr.shape[0] * image_bgr.shape[1] * 0.25:
            continue
        source = _order_points(approx.reshape(4, 2))
        destination = np.array(
            [[0, 0], [PAGE_WIDTH - 1, 0], [PAGE_WIDTH - 1, PAGE_HEIGHT - 1], [0, PAGE_HEIGHT - 1]],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(image_bgr, matrix, (PAGE_WIDTH, PAGE_HEIGHT)), (
            "Registration squares were not found; the app used the page boundary instead. "
            "For the most reliable grading, keep all four black corner squares visible."
        )

    resized = cv2.resize(image_bgr, (PAGE_WIDTH, PAGE_HEIGHT), interpolation=cv2.INTER_AREA)
    return resized, (
        "The app could not detect the corner squares or page boundary, so it resized the image directly. "
        "Results may be inaccurate; retake the photo with the full sheet visible."
    )


def align_sheet(image_bgr: np.ndarray) -> tuple[np.ndarray, bool, str | None]:
    markers = _find_registration_markers(image_bgr)
    if markers is None:
        aligned, warning = _fallback_page_warp(image_bgr)
        return aligned, False, warning

    matrix = cv2.getPerspectiveTransform(markers, MARKER_CENTERS)
    aligned = cv2.warpPerspective(
        image_bgr,
        matrix,
        (PAGE_WIDTH, PAGE_HEIGHT),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )
    return aligned, True, None


def _prepare_threshold(aligned_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Adaptive threshold handles mild shadows and inexpensive phone cameras better than a global threshold.
    threshold = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        41,
        12,
    )
    threshold = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return threshold


def bubble_centers() -> list[list[tuple[int, int]]]:
    rows: list[list[tuple[int, int]]] = []
    for question_index in range(QUESTION_COUNT):
        y = QUESTION_START_Y + question_index * QUESTION_GAP_Y
        rows.append([(int(x), int(y)) for x in BUBBLE_X])
    return rows


def _density_for_circle(binary: np.ndarray, center: tuple[int, int], radius: int = INNER_RADIUS) -> float:
    mask = np.zeros(binary.shape, dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    dark_pixels = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))
    total_pixels = cv2.countNonZero(mask)
    return float(dark_pixels / total_pixels) if total_pixels else 0.0


def classify_densities(
    densities: Sequence[float],
    *,
    answer_key_mode: bool,
) -> tuple[str | None, str, float]:
    values = np.asarray(densities, dtype=np.float32)
    order = np.argsort(values)[::-1]
    top_index = int(order[0])
    second_index = int(order[1])
    top = float(values[top_index])
    second = float(values[second_index])
    margin = top - second

    # Confidence expresses separation between the two strongest choices, not probability.
    confidence = float(np.clip((margin / 0.22) * 100.0, 0.0, 100.0))

    blank_threshold = 0.10 if answer_key_mode else 0.085
    clear_threshold = 0.20 if answer_key_mode else 0.17
    second_mark_threshold = 0.14 if answer_key_mode else 0.13
    minimum_margin = 0.085 if answer_key_mode else 0.070

    if top < blank_threshold:
        return None, "blank", 0.0

    if second >= second_mark_threshold and (margin < 0.13 or second / max(top, 1e-6) > 0.62):
        return None, "multiple", confidence

    if top < clear_threshold or margin < minimum_margin:
        return None, "unclear", confidence

    return CHOICES[top_index], "clear", confidence


def scan_sheet(
    image_bgr: np.ndarray,
    *,
    answer_key_mode: bool,
    scanner_mode: bool = False,
) -> ScanResult:
    scanner_bgr = None
    scanner_page_found = False
    scanner_warning = None

    working_image = image_bgr
    if scanner_mode:
        scanned = scan_document(image_bgr, output_size=(PAGE_WIDTH, PAGE_HEIGHT))
        scanner_bgr = scanned.image_bgr
        scanner_page_found = scanned.page_found
        working_image = scanned.image_bgr
        if not scanned.page_found:
            scanner_warning = scanned.message

    aligned, marker_used, warning = align_sheet(working_image)
    if scanner_warning and warning:
        warning = scanner_warning + " " + warning
    elif scanner_warning:
        warning = scanner_warning
    threshold = _prepare_threshold(aligned)

    detections: list[QuestionDetection] = []
    for q_index, row in enumerate(bubble_centers(), start=1):
        densities = tuple(_density_for_circle(threshold, center) for center in row)
        selected, status, confidence = classify_densities(densities, answer_key_mode=answer_key_mode)
        detections.append(
            QuestionDetection(
                question=q_index,
                densities=densities,
                selected=selected,
                status=status,
                confidence=confidence,
            )
        )

    return ScanResult(
        aligned_bgr=aligned,
        threshold=threshold,
        detections=tuple(detections),
        marker_detection_used=marker_used,
        warning=warning,
        scanner_bgr=scanner_bgr,
        scanner_page_found=scanner_page_found,
    )


def crop_question(aligned_bgr: np.ndarray, question_number: int) -> np.ndarray:
    index = question_number - 1
    y = QUESTION_START_Y + index * QUESTION_GAP_Y
    y1 = max(0, y - 38)
    y2 = min(PAGE_HEIGHT, y + 38)
    x1 = 430
    x2 = 1450
    return aligned_bgr[y1:y2, x1:x2].copy()


def draw_overlay(
    aligned_bgr: np.ndarray,
    detections: Sequence[QuestionDetection],
    *,
    key_answers: Sequence[str] | None = None,
    overrides: dict[int, str | None] | None = None,
) -> np.ndarray:
    overlay = aligned_bgr.copy()
    overrides = overrides or {}

    for detection in detections:
        q = detection.question
        y = QUESTION_START_Y + (q - 1) * QUESTION_GAP_Y
        x1, x2 = 430, 1450
        y1, y2 = y - 34, y + 34

        selected = overrides.get(q, detection.selected)
        if selected == "Blank":
            selected = None

        if key_answers is None:
            if detection.status == "clear" or q in overrides:
                color = (50, 170, 70)  # green
            elif detection.status == "blank":
                color = (145, 145, 145)  # gray
            else:
                color = (0, 165, 255)  # orange
        else:
            if detection.status not in {"clear"} and q not in overrides:
                color = (0, 165, 255)
            elif selected is None:
                color = (145, 145, 145)
            elif selected == key_answers[q - 1]:
                color = (50, 170, 70)
            else:
                color = (45, 45, 220)  # red in BGR

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 4)
        label = selected if selected is not None else detection.status.upper()
        cv2.putText(
            overlay,
            f"Q{q}: {label}",
            (1455, y + 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
            cv2.LINE_AA,
        )

    return overlay


def answers_from_detections(
    detections: Sequence[QuestionDetection],
    overrides: dict[int, str | None] | None = None,
) -> list[str | None]:
    overrides = overrides or {}
    answers: list[str | None] = []
    for detection in detections:
        value = overrides.get(detection.question, detection.selected)
        if value == "Blank":
            value = None
        answers.append(value)
    return answers


def grade_answers(student_answers: Sequence[str | None], key_answers: Sequence[str]) -> dict[str, object]:
    if len(student_answers) != len(key_answers):
        raise ValueError("Student answers and answer key must have the same number of questions.")

    rows: list[dict[str, object]] = []
    correct = 0
    blank = 0
    for index, (student, correct_answer) in enumerate(zip(student_answers, key_answers), start=1):
        is_correct = student == correct_answer
        if is_correct:
            correct += 1
        if student is None:
            blank += 1
        rows.append(
            {
                "Question": index,
                "Student answer": student or "Blank",
                "Correct answer": correct_answer,
                "Result": "Correct" if is_correct else ("Blank" if student is None else "Incorrect"),
            }
        )

    total = len(key_answers)
    return {
        "score": correct,
        "total": total,
        "percentage": (correct / total * 100.0) if total else 0.0,
        "blank": blank,
        "incorrect": total - correct - blank,
        "rows": rows,
    }


def encode_png(image_bgr: np.ndarray) -> bytes:
    ok, encoded = cv2.imencode(".png", image_bgr)
    if not ok:
        raise OMRProcessingError("Could not encode the processed image.")
    return encoded.tobytes()


def detection_rows(detections: Iterable[QuestionDetection]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for detection in detections:
        rows.append(
            {
                "Question": detection.question,
                "Detected": detection.selected or detection.status.title(),
                "Status": detection.status.title(),
                "Confidence": round(detection.confidence, 1),
                "A density": round(detection.densities[0], 3),
                "B density": round(detection.densities[1], 3),
                "C density": round(detection.densities[2], 3),
                "D density": round(detection.densities[3], 3),
            }
        )
    return rows
