"""Teacher overlay, developer debug overlay, review crops and metric export."""

from __future__ import annotations

from typing import Mapping, Sequence

import cv2
import numpy as np

from .classifier import BLANK, CLEAR, MULTIPLE, REVIEW
from .grading import BLANK_LABEL, final_answer
from .pipeline import ExamScan, PageResult, QuestionResult

GREEN = (60, 170, 60)
RED = (50, 50, 220)
ORANGE = (0, 150, 255)
GRAY = (150, 150, 150)
BLUE = (200, 120, 20)
PURPLE = (170, 60, 170)


def _row_box(q: QuestionResult, radius: float) -> tuple[int, int, int, int]:
    xs = [b.center[0] for b in q.bubbles]
    ys = [b.center[1] for b in q.bubbles]
    pad = radius * 1.35
    return int(min(xs) - pad), int(min(ys) - pad), int(max(xs) + pad), int(max(ys) + pad)


def _status_color(q: QuestionResult, answer: str | None, source: str, key_answer: str | None) -> tuple[int, int, int]:
    if source == "unresolved":
        return ORANGE
    if key_answer is None:
        if source == "teacher" or q.status == CLEAR:
            return GREEN
        if q.status == MULTIPLE:
            return PURPLE
        return GRAY
    if answer is None:
        return GRAY
    return GREEN if answer == key_answer else RED


def teacher_overlay(
    page: PageResult,
    *,
    key: Mapping[int, str] | None = None,
    overrides: Mapping[int, str | None] | None = None,
) -> np.ndarray:
    """Green = correct/clear, red = incorrect, orange = needs review, gray = blank, purple = multiple."""
    overrides = overrides or {}
    # Draw on the illumination-normalized scan: neutral white paper regardless of camera tint or shadows.
    img = cv2.cvtColor(page.normalized.flat, cv2.COLOR_GRAY2BGR) if page.normalized is not None else page.canonical_bgr.copy()
    radius = page.template.bubble_radius * (page.grid.radius_scale if page.grid else 1.0)
    thick = max(2, int(radius / 8))
    font_scale = max(0.5, radius / 34.0)
    for q in page.questions:
        answer, source = final_answer(q, overrides)
        key_answer = key.get(q.question) if key else None
        color = _status_color(q, answer, source, key_answer)
        x1, y1, x2, y2 = _row_box(q, radius)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
        chosen = set(answer.split("+")) if answer else set()
        for b in q.bubbles:
            if b.choice in chosen:
                cv2.circle(img, (int(b.center[0]), int(b.center[1])), int(radius * 1.15), color, thick)
            if key_answer and b.choice == key_answer and answer != key_answer:
                cv2.circle(img, (int(b.center[0]), int(b.center[1])), int(radius * 1.15), GREEN, max(1, thick // 2))
        if source == "unresolved":
            tag = "?"
        elif answer is None:
            tag = "-"
        else:
            tag = answer.replace("+", "")
        cv2.putText(img, tag, (x2 + 3, int((y1 + y2) / 2 + radius * 0.4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 2, cv2.LINE_AA)
    return img


def debug_overlay(page: PageResult) -> np.ndarray:
    """Developer view: expected vs corrected grid, candidates, outliers, row/column lines, scores."""
    img = cv2.cvtColor(page.normalized.flat, cv2.COLOR_GRAY2BGR) if page.normalized is not None else page.canonical_bgr.copy()
    if page.grid is None or page.template is None:
        return img
    grid = page.grid
    template = page.template
    layout = template.page(page.page_number)
    n_c = template.n_choices
    radius = template.bubble_radius * grid.radius_scale

    # Candidates: inliers small green dots, unexplained detections red crosses.
    matched = {tuple(np.round(p, 1)) for p in grid.observed[grid.inlier]}
    for p in grid.candidates:
        if tuple(np.round(p, 1)) in matched:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 200, 0), -1)
        else:
            cv2.drawMarker(img, (int(p[0]), int(p[1])), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 14, 2)

    centers = grid.centers.reshape(layout.n_rows, n_c, 2)
    # Row and column lines through the corrected grid.
    for r in range(layout.n_rows):
        pts = centers[r].astype(np.int32)
        cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (230, 200, 0), 1, cv2.LINE_AA)
    blocks = np.array(layout.block_index)
    for b in np.unique(blocks):
        rows = np.nonzero(blocks == b)[0]
        for c in range(n_c):
            pts = centers[rows, c].astype(np.int32)
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, (230, 200, 0), 1, cv2.LINE_AA)

    for i, (e, c, off) in enumerate(zip(grid.expected, grid.centers, grid.local_offset)):
        cv2.drawMarker(img, (int(e[0]), int(e[1])), BLUE, cv2.MARKER_CROSS, 10, 1)  # template position
        cv2.circle(img, (int(c[0]), int(c[1])), int(radius * 0.66), (0, 160, 0), 1, cv2.LINE_AA)  # sampled interior
        if np.linalg.norm(off) > 0.3:
            tip = c + off * 6.0  # exaggerated local correction
            cv2.arrowedLine(img, (int(c[0]), int(c[1])), (int(tip[0]), int(tip[1])), (0, 140, 255), 1, cv2.LINE_AA, tipLength=0.3)

    status_colors = {CLEAR: GREEN, BLANK: GRAY, MULTIPLE: PURPLE, REVIEW: ORANGE}
    fs = max(0.35, radius / 60.0)
    for q in page.questions:
        color = status_colors.get(q.status, ORANGE)
        x1, y1, x2, y2 = _row_box(q, radius)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        for b in q.bubbles:
            cv2.putText(img, f"{max(b.level, 0.0):.2f}", (int(b.center[0] - radius), int(b.center[1] + radius * 1.6)), cv2.FONT_HERSHEY_SIMPLEX, fs, (40, 40, 40), 1, cv2.LINE_AA)
        cv2.putText(img, f"{q.confidence:.0f}%", (x2 + 2, y1 + int(radius)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, 1, cv2.LINE_AA)

    s = grid.stats
    lines = [
        f"template {template.template_id} page {page.page_number}  registration={page.registration.method if page.registration else '?'}  model={grid.model_kind}",
        f"rotation {s.get('rotation_deg', 0):+.2f} deg  scale {s.get('scale', 1):.3f}  skew {s.get('skew_deg', 0):+.2f}  offset ({s.get('offset_x', 0):+.1f},{s.get('offset_y', 0):+.1f})",
        f"inliers {s.get('inliers')}/{s.get('expected')}  rms {s.get('rms_px', 0):.2f}px  max local {s.get('max_local_offset_px', 0):.1f}px  candidates {s.get('candidates')}",
        "  ".join(f"{k} {v * 100:.0f}%" for k, v in page.confidence_parts.items()),
    ]
    for i, text in enumerate(lines):
        cv2.putText(img, text, (160, 24 + i * 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 160), 1, cv2.LINE_AA)
    return img


def registration_view(image_bgr: np.ndarray, page: PageResult, max_side: int = 1100) -> np.ndarray:
    """The original photo with the detected marker quad drawn on it."""
    h, w = image_bgr.shape[:2]
    scale = min(1.0, max_side / float(max(h, w)))
    view = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    reg = page.registration
    if reg is not None and reg.quad is not None:
        quad = (reg.quad * scale).astype(np.int32)
        cv2.polylines(view, [quad.reshape(-1, 1, 2)], True, (0, 200, 0), 2, cv2.LINE_AA)
        for i, p in enumerate(quad):
            color = (0, 0, 255) if reg.missing_corner_slot == i else (0, 200, 0)
            cv2.circle(view, (int(p[0]), int(p[1])), 9, color, 2)
    return view


def crop_question(page: PageResult, question: int) -> np.ndarray:
    q = next(x for x in page.questions if x.question == question)
    radius = page.template.bubble_radius * (page.grid.radius_scale if page.grid else 1.0)
    x1, y1, x2, y2 = _row_box(q, radius)
    pad = int(radius * 2.2)  # include the question number label
    base = cv2.cvtColor(page.normalized.flat, cv2.COLOR_GRAY2BGR) if page.normalized is not None else page.canonical_bgr
    h, w = base.shape[:2]
    return base[max(0, y1 - 4) : min(h, y2 + 4), max(0, x1 - pad) : min(w, x2 + 4)].copy()


def metric_rows(scan: ExamScan | Sequence[PageResult]) -> list[dict]:
    pages = scan.pages if isinstance(scan, ExamScan) else scan
    rows = []
    for page in pages:
        if not page.ok:
            continue
        for q in page.questions:
            levels = sorted((b.level for b in q.bubbles), reverse=True)
            ratio = levels[1] / levels[0] if len(levels) > 1 and levels[0] > 1e-6 else 0.0
            for b in q.bubbles:
                f = b.features
                rows.append(
                    {
                        "Page": page.page_number,
                        "Question": q.question,
                        "Choice": b.choice,
                        "Fill score": round(b.score, 3),
                        "Level vs sheet": round(b.level, 3),
                        "Background": round(f.get("local_bg", 0.0), 3),
                        "Inner fill": round(f.get("inner_fill", 0.0), 3),
                        "Center fill": round(f.get("center_fill", 0.0), 3),
                        "Inner darkness": round(f.get("inner_mean", 0.0), 3),
                        "Ring darkness": round(f.get("ring_mean", 0.0), 3),
                        "Spill": round(f.get("spill", 0.0), 3),
                        "Extent": round(f.get("extent", 0.0), 3),
                        "Ink intensity": round(f.get("ink_intensity", 0.0), 3),
                        "Second/best": round(ratio, 3),
                        "Classification": q.label,
                        "Confidence": round(q.confidence, 1),
                        "Reason": q.reason,
                        "X": round(b.center[0], 1),
                        "Y": round(b.center[1], 1),
                    }
                )
    return rows


def question_rows(questions: Sequence[QuestionResult], overrides: Mapping[int, str | None] | None = None) -> list[dict]:
    overrides = overrides or {}
    out = []
    for q in questions:
        answer, source = final_answer(q, overrides)
        out.append(
            {
                "Question": q.question,
                "Detected": q.label if source != "teacher" else (answer or BLANK_LABEL).replace("+", "/") + " (teacher)",
                "Status": q.status.title(),
                "Confidence": f"{q.confidence:.0f}%",
                "Why": q.reason + (" " + " ".join(q.notes) if q.notes else ""),
            }
        )
    return out
