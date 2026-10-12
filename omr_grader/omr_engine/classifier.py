"""Row-level answer classification with dominance analysis and confidence.

Every row is judged as a group against the sheet's own calibration:

    level L = (score - empty_level) / (filled_level - empty_level)

so 0 means "looks like an empty printed bubble on this sheet" and 1 means "looks
like a typical deliberate fill on this sheet". No single fixed darkness
threshold decides an answer; the decision uses the best/second-best pattern,
distance from the empty baseline (in noise units), mark structure and the
page's alignment quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .features import F, Calibration

CLEAR = "clear"
BLANK = "blank"
MULTIPLE = "multiple"
REVIEW = "review"

STRONG_LEVEL = 0.5
WEAK_LEVEL = 0.2
REVIEW_CONFIDENCE = 70.0
KEY_REVIEW_CONFIDENCE = 85.0


@dataclass
class RowDecision:
    status: str
    selected: str | None  # single answer for CLEAR
    marked: tuple[str, ...]  # all choices with deliberate marks (MULTIPLE) or the suggestion (REVIEW)
    confidence: float
    reason: str
    notes: list[str] = field(default_factory=list)
    suggestion: str | None = None


def _ramp(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def classify_row(
    scores: np.ndarray,
    feats: np.ndarray,
    choices: tuple[str, ...],
    cal: Calibration,
    *,
    answer_key_mode: bool = False,
    alignment_quality: float = 1.0,
) -> RowDecision:
    span = max(cal.filled_level - cal.empty_level, 0.25)
    level = (scores - cal.empty_level) / span
    z = (scores - cal.empty_level) / cal.empty_spread
    order = np.argsort(level)[::-1]
    i1, i2 = int(order[0]), int(order[1]) if len(order) > 1 else int(order[0])
    l1, l2 = float(level[i1]), float(level[i2]) if len(order) > 1 else 0.0
    floor = float(level.min())
    notes: list[str] = []

    strong = [i for i in order if level[i] >= STRONG_LEVEL and z[i] >= 8]
    weak = [i for i in order if WEAK_LEVEL <= level[i] < STRONG_LEVEL and z[i] >= 4 and i not in strong]

    extent = feats[:, F["extent"]]
    center_fill = feats[:, F["center_fill"]]
    spill = feats[:, F["spill"]]
    intensity = feats[:, F["ink_intensity"]]
    max_component = feats[:, F["max_component"]]

    def letters(idx) -> tuple[str, ...]:
        return tuple(choices[i] for i in sorted(idx))

    if floor >= 0.25 and len(choices) > 1:
        decision = RowDecision(
            REVIEW,
            None,
            (choices[i1],),
            35.0,
            "Every choice in this row is darker than an empty bubble (smudge, shadow, or scribble).",
            suggestion=choices[i1],
        )
    elif not strong and not weak and (
        (l1 >= 0.1 and center_fill[i1] >= 0.3) or (l1 >= 0.04 and max_component[i1] >= 0.04 and intensity[i1] >= 0.3)
    ):
        decision = RowDecision(
            REVIEW,
            None,
            (choices[i1],),
            40.0,
            (
                f"A pencil stroke runs through {choices[i1]} and beyond the bubble; it may be a stray line."
                if spill[i1] >= 0.25
                else f"A small dot or faint mark near the centre of {choices[i1]}; cannot tell if it is an intended answer."
            ),
            suggestion=choices[i1],
        )
    elif not strong and not weak:
        conf = 99.0 - 35.0 * _ramp(l1, 0.05, WEAK_LEVEL) ** 1.5
        decision = RowDecision(BLANK, None, (), conf, "All choices look like empty bubbles.")
    elif len(strong) >= 2:
        top = level[strong[0]]
        close = [i for i in strong if level[i] >= 0.6 * top]
        if len(close) >= 2:
            second = sorted((level[i] for i in close), reverse=True)[1]
            conf = 70.0 + 29.0 * min(_ramp(second, STRONG_LEVEL, 0.8), _ramp(second / top, 0.6, 0.85) ** 0.5)
            decision = RowDecision(MULTIPLE, None, letters(close), conf, f"Strong marks in {' and '.join(letters(close))}.")
        else:
            decision = RowDecision(
                REVIEW,
                None,
                (choices[strong[0]],),
                45.0,
                f"{choices[strong[0]]} is strongest, but {choices[strong[1]]} also has a substantial mark (erasure or second answer?).",
                suggestion=choices[strong[0]],
            )
    elif len(strong) == 1:
        best = strong[0]
        if not weak:
            conf = 100.0 * min(0.8 + 0.2 * _ramp(l1, STRONG_LEVEL, 0.8), 1.0 - 0.15 * _ramp(l2, 0.08, WEAK_LEVEL))
            decision = RowDecision(CLEAR, choices[best], (choices[best],), conf, f"{choices[best]} clearly dominates the row.")
        else:
            ratio = l2 / max(l1, 1e-6)
            if ratio <= 0.42 and l1 >= 0.7:
                conf = 72.0 + 20.0 * _ramp(0.42 - ratio, 0.0, 0.25)
                notes.append(f"Light mark at {choices[weak[0]]} looks like an erasure.")
                decision = RowDecision(CLEAR, choices[best], (choices[best],), conf, f"{choices[best]} is a strong fill; {choices[weak[0]]} is much weaker.", notes)
            else:
                decision = RowDecision(
                    REVIEW,
                    None,
                    (choices[best],),
                    50.0,
                    f"{choices[best]} is marked, but {choices[weak[0]]} also has a noticeable mark.",
                    suggestion=choices[best],
                )
    else:
        # Only weak evidence: light pencil, a check, an X, a dot, or noise.
        best = weak[0]
        # A check, X or diagonal stroke: real pencil darkness, spans most of the bubble, stays inside it.
        # Erasure residue spans the bubble too but is faint; a dot is dark but tiny.
        deliberate = extent[best] >= 0.6 and intensity[best] >= 0.33 and spill[best] < 0.25
        if len(weak) == 1 and deliberate and l2 < 0.12:
            conf = 74.0 + 12.0 * _ramp(l1, WEAK_LEVEL, STRONG_LEVEL)
            notes.append(f"{choices[best]} has a partial mark (check, X, or light shading), not a full fill.")
            decision = RowDecision(CLEAR, choices[best], (choices[best],), conf, f"{choices[best]} has the only deliberate mark.", notes)
        else:
            why = "a faint or very small mark" if len(weak) == 1 else "several faint marks"
            decision = RowDecision(
                REVIEW,
                None,
                letters(weak),
                40.0,
                f"Only {why} found ({', '.join(letters(weak))}); cannot tell if it is intentional.",
                suggestion=choices[best],
            )

    # Alignment and image quality cap how certain any decision can be.
    decision.confidence = float(np.clip(decision.confidence * (0.55 + 0.45 * alignment_quality), 0.0, 99.0))

    if answer_key_mode:
        if decision.status in (BLANK, MULTIPLE):
            decision = RowDecision(
                REVIEW,
                None,
                decision.marked,
                decision.confidence,
                "An answer key needs exactly one answer per question. " + decision.reason,
                decision.notes,
                suggestion=decision.marked[0] if decision.marked else None,
            )
        elif decision.status == CLEAR and decision.confidence < KEY_REVIEW_CONFIDENCE:
            decision = RowDecision(REVIEW, None, decision.marked, decision.confidence, "Confidence too low for an answer key. " + decision.reason, decision.notes, suggestion=decision.selected)
    elif decision.status != REVIEW and decision.confidence < REVIEW_CONFIDENCE:
        suggestion = decision.selected or (decision.marked[0] if decision.marked else None)
        decision = RowDecision(REVIEW, None, decision.marked, decision.confidence, "Low confidence. " + decision.reason, decision.notes, suggestion=suggestion)
    return decision
