"""Scenario tests for the OMR engine.

Run with:  python -m tests.test_engine        (no extra packages needed)
or:        python -m pytest tests             (if pytest is installed)
"""

from __future__ import annotations

import sys
import time
import traceback

import cv2
import numpy as np

from omr_engine.classifier import BLANK, CLEAR, MULTIPLE, REVIEW
from omr_engine.grading import grade_exam
from omr_engine.image_io import decode_input
from omr_engine.pipeline import process_exam, process_page
from omr_engine.sheet_generator import pages_to_pdf, render_all
from omr_engine.template import build_template, legacy_template, parse_qr_payload, template_from_qr

from . import synth as S

# Rows with special marks; every other question gets one solid fill.
SPECIAL = {
    2: ([], BLANK, None),
    3: ([("A", "solid"), ("C", "solid")], MULTIPLE, "A+C"),
    4: ([("B", "erased"), ("D", "solid")], CLEAR, "D"),
    5: ([("C", "heavy")], CLEAR, "C"),
    6: ([("B", "x")], CLEAR, "B"),
    7: ([("D", "check")], CLEAR, "D"),
    8: ([("A", "dot")], REVIEW, None),
    9: ([("C", "half")], CLEAR, "C"),
    10: ([("B", "line")], {BLANK, REVIEW}, None),
    11: ([("A", "solid"), ("B", "solid"), ("D", "solid")], MULTIPLE, "A+B+D"),
    12: ([("C", "erased")], {BLANK, REVIEW}, None),
}


def answers_for(template, special=True):
    answers, expected = {}, {}
    for q in range(1, template.n_questions + 1):
        if special and q in SPECIAL:
            marks, status, answer = SPECIAL[q]
            answers[q] = marks
            expected[q] = (status, answer)
        else:
            letter = template.choices[(q * 7 + q // 3) % template.n_choices]
            answers[q] = [(letter, "solid")]
            expected[q] = (CLEAR, letter)
    return answers, expected


def check_page(result, expected, *, allow_review=0, label=""):
    assert result.ok, f"{label}: page rejected: {result.problems}"
    wrong = []
    reviews = 0
    for q in result.questions:
        status, answer = expected[q.question]
        statuses = status if isinstance(status, set) else {status}
        if q.status == REVIEW and REVIEW not in statuses:
            # Review is never wrong, but it costs the teacher time; budgeted per scenario.
            reviews += 1
            continue
        if q.status not in statuses or (answer is not None and q.answer != answer):
            wrong.append(f"Q{q.question}: got {q.label} ({q.confidence:.0f}%), expected {status} {answer or ''} [{q.reason}]")
    assert not wrong, f"{label}: misclassified:\n  " + "\n  ".join(wrong)
    assert reviews <= allow_review, f"{label}: {reviews} unexpected review rows (allowed {allow_review})"
    return reviews


def sheet(nq=30, nc=4, page=1, special=True):
    t = build_template(nq, nc, title="Scenario test")
    answers, expected = answers_for(t, special)
    return t, S.make_sheet(t, page, answers), expected


# ----------------------------- scenarios ------------------------------------


def test_clean_scan_all_mark_types():
    t, img, exp = sheet()
    check_page(process_page(img), exp, label="clean")


def test_phone_photo_rotation_perspective_tint_shadow():
    t, img, exp = sheet()
    photo = S.jpeg(S.noise(S.lighting(S.photo(img, rotate_deg=4, tilt=0.05), gain=(1.0, 0.93, 0.72), shadow=0.4), 4), 70)
    r = process_page(photo)
    check_page(r, exp, allow_review=1, label="phone photo")


def test_rotated_page_large_angle():
    t, img, exp = sheet()
    check_page(process_page(S.photo(img, rotate_deg=-12)), exp, allow_review=1, label="rotated 12deg")


def test_upside_down_photo():
    t, img, exp = sheet()
    check_page(process_page(S.photo(img, rotate_deg=178)), exp, allow_review=1, label="upside down")


def test_sideways_photo():
    t, img, exp = sheet()
    check_page(process_page(S.photo(img, rotate_deg=91, canvas=(3200, 2600))), exp, allow_review=1, label="sideways")


def test_strong_perspective():
    t, img, exp = sheet()
    check_page(process_page(S.photo(img, tilt=0.12, rotate_deg=2)), exp, allow_review=1, label="trapezoid")


def test_gray_paper_and_blue_light():
    t, img, exp = sheet()
    check_page(process_page(S.lighting(S.photo(img), gray=0.7, gain=(1.1, 0.95, 0.8))), exp, allow_review=1, label="gray paper")
    check_page(process_page(S.lighting(S.photo(img), gain=(0.75, 0.9, 1.0))), exp, allow_review=1, label="blue light")


def test_mild_blur_ok_heavy_blur_rejected():
    t, img, exp = sheet()
    check_page(process_page(S.blur(S.photo(img), 1.5)), exp, allow_review=2, label="mild blur")
    r = process_page(S.blur(S.photo(img), 7.0))
    assert not r.ok, "heavy blur should be rejected"
    assert any("blurry" in p or "grid" in p or "bubble" in p for p in r.problems), r.problems


def test_printer_scale_and_shift():
    t, img, exp = sheet()
    check_page(process_page(S.printer_scale(img, 0.96, shift=(12, -9))), exp, label="printer 96%")
    check_page(process_page(S.photo(S.printer_scale(img, 1.03))), exp, allow_review=1, label="printer 103%")


def test_paper_curl_local_correction():
    t, img, exp = sheet()
    r = process_page(S.photo(S.curl(img, 7.0)))
    check_page(r, exp, allow_review=1, label="curl")
    assert r.grid.stats["max_local_offset_px"] > 0.5


def test_one_marker_missing_recovers():
    t, img, exp = sheet()
    r = process_page(S.photo(S.remove_marker(img, 2), rotate_deg=3))
    check_page(r, exp, allow_review=1, label="missing marker")
    assert any("bottom-right" in w for w in r.warnings), r.warnings


def test_cropped_page_rejected_with_reason():
    t, img, exp = sheet()
    r = process_page(S.crop(img, 0.25))
    assert not r.ok
    assert r.problems


def test_light_pencil_whole_sheet():
    t = build_template(20, 4)
    answers = {q: [(t.choices[q % 4], "light")] for q in range(1, 21)}
    answers[5] = []
    r = process_page(S.photo(S.make_sheet(t, 1, answers)))
    assert r.ok, r.problems
    for q in r.questions:
        if q.question == 5:
            assert q.status == BLANK, q.label
        else:
            assert q.status == CLEAR and q.selected == t.choices[q.question % 4], (q.question, q.label, q.reason)


def test_sizes_and_choice_counts():
    for nq, nc in ((10, 4), (20, 5), (50, 4), (50, 5), (90, 5), (100, 4)):
        t = build_template(nq, nc)
        answers, exp = answers_for(t, special=False)
        for page in t.pages:
            img = S.make_sheet(t, page.page_number, answers)
            r = process_page(S.photo(img, rotate_deg=2, tilt=0.03))
            check_page(r, exp, label=f"{nq}x{nc} p{page.page_number}")


def test_multi_page_exam_and_missing_page():
    t = build_template(200, 5)
    assert t.n_pages >= 2
    answers, exp = answers_for(t, special=False)
    imgs = [S.photo(S.make_sheet(t, p.page_number, answers)) for p in t.pages]
    scan = process_exam([(im, False) for im in reversed(imgs)])
    assert scan.ok, scan.problems
    assert [q.question for q in scan.questions] == list(range(1, 201))
    key = [exp[q][1] for q in range(1, 201)]
    grade = grade_exam(scan.questions, key, overrides={})
    assert grade["score"] == 200, grade["score"]
    partial = process_exam([(imgs[0], False)])
    assert partial.missing_pages == list(range(2, t.n_pages + 1))
    assert any("Page 2 of the exam is missing" in p for p in partial.problems)


def test_wrong_template_detected():
    t1 = build_template(30, 4)
    t2 = build_template(30, 5)
    answers, _ = answers_for(t1, special=False)
    r = process_page(S.make_sheet(t1, 1, answers), template_hint=t2)
    assert not r.ok and any("different exam template" in p for p in r.problems), r.problems


def test_wrong_template_detected_without_qr():
    # Legacy sheet scanned against a new template: no QR, so the bubble pattern must reveal the mismatch.
    from sheet_template import create_sheet

    img = create_sheet(list("ABCDABCDABCDABCDABCD"))
    r = process_page(img, template_hint=build_template(20, 5))
    assert not r.ok, "a legacy sheet must not be graded against a 5-choice template"


def test_legacy_sheet_still_works():
    from sheet_template import create_sheet

    key = list("ABCDABCDABCDABCDABCD")
    r = process_page(S.photo(create_sheet(key)), template_hint=legacy_template(), answer_key_mode=True)
    assert r.ok, r.problems
    assert [q.selected for q in r.questions] == key


def test_fallback_template_does_not_override_qr():
    t1 = build_template(30, 4)
    answers, exp = answers_for(t1, special=False)
    img = S.make_sheet(t1, 1, answers)
    scan = process_exam([(img, False)], template_hint=legacy_template(), enforce_template=False)
    assert scan.ok and scan.template.template_id == t1.template_id, scan.problems
    strict = process_exam([(img, False)], template_hint=legacy_template(), enforce_template=True)
    assert not strict.ok


def test_answer_key_mode_requires_single_answers():
    t, img, exp = sheet(20, 4)
    r = process_page(img, answer_key_mode=True)
    assert r.ok
    by_q = {q.question: q for q in r.questions}
    assert by_q[2].status == REVIEW  # blank key row
    assert by_q[3].status == REVIEW  # multiple key row
    assert by_q[1].status == CLEAR


def test_pdf_input_multi_page():
    t = build_template(150, 4)
    answers, exp = answers_for(t, special=False)
    pages = [S.make_sheet(t, p.page_number, answers) for p in t.pages]
    pdf = pages_to_pdf(pages)
    images = decode_input(pdf)
    assert len(images) == t.n_pages
    scan = process_exam([(im, False) for im in images])
    assert scan.ok, scan.problems
    assert all(q.answer == exp[q.question][1] for q in scan.questions)


def test_qr_payload_roundtrip_has_no_answers():
    t = build_template(40, 5)
    payload = t.qr_payload(1)
    info = parse_qr_payload(payload)
    assert template_from_qr(info).template_id == t.template_id
    assert payload.count("|") == 6  # template id, page, counts only


def test_row_classifier_examples_from_spec():
    from omr_engine.classifier import classify_row
    from omr_engine.features import FEATURE_NAMES, F, Calibration

    cal = Calibration(empty_level=0.05, empty_spread=0.015, filled_level=0.66, ring_baseline=0.4, paper_noise=0.01, pixel_threshold=0.12, n_filled_reference=10)
    choices = ("A", "B", "C", "D")

    def run(values):
        feats = np.zeros((4, len(FEATURE_NAMES)), np.float32)
        feats[:, F["extent"]] = 0.9
        feats[:, F["ink_intensity"]] = 0.5
        return classify_row(np.array(values, np.float32), feats, choices, cal)

    assert run([0.04, 0.05, 0.63, 0.06]).selected == "C"
    assert run([0.06, 0.07, 0.09, 0.06]).status == BLANK
    d = run([0.51, 0.07, 0.48, 0.06])
    assert d.status == MULTIPLE and d.marked == ("A", "C")
    d = run([0.20, 0.69, 0.07, 0.06])
    assert d.status == CLEAR and d.selected == "B" and d.notes
    assert run([0.31, 0.35, 0.27, 0.29]).status == REVIEW
    d = run([0.58, 0.07, 0.08, 0.54])
    assert d.status == MULTIPLE and d.marked == ("A", "D")
    d = run([0.48, 0.07, 0.09, 0.44])
    assert d.status == MULTIPLE and d.marked == ("A", "D")


def test_original_mvp_api_still_works():
    import test_mvp

    test_mvp.main()


def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    if len(sys.argv) > 1:
        tests = [(n, f) for n, f in tests if any(a in n for a in sys.argv[1:])]
    failed = 0
    for name, fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {name} ({time.time() - t0:.1f}s)")
        except Exception:  # noqa: BLE001 - report every failure
            failed += 1
            print(f"FAIL {name} ({time.time() - t0:.1f}s)")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
