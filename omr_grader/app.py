from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import cv2
import pandas as pd
import streamlit as st

from omr import encode_png
from omr_engine.classifier import REVIEW
from omr_engine.grading import BLANK_LABEL, final_answer, grade_exam
from omr_engine.image_io import InputError, decode_input
from omr_engine.pipeline import process_exam
from omr_engine.sheet_generator import pages_to_pdf, render_all
from omr_engine.template import LEGACY_ID, MAX_CHOICES, MAX_QUESTIONS, MIN_CHOICES, TemplateStore, build_template, legacy_template
from omr_engine.visualize import crop_question, debug_overlay, metric_rows, question_rows, registration_view, teacher_overlay
from sheet_template import create_sheet

APP_DIR = Path(__file__).resolve().parent
STORE = TemplateStore(APP_DIR / "templates")

st.set_page_config(page_title="OMR Grader", page_icon="✅", layout="wide")

st.markdown(
    """
<style>
.block-container {max-width: 1180px; padding-top: 1.7rem; padding-bottom: 3rem;}
.omr-hero {padding: 1.2rem 1.4rem; border: 1px solid #dbe5f4; border-radius: 18px; background: linear-gradient(135deg,#f7fbff,#eef5ff); margin-bottom: 1rem;}
.omr-hero h1 {margin: 0; font-size: 2.25rem; color: #0b3975;}
.omr-hero p {margin: .35rem 0 0; color: #40536d;}
.status-card {padding: .8rem 1rem; border-radius: 12px; border: 1px solid #dfe6ef; background: white;}
.small-muted {font-size: .88rem; color: #66758a;}
[data-testid="stMetric"] {border: 1px solid #dfe6ef; padding: .8rem; border-radius: 12px; background: white;}
div[role="radiogroup"][aria-label="Step"] {gap: .25rem .9rem; flex-wrap: wrap;}
@media (max-width: 640px) {
  .block-container {padding: .8rem .7rem 2rem;}
  .omr-hero {padding: .7rem .9rem; border-radius: 14px; margin-bottom: .6rem;}
  .omr-hero h1 {font-size: 1.5rem;}
  .omr-hero p {font-size: .9rem;}
  [data-testid="stMetric"] {padding: .5rem;}
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div class="omr-hero">
  <h1>OMR Grader</h1>
  <p>Create an exam sheet, scan the answer key, then grade student sheets. Only answers the engine is unsure about need your review.</p>
</div>
""",
    unsafe_allow_html=True,
)


def initialize_state() -> None:
    defaults = {
        "active_template_id": None,
        "exam_title": "OMR ANSWER SHEET",
        "key_scan": None,
        "key_hash": None,
        "key_overrides": {},
        "key": None,  # {"template_id", "answers", "title", "saved_at"}
        "student_scan": None,
        "student_hash": None,
        "student_overrides": {},
        "key_camera_pages": [],
        "student_camera_pages": [],
        "history": [],
        "review_nonce": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


def bgr_to_rgb(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def get_template(template_id: str | None):
    if not template_id:
        return None
    if template_id == LEGACY_ID:
        return legacy_template()
    return STORE.load(template_id)


def template_label(t) -> str:
    if t.legacy:
        return "Original 20-question sheet (no QR code)"
    return t.describe()


CAMERA_TIPS = (
    "The live camera below needs the app to be opened over https (for example when hosted online). "
    "On a phone, the Upload tab's Take Photo option usually gives sharper pictures. "
    "Lay the sheet flat on a darker surface, fill the frame with the page, keep all four black corner squares "
    "and the QR code visible, avoid shadows and glare, and hold the phone steady and parallel to the paper."
)


def collect_inputs(prefix: str) -> list[tuple[bytes, bool, str]]:
    """Return (bytes, from_camera, name) for every uploaded file or captured page."""
    upload_tab, camera_tab = st.tabs(["Upload images or PDF", "Scan with camera"])
    items: list[tuple[bytes, bool, str]] = []
    with upload_tab:
        uploads = st.file_uploader(
            "Upload every page of the sheet (PNG, JPG, or PDF). Pages can be in any order. "
            "On a phone, tap Browse files and choose Take Photo (or Camera) to use the main camera at full resolution.",
            type=["jpg", "jpeg", "png", "pdf"],
            accept_multiple_files=True,
            key=f"{prefix}_upload",
        )
        for up in uploads or []:
            up.seek(0)
            items.append((up.read(), False, up.name))
    with camera_tab:
        st.caption(CAMERA_TIPS)
        pages_key = f"{prefix}_camera_pages"
        camera = st.camera_input("Capture one page at a time", key=f"{prefix}_camera_{st.session_state.review_nonce}")
        if camera is not None:
            camera.seek(0)
            data = camera.read()
            digest = hashlib.sha256(data).hexdigest()
            if digest not in [d for d, _ in st.session_state[pages_key]]:
                st.session_state[pages_key].append((digest, data))
        captured = st.session_state[pages_key]
        if captured:
            c1, c2 = st.columns([3, 1])
            c1.success(f"{len(captured)} page(s) captured. Take another photo for the next page, if any.")
            if c2.button("Clear captures", key=f"{prefix}_clear_captures"):
                st.session_state[pages_key] = []
                st.rerun()
            for i, (_, data) in enumerate(captured):
                items.append((data, True, f"camera page {i + 1}"))
    return items


def run_scan(items, *, template_hint, answer_key_mode: bool, enforce_template: bool):
    images = []
    for data, from_camera, name in items:
        try:
            for page in decode_input(data):
                images.append((page, from_camera))
        except InputError as exc:
            st.error(f"{name}: {exc}")
    if not images:
        return None, []
    scan = process_exam(images, template_hint=template_hint, enforce_template=enforce_template, answer_key_mode=answer_key_mode)
    return scan, [img for img, _ in images]


def show_messages(scan) -> None:
    for problem in scan.problems:
        st.error(problem)
    if scan.warnings:
        with st.expander(f"{len(scan.warnings)} note(s) about image quality and alignment"):
            for w in scan.warnings:
                st.write("• " + w)


def show_debug(scan, originals) -> None:
    if not st.session_state.get("debug_mode"):
        return
    st.markdown("#### Developer debug view")
    for page in scan.pages:
        title = f"Upload {page.source_index + 1}" + (f" · page {page.page_number}" if page.page_number else "") + (" · REJECTED" if not page.ok else "")
        with st.expander(title, expanded=False):
            if page.confidence_parts:
                st.dataframe(
                    pd.DataFrame([{"Check": k.replace("_", " ").title(), "Confidence": f"{v * 100:.0f}%"} for k, v in page.confidence_parts.items()]),
                    hide_index=True,
                    use_container_width=True,
                )
            for p in page.problems:
                st.error(p)
            c1, c2 = st.columns(2)
            if page.source_index < len(originals):
                c1.image(bgr_to_rgb(registration_view(originals[page.source_index], page)), caption="Input with detected registration markers (red = inferred)", use_container_width=True)
            if page.canonical_bgr is not None and page.grid is not None:
                c2.image(
                    bgr_to_rgb(debug_overlay(page)),
                    caption="Blue + template position · green circle = corrected sampling area · green dot = matched bubble · red × = rejected candidate · orange arrow = local correction ×6",
                    use_container_width=True,
                )
            details = {
                "registration": page.registration.method if page.registration else None,
                "grid": {k: v for k, v in (page.grid.stats.items() if page.grid else [])},
                "calibration": page.calibration.as_dict() if page.calibration else None,
                "quality": page.quality.metrics if page.quality else None,
            }
            st.json(details, expanded=False)
    rows = metric_rows(scan)
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, hide_index=True, use_container_width=True, height=320)
        st.download_button("Download detection metrics (CSV)", df.to_csv(index=False).encode("utf-8"), "omr_detection_metrics.csv", "text/csv")


def page_for(scan, question: int):
    for page in scan.pages:
        if page.ok and any(q.question == question for q in page.questions):
            return page
    return None


def review_controls(scan, *, state_key: str, options: list[str]) -> dict:
    overrides = dict(st.session_state[state_key])
    problems = [q for q in scan.questions if q.status == REVIEW]
    if not problems:
        st.success(f"All {len(scan.questions)} questions were read with enough confidence. Nothing needs review.")
        return overrides
    st.warning(f"{len(problems)} question(s) need your review. The engine will not guess when the evidence is ambiguous.")
    choices = ["Select..."] + options
    for q in problems:
        with st.expander(f"Question {q.question}: {q.reason}", expanded=True):
            page = page_for(scan, q.question)
            if page is not None:
                st.image(bgr_to_rgb(crop_question(page, q.question)), use_container_width=True)
            levels = " · ".join(f"{b.choice}: {max(b.level, 0) * 100:.0f}%" for b in q.bubbles)
            st.caption(f"Mark strength compared with a typical fill on this sheet: {levels}. Confidence {q.confidence:.0f}%.")
            if q.notes:
                st.caption(" ".join(q.notes))
            current = overrides.get(q.question)
            index = choices.index(current) if current in choices else 0
            selected = st.radio(
                f"What did the student intend for question {q.question}?",
                choices,
                index=index,
                horizontal=True,
                key=f"{state_key}_{q.question}_{st.session_state.review_nonce}",
            )
            if selected == "Select...":
                overrides.pop(q.question, None)
            else:
                overrides[q.question] = selected
    st.session_state[state_key] = overrides
    return overrides


def correction_controls(scan, *, state_key: str, options: list[str]) -> dict:
    overrides = dict(st.session_state[state_key])
    items = [q for q in scan.questions if q.status != REVIEW]
    if not items:
        return overrides
    with st.expander("Optional: correct an answer that was read automatically"):
        st.caption("Use this only when the marked image shows the automatic reading is wrong.")
        cols = st.columns(4)
        for i, q in enumerate(items):
            auto = q.answer if q.answer is not None else BLANK_LABEL
            opts = list(dict.fromkeys([auto] + options))
            current = overrides.get(q.question, auto)
            picked = cols[i % 4].selectbox(
                f"Question {q.question}",
                opts,
                index=opts.index(current) if current in opts else 0,
                key=f"{state_key}_fix_{q.question}_{st.session_state.review_nonce}",
            )
            if picked == auto:
                overrides.pop(q.question, None)
            else:
                overrides[q.question] = picked
    st.session_state[state_key] = overrides
    return overrides


def reset_review(state_key: str) -> None:
    st.session_state[state_key] = {}
    st.session_state.review_nonce += 1


# ------------------------------------------------------------------ sidebar
# Step navigation lives in the main area so it is always visible on a phone (the sidebar is hidden there).
STEP_LABELS = {
    "1. Create exam & sheet": "1 · Sheet",
    "2. Create answer key": "2 · Answer key",
    "3. Grade student paper": "3 · Grade",
    "4. Session results": "4 · Results",
}
page = st.radio("Step", list(STEP_LABELS), format_func=STEP_LABELS.get, horizontal=True, label_visibility="collapsed", key="step")

with st.sidebar:
    st.header("Status & settings")
    active = get_template(st.session_state.active_template_id)
    if active is not None:
        st.caption("Active exam template")
        st.write(f"**{st.session_state.exam_title}**")
        st.caption(template_label(active))
    if st.session_state.key:
        key_t = get_template(st.session_state.key["template_id"])
        st.success("Answer key ready")
        st.caption(f"{key_t.n_questions if key_t else '?'} questions · saved {st.session_state.key['saved_at']}")
    else:
        st.info("No answer key saved yet")
    st.divider()
    multiple_policy = st.selectbox(
        "Multiple marks on one question",
        ["incorrect", "partial"],
        format_func=lambda v: "Count as incorrect" if v == "incorrect" else "Correct if the key answer is among them",
    )
    st.toggle("Developer debug mode", key="debug_mode", help="Shows alignment, grid fitting, and per-bubble measurements.")


# ------------------------------------------------------------------ 1. sheet generator
if page == "1. Create exam & sheet":
    left, right = st.columns([0.95, 1.05], gap="large")
    with left:
        st.subheader("Design the answer sheet")
        with st.form("sheet_form"):
            title = st.text_input("Exam title (printed at the top)", value=st.session_state.exam_title, max_chars=60)
            n_questions = st.number_input("Number of questions", min_value=1, max_value=MAX_QUESTIONS, value=20, step=1)
            n_choices = st.selectbox(
                "Answer choices",
                list(range(MIN_CHOICES, MAX_CHOICES + 1)),
                index=2,
                format_func=lambda n: f"A–{'ABCDEFGH'[n - 1]} ({n} choices)",
            )
            name_field = st.checkbox("Include student name / ID section", value=True)
            submitted = st.form_submit_button("Generate sheet", type="primary", use_container_width=True)
        if submitted:
            template = build_template(int(n_questions), int(n_choices), name_field=name_field, title=title or "OMR ANSWER SHEET")
            STORE.save(template)
            st.session_state.active_template_id = template.template_id
            st.session_state.exam_title = title or "OMR ANSWER SHEET"
            st.rerun()

        saved = STORE.list()
        ids = [t.template_id for t in saved]
        current = st.session_state.active_template_id
        picked = st.selectbox(
            "Or reuse a saved template",
            ids,
            index=ids.index(current) if current in ids else None,
            format_func=lambda i: template_label(next(t for t in saved if t.template_id == i)),
            placeholder="Choose a template",
        )
        if picked and picked != current:
            st.session_state.active_template_id = picked
            chosen = get_template(picked)
            st.session_state.exam_title = chosen.title if chosen else st.session_state.exam_title
            st.rerun()

        st.markdown(
            """
**How it works**
1. Generate and print the sheet (all pages).
2. Fill one copy with the correct answers and scan it under **Create answer key**.
3. Scan each student's sheet under **Grade student paper**.
4. Review only the questions the engine flags.

The QR code tells the scanner exactly which layout and page it is reading. It never contains the answers.

**Overlay colors:** green = correct or clear · red = incorrect · orange = needs review · gray = blank · purple = multiple marks
            """
        )

    with right:
        template = get_template(st.session_state.active_template_id)
        if template is None:
            st.info("Generate a sheet on the left to see the printable preview.")
        elif template.legacy:
            st.subheader("Original 20-question sheet")
            sheet = create_sheet()
            st.image(bgr_to_rgb(sheet), use_container_width=True)
            st.download_button("Download printable sheet (PNG)", encode_png(sheet), "omr_answer_sheet_20_questions.png", "image/png", use_container_width=True)
        else:
            st.subheader(st.session_state.exam_title)
            st.caption(
                f"{template.describe()} · bubble diameter {template.bubble_radius * 2 / 200 * 25.4:.1f} mm · "
                f"row spacing {template.row_pitch / 200 * 25.4:.1f} mm · layout version {template.version}"
            )
            pages = render_all(template, title=st.session_state.exam_title)
            tabs = st.tabs([f"Page {i + 1}" for i in range(len(pages))]) if len(pages) > 1 else [st.container()]
            for i, (tab, img) in enumerate(zip(tabs, pages)):
                with tab:
                    st.image(bgr_to_rgb(img), use_container_width=True)
                    st.download_button(
                        f"Download page {i + 1} (PNG)",
                        encode_png(img),
                        f"omr_{template.template_id}_page{i + 1}.png",
                        "image/png",
                        key=f"png_{i}",
                        use_container_width=True,
                    )
            st.download_button(
                "Download all pages (PDF, print at 100% / actual size)",
                pages_to_pdf(pages),
                f"omr_{template.template_id}.pdf",
                "application/pdf",
                type="primary",
                use_container_width=True,
            )
            st.download_button(
                "Download layout definition (JSON)",
                json.dumps(template.to_json(), indent=1).encode("utf-8"),
                f"omr_template_{template.template_id}.json",
                "application/json",
                use_container_width=True,
            )

# ------------------------------------------------------------------ 2. answer key
elif page == "2. Create answer key":
    st.subheader("Create and validate the answer key")
    st.write("Upload or photograph a correctly completed copy of the sheet. The template is identified automatically from its QR code.")
    saved = STORE.list()
    ids = [t.template_id for t in saved]
    current = st.session_state.active_template_id
    fallback_id = st.selectbox(
        "Template to use when a sheet has no readable QR code (for example the original 20-question sheet)",
        ids,
        index=ids.index(current) if current in ids else 0,
        format_func=lambda i: template_label(next(t for t in saved if t.template_id == i)),
    )
    hint = get_template(fallback_id)
    with st.expander("Restore a previously downloaded answer key"):
        restore = st.file_uploader("Answer key file (JSON)", type=["json"], key="key_restore")
        if restore is not None:
            try:
                data = json.loads(restore.read().decode("utf-8"))
                restored_t = get_template(data["template_id"]) or build_template(len(data["answers"]), len(data.get("choices", "ABCD")), name_field=data.get("name_field", True))
                if restored_t.template_id != data["template_id"]:
                    raise ValueError("template mismatch")
                if not restored_t.legacy:
                    STORE.save(restored_t)
                st.session_state.key = {"template_id": data["template_id"], "answers": list(data["answers"]), "title": data.get("title", ""), "saved_at": data.get("saved_at", "")}
                st.session_state.active_template_id = data["template_id"]
                st.success("Answer key restored.")
            except (ValueError, KeyError, json.JSONDecodeError):
                st.error("That file is not a valid answer key.")

    items = collect_inputs("key")
    if items:
        digest = hashlib.sha256(b"".join(hashlib.sha256(d).digest() for d, _, _ in items)).hexdigest()
        if digest != st.session_state.key_hash:
            with st.spinner("Finding the sheet, aligning the bubble grid, and reading marks..."):
                scan, originals = run_scan(items, template_hint=hint, answer_key_mode=True, enforce_template=False)
            st.session_state.key_scan = (scan, originals) if scan else None
            st.session_state.key_hash = digest
            reset_review("key_overrides")

    if st.session_state.key_scan is not None:
        scan, originals = st.session_state.key_scan
        show_messages(scan)
        good_pages = [p for p in scan.pages if p.ok]
        if good_pages and scan.template is not None:
            template = scan.template
            st.caption(f"Detected {template_label(template)}")
            c1, c2 = st.columns([1.25, 0.75], gap="large")
            with c1:
                for p in sorted(good_pages, key=lambda x: x.page_number):
                    st.image(bgr_to_rgb(teacher_overlay(p, overrides=st.session_state.key_overrides)), caption=f"Answer key, page {p.page_number}", use_container_width=True)
            with c2:
                rows = pd.DataFrame(question_rows(scan.questions, st.session_state.key_overrides))
                review = int((rows["Status"] == "Review").sum()) if len(rows) else 0
                m1, m2 = st.columns(2)
                m1.metric("Read clearly", len(rows) - review)
                m2.metric("Needs review", review)
                st.dataframe(rows, hide_index=True, use_container_width=True, height=420)
            show_debug(scan, originals)

            st.divider()
            overrides = review_controls(scan, state_key="key_overrides", options=list(template.choices))
            overrides = correction_controls(scan, state_key="key_overrides", options=list(template.choices))
            answers = [final_answer(q, overrides)[0] for q in scan.questions]
            unresolved = [q.question for q, a in zip(scan.questions, answers) if a is None or "+" in a]
            if scan.missing_pages:
                st.error("Upload every page of the answer key before saving it.")
            elif unresolved:
                st.error("The answer key cannot be saved until every question has exactly one confirmed answer. Still open: " + ", ".join(map(str, unresolved)))
            else:
                st.code(" · ".join(f"{q.question}:{a}" for q, a in zip(scan.questions, answers)), language=None)
                if st.button("Confirm and save answer key", type="primary", use_container_width=True):
                    stored = None if template.legacy else STORE.load(template.template_id)
                    if stored is not None:
                        st.session_state.exam_title = stored.title
                    elif not template.legacy:
                        STORE.save(template)
                    st.session_state.key = {
                        "template_id": template.template_id,
                        "answers": [str(a) for a in answers],
                        "title": st.session_state.exam_title,
                        "saved_at": datetime.now().strftime("%Y-%m-%d %I:%M %p"),
                    }
                    st.session_state.active_template_id = template.template_id
                    st.session_state.student_scan = None
                    st.session_state.student_hash = None
                    st.success("Answer key saved. You can now grade student papers.")
                    st.rerun()
    if st.session_state.key:
        k = st.session_state.key
        key_t = get_template(k["template_id"])
        export = dict(k, choices="".join(key_t.choices) if key_t else "ABCD", name_field=key_t.name_field if key_t else True)
        st.download_button("Download saved answer key (JSON, keep it private)", json.dumps(export, indent=1).encode("utf-8"), "omr_answer_key.json", "application/json")

# ------------------------------------------------------------------ 3. grading
elif page == "3. Grade student paper":
    st.subheader("Grade a student answer sheet")
    if not st.session_state.key:
        st.warning("Create and confirm an answer key first.")
        st.stop()
    key = st.session_state.key
    key_template = get_template(key["template_id"])
    if key_template is None:
        st.error("The template for the saved answer key is missing. Restore the key from its JSON file.")
        st.stop()
    st.caption(f"{template_label(key_template)} · key: " + " · ".join(f"{i + 1}:{a}" for i, a in enumerate(key["answers"])))
    student_name = st.text_input("Student name (optional, for the session results)", key="student_name")

    items = collect_inputs("student")
    if items:
        digest = hashlib.sha256(b"".join(hashlib.sha256(d).digest() for d, _, _ in items)).hexdigest()
        if digest != st.session_state.student_hash:
            with st.spinner("Reading student answers..."):
                scan, originals = run_scan(items, template_hint=key_template, answer_key_mode=False, enforce_template=True)
            st.session_state.student_scan = (scan, originals) if scan else None
            st.session_state.student_hash = digest
            reset_review("student_overrides")

    if st.session_state.student_scan is not None:
        scan, originals = st.session_state.student_scan
        show_messages(scan)
        good_pages = sorted([p for p in scan.pages if p.ok], key=lambda x: x.page_number)
        if good_pages and not scan.missing_pages and scan.template is not None and scan.template.template_id == key_template.template_id:
            key_map = {i + 1: a for i, a in enumerate(key["answers"])}
            overrides = st.session_state.student_overrides
            grade = grade_exam(scan.questions, key["answers"], overrides=overrides, multiple_policy=multiple_policy)

            left, right = st.columns([1.2, 0.8], gap="large")
            with left:
                for p in good_pages:
                    st.image(
                        bgr_to_rgb(teacher_overlay(p, key=key_map, overrides=overrides)),
                        caption=f"Page {p.page_number}: green = correct, red = incorrect, orange = review, gray = blank",
                        use_container_width=True,
                    )
            with right:
                r1, r2 = st.columns(2)
                r1.metric("Current score", f"{grade['score']}/{grade['total']}")
                r2.metric("Percentage", f"{grade['percentage']:.1f}%")
                r3, r4, r5 = st.columns(3)
                r3.metric("Incorrect", grade["incorrect"])
                r4.metric("Blank", grade["blank"])
                r5.metric("Review", grade["unresolved"])
                st.dataframe(pd.DataFrame(grade["rows"]), hide_index=True, use_container_width=True, height=420)
            show_debug(scan, originals)

            st.divider()
            options = list(key_template.choices) + [BLANK_LABEL]
            overrides = review_controls(scan, state_key="student_overrides", options=options)
            overrides = correction_controls(scan, state_key="student_overrides", options=options)
            unresolved = [q.question for q in scan.questions if q.status == REVIEW and q.question not in overrides]
            if unresolved:
                st.warning("Review the orange questions to finalize the grade: " + ", ".join(map(str, unresolved)))
            else:
                final = grade_exam(scan.questions, key["answers"], overrides=overrides, multiple_policy=multiple_policy)
                st.success(f"Final score: {final['score']}/{final['total']} ({final['percentage']:.1f}%)")
                result_df = pd.DataFrame(final["rows"])
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.download_button("Download question results (CSV)", result_df.to_csv(index=False).encode("utf-8"), "omr_question_results.csv", "text/csv", use_container_width=True)
                with c2:
                    for p in good_pages:
                        st.download_button(
                            f"Download marked page {p.page_number} (PNG)",
                            encode_png(teacher_overlay(p, key=key_map, overrides=overrides)),
                            f"omr_graded_page{p.page_number}.png",
                            "image/png",
                            key=f"marked_{p.page_number}",
                            use_container_width=True,
                        )
                with c3:
                    if st.button("Add result to session", use_container_width=True):
                        st.session_state.history.append(
                            {
                                "Scan": len(st.session_state.history) + 1,
                                "Student": student_name or "",
                                "Time": datetime.now().strftime("%I:%M:%S %p"),
                                "Template": key_template.template_id,
                                "Score": final["score"],
                                "Total": final["total"],
                                "Percentage": round(final["percentage"], 1),
                                "Incorrect": final["incorrect"],
                                "Blank": final["blank"],
                                "Multiple": final["multiple"],
                                "Teacher reviewed": sum(1 for r in final["rows"] if r["Decided by"] == "teacher"),
                            }
                        )
                        st.success("Result added to this session.")
        elif scan.missing_pages:
            st.info("Upload the missing page(s) together with the others to finish grading.")

# ------------------------------------------------------------------ 4. results
elif page == "4. Session results":
    st.subheader("Session results")
    if not st.session_state.history:
        st.info("No student results have been added during this session yet.")
    else:
        history = pd.DataFrame(st.session_state.history)
        st.dataframe(history, hide_index=True, use_container_width=True)
        st.download_button(
            "Download session results (CSV)",
            data=history.to_csv(index=False).encode("utf-8"),
            file_name="omr_session_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
        if st.button("Clear session results"):
            st.session_state.history = []
            st.rerun()

st.divider()
st.caption("Template-driven OMR engine · custom sheets with QR identification · grid fitting · calibrated, explainable decisions")
