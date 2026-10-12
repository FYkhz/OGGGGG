# OMR Grader

An offline OMR (optical mark recognition) grader for printed bubble sheets. A teacher designs an exam sheet,
scans a filled answer key, then photographs or uploads student sheets. The engine reads every answer as
**a letter, BLANK, MULTIPLE, or REVIEW** with a confidence score, and only asks the teacher about rows it
cannot read reliably.

## Run locally

1. Install Python 3.10 or newer.
2. Open a terminal in this folder (optionally create and activate a virtual environment).
3. `pip install -r requirements.txt`
4. `streamlit run app.py` (or double-click `run_app.bat` on Windows, `./run_app.sh` on Mac/Linux)

Streamlit opens the app at `http://localhost:8501`.

On Replit: upload this folder, install `requirements.txt`, and run
`streamlit run app.py --server.address 0.0.0.0 --server.port 8080`.

## Use it on a phone

**Same Wi-Fi (free, private):** on the computer run `python run_on_phone.py` (or double-click `run_on_phone.bat` /
`./run_on_phone.sh`). It prints an address like `http://192.168.1.20:8501`; open it in the phone's browser while the
phone is on the same Wi-Fi. Allow Python through the firewall if asked. To photograph sheets, use
**Upload → Take Photo**: it uses the phone's main camera at full resolution. The live "Scan with camera" tab only
works over an https address.

**Anywhere (hosted):** deploy the folder to a host such as Streamlit Community Cloud (free, needs a GitHub
repository). The app then gets an https address that works on any phone, including the live camera tab.

## Teacher workflow

1. **Create exam & sheet**: choose the title, number of questions (1–300), choices (A–B up to A–H), and
   whether to print a name/ID section. Download the PNG pages or one PDF and print at 100% (actual size).
   Large exams automatically flow onto several pages (for example 120 × A–D fits on one page, 200 × A–E on three).
2. **Create answer key**: upload or photograph a filled copy. The QR code tells the app which layout and page
   it is reading. Every question must end up with exactly one answer before the key can be saved.
   The key can be downloaded as JSON and restored later; it is never encoded in the sheet.
3. **Grade student paper**: upload every page (PNG, JPG, or a multi-page PDF, in any order) or capture pages
   with the camera. Confident answers are graded automatically; REVIEW rows show a close-up and the measured
   mark strength so the teacher can decide.
4. **Session results**: running table of graded students, downloadable as CSV.

The original fixed 20-question sheet still works: pick "Original 20-question sheet" as the fallback template on
the answer-key page.

## How recognition works

```
image → page / registration-marker detection → perspective correction (any rotation, incl. upside down)
      → QR template identification → illumination normalization → expected grid from the template
      → bubble pattern detection → RANSAC grid fit → local grid refinement → per-bubble features
      → whole-sheet calibration → row-level classification → confidence → grading → teacher review
```

- **Registration**: the four black corner squares define the page. If one is missing, the fourth corner is
  inferred and the bubble pattern recovers the exact geometry. If fewer than three are found, the paper edge is
  tried, and the page is only graded if the bubble grid then confirms the alignment.
- **Template**: every generated sheet has an exact, saved layout (`templates/<ID>.json`) and a QR code carrying
  the template ID, page number, question count, and choices. The scanner never rediscovers the layout.
- **Grid fitting** (`omr_engine/grid.py`): printed bubbles are found with two independent detectors (contour shape
  and ring template matching), matched to the template, and fitted with RANSAC (affine or homography). Stray
  circles never influence the fit. Smooth local offsets from neighbouring bubbles absorb paper curl and printer
  distortion while staying clamped, so one odd detection cannot move a bubble. Heavily filled bubbles whose
  outline disappeared are still sampled at the position the grid predicts.
- **Features** (`omr_engine/features.py`): each bubble is split into centre, interior, printed ring, and the paper
  just outside it. Darkness is measured against that local paper, so shadows, gray paper, and coloured light do
  not matter. The printed outline carries almost no weight.
- **Calibration**: the many empty bubbles on each sheet define what "empty" looks like; the clear fills define a
  typical mark. Every bubble gets a level: 0 = empty, 1 = a typical fill on this sheet.
- **Row classification** (`omr_engine/classifier.py`): the whole row is judged together using the best and
  second-best marks, distance from the empty baseline, and mark structure:
  - one strong mark, others empty → the letter (an erased light mark next to it is noted, not counted);
  - two or more strong, similar marks → MULTIPLE (e.g. A/D);
  - nothing above the empty baseline → BLANK;
  - a check or X inside one bubble → that letter at reduced confidence;
  - dots, faint marks, strokes running through bubbles, uniformly dark rows, or a strong mark next to a fairly
    strong one → REVIEW.
  Anything under 70% confidence (85% for answer keys) becomes REVIEW.
- **Quality checks** (`omr_engine/quality.py`): blur, darkness, washed-out images, strong shadow, noise, steep camera
  angle, sheet too small in the photo, missing markers, cropped pages, wrong template, and missing pages are
  reported in plain language and stop grading when they make results unreliable.

## Developer debug mode

Turn on **Developer debug mode** in the sidebar to see, for every page: the input photo with the detected marker
quad, a debug overlay (template positions, corrected sampling circles, matched and rejected bubble candidates,
row and column lines, local corrections, per-bubble levels, confidence), the confidence breakdown
(page detection, perspective, template match, bubble grid, image quality), grid statistics (rotation, scale,
skew, offset, residuals), sheet calibration, and a per-bubble metrics table you can download as CSV.

## Project layout

| File | Role |
| --- | --- |
| `app.py` | Streamlit teacher UI |
| `omr_engine/template.py` | template manager: layouts, IDs, QR payload, JSON store, legacy sheet |
| `omr_engine/sheet_generator.py` | printable pages and PDF |
| `omr_engine/image_io.py` | PNG/JPG/PDF input |
| `omr_engine/registration.py` | page and corner-marker detection, perspective, orientation |
| `omr_engine/grid.py` | bubble pattern detection, RANSAC grid fit, local refinement, validation |
| `omr_engine/features.py` | illumination normalization, bubble features, sheet calibration |
| `omr_engine/classifier.py` | CLEAR / BLANK / MULTIPLE / REVIEW and confidence |
| `omr_engine/quality.py` | image-quality checks |
| `omr_engine/pipeline.py` | one page and multi-page exam processing |
| `omr_engine/grading.py` | scoring, teacher overrides, multiple-mark policy |
| `omr_engine/visualize.py` | teacher overlay, debug overlay, review crops, metrics |
| `document_scanner.py` | camera document cleanup (used for the camera preview and as a fallback) |
| `omr.py`, `sheet_template.py` | original 20-question reader and sheet, kept for compatibility |

## Tests

```bash
python -m tests.test_engine        # 24 scenario tests, no extra packages needed
python test_mvp.py                 # original prototype test
```

The scenarios cover solid, heavy, light, erased, half, X, check, dot, and stray-line marks; blank and multiple
rows; phone photos with rotation, upside-down and sideways pages, strong perspective, tinted and gray lighting,
shadows, noise, JPEG compression, blur, printer scaling and shift, paper curl, a missing corner marker, cropped
pages, wrong templates (with and without QR), 10–200 questions with A–D and A–E, multi-page exams with a
missing page, PDF input, answer-key validation, and the original sheet.

## Known limits

- Tests use realistic synthetic photos; real phone photos of real sheets should be collected to fine-tune the
  thresholds in `classifier.py` and `quality.py`.
- Sheets must be printed from this app (or be the original 20-question sheet); arbitrary third-party bubble
  sheets are not recognized.
- Grading is per student; batch upload of a whole class is not implemented yet.
