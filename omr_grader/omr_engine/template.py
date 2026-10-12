"""Template manager: exact, machine-readable layout of every generated sheet.

A template is fully determined by its geometry parameters (question count,
choices, name field, layout version). The same parameters always produce the
same template ID and the same bubble coordinates, so a QR code that carries
those parameters is enough to rebuild the layout even when no JSON file is
stored next to the app.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LAYOUT_VERSION = 1

# Canonical page: US Letter at 200 dpi. Every page is processed in this frame.
PAGE_WIDTH = 1700
PAGE_HEIGHT = 2200
MARKER_SIZE = 70
MARKER_MARGIN = 75

ALPHABET = "ABCDEFGH"
MIN_CHOICES = 2
MAX_CHOICES = len(ALPHABET)
MAX_QUESTIONS = 300

QR_PREFIX = "OMR"


def marker_centers() -> np.ndarray:
    half = MARKER_SIZE / 2.0
    lo = MARKER_MARGIN + half
    return np.array(
        [
            [lo, lo],
            [PAGE_WIDTH - lo, lo],
            [PAGE_WIDTH - lo, PAGE_HEIGHT - lo],
            [lo, PAGE_HEIGHT - lo],
        ],
        dtype=np.float32,
    )


CORNER_NAMES = ("top-left", "top-right", "bottom-right", "bottom-left")


@dataclass(frozen=True)
class PageLayout:
    page_number: int  # 1-based
    questions: tuple[int, ...]  # question numbers on this page, in row order
    centers: np.ndarray  # (n_questions_on_page, n_choices, 2) float32, canonical page pixels
    block_index: tuple[int, ...]  # which column block each question row belongs to
    block_header_y: tuple[float, ...]  # y of the A/B/C header for each block
    label_x: tuple[float, ...]  # x of the question number label for each block

    @property
    def n_rows(self) -> int:
        return len(self.questions)


@dataclass(frozen=True)
class Template:
    template_id: str
    version: int
    n_questions: int
    choices: tuple[str, ...]
    name_field: bool
    bubble_radius: float
    ring_width: int
    choice_pitch: float
    row_pitch: float
    pages: tuple[PageLayout, ...]
    page_width: int = PAGE_WIDTH
    page_height: int = PAGE_HEIGHT
    has_qr: bool = True
    legacy: bool = False
    qr_box: tuple[int, int, int, int] = (1415, 175, 1565, 325)  # x1, y1, x2, y2
    title: str = "OMR ANSWER SHEET"
    extra: dict = field(default_factory=dict)

    @property
    def n_choices(self) -> int:
        return len(self.choices)

    @property
    def n_pages(self) -> int:
        return len(self.pages)

    @property
    def marker_centers(self) -> np.ndarray:
        return marker_centers()

    def page(self, page_number: int) -> PageLayout:
        return self.pages[page_number - 1]

    def page_of_question(self, question: int) -> int:
        for layout in self.pages:
            if question in layout.questions:
                return layout.page_number
        raise KeyError(question)

    def describe(self) -> str:
        choice_text = f"{self.choices[0]}–{self.choices[-1]}"
        pages = "1 page" if self.n_pages == 1 else f"{self.n_pages} pages"
        return f"Template {self.template_id}: {self.n_questions} questions, {choice_text}, {pages}"

    def qr_payload(self, page_number: int) -> str:
        return "|".join(
            [
                QR_PREFIX,
                str(self.version),
                self.template_id,
                f"{page_number}/{self.n_pages}",
                str(self.n_questions),
                "".join(self.choices),
                "1" if self.name_field else "0",
            ]
        )

    def to_json(self) -> dict:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "title": self.title,
            "n_questions": self.n_questions,
            "choices": list(self.choices),
            "name_field": self.name_field,
            "legacy": self.legacy,
            "bubble_radius": self.bubble_radius,
            "ring_width": self.ring_width,
            "choice_pitch": self.choice_pitch,
            "row_pitch": self.row_pitch,
            "page_width": self.page_width,
            "page_height": self.page_height,
            "marker_centers": marker_centers().tolist(),
            "pages": [
                {
                    "page_number": p.page_number,
                    "questions": list(p.questions),
                    "rows": [
                        {
                            "question": q,
                            "block": b,
                            "bubbles": {c: [round(float(x), 2), round(float(y), 2)] for c, (x, y) in zip(self.choices, row)},
                        }
                        for q, b, row in zip(p.questions, p.block_index, p.centers)
                    ],
                }
                for p in self.pages
            ],
        }


@dataclass(frozen=True)
class QRInfo:
    version: int
    template_id: str
    page_number: int
    n_pages: int
    n_questions: int
    choices: tuple[str, ...]
    name_field: bool


def parse_qr_payload(text: str) -> QRInfo | None:
    try:
        parts = text.strip().split("|")
        if len(parts) != 7 or parts[0] != QR_PREFIX:
            return None
        page, n_pages = parts[3].split("/")
        return QRInfo(
            version=int(parts[1]),
            template_id=parts[2],
            page_number=int(page),
            n_pages=int(n_pages),
            n_questions=int(parts[4]),
            choices=tuple(parts[5]),
            name_field=parts[6] == "1",
        )
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Layout computation
# ---------------------------------------------------------------------------

GRID_TOP = 430.0  # y of the column-letter header row
GRID_BOTTOM = 2030.0  # last allowed bubble-row centre
GRID_LEFT = 120.0
GRID_RIGHT = PAGE_WIDTH - 120.0


@dataclass(frozen=True)
class _Geometry:
    scale: float
    radius: float
    choice_pitch: float
    row_pitch: float
    label_width: float
    block_gap: float
    ring_width: int

    @property
    def first_row_offset(self) -> float:
        return 0.95 * self.row_pitch

    def rows_per_column(self) -> int:
        usable = GRID_BOTTOM - (GRID_TOP + self.first_row_offset)
        return int(math.floor(usable / self.row_pitch)) + 1

    def block_width(self, n_choices: int) -> float:
        return self.label_width + n_choices * self.choice_pitch

    def max_blocks(self, n_choices: int) -> int:
        width = GRID_RIGHT - GRID_LEFT
        return max(1, int(math.floor((width + self.block_gap) / (self.block_width(n_choices) + self.block_gap))))

    def capacity(self, n_choices: int) -> int:
        return self.rows_per_column() * self.max_blocks(n_choices)


def _geometry(scale: float) -> _Geometry:
    return _Geometry(
        scale=scale,
        radius=round(20.0 * scale, 1),
        choice_pitch=round(58.0 * scale, 1),
        row_pitch=round(52.0 * scale, 1),
        label_width=round(80.0 * scale, 1),
        block_gap=round(40.0 * scale, 1),
        ring_width=3 if scale < 1.25 else 4,
    )


SCALES = (1.4, 1.2, 1.0)


def _choose_geometry(n_questions: int, n_choices: int) -> tuple[_Geometry, int]:
    for scale in SCALES:
        geo = _geometry(scale)
        if n_questions <= geo.capacity(n_choices):
            return geo, 1
    geo = _geometry(SCALES[-1])
    return geo, int(math.ceil(n_questions / geo.capacity(n_choices)))


def _page_layout(geo: _Geometry, page_number: int, questions: list[int], n_choices: int) -> PageLayout:
    rows_max = geo.rows_per_column()
    n_blocks = int(math.ceil(len(questions) / rows_max))
    rows_per_block = int(math.ceil(len(questions) / n_blocks))
    block_w = geo.block_width(n_choices)
    total_w = n_blocks * block_w + (n_blocks - 1) * geo.block_gap
    left = (PAGE_WIDTH - total_w) / 2.0

    centers = np.zeros((len(questions), n_choices, 2), dtype=np.float32)
    block_index: list[int] = []
    label_x: list[float] = []
    for b in range(n_blocks):
        label_x.append(left + b * (block_w + geo.block_gap) + geo.label_width * 0.45)
    for i, _q in enumerate(questions):
        b = i // rows_per_block
        r = i % rows_per_block
        x0 = left + b * (block_w + geo.block_gap) + geo.label_width + geo.choice_pitch / 2.0
        y = GRID_TOP + geo.first_row_offset + r * geo.row_pitch
        for c in range(n_choices):
            centers[i, c] = (x0 + c * geo.choice_pitch, y)
        block_index.append(b)
    return PageLayout(
        page_number=page_number,
        questions=tuple(questions),
        centers=centers,
        block_index=tuple(block_index),
        block_header_y=tuple(GRID_TOP for _ in range(n_blocks)),
        label_x=tuple(label_x),
    )


def template_id_for(n_questions: int, choices: tuple[str, ...], name_field: bool, version: int = LAYOUT_VERSION) -> str:
    key = f"v{version}|{PAGE_WIDTH}x{PAGE_HEIGHT}|q{n_questions}|c{''.join(choices)}|n{int(name_field)}"
    return hashlib.sha1(key.encode("ascii")).hexdigest()[:6].upper()


def build_template(
    n_questions: int,
    n_choices: int = 4,
    *,
    name_field: bool = True,
    title: str = "OMR ANSWER SHEET",
    version: int = LAYOUT_VERSION,
) -> Template:
    if not 1 <= n_questions <= MAX_QUESTIONS:
        raise ValueError(f"Number of questions must be between 1 and {MAX_QUESTIONS}.")
    if not MIN_CHOICES <= n_choices <= MAX_CHOICES:
        raise ValueError(f"Number of choices must be between {MIN_CHOICES} and {MAX_CHOICES}.")
    if version != LAYOUT_VERSION:
        raise ValueError(f"Unsupported template layout version {version}.")
    choices = tuple(ALPHABET[:n_choices])
    geo, n_pages = _choose_geometry(n_questions, n_choices)
    per_page = int(math.ceil(n_questions / n_pages))
    pages: list[PageLayout] = []
    for p in range(n_pages):
        qs = list(range(p * per_page + 1, min(n_questions, (p + 1) * per_page) + 1))
        pages.append(_page_layout(geo, p + 1, qs, n_choices))
    return Template(
        template_id=template_id_for(n_questions, choices, name_field, version),
        version=version,
        n_questions=n_questions,
        choices=choices,
        name_field=name_field,
        bubble_radius=geo.radius,
        ring_width=geo.ring_width,
        choice_pitch=geo.choice_pitch,
        row_pitch=geo.row_pitch,
        pages=tuple(pages),
        title=title,
    )


def template_from_qr(info: QRInfo) -> Template | None:
    try:
        template = build_template(info.n_questions, len(info.choices), name_field=info.name_field, version=info.version)
    except ValueError:
        return None
    if template.template_id != info.template_id or template.choices != info.choices:
        return None
    return template


# Geometry of the original fixed 20-question prototype sheet (sheet_template.py).
LEGACY_ID = "LEGACY20"


def legacy_template() -> Template:
    xs = np.array([690, 900, 1110, 1320], dtype=np.float32)
    ys = 430 + 78 * np.arange(20, dtype=np.float32)
    centers = np.zeros((20, 4, 2), dtype=np.float32)
    centers[:, :, 0] = xs[None, :]
    centers[:, :, 1] = ys[:, None]
    layout = PageLayout(
        page_number=1,
        questions=tuple(range(1, 21)),
        centers=centers,
        block_index=tuple(0 for _ in range(20)),
        block_header_y=(365.0,),
        label_x=(505.0,),
    )
    return Template(
        template_id=LEGACY_ID,
        version=0,
        n_questions=20,
        choices=("A", "B", "C", "D"),
        name_field=True,
        bubble_radius=28.0,
        ring_width=4,
        choice_pitch=210.0,
        row_pitch=78.0,
        pages=(layout,),
        has_qr=False,
        legacy=True,
        title="OMR ANSWER SHEET",
    )


class TemplateStore:
    """Persists generated templates as JSON so the app can list and reuse them."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def save(self, template: Template) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{template.template_id}.json"
        path.write_text(json.dumps(template.to_json(), indent=1), encoding="utf-8")
        return path

    def load(self, template_id: str) -> Template | None:
        if template_id == LEGACY_ID:
            return legacy_template()
        path = self.directory / f"{template_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        template = build_template(
            int(data["n_questions"]),
            len(data["choices"]),
            name_field=bool(data["name_field"]),
            title=data.get("title", "OMR ANSWER SHEET"),
            version=int(data["version"]),
        )
        return template if template.template_id == template_id else None

    def list(self) -> list[Template]:
        templates = [legacy_template()]
        if self.directory.exists():
            for path in sorted(self.directory.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
                loaded = self.load(path.stem)
                if loaded is not None:
                    templates.append(loaded)
        return templates
