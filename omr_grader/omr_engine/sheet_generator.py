"""Sheet generator: renders printable pages from a Template."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .template import MARKER_MARGIN, MARKER_SIZE, Template


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf" if bold else "/System/Library/Fonts/Supplemental/Arial.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def qr_image(payload: str, size: int) -> Image.Image:
    encoder = cv2.QRCodeEncoder.create()
    modules = encoder.encode(payload)  # one pixel per module, includes a quiet zone
    modules = cv2.resize(modules, (size, size), interpolation=cv2.INTER_NEAREST)
    return Image.fromarray(modules).convert("RGB")


def _fit_title(draw: ImageDraw.ImageDraw, title: str, max_width: int) -> ImageFont.ImageFont:
    for size in (54, 48, 42, 36, 32, 28):
        font = _font(size, bold=True)
        if draw.textlength(title, font=font) <= max_width:
            return font
    return _font(26, bold=True)


def render_page(
    template: Template,
    page_number: int,
    *,
    title: str | None = None,
    marks: Mapping[int, str | Sequence[str] | None] | None = None,
) -> np.ndarray:
    """Render one page as a BGR image. ``marks`` fills bubbles (used for tests/demos)."""
    layout = template.page(page_number)
    W, H = template.page_width, template.page_height
    image = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(image)

    # Registration markers.
    for x, y in (
        (MARKER_MARGIN, MARKER_MARGIN),
        (W - MARKER_MARGIN - MARKER_SIZE, MARKER_MARGIN),
        (W - MARKER_MARGIN - MARKER_SIZE, H - MARKER_MARGIN - MARKER_SIZE),
        (MARKER_MARGIN, H - MARKER_MARGIN - MARKER_SIZE),
    ):
        draw.rectangle((x, y, x + MARKER_SIZE - 1, y + MARKER_SIZE - 1), fill="black")

    # QR code with template ID + page; never the answer key.
    x1, y1, x2, y2 = template.qr_box
    image.paste(qr_image(template.qr_payload(page_number), x2 - x1), (x1, y1))

    title = title or template.title
    draw.text((W // 2, 105), title, font=_fit_title(draw, title, 1100), fill="black", anchor="ma")
    draw.text(
        (W // 2, 178),
        "Fill one bubble completely per question. Keep all four corner squares visible.",
        font=_font(23),
        fill="black",
        anchor="ma",
    )
    page_text = f"Template {template.template_id}  ·  Page {page_number} of {template.n_pages}"
    draw.text((W // 2, 214), page_text, font=_font(21), fill=(70, 70, 70), anchor="ma")

    if template.name_field:
        draw.text((190, 275), "Name: ________________________________", font=_font(27), fill="black")
        draw.text((190, 330), "Student ID: ______________", font=_font(27), fill="black")
        draw.text((900, 330), "Date: ______________", font=_font(27), fill="black")
    draw.line((150, 395, W - 150, 395), fill="black", width=2)

    R = template.bubble_radius
    header_font = _font(int(round(R * 1.05)), bold=True)
    label_font = _font(int(round(R * 0.95)), bold=True)

    rows_by_block: dict[int, list[int]] = {}
    for i, b in enumerate(layout.block_index):
        rows_by_block.setdefault(b, []).append(i)
    for b, rows in rows_by_block.items():
        xs = layout.centers[rows[0], :, 0]
        for c, letter in enumerate(template.choices):
            draw.text((int(xs[c]), int(layout.block_header_y[b])), letter, font=header_font, fill="black", anchor="mm")

    marks = marks or {}
    for i, q in enumerate(layout.questions):
        b = layout.block_index[i]
        row = layout.centers[i]
        y = float(row[0, 1])
        draw.text((int(layout.label_x[b]), int(y)), str(q), font=label_font, fill="black", anchor="mm")
        chosen = marks.get(q)
        if isinstance(chosen, str):
            chosen = [chosen]
        chosen = set(chosen or [])
        for c, (x, yy) in enumerate(row):
            box = (x - R, yy - R, x + R, yy + R)
            draw.ellipse(box, outline="black", width=template.ring_width)
            if template.choices[c] in chosen:
                inner = R - template.ring_width - 2
                draw.ellipse((x - inner, yy - inner, x + inner, yy + inner), fill=(25, 25, 25))

    draw.text(
        (W // 2, H - 92),
        f"{template.n_questions} questions · choices {template.choices[0]}–{template.choices[-1]} · do not fold or staple",
        font=_font(20),
        fill=(70, 70, 70),
        anchor="ma",
    )
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def render_all(template: Template, *, title: str | None = None, marks=None) -> list[np.ndarray]:
    return [render_page(template, p.page_number, title=title, marks=marks) for p in template.pages]


def pages_to_pdf(pages: Sequence[np.ndarray], dpi: int = 200) -> bytes:
    images = [Image.fromarray(cv2.cvtColor(p, cv2.COLOR_BGR2RGB)) for p in pages]
    buffer = io.BytesIO()
    images[0].save(buffer, format="PDF", save_all=True, append_images=images[1:], resolution=dpi)
    return buffer.getvalue()
