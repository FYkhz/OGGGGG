from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from omr import (
    BUBBLE_RADIUS,
    BUBBLE_X,
    CHOICES,
    MARKER_MARGIN,
    MARKER_SIZE,
    PAGE_HEIGHT,
    PAGE_WIDTH,
    QUESTION_COUNT,
    QUESTION_GAP_Y,
    QUESTION_START_Y,
)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def create_sheet(answer_marks: Sequence[str | None] | None = None, *, title: str = "OMR ANSWER SHEET") -> np.ndarray:
    if answer_marks is not None and len(answer_marks) != QUESTION_COUNT:
        raise ValueError(f"Expected {QUESTION_COUNT} answers.")

    image = Image.new("RGB", (PAGE_WIDTH, PAGE_HEIGHT), "white")
    draw = ImageDraw.Draw(image)

    marker_positions = [
        (MARKER_MARGIN, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, MARKER_MARGIN),
        (PAGE_WIDTH - MARKER_MARGIN - MARKER_SIZE, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
        (MARKER_MARGIN, PAGE_HEIGHT - MARKER_MARGIN - MARKER_SIZE),
    ]
    for x, y in marker_positions:
        draw.rectangle((x, y, x + MARKER_SIZE, y + MARKER_SIZE), fill="black")

    draw.text((PAGE_WIDTH // 2, 115), title, font=_font(54, bold=True), fill="black", anchor="ma")
    draw.text(
        (PAGE_WIDTH // 2, 180),
        "Fill one bubble completely for each question. Keep all four corner squares visible when photographing.",
        font=_font(24),
        fill="black",
        anchor="ma",
    )
    draw.line((250, 250, 1450, 250), fill="black", width=3)
    draw.text((300, 295), "Name (optional): ______________________________", font=_font(28), fill="black")
    draw.text((1080, 295), "Date: ______________", font=_font(28), fill="black")

    header_y = 365
    draw.text((505, header_y), "Question", font=_font(29, bold=True), fill="black", anchor="mm")
    for x, choice in zip(BUBBLE_X, CHOICES):
        draw.text((int(x), header_y), choice, font=_font(31, bold=True), fill="black", anchor="mm")

    for q_index in range(QUESTION_COUNT):
        q = q_index + 1
        y = QUESTION_START_Y + q_index * QUESTION_GAP_Y
        if q_index % 2 == 1:
            draw.rectangle((430, y - 34, 1450, y + 34), fill=(245, 247, 250))
        draw.text((505, y), str(q), font=_font(30, bold=True), fill="black", anchor="mm")
        for choice_index, x in enumerate(BUBBLE_X):
            box = (int(x - BUBBLE_RADIUS), y - BUBBLE_RADIUS, int(x + BUBBLE_RADIUS), y + BUBBLE_RADIUS)
            draw.ellipse(box, outline="black", width=4)
            if answer_marks is not None and answer_marks[q_index] == CHOICES[choice_index]:
                inner = BUBBLE_RADIUS - 7
                fill_box = (int(x - inner), y - inner, int(x + inner), y + inner)
                draw.ellipse(fill_box, fill="black")

    footer_y = QUESTION_START_Y + QUESTION_COUNT * QUESTION_GAP_Y + 25
    draw.line((250, footer_y, 1450, footer_y), fill="black", width=3)
    draw.text(
        (PAGE_WIDTH // 2, footer_y + 45),
        "Prototype sheet: 20 questions, choices A-D",
        font=_font(23),
        fill="black",
        anchor="ma",
    )

    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def save_sheet(path: str | Path, answer_marks: Sequence[str | None] | None = None, *, title: str = "OMR ANSWER SHEET") -> None:
    image = create_sheet(answer_marks, title=title)
    cv2.imwrite(str(path), image)


if __name__ == "__main__":
    output = Path(__file__).resolve().parent / "assets" / "omr_answer_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    save_sheet(output)
    print(output)
