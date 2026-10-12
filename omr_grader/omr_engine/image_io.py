"""Input decoding: PNG/JPG/other raster images and (multi-page) PDFs."""

from __future__ import annotations

import cv2
import numpy as np


class InputError(RuntimeError):
    pass


PDF_RENDER_DPI = 220
MAX_SIDE = 3200


def _limit_size(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    largest = max(h, w)
    if largest <= MAX_SIDE:
        return image
    scale = MAX_SIDE / float(largest)
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def is_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def render_pdf(data: bytes, dpi: int = PDF_RENDER_DPI) -> list[np.ndarray]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:  # pragma: no cover - depends on install
        raise InputError("PDF support needs the 'pypdfium2' package. Install it with: pip install pypdfium2") from exc
    try:
        document = pdfium.PdfDocument(data)
    except Exception as exc:  # pdfium raises its own error types
        raise InputError("The PDF could not be opened. It may be damaged or password-protected.") from exc
    pages: list[np.ndarray] = []
    try:
        for index in range(len(document)):
            page = document[index]
            bitmap = page.render(scale=dpi / 72.0, grayscale=False)
            rgb = np.asarray(bitmap.to_numpy())
            if rgb.ndim == 2:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
            elif rgb.shape[2] == 4:
                bgr = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR)
            else:
                bgr = rgb[:, :, :3].copy()
            # pdfium renders BGR(A) by default (rev_byteorder=False), which matches OpenCV.
            pages.append(_limit_size(bgr))
    finally:
        document.close()
    if not pages:
        raise InputError("The PDF has no pages.")
    return pages


def decode_input(data: bytes) -> list[np.ndarray]:
    """Return one BGR image per page."""
    if not data:
        raise InputError("The uploaded file is empty.")
    if is_pdf(data):
        return render_pdf(data)
    array = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if image is None:
        raise InputError("The uploaded file could not be read as an image. Use PNG, JPG, or PDF.")
    return [_limit_size(image)]
