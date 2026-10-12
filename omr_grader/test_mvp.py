from __future__ import annotations

import cv2
import numpy as np

from omr import QUESTION_COUNT, answers_from_detections, grade_answers, scan_sheet
from sheet_template import create_sheet


def perspective_photo(image: np.ndarray) -> np.ndarray:
    h, w = image.shape[:2]
    canvas_w, canvas_h = 2200, 2600
    src = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])
    dst = np.float32([[280, 180], [1900, 320], [1750, 2380], [180, 2250]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (canvas_w, canvas_h), borderValue=(180, 180, 180))


def main() -> None:
    key = list("ABCDABCDABCDABCDABCD")
    key_sheet = perspective_photo(create_sheet(key, title="TEST KEY"))
    key_scan = scan_sheet(key_sheet, answer_key_mode=True)
    detected_key = answers_from_detections(key_scan.detections)
    assert detected_key == key, detected_key

    student = key.copy()
    student[2] = "A"
    student[8] = None
    student_sheet = perspective_photo(create_sheet(student, title="TEST STUDENT"))
    student_scan = scan_sheet(student_sheet, answer_key_mode=False)
    detected_student = answers_from_detections(student_scan.detections)
    grade = grade_answers(detected_student, key)

    assert grade["score"] == QUESTION_COUNT - 2, grade
    assert grade["incorrect"] == 1, grade
    assert grade["blank"] == 1, grade
    print("OMR MVP tests passed.")


if __name__ == "__main__":
    main()
