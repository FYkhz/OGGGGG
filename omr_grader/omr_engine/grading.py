"""Grading engine: compares final answers with the answer key."""

from __future__ import annotations

from typing import Mapping, Sequence

from .classifier import BLANK, CLEAR, MULTIPLE, REVIEW

BLANK_LABEL = "Blank"


def final_answer(question, overrides: Mapping[int, str | None]) -> tuple[str | None, str]:
    """(answer, source). Answer is a letter, 'A+C' for multiple, None for blank.

    Source is 'teacher' for a manual decision, 'auto' for a confident machine
    decision, and 'unresolved' for a REVIEW row nobody has confirmed yet.
    """
    if question.question in overrides:
        value = overrides[question.question]
        return (None if value in (None, BLANK_LABEL) else value), "teacher"
    if question.status == REVIEW:
        return None, "unresolved"
    return question.answer, "auto"


def grade_exam(
    questions: Sequence,
    key: Sequence[str],
    *,
    overrides: Mapping[int, str | None] | None = None,
    multiple_policy: str = "incorrect",
) -> dict:
    """``multiple_policy``: 'incorrect' (default) or 'partial' (credit if the key letter is among the marks)."""
    overrides = overrides or {}
    if len(questions) != len(key):
        raise ValueError(f"The sheet has {len(questions)} questions but the answer key has {len(key)}.")
    rows = []
    correct = blank = multiple = unresolved = 0
    for question, key_answer in zip(questions, key):
        answer, source = final_answer(question, overrides)
        is_multiple = answer is not None and "+" in answer
        if source == "unresolved":
            result = "Needs review"
            unresolved += 1
        elif answer is None:
            result = "Blank"
            blank += 1
        elif is_multiple:
            multiple += 1
            if multiple_policy == "partial" and key_answer in answer.split("+"):
                result = "Correct"
                correct += 1
            else:
                result = "Multiple"
        elif answer == key_answer:
            result = "Correct"
            correct += 1
        else:
            result = "Incorrect"
        rows.append(
            {
                "Question": question.question,
                "Student answer": (answer or ("?" if source == "unresolved" else BLANK_LABEL)).replace("+", "/"),
                "Correct answer": key_answer,
                "Result": result,
                "Detection": question.status.title(),
                "Confidence": round(float(question.confidence)),
                "Decided by": source,
            }
        )
    total = len(key)
    incorrect = total - correct - blank - unresolved
    return {
        "score": correct,
        "total": total,
        "percentage": (correct / total * 100.0) if total else 0.0,
        "blank": blank,
        "multiple": multiple,
        "incorrect": incorrect,
        "unresolved": unresolved,
        "rows": rows,
    }


__all__ = ["grade_exam", "final_answer", "BLANK_LABEL", "CLEAR", "BLANK", "MULTIPLE", "REVIEW"]
