"""Template-driven OMR recognition engine.

Pipeline (see ``pipeline.process_page``):

    image -> page/marker registration -> perspective correction -> illumination
    normalization -> template identification (QR) -> expected grid -> bubble
    pattern detection -> RANSAC grid fit -> local refinement -> feature
    extraction -> sheet calibration -> row classification -> confidence -> grade
"""


from .grading import grade_exam
from .pipeline import ExamScan, PageResult, QuestionResult, process_exam, process_page
from .template import Template, TemplateStore, build_template, legacy_template

__all__ = [
    "ExamScan",
    "PageResult",
    "QuestionResult",
    "Template",
    "TemplateStore",
    "build_template",
    "grade_exam",
    "legacy_template",
    "process_exam",
    "process_page",
]
