"""Evaluation dataset: questions with the sections that answer them.

One JSON object per line:

    {"id": "http-01", "type": "paraphrase", "question": "...",
     "expected": [["HTTP Caching", "Cache busting"]], "reference": "..."}

`expected` lists (document title, section path) pairs. Sections, unlike chunk ids,
survive re-ingestion, so labels stay valid when documents are re-chunked.
An empty list marks an unanswerable question: the right behaviour is to decline.
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class QuestionType(StrEnum):
    PARAPHRASE = "paraphrase"  # worded unlike the source
    EXACT = "exact"  # identifiers, numbers, defaults
    MULTI_PART = "multi_part"  # needs two or more sections
    UNANSWERABLE = "unanswerable"  # not in the documents


Section = tuple[str, str]  # (document title, "Heading > Subheading"); "" = intro


class EvalQuestion(BaseModel):
    id: str
    type: QuestionType
    question: str
    expected: list[Section] = Field(default_factory=list)
    reference: str

    @property
    def answerable(self) -> bool:
        return self.type is not QuestionType.UNANSWERABLE


def load_dataset(path: Path) -> list[EvalQuestion]:
    questions = [
        EvalQuestion.model_validate_json(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    ids = [q.id for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate question ids in {path}")
    for q in questions:
        if q.answerable != bool(q.expected):
            raise ValueError(f"{q.id}: answerable questions need `expected`, others must not")
    return questions


def section_of(document_title: str, heading_path: tuple[str, ...] | list[str]) -> Section:
    return (document_title, " > ".join(heading_path))
