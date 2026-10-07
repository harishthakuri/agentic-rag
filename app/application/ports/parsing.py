"""Parsing: raw bytes → a format-independent structure the chunker understands."""

from dataclasses import dataclass, field
from typing import Protocol

from app.domain.models import DocumentType


@dataclass(frozen=True, slots=True)
class Section:
    """A run of text under one heading (Markdown) or on one page (PDF)."""

    text: str
    heading_path: tuple[str, ...] = ()  # e.g. ("Services", "ClusterIP")
    page: int | None = None  # 1-based, PDFs only


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    title: str | None
    sections: list[Section] = field(default_factory=list)


class DocumentParser(Protocol):
    def parse(self, content: bytes) -> ParsedDocument: ...


class ParserRegistry(Protocol):
    def for_type(self, document_type: DocumentType) -> DocumentParser: ...


class UnparseableDocumentError(Exception):
    """The file is corrupt or not what its type claims. Retrying will not help."""
