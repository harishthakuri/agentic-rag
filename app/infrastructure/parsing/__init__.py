from app.application.ports.parsing import DocumentParser
from app.domain.models import DocumentType
from app.infrastructure.parsing.docling_parser import DoclingPdfParser, load_docling_converter
from app.infrastructure.parsing.markdown_parser import MarkdownParser
from app.infrastructure.parsing.pdf_parser import PdfParser
from app.infrastructure.parsing.text_parser import PlainTextParser


class DefaultParserRegistry:
    def __init__(self, *, pdf: DocumentParser | None = None) -> None:
        self._parsers: dict[DocumentType, DocumentParser] = {
            DocumentType.MARKDOWN: MarkdownParser(),
            DocumentType.PLAIN_TEXT: PlainTextParser(),
            DocumentType.PDF: pdf or PdfParser(),
        }

    def for_type(self, document_type: DocumentType) -> DocumentParser:
        return self._parsers[document_type]


__all__ = [
    "DefaultParserRegistry",
    "DoclingPdfParser",
    "MarkdownParser",
    "PdfParser",
    "PlainTextParser",
    "load_docling_converter",
]
