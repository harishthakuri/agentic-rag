"""PDF → one section per page, so answers can cite page numbers.

Text extraction quality depends on how the PDF was produced: digitally
created PDFs work well; scanned PDFs contain only images and yield no text
(they would need OCR). Multi-column layouts and tables may come out jumbled;
layout-aware parsers (e.g. Docling) are the upgrade path behind the same port.
"""

import io
import re

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from app.application.ports.parsing import ParsedDocument, Section, UnparseableDocumentError

_HYPHENATED_LINE_BREAK = re.compile(r"(\w)-\n(\w)")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_SPACES = re.compile(r"[ \t]+\n")


class PdfParser:
    def parse(self, content: bytes) -> ParsedDocument:
        try:
            reader = PdfReader(io.BytesIO(content))
            if reader.is_encrypted and not reader.decrypt(""):
                raise UnparseableDocumentError("the PDF is password-protected")
            sections = [
                Section(text=text, page=number)
                for number, page in enumerate(reader.pages, start=1)
                if (text := _clean(page.extract_text() or ""))
            ]
            title = _title(reader)
        except (PdfReadError, ValueError, KeyError) as exc:
            raise UnparseableDocumentError(f"invalid PDF ({exc})") from exc
        return ParsedDocument(title=title, sections=sections)


def _clean(text: str) -> str:
    text = _HYPHENATED_LINE_BREAK.sub(r"\1\2", text)  # "embed-\nding" → "embedding"
    text = _TRAILING_SPACES.sub("\n", text)
    return _EXCESS_BLANK_LINES.sub("\n\n", text).strip()


def _title(reader: PdfReader) -> str | None:
    metadata = reader.metadata
    title = (metadata.title or "").strip() if metadata else ""
    # Word exports often set titles like "Microsoft Word - draft_v3.docx": not useful.
    if not title or title.lower().startswith("microsoft word") or title.endswith(".docx"):
        return None
    return title
