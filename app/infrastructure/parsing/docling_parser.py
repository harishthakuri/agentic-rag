"""PDF → sections with Docling: real headings, reading order, tables and OCR.

pypdf (pdf_parser.py) only reads the text layer, page by page. Docling looks at
each page with small local models (see docs/diagrams or the Docling docs):

- a layout model (Heron) finds titles, section headings, paragraphs, lists,
  tables and pictures, and the reading order (multi-column pages included);
- TableFormer rebuilds table rows and columns, which we keep as Markdown tables;
- an OCR engine reads pages that have no text layer (scans). "auto" picks macOS
  Vision (ocrmac) on a Mac, otherwise RapidOCR.

So a PDF gets the same structure as a Markdown file: one section per heading,
with its heading path, and still the page number for citations. A section that
continues onto the next page is split at the page break, so every section keeps
a single, correct page number.

Design choices:
- Docling (and PyTorch) is an optional extra: `uv sync --extra docling`. It is
  imported only when the converter is loaded, once per worker process.
- Conversion is slow, synchronous work (about 0.1 s per text page and more per
  scanned page), so the ingestion use case runs parsers in a worker thread.
- If Docling fails or exceeds its time limit, the PDF is parsed with pypdf
  instead: plain text per page is better than a failed upload.
"""

import dataclasses
import io
import logging
import threading
from collections.abc import Callable
from typing import Any, Protocol

from app.application.ports.parsing import (
    DocumentParser,
    ParsedDocument,
    Section,
    UnparseableDocumentError,
)
from app.infrastructure.parsing.pdf_parser import metadata_title

logger = logging.getLogger(__name__)

# Block kinds whose text is skipped: a picture item has no text of its own (text
# found inside it, like the labels of a diagram, comes as separate child items,
# which we keep), and Docling already keeps page headers/footers out of the body.
_SKIPPED_LABELS = {"picture", "chart", "page_header", "page_footer"}

OCR_ENGINES = ("auto", "ocrmac", "rapidocr", "easyocr", "tesseract", "off")


class PdfConverter(Protocol):
    def convert(self, content: bytes) -> Any:
        """Return a `docling_core` DoclingDocument, or raise UnparseableDocumentError."""
        ...


def load_docling_converter(
    *, ocr: str = "auto", tables: bool = True, timeout_seconds: float | None = None
) -> PdfConverter:
    """Build Docling's PDF pipeline (downloads its models from Hugging Face on first use)."""
    # Optional dependency: imported here, not at module import time.
    from docling.datamodel import pipeline_options as options
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption

    pipeline = options.PdfPipelineOptions()
    pipeline.do_table_structure = tables
    # Headings come out of the layout model all at level 1. This step infers real
    # levels from numbering ("2.1" under "2."), font size and the PDF's bookmarks,
    # so "2.1 North" keeps its parent: ("2. Regional results", "2.1 North").
    pipeline.heading_hierarchy_options.enabled = True
    pipeline.generate_parsed_pages = True  # the font sizes it needs
    pipeline.document_timeout = timeout_seconds
    pipeline.do_ocr = ocr != "off"
    match ocr:
        case "ocrmac":
            pipeline.ocr_options = options.OcrMacOptions()
        case "rapidocr":
            pipeline.ocr_options = options.RapidOcrOptions()
        case "easyocr":
            pipeline.ocr_options = options.EasyOcrOptions()
        case "tesseract":
            pipeline.ocr_options = options.TesseractOcrOptions()
        case _:  # "auto" (Docling's default) or "off"
            pass
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline)}
    )
    converter.initialize_pipeline(InputFormat.PDF)  # load the models now, not mid-document
    return _DoclingConverter(converter)


class _DoclingConverter:
    def __init__(self, converter: Any) -> None:
        self._converter = converter

    def convert(self, content: bytes) -> Any:
        from docling.datamodel.base_models import ConversionStatus
        from docling_core.types.io import DocumentStream

        source = DocumentStream(name="document.pdf", stream=io.BytesIO(content))
        result = self._converter.convert(source, raises_on_error=False)
        if result.status is not ConversionStatus.SUCCESS:
            # PARTIAL_SUCCESS: some pages failed or the time limit was reached.
            errors = "; ".join(e.error_message for e in result.errors)[:300]
            raise UnparseableDocumentError(f"Docling: {result.status.value} {errors}".strip())
        return result.document


class DoclingPdfParser:
    def __init__(self, load: Callable[[], PdfConverter], *, fallback: DocumentParser) -> None:
        self._load = load
        self._fallback = fallback
        self._converter: PdfConverter | None = None
        self._lock = threading.Lock()

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            document = self._get_converter().convert(content)
        except Exception as exc:  # any Docling failure: plain text is better than nothing
            logger.warning("Docling could not parse the PDF, using pypdf instead: %s", exc)
            fallback = self._fallback.parse(content)
            return dataclasses.replace(fallback, parser=f"{fallback.parser}-fallback")
        parsed = to_parsed_document(document)
        if parsed.title is None:  # the layout model rarely labels a title: use the metadata
            parsed = dataclasses.replace(parsed, title=metadata_title(content))
        return parsed

    def _get_converter(self) -> PdfConverter:
        with self._lock:  # load the models once, even if two threads ask at once
            if self._converter is None:
                self._converter = self._load()
            return self._converter


# --- DoclingDocument → ParsedDocument ------------------------------------------
def to_parsed_document(document: Any) -> ParsedDocument:
    """Walk the document's body in reading order and group blocks into sections.

    A new section starts at every heading and at every page break, so each section
    has one heading path and one page.
    """
    title: str | None = None
    headings: list[tuple[int, str]] = []  # the open headings: (level, text)
    sections: list[Section] = []
    blocks: list[str] = []
    key: tuple[tuple[str, ...], int | None] = ((), None)

    def flush() -> None:
        text = "\n\n".join(blocks).strip()
        if text:
            sections.append(Section(text=text, heading_path=key[0], page=key[1]))
        blocks.clear()

    for item, _depth in document.iterate_items(traverse_pictures=True):
        label = item.label.value
        page = item.prov[0].page_no if item.prov else key[1]
        if label == "title":
            text = _clean(item.text)
            if title is None and text:
                title = text
                continue
            label = "section_header"  # a second title acts as a top-level heading
        if label == "section_header":
            flush()
            level = getattr(item, "level", 1)
            while headings and headings[-1][0] >= level:
                headings.pop()
            if text := _clean(item.text):
                headings.append((level, text))
            key = (tuple(h for _, h in headings), page)
            continue
        if label in _SKIPPED_LABELS:
            continue
        block = _block_text(item, label, document)
        if not block:
            continue
        if page != key[1]:
            flush()
            key = (key[0], page)
        if label == "list_item" and blocks and blocks[-1].startswith("- "):
            blocks[-1] += "\n" + block  # keep a list together as one paragraph
        else:
            blocks.append(block)
    flush()
    return ParsedDocument(title=title, sections=sections, parser="docling")


def _block_text(item: Any, label: str, document: Any) -> str:
    if label == "table":
        return str(item.export_to_markdown(doc=document)).strip()
    text = _clean(getattr(item, "text", ""))
    if not text:
        return ""
    if label == "list_item":
        return f"- {text}"
    if label == "code":
        return f"```\n{text}\n```"
    return text


def _clean(text: str) -> str:
    return " ".join(text.split()) if "\n" not in text else text.strip()
