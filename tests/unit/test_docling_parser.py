from typing import Any

import pytest

from app.application.ports.parsing import ParsedDocument, Section, UnparseableDocumentError
from app.infrastructure.parsing import DoclingPdfParser
from app.infrastructure.parsing.docling_parser import to_parsed_document


class _Fallback:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, content: bytes) -> ParsedDocument:
        self.calls += 1
        return ParsedDocument(title="from pypdf", sections=[Section("plain text", page=1)])


class _Converter:
    def __init__(self, result: Any) -> None:
        self.result = result

    def convert(self, content: bytes) -> Any:
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


# --- Fallback and loading (no Docling needed) -----------------------------------
@pytest.mark.parametrize(
    "failure", [UnparseableDocumentError("Docling: partial_success"), RuntimeError("boom")]
)
def test_falls_back_to_pypdf_when_docling_fails(failure: Exception) -> None:
    fallback = _Fallback()
    parser = DoclingPdfParser(lambda: _Converter(failure), fallback=fallback)

    assert parser.parse(b"%PDF").title == "from pypdf"
    assert fallback.calls == 1


def test_falls_back_when_the_models_cannot_load() -> None:
    def load() -> _Converter:
        raise OSError("no network to download the models")

    fallback = _Fallback()
    assert DoclingPdfParser(load, fallback=fallback).parse(b"%PDF").title == "from pypdf"


def test_models_load_once() -> None:
    loads = 0

    def load() -> _Converter:
        nonlocal loads
        loads += 1
        return _Converter(RuntimeError("unused"))

    parser = DoclingPdfParser(load, fallback=_Fallback())
    parser.parse(b"%PDF")
    parser.parse(b"%PDF")
    assert loads == 1


# --- DoclingDocument → sections (needs docling-core: uv sync --extra docling) ----
def _document() -> Any:
    doc = pytest.importorskip("docling_core.types.doc", reason="uv sync --extra docling")
    document = doc.DoclingDocument(name="guide")

    def prov(page: int) -> Any:
        return doc.ProvenanceItem(
            page_no=page, bbox=doc.BoundingBox(l=0, t=0, r=1, b=1), charspan=(0, 0)
        )

    document.add_title("Business English", prov=prov(1))
    document.add_heading("Idioms", level=1, prov=prov(2))
    document.add_heading("Quick buck", level=2, prov=prov(2))
    document.add_text(doc.DocItemLabel.TEXT, "A quick buck is easy profit.", prov=prov(2))
    group = document.add_list_group()
    document.add_list_item("Bet you made a quick buck?", parent=group, prov=prov(2))
    document.add_list_item("We did.", parent=group, prov=prov(2))
    document.add_text(doc.DocItemLabel.TEXT, "It is informal.", prov=prov(3))  # next page
    document.add_heading("Raise the bar", level=2, prov=prov(3))
    cells = [
        doc.TableCell(
            text=text,
            start_row_offset_idx=r,
            end_row_offset_idx=r + 1,
            start_col_offset_idx=c,
            end_col_offset_idx=c + 1,
            column_header=r == 0,
        )
        for r, row in enumerate([["Idiom", "Meaning"], ["Raise the bar", "Expect more"]])
        for c, text in enumerate(row)
    ]
    document.add_table(doc.TableData(num_rows=2, num_cols=2, table_cells=cells), prov=prov(3))
    document.add_picture(prov=prov(3))
    return document


def test_headings_become_section_paths_and_pages_are_kept() -> None:
    parsed = to_parsed_document(_document())

    assert parsed.title == "Business English"
    paths_and_pages = [(s.heading_path, s.page) for s in parsed.sections]
    assert paths_and_pages == [
        (("Idioms", "Quick buck"), 2),
        (("Idioms", "Quick buck"), 3),  # same section, split at the page break
        (("Idioms", "Raise the bar"), 3),  # a level-2 heading replaces its sibling
    ]
    assert parsed.sections[0].text == (
        "A quick buck is easy profit.\n\n- Bet you made a quick buck?\n- We did."
    )


def test_tables_become_markdown_tables() -> None:
    table = to_parsed_document(_document()).sections[-1].text
    assert "| Idiom" in table and "| Raise the bar" in table and "---" in table


def test_title_falls_back_to_the_pdf_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    from app.infrastructure.parsing import docling_parser

    monkeypatch.setattr(
        docling_parser, "to_parsed_document", lambda _doc: ParsedDocument(title=None, sections=[])
    )
    monkeypatch.setattr(docling_parser, "metadata_title", lambda _content: "From metadata")
    parser = DoclingPdfParser(lambda: _Converter(object()), fallback=_Fallback())
    assert parser.parse(b"%PDF").title == "From metadata"
