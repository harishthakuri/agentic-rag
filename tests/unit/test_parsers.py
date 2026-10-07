import pytest

from app.application.ports.parsing import UnparseableDocumentError
from app.infrastructure.parsing import MarkdownParser, PdfParser, PlainTextParser

MARKDOWN = b"""---
author: someone
---
# Kubernetes Networking

Intro paragraph.

## Services

Services expose pods.

### ClusterIP

Internal only.

```bash
# this is a shell comment, not a heading
kubectl get svc
```

## Ingress

Routes HTTP traffic.
"""


def test_markdown_title_and_heading_paths() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)

    assert parsed.title == "Kubernetes Networking"
    assert [(s.heading_path, s.text.split("\n")[0]) for s in parsed.sections] == [
        ((), "Intro paragraph."),  # H1 is the title, not part of the path
        (("Services",), "Services expose pods."),
        (("Services", "ClusterIP"), "Internal only."),
        (("Ingress",), "Routes HTTP traffic."),
    ]


def test_markdown_hash_lines_inside_code_fences_are_not_headings() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)

    cluster_ip = parsed.sections[2]
    assert "# this is a shell comment, not a heading" in cluster_ip.text
    assert "kubectl get svc" in cluster_ip.text


def test_markdown_front_matter_is_dropped() -> None:
    parsed = MarkdownParser().parse(MARKDOWN)
    assert all("author:" not in s.text for s in parsed.sections)


def test_markdown_without_h1_has_no_title_and_skips_empty_sections() -> None:
    parsed = MarkdownParser().parse(b"## Empty\n\n## Filled\n\nText.\n")

    assert parsed.title is None
    assert [s.heading_path for s in parsed.sections] == [("Filled",)]


def test_markdown_inline_markup_is_stripped_from_headings() -> None:
    parsed = MarkdownParser().parse(b"## The `kubectl` **CLI**\n\nText.")
    assert parsed.sections[0].heading_path == ("The kubectl CLI",)


def test_plain_text_is_one_section() -> None:
    parsed = PlainTextParser().parse(b"line one\r\n\r\nline two\r\n")
    assert [s.text for s in parsed.sections] == ["line one\n\nline two"]


def _minimal_pdf(*pages: str, title: str | None = None) -> bytes:
    """Build a tiny valid PDF with one line of text per page (no PDF library needed)."""
    objects: list[str] = []
    page_ids = [3 + 2 * i for i in range(len(pages))]
    font_id = 3 + 2 * len(pages)
    info_id = font_id + 1
    objects.append("<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(
        f"<< /Type /Pages /Kids [{' '.join(f'{i} 0 R' for i in page_ids)}] /Count {len(pages)} >>"
    )
    for page_id, text in zip(page_ids, pages, strict=True):
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Contents {page_id + 1} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
        )
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
    objects.append("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Title ({title}) >>" if title else "<< >>")

    out = "%PDF-1.4\n"
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    out += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R /Info {info_id} 0 R >>\n"
    out += f"startxref\n{xref}\n%%EOF\n"
    return out.encode("latin-1")


def test_pdf_one_section_per_page_with_page_numbers() -> None:
    parsed = PdfParser().parse(_minimal_pdf("First page text", "Second page text", title="Guide"))

    assert parsed.title == "Guide"
    assert [(s.page, s.text) for s in parsed.sections] == [
        (1, "First page text"),
        (2, "Second page text"),
    ]


def test_pdf_word_export_titles_are_ignored() -> None:
    parsed = PdfParser().parse(_minimal_pdf("Text", title="Microsoft Word - draft.docx"))
    assert parsed.title is None


def test_corrupt_pdf_is_unparseable() -> None:
    with pytest.raises(UnparseableDocumentError):
        PdfParser().parse(b"%PDF-1.4\nthis is not really a pdf")
