from app.application.ports.parsing import ParsedDocument, Section, UnparseableDocumentError


class PlainTextParser:
    """Plain text has no structure: one section; the chunker splits it by paragraphs."""

    def parse(self, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8-sig").replace("\r\n", "\n").strip()
        except UnicodeDecodeError as exc:
            raise UnparseableDocumentError("not valid UTF-8") from exc
        return ParsedDocument(
            title=None, sections=[Section(text=text)] if text else [], parser="text"
        )
