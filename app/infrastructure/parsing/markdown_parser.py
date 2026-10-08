"""Markdown → sections, one per heading, each knowing its heading path.

    # Kubernetes Networking        ← document title (first H1)
    ## Services                    ← section path: ("Services",)
    ### ClusterIP                  ← section path: ("Services", "ClusterIP")

A small line-based parser is enough here, and it gets the one subtle case
right: `#` lines inside fenced code blocks (shell comments!) are not headings.
"""

import re

from app.application.ports.parsing import ParsedDocument, Section, UnparseableDocumentError

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})")
_FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
# Emphasis and code markers. Underscores only at word edges: `work_mem` keeps its own.
_INLINE_MARKUP = re.compile(r"[*`]|(?<!\w)_+|_+(?!\w)")


class MarkdownParser:
    def parse(self, content: bytes) -> ParsedDocument:
        try:
            text = content.decode("utf-8-sig")
        except UnicodeDecodeError as exc:
            raise UnparseableDocumentError("not valid UTF-8") from exc
        text = _FRONT_MATTER.sub("", text.replace("\r\n", "\n"))

        title: str | None = None
        sections: list[Section] = []
        headings: list[tuple[int, str]] = []  # the current path, as (level, text)
        buffer: list[str] = []
        open_fence: str | None = None

        def flush() -> None:
            body = "\n".join(buffer).strip()
            buffer.clear()
            if body:
                path = tuple(h for level, h in headings if not (level == 1 and h == title))
                sections.append(Section(text=body, heading_path=path))

        for line in text.split("\n"):
            fence = _FENCE.match(line)
            if fence:
                marker = fence.group(1)
                if open_fence is None:
                    open_fence = marker
                elif marker[0] == open_fence[0] and len(marker) >= len(open_fence):
                    open_fence = None
            elif open_fence is None and (heading := _HEADING.match(line)):
                flush()
                level = len(heading.group(1))
                heading_text = _INLINE_MARKUP.sub("", heading.group(2)).strip()
                if level == 1 and title is None:
                    title = heading_text
                headings = [(lvl, h) for lvl, h in headings if lvl < level]
                headings.append((level, heading_text))
                continue
            buffer.append(line)
        flush()

        return ParsedDocument(title=title, sections=sections, parser="markdown")
