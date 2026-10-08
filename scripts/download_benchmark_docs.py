"""Download the benchmark corpus: official docs pages that overlap on purpose.

    uv run python scripts/download_benchmark_docs.py [--out data/benchmark/docs]

The sample corpus is too easy to compare rerankers (everything scores 1.00), so
the benchmark uses ~45 Kubernetes, PostgreSQL and MDN HTTP pages plus the pgvector
README, with many look-alike sections (GIN vs GiST, readiness vs liveness probes,
ETag vs Last-Modified). See docs/RERANKING.md, section 7.

The pages are third-party content under their own licences (not this repository's
MIT licence). This script downloads pinned versions, converts them to plain
Markdown the app can ingest, and writes the notices the licences require:
ATTRIBUTION.md (every source URL, licence and what was changed) and the PostgreSQL
License text next to the PostgreSQL and pgvector files.
"""

import argparse
import html
import re
import sys
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

import httpx

ROOT = Path(__file__).resolve().parents[1]

# Pinned upstream versions: the same commit gives the same corpus, so results
# stay comparable over time. Bump deliberately, then re-run the evaluation.
KUBERNETES_SHA = "aac74ad6c5291fe88f2d7d0c4c13ccd37203984c"  # kubernetes/website, 2026-10-08
MDN_SHA = "259ae6f55fa8009dd492e010e497c8a18737523c"  # mdn/content, 2026-10-07
POSTGRES_VERSION = "16"  # matches the database we run; postgresql.org serves only the latest 16.x
PGVECTOR_TAG = "v0.8.2"  # matches the extension we run

KUBERNETES_PAGES = [
    "concepts/services-networking/service",
    "concepts/services-networking/ingress",
    "concepts/services-networking/ingress-controllers",
    "concepts/services-networking/gateway",
    "concepts/services-networking/endpoint-slices",
    "concepts/services-networking/network-policies",
    "concepts/services-networking/dns-pod-service",
    "concepts/services-networking/service-traffic-policy",
    "concepts/services-networking/topology-aware-routing",
    "concepts/workloads/controllers/deployment",
    "concepts/workloads/controllers/statefulset",
    "concepts/workloads/controllers/daemonset",
    "concepts/workloads/controllers/replicaset",
    "concepts/workloads/controllers/job",
    "concepts/configuration/configmap",
    "concepts/configuration/secret",
    "concepts/workloads/pods/probes",
    "concepts/workloads/pods/pod-lifecycle",
    "tasks/configure-pod-container/configure-liveness-readiness-startup-probes",
]

POSTGRES_PAGES = [
    "indexes-intro",
    "indexes-types",
    "indexes-multicolumn",
    "indexes-partial",
    "indexes-expressional",
    "indexes-index-only-scans",
    "brin-intro",
    "textsearch-intro",
    "textsearch-controls",
    "textsearch-tables",
    "textsearch-indexes",
    "routine-vacuuming",
    "runtime-config-autovacuum",
    "mvcc-intro",
    "transaction-iso",
    "using-explain",
    "runtime-config-resource",
]

MDN_PAGES = [
    "guides/caching",
    "guides/conditional_requests",
    "reference/headers/cache-control",
    "reference/headers/etag",
    "reference/headers/last-modified",
    "reference/headers/vary",
    "reference/headers/if-none-match",
    "reference/headers/if-modified-since",
    "reference/status/304",
]


@dataclass(frozen=True)
class Source:
    name: str
    licence: str
    licence_url: str
    homepage: str


KUBERNETES = Source(
    "Kubernetes documentation",
    "CC BY 4.0",
    "https://creativecommons.org/licenses/by/4.0/",
    "https://github.com/kubernetes/website",
)
POSTGRES = Source(
    "PostgreSQL documentation",
    "PostgreSQL License",
    "https://www.postgresql.org/about/licence/",
    "https://www.postgresql.org/docs/",
)
MDN = Source(
    "MDN Web Docs",
    "CC BY-SA 2.5",
    "https://creativecommons.org/licenses/by-sa/2.5/",
    "https://github.com/mdn/content",
)
PGVECTOR = Source(
    "pgvector README",
    "PostgreSQL License",
    "https://github.com/pgvector/pgvector/blob/master/LICENSE",
    "https://github.com/pgvector/pgvector",
)


@dataclass(frozen=True)
class LicenceFile:
    url: str
    path: Path  # relative to the output directory; no extension, so it is never ingested


# The PostgreSQL License asks for its copyright notice and text in every copy.
LICENCE_FILES = [
    LicenceFile(
        f"https://raw.githubusercontent.com/postgres/postgres/REL_{POSTGRES_VERSION}_STABLE/COPYRIGHT",
        Path("postgresql") / "LICENSE",
    ),
    LicenceFile(
        f"https://raw.githubusercontent.com/pgvector/pgvector/{PGVECTOR_TAG}/LICENSE",
        Path("pgvector") / "LICENSE",
    ),
]


@dataclass(frozen=True)
class Page:
    source: Source
    url: str  # what we download
    view_url: str  # where a person reads it
    path: Path  # relative to the output directory


def pages() -> list[Page]:
    result: list[Page] = []
    for page in KUBERNETES_PAGES:
        result.append(
            Page(
                KUBERNETES,
                f"https://raw.githubusercontent.com/kubernetes/website/{KUBERNETES_SHA}"
                f"/content/en/docs/{page}.md",
                f"https://kubernetes.io/docs/{page}/",
                Path("kubernetes") / f"{page.rsplit('/', 1)[-1]}.md",
            )
        )
    for page in POSTGRES_PAGES:
        url = f"https://www.postgresql.org/docs/{POSTGRES_VERSION}/{page}.html"
        result.append(Page(POSTGRES, url, url, Path("postgresql") / f"{page}.md"))
    for page in MDN_PAGES:
        result.append(
            Page(
                MDN,
                f"https://raw.githubusercontent.com/mdn/content/{MDN_SHA}"
                f"/files/en-us/web/http/{page}/index.md",
                f"https://developer.mozilla.org/en-US/docs/Web/HTTP/{page}",  # MDN fixes the case
                Path("mdn-http") / f"{page.rsplit('/', 1)[-1]}.md",
            )
        )
    result.append(
        Page(
            PGVECTOR,
            f"https://raw.githubusercontent.com/pgvector/pgvector/{PGVECTOR_TAG}/README.md",
            f"https://github.com/pgvector/pgvector/blob/{PGVECTOR_TAG}/README.md",
            Path("pgvector") / "pgvector-readme.md",
        )
    )
    return result


# --- Conversion: everything becomes plain Markdown with one H1 title --------------
_FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_HEADING_ID = re.compile(r"^(#+ .*?)[ \t]*\{#[^}]*\}[ \t]*$", re.MULTILINE)  # "## Foo {#foo}"
_MD_LINK = re.compile(r"(?<!!)\[([^\]]+)\]\([^)]*\)")
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)


def _title_and_body(text: str) -> tuple[str, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        raise ValueError("no front matter")
    title_match = re.search(r"^title:\s*(.+)$", match.group(1), re.MULTILINE)
    if not title_match:
        raise ValueError("no title in front matter")
    return title_match.group(1).strip().strip("\"'"), text[match.end() :]


def _outside_code(text: str, transform: Callable[[str], str]) -> str:
    """Apply `transform` to the prose only, never inside fenced code blocks."""
    parts = re.split(r"(^```.*?^```[^\n]*$)", text, flags=re.MULTILINE | re.DOTALL)
    return "".join(p if p.startswith("```") else transform(p) for p in parts)


def _tidy(title: str, body: str) -> str:
    body = re.sub(r"[ \t]+$", "", body, flags=re.MULTILINE)  # trailing spaces
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    return f"# {title}\n\n{body}\n"


_K8S_HEADINGS = {
    "whatsnext": "What's next",
    "prerequisites": "Before you begin",
    "cleanup": "Cleaning up",
    "objectives": "Objectives",
}


def _glossary_text(match: re.Match[str]) -> str:
    attrs: dict[str, str] = dict(re.findall(r'(\w+)="([^"]*)"', match.group(1)))
    return attrs.get("text") or attrs.get("term_id", "")


def convert_kubernetes(text: str) -> str:
    """Hugo Markdown: strip front matter and shortcodes, keep the prose."""
    title, body = _title_and_body(text)

    def prose(part: str) -> str:
        part = _HTML_COMMENT.sub("", part)
        # {{< glossary_tooltip text="kubelet" term_id="kubelet" >}} -> kubelet
        part = re.sub(r"\{\{[<%]\s*glossary_tooltip\s+([^}]*?)[%>]\}\}", _glossary_text, part)
        part = re.sub(r"\{\{[<%]\s*(note|caution|warning)\s*[%>]\}\}", r"**\1:**", part)
        # Standard headings are shortcodes too: ## {{% heading "whatsnext" %}}
        part = re.sub(
            r'\{\{%\s*heading\s+"(\w+)"\s*%\}\}',
            lambda m: _K8S_HEADINGS.get(m.group(1), m.group(1).capitalize()),
            part,
        )
        part = re.sub(r"\{\{[<%].*?[%>]\}\}", "", part, flags=re.DOTALL)  # everything else
        part = _HEADING_ID.sub(r"\1", part)
        return _MD_LINK.sub(r"\1", part)

    return _tidy(title, _outside_code(body, prose))


_PROPERTIES_TABLE = re.compile(r'<table class="properties">(.*?)</table>', re.DOTALL)


def _properties_list(match: re.Match[str]) -> str:
    """The infobox at the top of header pages -> "- Header type: Response header"."""
    rows = re.findall(r"<tr>(.*?)</tr>", match.group(1), re.DOTALL)
    cells = [re.findall(r"<t[hd][^>]*>(.*?)</t[hd]>", row, re.DOTALL) for row in rows]
    return "\n".join(f"- {' '.join(c[0].split())}: {' '.join(c[1].split())}" for c in cells)


def convert_mdn(text: str) -> str:
    """MDN Markdown: strip front matter, expand KumaScript macros to their text."""
    title, body = _title_and_body(text)

    def macro(match: re.Match[str]) -> str:
        args = re.findall(r'"([^"]*)"|\'([^\']*)\'', match.group(2) or "")
        values = [a or b for a, b in args]
        if not values:
            return ""  # {{Specifications}}, {{optional_inline}}, sidebars
        # {{HTTPHeader("ETag")}} -> `ETag`; {{Glossary("ASCII")}} -> ASCII;
        # with a second argument, that is the label: {{HTTPStatus("304", "304 Not Modified")}}
        label = values[1] if len(values) > 1 else values[0]
        return f"`{label}`" if match.group(1).startswith("HTTP") else label

    def prose(part: str) -> str:
        part = _HTML_COMMENT.sub("", part)
        # Sections that only hold data tables rendered by the website.
        part = re.sub(r"^## (Specifications|Browser compatibility)[ \t]*$", "", part, flags=re.M)
        part = re.sub(r"\{\{\s*(\w+)\s*(?:\((.*?)\))?\s*\}\}", macro, part)
        part = _PROPERTIES_TABLE.sub(_properties_list, part)
        part = re.sub(r"</?kbd>", "", part)
        return _MD_LINK.sub(r"\1", part)

    return _tidy(title, _outside_code(body, prose))


def convert_pgvector(text: str) -> str:
    def prose(part: str) -> str:
        part = re.sub(r"^\[!\[.*$", "", part, flags=re.MULTILINE)  # CI badges
        return _MD_LINK.sub(r"\1", part)

    body = re.sub(r"[ \t]+$", "", _outside_code(text, prose), flags=re.MULTILINE)
    if not body.lstrip().startswith("# "):
        raise ValueError("pgvector README has no H1 title")
    return re.sub(r"\n{3,}", "\n\n", body).strip() + "\n"


class _PostgresHTML(HTMLParser):
    """postgresql.org page -> Markdown: only the documentation body (div#docContent),
    without navigation; headings, paragraphs, lists, code, definition lists, tables.

    Two structural choices keep sections meaningful: Tip/Note/Warning boxes become
    inline labels, not headings, and each configuration parameter (work_mem, ...)
    becomes a heading of its own, so it is a section that questions can point to."""

    BLOCKS: ClassVar = {"p", "div", "dl", "ul", "ol", "table", "blockquote", "dd"}
    ADMONITIONS: ClassVar = {"tip", "note", "warning", "important", "caution"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self.title: str | None = None
        self._chapter: str | None = None  # from the navigation's "Up" link
        self._depth = 0  # div depth inside docContent; 0 = outside
        self._skip_depth = 0  # >0 while inside navigation or the table of contents
        self._pre = False
        # What the captured text becomes: ("heading", level), ("label", 0), ("param", 0).
        self._heading: tuple[str, int] | None = None
        self._heading_text: list[str] = []
        self._level = 1  # Markdown level of the last heading written
        self._admonition_depth = 0  # div depth of the open Tip/Note box; 0 = none
        self._row: list[str] | None = None
        self._cell: list[str] | None = None
        self._rows_in_table = 0
        self._skip_anchor = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = dict(attrs)
        if tag == "div" and a.get("id") == "docContent":
            self._depth = 1
            return
        if not self._depth:
            return
        if tag == "div":
            self._depth += 1
            if self._skip_depth:
                self._skip_depth += 1
            elif a.get("class") in ("navheader", "navfooter", "toc"):
                self._skip_depth = 1
            elif a.get("class") in self.ADMONITIONS and not self._admonition_depth:
                self._admonition_depth = self._depth
        if self._skip_depth:
            if tag == "a" and a.get("accesskey") == "u" and self._chapter is None:
                self._chapter = _strip_number((a.get("title") or "").replace("\xa0", " "))
            return
        if tag == "a" and a.get("class") == "id_link":
            self._skip_anchor = True
        elif tag in ("h1", "h2", "h3", "h4", "h5"):
            self._heading = ("label", 0) if self._admonition_depth else ("heading", int(tag[1]))
            self._heading_text = []
        elif tag == "dt" and (a.get("id") or "").startswith("GUC-"):
            self._heading = ("param", 0)
            self._heading_text = []
        elif tag == "pre":
            self._pre = True
            self._emit("\n\n```\n")
        elif tag in self.BLOCKS:
            self._emit("\n\n")
        elif tag == "li":
            self._emit("\n- ")
        elif tag == "dt":
            self._emit("\n\n**")
        elif tag == "br":
            self._emit("\n")
        elif tag == "tr":
            self._row = []
        elif tag in ("td", "th") and self._row is not None:
            self._cell = []
        elif tag == "code" and not self._pre and self._heading is None:
            self._emit("`")

    def handle_endtag(self, tag: str) -> None:
        if not self._depth:
            return
        if tag == "div":
            self._depth -= 1
            if self._depth < self._admonition_depth:
                self._admonition_depth = 0
            if self._skip_depth:
                self._skip_depth -= 1
                return
        if self._skip_depth:
            return
        if tag == "a" and self._skip_anchor:
            self._skip_anchor = False
        elif tag in ("h1", "h2", "h3", "h4", "h5", "dt") and self._heading is not None:
            self._end_heading(*self._heading)
            self._heading = None
        elif tag == "pre":
            self._pre = False
            self._emit("\n```\n\n")
        elif tag == "dt":
            self._emit("**\n")
        elif tag in ("td", "th") and self._cell is not None and self._row is not None:
            self._row.append(re.sub(r"\s+", " ", "".join(self._cell)).strip().replace("|", "\\|"))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._emit("\n| " + " | ".join(self._row) + " |")
            if self._rows_in_table == 0:
                self._emit("\n|" + " --- |" * len(self._row))
            self._rows_in_table += 1
            self._row = None
        elif tag == "table":
            self._rows_in_table = 0
            self._emit("\n\n")
        elif tag == "code" and not self._pre and self._heading is None:
            self._emit("`")

    def _end_heading(self, kind: str, html_level: int) -> None:
        text = _strip_number(re.sub(r"\s+", " ", "".join(self._heading_text)))
        if kind == "label":  # "Tip" -> **Tip:** at the start of the box's text
            self._emit(f"\n\n**{text}:** ")
        elif kind == "param":  # "work_mem (integer)" -> a heading below the current one
            name = re.sub(r"\s*\([\w ]+\)$", "", text)
            self._emit(f"\n\n{'#' * min(self._level + 1, 6)} {name}\n\n")
        elif self.title is None:
            # Section names repeat across chapters ("Introduction"), so the title
            # names the chapter too: "Full Text Search: Introduction".
            self.title = f"{self._chapter}: {text}" if self._chapter else text
            self._emit(f"# {self.title}\n\n")
        else:
            self._level = max(2, html_level - 1)  # page H2 is the title; H3 -> ##
            self._emit(f"\n\n{'#' * self._level} {text}\n\n")

    def handle_data(self, data: str) -> None:
        if not self._depth or self._skip_depth or self._skip_anchor:
            return
        data = data.replace("\xa0", " ")
        if self._heading is not None:
            self._heading_text.append(data)
        elif self._cell is not None:
            self._cell.append(data)
        elif self._pre:
            self.out.append(data)
        else:
            self._emit(re.sub(r"\s+", " ", data))

    def _emit(self, text: str) -> None:
        if self._cell is not None:
            self._cell.append(text.replace("\n", " "))
        else:
            self.out.append(text)


def _strip_number(text: str) -> str:
    """ "11.8. Partial Indexes" / "Chapter 12. Full Text Search" -> the name only."""
    return re.sub(r"^(Chapter\s+)?[A-Z]?[\d.]+\s+", "", text.strip())


def convert_postgres(text: str) -> str:
    parser = _PostgresHTML()
    parser.feed(text)
    if parser.title is None:
        raise ValueError("no documentation content found")
    body = "".join(parser.out)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"[ \t]+\*\*\n", "**\n", body)  # "**term  **"
    body = re.sub(r"^- *\n+", "- ", body, flags=re.MULTILINE)  # <li><p>: keep "- text"
    body = re.sub(r"(\*\*[\w ]+:\*\*)[ \t]*\n+", r"\1 ", body)  # "**Tip:**" joins its paragraph
    body = re.sub(  # <pre> starts and ends with a newline
        r"```\n+(.*?)\n+```", lambda m: f"```\n{m.group(1)}\n```", body, flags=re.DOTALL
    )
    return html.unescape(re.sub(r"\n{3,}", "\n\n", body).strip()) + "\n"


CONVERTERS = {
    KUBERNETES: convert_kubernetes,
    POSTGRES: convert_postgres,
    MDN: convert_mdn,
    PGVECTOR: convert_pgvector,
}


def write_attribution(out: Path, downloaded: list[Page]) -> None:
    lines = [
        "# Benchmark corpus: sources and licences",
        "",
        "These pages are third-party documentation, kept here as a fixed test corpus for",
        "the evaluation. **They are not covered by this repository's MIT licence**: each",
        "folder stays under its original licence, listed below. The converted MDN pages",
        "are shared under the same licence as the originals (CC BY-SA 2.5).",
        "",
        "**Changes:** downloaded by `scripts/download_benchmark_docs.py` and converted to",
        "Markdown. Navigation, templates, website-only markup and embedded code-sample",
        "references were removed, and PostgreSQL titles were prefixed with their chapter",
        "name. The text is otherwise unchanged.",
        "",
        f"- Kubernetes: kubernetes/website at commit `{KUBERNETES_SHA}`",
        f"- MDN: mdn/content at commit `{MDN_SHA}`",
        f"- PostgreSQL: version {POSTGRES_VERSION} documentation",
        f"- pgvector: tag `{PGVECTOR_TAG}`",
        "",
    ]
    for source in (KUBERNETES, POSTGRES, MDN, PGVECTOR):
        folders = {p.path.parts[0] for p in downloaded if p.source is source}
        licence_notes = [
            f"The licence text is in `{f.path}`."
            for f in LICENCE_FILES
            if f.path.parts[0] in folders
        ]
        lines += [
            f"## {source.name}",
            "",
            " ".join(
                [
                    f"© the respective authors, [{source.homepage}]({source.homepage}).",
                    f"Licence: [{source.licence}]({source.licence_url}).",
                    *licence_notes,
                ]
            ),
            "",
            "| File | Source |",
            "| --- | --- |",
        ]
        lines += [f"| `{p.path}` | {p.view_url} |" for p in downloaded if p.source is source]
        lines.append("")
    (out / "ATTRIBUTION.md").write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "benchmark" / "docs")
    args = parser.parse_args()
    out: Path = args.out

    failures: list[str] = []
    downloaded: list[Page] = []
    with httpx.Client(timeout=30, follow_redirects=True) as client:
        for page in pages():
            try:
                response = client.get(page.url)
                response.raise_for_status()
                markdown = CONVERTERS[page.source](response.text)
            except (httpx.HTTPError, ValueError) as exc:
                failures.append(f"{page.url}: {exc}")
                print(f"  ✗ {page.path}  ({exc})")
                continue
            target = out / page.path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(markdown)
            downloaded.append(page)
            print(f"  ✓ {page.path}  ({len(markdown.split()):,} words)")

        for licence in LICENCE_FILES:
            try:
                response = client.get(licence.url)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                failures.append(f"{licence.url}: {exc}")
                continue
            (out / licence.path).write_text(response.text)

    write_attribution(out, downloaded)
    print(f"\n{len(downloaded)} pages in {out} (sources: ATTRIBUTION.md)")
    if failures:
        sys.exit(f"{len(failures)} page(s) failed")


if __name__ == "__main__":
    main()
