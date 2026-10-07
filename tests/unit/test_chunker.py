"""Chunker behaviour, using a word-count "tokenizer" so expectations are easy to read."""

import pytest

from app.application.ports.parsing import ParsedDocument, Section
from app.infrastructure.chunking import StructureAwareChunker


class WordCounter:
    def count(self, text: str) -> int:
        return len(text.split())

    def split(self, text: str, max_tokens: int) -> list[str]:
        words = text.split()
        return [" ".join(words[i : i + max_tokens]) for i in range(0, len(words), max_tokens)]


def _words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def _chunker(target: int = 20, overlap: int = 5) -> StructureAwareChunker:
    return StructureAwareChunker(WordCounter(), target_tokens=target, overlap_tokens=overlap)


def _doc(*sections: Section) -> ParsedDocument:
    return ParsedDocument(title=None, sections=list(sections))


def test_small_section_is_one_chunk_with_contextual_header() -> None:
    section = Section(text="Pods get a virtual IP.", heading_path=("Services", "ClusterIP"))

    [chunk] = _chunker().chunk(_doc(section), title="K8s Guide")

    assert chunk.text == "Pods get a virtual IP."
    assert chunk.contextual_text == (
        "Document: K8s Guide\nSection: Services > ClusterIP\n\nPods get a virtual IP."
    )
    assert chunk.heading_path == ("Services", "ClusterIP")
    assert chunk.token_count == 5


def test_sections_are_never_merged() -> None:
    chunks = _chunker().chunk(
        _doc(Section(text="alpha", heading_path=("A",)), Section(text="beta", heading_path=("B",))),
        title="T",
    )
    assert [c.text for c in chunks] == ["alpha", "beta"]


def test_large_section_splits_on_paragraphs_and_respects_budget() -> None:
    paragraphs = [_words(8, p) for p in "abcde"]  # 5 paragraphs x 8 words = 40 words
    chunks = _chunker(target=20, overlap=0).chunk(
        _doc(Section(text="\n\n".join(paragraphs))), title="T"
    )

    assert [c.text.split("\n\n") for c in chunks] == [
        paragraphs[0:2],
        paragraphs[2:4],
        [paragraphs[4]],
    ]
    assert all(c.token_count <= 20 for c in chunks)


def test_consecutive_chunks_overlap_by_whole_paragraphs() -> None:
    paragraphs = [_words(8, "a"), _words(8, "b"), _words(4, "c"), _words(8, "d")]
    chunks = _chunker(target=20, overlap=5).chunk(
        _doc(Section(text="\n\n".join(paragraphs))), title="T"
    )

    first, second = chunks
    assert first.text.endswith(paragraphs[2])
    assert second.text.startswith(paragraphs[2])  # the 4-word paragraph is carried over


def test_code_block_with_blank_lines_is_kept_whole() -> None:
    code = "```python\n" + _words(6, "x") + "\n\n" + _words(6, "y") + "\n```"
    text = "\n\n".join([_words(10, "a"), code, _words(10, "b")])

    chunks = _chunker(target=20, overlap=0).chunk(_doc(Section(text=text)), title="T")

    assert any(code in c.text for c in chunks)
    assert not any(c.text.count("```") == 1 for c in chunks)


def test_oversized_paragraph_splits_on_sentences() -> None:
    sentences = [f"Sentence {i} " + _words(6, f"s{i}") + "." for i in range(5)]
    chunks = _chunker(target=20, overlap=0).chunk(
        _doc(Section(text=" ".join(sentences))), title="T"
    )

    assert len(chunks) > 1
    assert all(c.text.endswith(".") for c in chunks)  # never cut mid-sentence


def test_giant_sentence_is_hard_split_as_last_resort() -> None:
    chunks = _chunker(target=20, overlap=0).chunk(_doc(Section(text=_words(50))), title="T")
    assert [c.token_count for c in chunks] == [20, 20, 10]


def test_page_number_is_carried_through() -> None:
    [chunk] = _chunker().chunk(_doc(Section(text="On page three.", page=3)), title="T")
    assert chunk.page == 3
    assert "Section:" not in chunk.contextual_text


def test_overlap_must_be_smaller_than_target() -> None:
    with pytest.raises(ValueError, match="overlap"):
        StructureAwareChunker(WordCounter(), target_tokens=10, overlap_tokens=10)
