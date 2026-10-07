"""Citation checking: every [n] in an answer must refer to a source we provided.

An answer citing [7] when only 5 sources were given is a sign of hallucination.
We report such citations instead of hiding them, so clients and evaluations
can see them.
"""

import re
from dataclasses import dataclass

# [1]  [1, 3]  [2][3]  (but not markdown links like [text](url) or code like a[i])
_CITATION = re.compile(r"(?<!\w)\[(\d+(?:\s*,\s*\d+)*)\](?!\()")


# Some models (gpt-oss) were trained to cite with lenticular brackets: 【1】.
# A per-character mapping also works on streamed tokens split mid-citation.
_CITATION_MARKS = str.maketrans({"【": "[", "】": "]"})


def normalize_citation_marks(text: str) -> str:
    """Rewrite model-specific citation brackets to our [n] format."""
    return text.translate(_CITATION_MARKS)


@dataclass(frozen=True, slots=True)
class CitationCheck:
    cited: list[int]  # valid source numbers, in order of first citation
    invalid: list[int]  # numbers that match no source


def check_citations(answer: str, source_count: int) -> CitationCheck:
    seen: list[int] = []
    for match in _CITATION.finditer(answer):
        for number in (int(n) for n in match.group(1).split(",")):
            if number not in seen:
                seen.append(number)
    valid = [n for n in seen if 1 <= n <= source_count]
    return CitationCheck(cited=valid, invalid=[n for n in seen if n not in valid])
