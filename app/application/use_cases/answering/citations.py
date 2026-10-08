"""Citation checking: every [n] in an answer must refer to a source we provided.

An answer citing [7] when only 5 sources were given is a sign of hallucination.
We report such citations instead of hiding them, so clients and evaluations
can see them.

Grounding guardrail: an answer that cites no source at all, and is not a "couldn't
find" reply, came from the model's own memory rather than the documents. The
prompts forbid that, but a prompt is a request, not a guarantee (gpt-oss answered
a question the documents don't cover, from memory and partly wrong). So the use
cases check it in code and withhold such answers (see `is_grounded`).
"""

import re
from dataclasses import dataclass

# [1]  [1, 3]  [2][3]  mode[1]  (but not markdown links like [1](url)).
# Models often attach citations directly to a word ("mode[1]"), so no word-boundary
# rule; code like `items[0]` is excluded by removing code before scanning.
_CITATION = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\](?!\()")
_CODE = re.compile(r"```.*?```|`[^`\n]*`", re.DOTALL)
# The reply both prompts ask for when the documents don't answer the question.
_DECLINE = re.compile(r"couldn(?:'|\u2019)?t find", re.IGNORECASE)  # ' or U+2019


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
    for match in _CITATION.finditer(_CODE.sub(" ", answer)):
        for number in (int(n) for n in match.group(1).split(",")):
            if number not in seen:
                seen.append(number)
    valid = [n for n in seen if 1 <= n <= source_count]
    return CitationCheck(cited=valid, invalid=[n for n in seen if n not in valid])


def is_grounded(answer: str, check: CitationCheck) -> bool:
    """True if the answer cites at least one provided source, or declines to answer."""
    return bool(check.cited) or bool(_DECLINE.search(answer))
