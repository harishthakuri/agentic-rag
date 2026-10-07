"""LLM-as-judge: grade answers that can't be checked by string matching.

Two judgements, each with its own narrow prompt:

- Faithfulness: split the answer into factual claims and mark each as supported
  by the given sources or not. Unsupported claims are what we call
  hallucinations, even when they happen to be true.
- Correctness: compare the answer with a short reference answer.

Caveats worth knowing: a judge is itself a model and makes mistakes, and a
model judging its own answers may be lenient (self-preference bias). Use a
different, stronger judge model when you can (`--judge-model`). Treat
small differences between runs as noise.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

FAITHFULNESS_PROMPT = """You check whether an answer is supported by its sources.

1. Split the ANSWER into its distinct factual claims (skip greetings, headings and statements that information is missing).
2. For each claim, decide if the SOURCES state or directly imply it. General knowledge that is not in the sources counts as NOT supported, even if it is true.

Be strict and literal."""  # noqa: E501

CORRECTNESS_PROMPT = """You grade an answer against a reference answer.

- "correct": the answer contains the key facts of the reference and does not contradict it. Extra correct detail is fine.
- "partially_correct": some key facts are present, others are missing or wrong.
- "incorrect": the key facts are missing or contradicted.

If the reference starts with "NOT IN DOCUMENTS", the question cannot be answered from the documents: the answer is "correct" only if it says the information wasn't found (it may mention what it searched), and "incorrect" if it answers anyway."""  # noqa: E501


class Verdict(StrEnum):
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"

    @property
    def score(self) -> float:
        return {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}[self.value]


class _Claim(BaseModel):
    claim: str
    supported: bool


class _Claims(BaseModel):
    claims: list[_Claim] = Field(min_length=1)


class _Correctness(BaseModel):
    verdict: Verdict
    reason: str


@dataclass(frozen=True, slots=True)
class Faithfulness:
    supported: int
    total: int
    unsupported: list[str]

    @property
    def score(self) -> float:
        return self.supported / self.total if self.total else 1.0


@dataclass(frozen=True, slots=True)
class Correctness:
    verdict: Verdict
    reason: str


class Judge:
    def __init__(
        self, client: AsyncOpenAI, model: str, *, reasoning_effort: str | None = "low"
    ) -> None:
        self._client = client
        self.model = model
        self._reasoning_effort = reasoning_effort

    async def faithfulness(
        self, question: str, sources: list[tuple[int, str]], answer: str
    ) -> Faithfulness:
        source_text = "\n\n".join(f"[{n}] {text}" for n, text in sources) or "(no sources)"
        user = f"QUESTION:\n{question}\n\nSOURCES:\n{source_text}\n\nANSWER:\n{answer}"
        result = await self._ask(FAITHFULNESS_PROMPT, user, _claims_schema(), _Claims)
        unsupported = [c.claim for c in result.claims if not c.supported]
        return Faithfulness(
            supported=len(result.claims) - len(unsupported),
            total=len(result.claims),
            unsupported=unsupported,
        )

    async def correctness(self, question: str, reference: str, answer: str) -> Correctness:
        user = f"QUESTION:\n{question}\n\nREFERENCE ANSWER:\n{reference}\n\nANSWER:\n{answer}"
        result = await self._ask(CORRECTNESS_PROMPT, user, _correctness_schema(), _Correctness)
        return Correctness(verdict=result.verdict, reason=result.reason)

    async def _ask[M: BaseModel](
        self, system: str, user: str, schema: dict[str, Any], model: type[M]
    ) -> M:
        extra: dict[str, Any] = {}
        if self._reasoning_effort:
            extra["reasoning_effort"] = self._reasoning_effort
        response = await self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "judgement", "schema": schema, "strict": True},
            },
            temperature=0,
            **extra,
        )
        content: str | None = response.choices[0].message.content
        return model.model_validate_json(content or "{}")


def _claims_schema() -> dict[str, Any]:
    # minItems: with a bare "array" schema, gpt-oss tends to return an empty list.
    claim: dict[str, Any] = {
        "type": "object",
        "properties": {"claim": {"type": "string"}, "supported": {"type": "boolean"}},
        "required": ["claim", "supported"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"claims": {"type": "array", "minItems": 1, "maxItems": 20, "items": claim}},
        "required": ["claims"],
        "additionalProperties": False,
    }


def _correctness_schema() -> dict[str, Any]:
    verdicts: list[Literal["correct", "partially_correct", "incorrect"]] = [
        "correct",
        "partially_correct",
        "incorrect",
    ]
    return {
        "type": "object",
        "properties": {
            "verdict": {"type": "string", "enum": verdicts},
            "reason": {"type": "string"},
        },
        "required": ["verdict", "reason"],
        "additionalProperties": False,
    }
