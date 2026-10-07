from collections.abc import AsyncIterator
from uuid import UUID

import structlog
from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.application.use_cases.agent import (
    AgentCompleted,
    AgentEvent,
    AgentQuery,
    RunStarted,
    ToolCalled,
    ToolReturned,
)
from app.application.use_cases.answering import AnswerDelta
from app.presentation.api.dependencies import AgentAskDep, GetAgentRunDep
from app.presentation.api.errors import PROBLEM_RESPONSES
from app.presentation.api.schemas.agent import (
    AgentAskRequest,
    AgentAskResponse,
    AgentRunResponse,
    ToolCallEvent,
    ToolResultEvent,
)
from app.presentation.api.schemas.ask import TokenEvent
from app.presentation.api.sse import sse_event, sse_response

logger = structlog.get_logger(__name__)

router = APIRouter(tags=["agent"], responses=PROBLEM_RESPONSES)


@router.post(
    "/collections/{collection_id}/agent/ask",
    response_model=AgentAskResponse,
    responses={
        200: {
            "description": "JSON result, or with `stream: true` a `text/event-stream` of "
            "`run`, `tool_call`/`tool_result` pairs, `token` events and `done` (or `error`)",
            "content": {"text/event-stream": {}},
        },
        503: PROBLEM_RESPONSES[422],
        504: PROBLEM_RESPONSES[422],
    },
)
async def agent_ask(
    collection_id: UUID, body: AgentAskRequest, use_case: AgentAskDep
) -> AgentAskResponse | StreamingResponse:
    """Agentic search: the model decides what to search for, how often, and when to stop.

    It can split the question into several searches, rephrase, and read around a
    passage before answering with citations. Slower than `/ask`, better for
    multi-part questions. Every run is stored: see `GET /agent/runs/{run_id}`.
    """
    query = AgentQuery(
        collection_id=collection_id, question=body.question, max_tool_calls=body.max_tool_calls
    )
    if not body.stream:
        completed, events = await use_case.execute(query)
        steps = [ToolCallEvent.from_domain(e) for e in events if isinstance(e, ToolCalled)]
        return AgentAskResponse.from_domain(completed, steps)

    stream = aiter(use_case.stream(query))
    first = await anext(stream)  # validates the collection before headers are sent
    return sse_response(_sse(first, stream))


@router.get("/agent/runs/{run_id}")
async def get_agent_run(run_id: UUID, use_case: GetAgentRunDep) -> AgentRunResponse:
    """Replay a run: the question, every tool call with its result, and the outcome."""
    return AgentRunResponse.from_domain(await use_case.execute(run_id))


async def _sse(first: AgentEvent, rest: AsyncIterator[AgentEvent]) -> AsyncIterator[str]:
    steps: list[ToolCallEvent] = []
    try:
        yield _to_sse(first, steps)
        async for event in rest:
            yield _to_sse(event, steps)
    except Exception as exc:  # headers are sent: report in-band, then end the stream
        logger.exception("agent.stream_failed")
        yield sse_event("error", {"title": "Agent run failed", "detail": type(exc).__name__})


def _to_sse(event: AgentEvent, steps: list[ToolCallEvent]) -> str:
    match event:
        case RunStarted(run_id=run_id):
            return sse_event("run", {"run_id": str(run_id)})
        case ToolCalled():
            call = ToolCallEvent.from_domain(event)
            steps.append(call)
            return sse_event("tool_call", call)
        case ToolReturned():
            return sse_event("tool_result", ToolResultEvent.from_domain(event))
        case AnswerDelta(text=text):
            return sse_event("token", TokenEvent(text=text))
        case AgentCompleted():
            return sse_event("done", AgentAskResponse.from_domain(event, steps))
