"""Composition root.

This is the only module that knows about every concrete adapter. It wires
infrastructure implementations to application ports and hands fully built use
cases to the presentation layer. Swapping an adapter (e.g. Ollama → OpenAI,
LLM reranker → cross-encoder) is a change here and in settings, nowhere else.
"""

import importlib.util
from datetime import timedelta
from functools import partial

from openai import AsyncOpenAI

from app.application.ports.parsing import DocumentParser
from app.application.ports.reranking import Reranker
from app.application.ports.telemetry import NOOP_TELEMETRY, Telemetry
from app.application.ports.unit_of_work import UnitOfWork
from app.application.use_cases.agent import AgentAsk, GetAgentRun
from app.application.use_cases.answering import AskQuestion, ContextBuilder
from app.application.use_cases.api_keys import (
    AuthenticateApiKey,
    IssueApiKey,
    ListApiKeys,
    RevokeApiKey,
)
from app.application.use_cases.collections import (
    CreateCollection,
    DeleteCollection,
    GetCollection,
    ListCollections,
)
from app.application.use_cases.documents import DeleteDocument, GetDocument, ListDocuments
from app.application.use_cases.ingestion import (
    GetIngestionJob,
    ProcessNextIngestionJob,
    UploadDocument,
)
from app.application.use_cases.retrieval import SearchCollection
from app.application.use_cases.system.check_readiness import CheckReadiness
from app.core.config import PdfParserKind, RerankerKind, Settings
from app.domain.value_objects import EmbeddingSpec
from app.infrastructure.chunking import StructureAwareChunker, TiktokenCounter
from app.infrastructure.llm.health import ModelHealthCheck
from app.infrastructure.llm.openai_chat import OpenAICompatibleChatModel
from app.infrastructure.llm.openai_embedder import OpenAICompatibleEmbedder
from app.infrastructure.observability import OpenTelemetryAdapter, instrument_database
from app.infrastructure.parsing import (
    DefaultParserRegistry,
    DoclingPdfParser,
    PdfParser,
    load_docling_converter,
)
from app.infrastructure.persistence.database import Database
from app.infrastructure.persistence.health import DatabaseHealthCheck
from app.infrastructure.persistence.orm import EMBEDDING_DIMENSIONS
from app.infrastructure.persistence.unit_of_work import SqlAlchemyUnitOfWork
from app.infrastructure.reranking import CrossEncoderReranker, LLMReranker, load_cross_encoder
from app.infrastructure.storage.local import LocalFileStorage


class ConfigurationError(RuntimeError):
    pass


class Container:
    def __init__(self, settings: Settings) -> None:
        if settings.embedding_dim != EMBEDDING_DIMENSIONS:
            raise ConfigurationError(
                f"EMBEDDING_DIM={settings.embedding_dim} does not match the database column "
                f"vector({EMBEDDING_DIMENSIONS}). Changing it requires a migration and "
                "re-embedding every collection."
            )
        self.settings = settings
        # Spans and metrics; providers and exporters are set up per process (see
        # bootstrap/observability.py). Off: a no-op that records nothing.
        self.telemetry: Telemetry = (
            OpenTelemetryAdapter(capture_content=settings.observability_capture_content)
            if settings.observability_enabled
            else NOOP_TELEMETRY
        )
        self.database = Database(settings)
        if settings.observability_enabled:
            instrument_database(self.database.engine)
        self.storage = LocalFileStorage(settings.storage_dir)
        self.parsers = DefaultParserRegistry(pdf=self._build_pdf_parser(settings))
        self.tokens = TiktokenCounter()
        self.chunker = StructureAwareChunker(
            self.tokens,
            target_tokens=settings.chunk_target_tokens,
            overlap_tokens=settings.chunk_overlap_tokens,
        )
        self._embedding_client = AsyncOpenAI(
            base_url=settings.embedding_base_url,
            api_key=settings.embedding_api_key.get_secret_value(),
            timeout=settings.embedding_timeout_seconds,
            max_retries=2,
        )
        self.embedder = OpenAICompatibleEmbedder(
            self._embedding_client,
            EmbeddingSpec(model=settings.embedding_model, dimensions=settings.embedding_dim),
            batch_size=settings.embedding_batch_size,
            telemetry=self.telemetry,
            provider=_provider_name(settings.embedding_base_url),
        )
        self._llm_client = AsyncOpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key.get_secret_value(),
            timeout=settings.llm_timeout_seconds,
            max_retries=1,
        )
        self.reranker = self._build_reranker(settings)
        self.chat_model = OpenAICompatibleChatModel(
            self._llm_client,
            settings.llm_model,
            reasoning_effort=settings.llm_reasoning_effort or None,
            telemetry=self.telemetry,
            provider=_provider_name(settings.llm_base_url),
        )

    def _build_pdf_parser(self, settings: Settings) -> DocumentParser:
        if settings.pdf_parser is PdfParserKind.PYPDF:
            return PdfParser()
        if importlib.util.find_spec("docling") is None:
            raise ConfigurationError(
                "PDF_PARSER=docling needs the optional dependencies: uv sync --extra docling"
            )
        # The models load on the first PDF, so only the worker pays for them.
        return DoclingPdfParser(
            partial(
                load_docling_converter,
                ocr=settings.docling_ocr.value,
                tables=settings.docling_tables,
                timeout_seconds=settings.docling_timeout_seconds,
            ),
            fallback=PdfParser(),
        )

    def _build_reranker(self, settings: Settings) -> Reranker | None:
        match settings.reranker:
            case RerankerKind.NONE:
                return None
            case RerankerKind.LLM:
                return LLMReranker(
                    self._llm_client,
                    settings.reranker_model or settings.llm_model,
                    batch_size=settings.reranker_batch_size,
                    max_concurrency=settings.reranker_max_concurrency,
                    reasoning_effort=settings.reranker_reasoning_effort or None,
                )
            case RerankerKind.CROSS_ENCODER:
                if importlib.util.find_spec("sentence_transformers") is None:
                    raise ConfigurationError(
                        "RERANKER=cross_encoder needs the optional dependencies: "
                        "uv sync --extra rerank"
                    )
                return CrossEncoderReranker(
                    settings.cross_encoder_model,
                    partial(
                        load_cross_encoder,
                        settings.cross_encoder_model,
                        device=settings.cross_encoder_device or None,
                        max_length=settings.cross_encoder_max_length,
                    ),
                    batch_size=settings.cross_encoder_batch_size,
                )

    def _min_rerank_score(self) -> float:
        """The /ask relevance cut-off, on the scale of the configured reranker."""
        if self.settings.reranker is RerankerKind.CROSS_ENCODER:
            return self.settings.answer_min_cross_encoder_score
        return self.settings.answer_min_rerank_grade

    # --- Infrastructure ----------------------------------------------------
    def unit_of_work(self) -> UnitOfWork:
        return SqlAlchemyUnitOfWork(self.database.session_factory)

    # --- System ------------------------------------------------------------
    def check_readiness(self) -> CheckReadiness:
        settings = self.settings
        return CheckReadiness(
            checks=[
                DatabaseHealthCheck(self.database),
                ModelHealthCheck(
                    "embedding_model", self._embedding_client, settings.embedding_model
                ),
                ModelHealthCheck("chat_model", self._llm_client, settings.llm_model),
            ]
        )

    # --- Collections -------------------------------------------------------
    def create_collection(self) -> CreateCollection:
        return CreateCollection(self.unit_of_work, self.embedder.spec)

    def get_collection(self) -> GetCollection:
        return GetCollection(self.unit_of_work)

    def list_collections(self) -> ListCollections:
        return ListCollections(self.unit_of_work)

    def delete_collection(self) -> DeleteCollection:
        return DeleteCollection(self.unit_of_work, self.storage)

    # --- Documents and ingestion -------------------------------------------
    def upload_document(self) -> UploadDocument:
        return UploadDocument(
            self.unit_of_work,
            self.storage,
            max_bytes=self.settings.max_upload_mb * 1024 * 1024,
            max_attempts=self.settings.ingestion_max_attempts,
        )

    def process_next_ingestion_job(self) -> ProcessNextIngestionJob:
        return ProcessNextIngestionJob(
            self.unit_of_work,
            self.storage,
            self.parsers,
            self.chunker,
            self.embedder,
            lease=timedelta(seconds=self.settings.worker_job_lease_seconds),
            telemetry=self.telemetry,
        )

    def get_ingestion_job(self) -> GetIngestionJob:
        return GetIngestionJob(self.unit_of_work)

    def list_documents(self) -> ListDocuments:
        return ListDocuments(self.unit_of_work)

    def get_document(self) -> GetDocument:
        return GetDocument(self.unit_of_work)

    def delete_document(self) -> DeleteDocument:
        return DeleteDocument(self.unit_of_work, self.storage)

    # --- Retrieval ---------------------------------------------------------
    def search_collection(self) -> SearchCollection:
        return SearchCollection(
            self.unit_of_work,
            self.embedder,
            self.reranker,
            rerank_depth=self.settings.rerank_depth,
            telemetry=self.telemetry,
        )

    # --- Answering ---------------------------------------------------------
    def ask_question(self) -> AskQuestion:
        return AskQuestion(
            self.search_collection(),
            self.chat_model,
            ContextBuilder(
                self.tokens,
                token_budget=self.settings.answer_context_tokens,
                min_rerank_score=self._min_rerank_score(),
            ),
            telemetry=self.telemetry,
        )

    # --- Agent -------------------------------------------------------------
    def agent_ask(self) -> AgentAsk:
        settings = self.settings
        return AgentAsk(
            self.unit_of_work,
            self.search_collection(),
            self.chat_model,
            max_tool_calls=settings.agent_max_tool_calls,
            max_prompt_tokens=settings.agent_max_prompt_tokens,
            timeout_seconds=settings.agent_timeout_seconds,
            search_top_k=settings.agent_search_top_k,
            rerank=settings.agent_rerank and self.reranker is not None,
            telemetry=self.telemetry,
        )

    def get_agent_run(self) -> GetAgentRun:
        return GetAgentRun(self.unit_of_work)

    # --- API keys ----------------------------------------------------------
    def authenticate_api_key(self) -> AuthenticateApiKey:
        return AuthenticateApiKey(self.unit_of_work)

    def issue_api_key(self) -> IssueApiKey:
        return IssueApiKey(self.unit_of_work)

    def list_api_keys(self) -> ListApiKeys:
        return ListApiKeys(self.unit_of_work)

    def revoke_api_key(self) -> RevokeApiKey:
        return RevokeApiKey(self.unit_of_work)

    # --- Lifecycle ---------------------------------------------------------
    async def warm_up(self) -> None:
        """Load local models now, so the first request doesn't pay for it. Only the
        API calls this: the worker and the CLI never rerank."""
        if isinstance(self.reranker, CrossEncoderReranker):
            await self.reranker.warm_up()

    async def aclose(self) -> None:
        await self._embedding_client.close()
        await self._llm_client.close()
        await self.database.dispose()


def _provider_name(base_url: str) -> str:
    """`gen_ai.provider.name` for traces, guessed from the endpoint."""
    url = base_url.lower()
    if "openai.com" in url:
        return "openai"
    if ":11434" in url or "ollama" in url:
        return "ollama"
    return "openai_compatible"
