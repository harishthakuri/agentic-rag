"""Typed application settings, loaded from environment variables and `.env`.

Settings are created once (see `get_settings`) and passed explicitly to the
composition root. Nothing in the codebase reads `os.environ` directly.
"""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    TEST = "test"
    PROD = "prod"


class RerankerKind(StrEnum):
    LLM = "llm"
    NONE = "none"
    CROSS_ENCODER = "cross_encoder"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Application -------------------------------------------------------
    app_name: str = "Simple RAG"
    app_env: Environment = Environment.DEV
    log_level: str = "INFO"
    log_json: bool = False
    cors_origins: list[str] = Field(default_factory=list)
    # Require `Authorization: Bearer <api key>` on /api/v1. Disable only for local experiments.
    auth_enabled: bool = True

    # --- Database ----------------------------------------------------------
    # Runtime connection: least-privilege role (DML only).
    database_url: SecretStr
    # Migration connection: schema owner role (DDL). Only Alembic uses this.
    migrations_database_url: SecretStr | None = None
    db_pool_size: int = 5
    db_max_overflow: int = 10
    db_pool_timeout_seconds: int = 30
    db_pool_recycle_seconds: int = 1800
    db_echo: bool = False

    # --- Chat / agent LLM (any OpenAI-compatible endpoint) -----------------
    llm_base_url: str = "http://localhost:11434/v1"
    llm_api_key: SecretStr = SecretStr("ollama")
    llm_model: str = "gpt-oss:20b"
    llm_timeout_seconds: float = 120.0
    # Reasoning models only ("low" | "medium" | "high"); empty for models without it.
    llm_reasoning_effort: str | None = "low"

    # --- Embeddings (configured separately from the chat model) ------------
    embedding_base_url: str = "http://localhost:11434/v1"
    embedding_api_key: SecretStr = SecretStr("ollama")
    embedding_model: str = "qwen3-embedding:8b"
    embedding_dim: int = Field(default=1024, gt=0, le=2000)  # pgvector HNSW limit for `vector`
    embedding_batch_size: int = Field(default=32, gt=0)
    embedding_timeout_seconds: float = 120.0

    # --- Ingestion ---------------------------------------------------------
    max_upload_mb: int = Field(default=25, gt=0)
    chunk_target_tokens: int = Field(default=500, ge=100)
    chunk_overlap_tokens: int = Field(default=60, ge=0)
    ingestion_max_attempts: int = Field(default=3, ge=1)
    worker_poll_interval_seconds: float = Field(default=2.0, gt=0)
    # A running job whose worker hasn't finished within the lease is assumed crashed
    # and handed to another worker.
    worker_job_lease_seconds: int = Field(default=900, gt=0)

    # --- Retrieval ---------------------------------------------------------
    reranker: RerankerKind = RerankerKind.LLM
    reranker_model: str | None = None  # defaults to llm_model
    # Reasoning models (gpt-oss, o-series) only; set empty for models without it.
    reranker_reasoning_effort: str | None = "low"
    # First-stage candidates the reranker re-orders. Measured locally (gpt-oss:20b,
    # M1 Max): 10 ≈ 3.4 s (one batch), 20 ≈ 7.8 s; Ollama runs batches sequentially.
    rerank_depth: int = Field(default=10, ge=1, le=100)
    reranker_batch_size: int = Field(default=10, ge=1)
    reranker_max_concurrency: int = Field(default=2, ge=1)

    # --- Answering (/ask) --------------------------------------------------
    answer_context_tokens: int = Field(default=3000, ge=200)  # budget for sources in the prompt
    answer_min_rerank_grade: int = Field(default=1, ge=0, le=3)  # drop reranked chunks below

    # --- Agent (/agent/ask) ------------------------------------------------
    agent_max_tool_calls: int = Field(default=6, ge=1, le=12)
    agent_max_prompt_tokens: int = Field(default=24_000, ge=2000)  # then: answer, no more tools
    agent_timeout_seconds: float = Field(default=180.0, gt=0)
    agent_search_top_k: int = Field(default=5, ge=1, le=20)  # passages per search
    # The agent judges relevance itself; reranking each of its searches costs seconds.
    agent_rerank: bool = False

    # --- Storage -----------------------------------------------------------
    storage_dir: Path = Path("./data/uploads")

    @field_validator("database_url", "migrations_database_url")
    @classmethod
    def _require_asyncpg_driver(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().startswith("postgresql+asyncpg://"):
            raise ValueError("database URLs must use the 'postgresql+asyncpg://' scheme")
        return value

    @property
    def is_dev(self) -> bool:
        return self.app_env is Environment.DEV


@lru_cache
def get_settings() -> Settings:
    """Return the process-wide settings instance (validated on first call)."""
    return Settings()  # pyright: ignore[reportCallIssue]  # required fields come from env
