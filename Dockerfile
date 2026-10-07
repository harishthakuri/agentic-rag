# syntax=docker/dockerfile:1
# One image, three roles: API (default), ingestion worker (rag-worker) and
# migrations (alembic upgrade head). See docker-compose.yml.

# --- Build: resolve and install dependencies with uv --------------------------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app

# Dependencies first: this layer is cached until pyproject.toml or uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-install-project

COPY README.md LICENSE ./
COPY app ./app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# Fetch the tokenizer data now: at runtime the first download can take a minute.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN /app/.venv/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"

# --- Runtime: slim image, non-root, no build tools ------------------------------
FROM python:3.13-slim-bookworm AS runtime

RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /app --no-create-home app \
    && mkdir -p /data/uploads \
    && chown -R app:app /data

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TIKTOKEN_CACHE_DIR=/opt/tiktoken \
    STORAGE_DIR=/data/uploads \
    LOG_JSON=true

WORKDIR /app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --from=builder /opt/tiktoken /opt/tiktoken
COPY --chown=app:app alembic.ini ./
COPY --chown=app:app alembic ./alembic

USER app
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/live', timeout=2)"]

CMD ["uvicorn", "app.main:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers"]
