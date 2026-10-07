.DEFAULT_GOAL := help
.PHONY: help install run fmt lint typecheck arch test test-integration check migrate migration downgrade

help: ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies and git hooks
	uv sync
	uv run pre-commit install

run: ## Run the API with auto-reload
	uv run uvicorn app.main:create_app --factory --reload

fmt: ## Format code
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint code
	uv run ruff check .

typecheck: ## Static type checking
	uv run mypy app tests

arch: ## Verify Clean Architecture import contracts
	uv run lint-imports

test: ## Run unit and e2e tests
	uv run pytest -m "not integration and not live"

test-integration: ## Run integration tests (requires Docker)
	uv run pytest -m integration

check: lint typecheck arch test ## Everything CI runs

migrate: ## Apply all migrations (uses MIGRATIONS_DATABASE_URL)
	uv run alembic upgrade head

migration: ## Create a migration: make migration m="add chunks table"
	uv run alembic revision --autogenerate -m "$(m)"

downgrade: ## Roll back one migration
	uv run alembic downgrade -1
