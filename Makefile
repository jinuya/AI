.PHONY: help sync fmt lint types imports test test-all cov check clean

help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

sync: ## Install all dependencies into .venv
	uv sync --all-extras

fmt: ## Format the codebase
	uv run ruff format .
	uv run ruff check --fix .

lint: ## Lint without modifying
	uv run ruff check .
	uv run ruff format --check .

types: ## mypy strict — a CI gate (spec §4.1)
	uv run mypy

imports: ## Verify the risk engine cannot be bypassed (acceptance criterion #1)
	uv run lint-imports

test: ## Unit + property + replay + chaos, no external infrastructure
	uv run pytest -m "not integration and not live_llm and not slow"

test-all: ## Everything, including tests that need Postgres/Redis or an API key
	uv run pytest

cov: ## Coverage report
	uv run pytest -m "not integration and not live_llm and not slow" \
		--cov=atrader --cov-report=term-missing --cov-report=html

check: lint types imports test ## Everything CI runs

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
