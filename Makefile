# Convenience targets. Run `make help` for the list.
.DEFAULT_GOAL := help
PY ?= python
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: help venv install install-dev run frontend test lint format coupons docker-up docker-down clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: ## Create a virtual environment
	$(PY) -m venv $(VENV)

install: ## Install runtime dependencies
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt

install-dev: ## Install runtime + dev/test dependencies
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements-dev.txt

run: ## Run the FastAPI backend (reload) from the repo root
	$(BIN)/uvicorn app.main:app --reload --app-dir backend --host 0.0.0.0 --port 8000

frontend: ## Run the Streamlit frontend
	BACKEND_URL=$${BACKEND_URL:-http://localhost:8000} $(BIN)/streamlit run frontend/streamlit_app.py

test: ## Run the test suite
	$(BIN)/pytest

lint: ## Lint with ruff
	$(BIN)/ruff check backend tests

format: ## Auto-format with ruff
	$(BIN)/ruff format backend tests
	$(BIN)/ruff check --fix backend tests

coupons: ## Refresh the coupon catalogue
	$(BIN)/python scripts/refresh_coupons.py

docker-up: ## Build and start the full stack with Docker Compose
	docker compose up --build

docker-down: ## Stop the Docker Compose stack
	docker compose down

clean: ## Remove caches and build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache *.egg-info
