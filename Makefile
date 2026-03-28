.PHONY: install dev lint format typecheck test test-integration check docker-up docker-down clean

PYTHON_DIRS := $(strip $(wildcard axiom/) $(wildcard tests/) $(wildcard agents/))

install:
	pip install -e .

dev:
	pip install -e ".[dev]"
	playwright install chromium

lint:
	ruff check $(PYTHON_DIRS)

format:
	ruff format $(PYTHON_DIRS)

typecheck:
	mypy --strict axiom/

test:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

check: lint typecheck test

docker-up:
	docker-compose up -d --build

docker-down:
	docker-compose down -v

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/
