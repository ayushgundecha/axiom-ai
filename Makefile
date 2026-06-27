.PHONY: install dev lint format typecheck test test-integration check docker-up docker-down clean \
	axiomchat-install axiomchat-build axiomchat-run axiomchat-test

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

# --- AxiomChat (deterministic mini-Slack environment) ---------------------
# Build the SPA (vite) + server (tsc), then run locally on :3100. Typical flow:
#   make axiomchat-build && make axiomchat-run   # serves http://localhost:3100
# Backend unit tests (vitest): make axiomchat-test

axiomchat-install:
	cd apps/axiomchat && npm install
	cd apps/axiomchat/web && npm install

axiomchat-build: axiomchat-install
	cd apps/axiomchat/web && npm run build   # -> apps/axiomchat/web/dist (Vite SPA)
	cd apps/axiomchat && npm run build       # -> apps/axiomchat/dist (tsc server)

axiomchat-run:
	cd apps/axiomchat && PORT=3100 node dist/server.js

axiomchat-test:
	cd apps/axiomchat && npm test

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
