.PHONY: install lint format typecheck test test-integration check docker-up docker-down

# Directories to lint/check — only include those that exist
PYTHON_DIRS := $(strip $(wildcard axiom/) $(wildcard tests/))

install:
	pip install -e ".[dev]"
	playwright install chromium

lint:
	@if command -v ruff >/dev/null 2>&1; then \
		if [ -n "$(PYTHON_DIRS)" ]; then \
			ruff check $(PYTHON_DIRS); \
		else \
			echo "  No Python source directories found, skipping lint"; \
		fi; \
	else \
		echo "  ruff not installed, skipping lint (run: make install)"; \
	fi

format:
	@if command -v ruff >/dev/null 2>&1; then \
		if [ -n "$(PYTHON_DIRS)" ]; then ruff format $(PYTHON_DIRS); fi; \
	else \
		echo "  ruff not installed (run: make install)"; \
	fi

typecheck:
	@if command -v mypy >/dev/null 2>&1; then \
		if [ -d "axiom" ]; then \
			mypy --strict axiom/; \
		else \
			echo "  axiom package not yet implemented, skipping typecheck"; \
		fi; \
	else \
		echo "  mypy not installed, skipping typecheck (run: make install)"; \
	fi

test:
	@if command -v pytest >/dev/null 2>&1; then \
		if [ -d "axiom" ]; then \
			python -m pytest tests/ -v -m "not integration"; \
		else \
			echo "  axiom package not yet implemented, skipping tests"; \
		fi; \
	else \
		echo "  pytest not installed, skipping tests (run: make install)"; \
	fi

test-integration:
	python -m pytest tests/ -v -m integration

check: lint typecheck test

docker-up:
	docker-compose up -d

docker-down:
	docker-compose down
