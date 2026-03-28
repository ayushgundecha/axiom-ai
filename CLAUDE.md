# Axiom-AI Project Guidelines

## Project Overview
axiom-ai is a training gym for AI agents operating in real digital environments.
Python package name: `axiom`. Repository name: `axiom-ai`.

## Build & Test Commands
- Install: `make install`
- Lint: `make lint`
- Format: `make format`
- Type check: `make typecheck`
- Test (unit): `make test`
- Test (integration): `make test-integration`
- All checks: `make check` (lint + typecheck + test)
- Docker up: `make docker-up`
- Docker down: `make docker-down`
- Single test: `python -m pytest tests/test_models.py -v`

## Architecture
```
axiom/                  # Python package (the framework)
  models.py             # ALL Pydantic models (Action, Observation, StepResult, TaskConfig, etc.)
  config.py             # pydantic-settings: AxiomSettings (env vars, typed config)
  exceptions.py         # Exception hierarchy (AxiomError -> EnvironmentError, SessionError, etc.)
  logging.py            # structlog configuration (session correlation IDs)
  core/                 # Abstract framework machinery
    base_env.py         # BaseEnvironment ABC (async context manager, reset/step/observe/evaluate)
    registry.py         # @register_env decorator, EnvironmentRegistry
    session.py          # SessionManager (create/get/close sessions, asyncio.Lock)
    trajectory.py       # TrajectoryRecorder (separated screenshots from JSON)
    evaluator.py        # Evaluator protocol
    task_loader.py      # YAML task loading + Pydantic validation
  envs/                 # Concrete environment implementations
    json_env.py         # JSONEnvironment (simple state machine, baseline)
    webapp_env.py       # WebAppEnvironment (Playwright, screenshots, DOM) -- maps to OSWorld
    cli_env.py          # CLIEnvironment (subprocess, sandboxed) -- maps to Terminal-Bench
  api/                  # FastAPI REST API
    app.py              # Application factory + lifespan
    schemas.py          # API-specific request/response models (separate from domain models)
    middleware.py       # Request ID, error handling
    routes/             # Route modules (sessions, environments, tasks, health)
  utils/                # Utilities
    dom_parser.py       # HTML -> simplified DOM tree (token-efficient for LLMs)
    screenshot.py       # Screenshot save/encode/resize
agents/                 # AI agents (interact via HTTP API)
  claude_agent.py       # Claude with vision (screenshots + DOM)
  random_agent.py       # Random baseline
apps/todo-app/          # TypeScript Express todo app (Docker container)
tasks/                  # YAML task definitions (json/, webapp/, cli/)
trajectories/           # Saved trajectories + screenshots
```

## Code Style & Standards
- Python 3.11+ (use modern syntax: `list[str]`, `dict[str, Any]`, `X | None`)
- ALL code must pass `mypy --strict` -- no `Any` escape hatches in core code
- Use `ruff` for linting and formatting (line-length 100)
- Every function must have type annotations
- Use `async`/`await` throughout -- this is an async-first codebase
- Pydantic v2 for all data models
- structlog for all logging (never use `print()` for operational output)
- Custom exceptions from `axiom/exceptions.py` -- never raise bare `Exception`

## Key Design Principles
- Environments implement `__aenter__`/`__aexit__` for guaranteed resource cleanup
- API schemas (axiom/api/schemas.py) are SEPARATE from domain models (axiom/models.py)
- Trajectory screenshots saved as separate PNG files, never inline base64 in JSON
- Environment `reset()` must reset BOTH client state AND server state (e.g., call /api/reset)
- CLI commands inspected fully (not just first word) for safety
- All environment operations are idempotent where possible

## Testing
- TDD approach: tests written BEFORE implementation
- `pytest` + `pytest-asyncio` for async tests
- Unit tests: `tests/test_*.py` (no external dependencies)
- Integration tests: marked with `@pytest.mark.integration` (require Docker/todo-app)
- Test fixtures in `tests/conftest.py`
- Run `make check` before every commit (enforced by Husky pre-commit hook)

## Git Workflow
- Feature branches from `main`
- Conventional commits: `feat:`, `fix:`, `refactor:`, `test:`, `docs:`, `chore:`
- Husky pre-commit: runs `make check` (lint + typecheck + tests)
- Never commit `.env`, `trajectories/` data, or `node_modules/`

## Naming Conventions
- Package: `axiom` (not minigym)
- Environment classes: `{Type}Environment` (e.g., `WebAppEnvironment`, `CLIEnvironment`)
- Test files: `test_{module}.py`
- Task configs: `{task_name}.yaml` in `tasks/{env_type}/`
- All references to "minigym" should be "axiom"

## Common Gotchas
- Playwright needs `playwright install chromium` after pip install
- Todo app must be running for webapp env integration tests
- Always use `async with` for environments, never manual `cleanup()` calls
- TaskConfig.goal is a discriminated union -- check the `type` field
- Session IDs are `uuid4().hex[:12]` -- short, URL-friendly
