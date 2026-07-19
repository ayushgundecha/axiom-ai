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
    axiomchat_env.py    # AxiomChatEnvironment(WebAppEnvironment) -- the deterministic mini-Slack (4th env)
    cli_env.py          # CLIEnvironment (subprocess, sandboxed) -- maps to Terminal-Bench
    __init__.py         # imports all four envs so registration side-effects fire
  robustness/           # Reward Robustness Benchmark (the star deliverable)
    proxies.py          # v0/v1 gameable PROXY rewards (DomRegex/Keyword/Count/Judge) -- what the agent is scored on
    oracles.py          # deterministic ORACLE (privileged ground truth, referee only, never the live reward)
    oracle_client.py    # fetch oracle state via X-Oracle-Token (harness-side ONLY)
    hardening.py        # named-defense catalog ("TDD for rewards"); v1 = v0 + defenses_for_task
    exploits/library.py # exploit action generators (incl. live-discovered ones)
    corpus.py           # permanent regression corpus (exploit + honest cases); the CI no-false-negative gate
    metrics.py          # RRS = (1 - hack_rate) * honest_fidelity; macro over tasks
    judge_reward.py     # LLM-as-judge proxy + 5-defense hardened judge
    judge_substudy.py   # naive-vs-hardened judge under prompt injection
    simulator.py        # deterministic in-memory AxiomChat (offline mode, zero API keys)
    labeler.py          # label an episode: proxy/oracle pass, hack, mode
    report.py           # build_report (top-level meta: mode + model labels) + write_report
    seeds.py            # held-out train/eval split helpers
  api/                  # FastAPI REST API
    app.py              # Application factory + lifespan (registers all four envs incl. axiomchat)
    schemas.py          # API-specific request/response models (separate from domain models)
    middleware.py       # Request ID, error handling
    routes/             # Route modules (sessions, environments, tasks, trajectories, health)
  utils/                # Utilities
    dom_parser.py       # HTML -> simplified DOM tree (token-efficient for LLMs)
    screenshot.py       # Screenshot save/encode/resize
agents/                 # AI agents (interact via HTTP API or drive envs directly)
  claude_agent.py       # ClaudeAgent -- vision + DOM; multi-provider (Anthropic OR Gemini, routed by model id)
  exploiter_agent.py    # ExploiterAgent -- briefed on the PROXY spec only, takes the laziest path to reward
  random_agent.py       # Random baseline
apps/todo-app/          # TypeScript Express todo app (Docker container)
apps/axiomchat/         # AxiomChat: React+Vite+TS+Tailwind SPA + Express (seeded, token-gated oracle)
tasks/                  # YAML task definitions (json/, webapp/, axiomchat/, cli/)
  axiomchat/exploits/catalog.yaml   # named reward-hacking patterns (24; scripted ones become corpus cases)
scripts/
  run_robustness.py     # the benchmark harness (offline sim | live browser | live LLM); holds the oracle token
  run_demo.py           # single-agent episode via the HTTP API
static/                 # the Axiom Console -- demo.html - replay.html - robustness.html (shared theme + nav)
reports/                # robustness.json (offline) + robustness_live*.json + transcripts/ (curated evidence)
trajectories/           # Saved trajectories + screenshots (never committed; curated copies live in reports/transcripts/)
```

**Two runners, one interface.** Every env implements `BaseEnvironment` + registry. The
**interactive API** (`run_demo.py` -> FastAPI :8000) answers "can the agent do the task?"
(4-dim evaluation). The **benchmark harness** (`run_robustness.py`, drives envs directly)
answers "can the agent cheat the reward?" (proxy vs oracle out-of-band -> RRS). The harness
holds `X-Oracle-Token` and grades from the oracle-state diff AFTER each episode -- it is never
in the serving path, so the agent can never reach ground truth. The **Axiom Console** (static
pages on GitHub Pages) reads the shared trajectory + report outputs: Demo, Replay (with the
REWARD HACK verdict banner), Leaderboard.

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


<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:ca08a54f -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

## Session Completion

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd dolt push
   git push
   git status  # MUST show "up to date with origin"
   ```
5. **Clean up** - Clear stashes, prune remote branches
6. **Verify** - All changes committed AND pushed
7. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds
<!-- END BEADS INTEGRATION -->
