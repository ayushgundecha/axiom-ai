# axiom-ai

A training gym for AI agents operating in real digital environments.

axiom-ai provides infrastructure for training and evaluating AI agents in controlled, reproducible environments. Agents interact with real web applications via Playwright browser automation, execute shell commands in sandboxed terminals, and are evaluated across multiple dimensions — completion, efficiency, accuracy, and safety.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                      axiom-ai Server                          │
│                    (Python + FastAPI)                          │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                  Environment Registry                    │ │
│  │  Pluggable: register any env that implements BaseEnv     │ │
│  └───────┬──────────────┬──────────────────┬───────────────┘ │
│          │              │                  │                  │
│  ┌───────▼─────┐ ┌──────▼───────┐ ┌───────▼──────────┐      │
│  │  JSON Env   │ │  WebApp Env  │ │   CLI Env        │      │
│  │  (baseline) │ │  (Playwright)│ │   (subprocess)   │      │
│  │             │ │              │ │                   │      │
│  │ Pure Python │ │ Real HTML/JS │ │  Shell commands   │      │
│  │ dict state  │ │ app in Docker│ │  in sandboxed dir │      │
│  └─────────────┘ │              │ └───────────────────┘      │
│                  │ Screenshots  │                             │
│                  │ DOM/a11y tree│                             │
│                  │ Click / Type │                             │
│                  └──────────────┘                             │
│                                                               │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │                    Core Engine                           │ │
│  │  Session Manager  ·  Trajectory Recorder                 │ │
│  │  Evaluation Engine  ·  Task Loader (YAML)                │ │
│  └─────────────────────┬───────────────────────────────────┘ │
│                        │                                      │
│  ┌─────────────────────▼───────────────────────────────────┐ │
│  │                  REST API (FastAPI)                       │ │
│  │  POST /sessions           → create session                │ │
│  │  POST /sessions/:id/step  → take action                   │ │
│  │  GET  /sessions/:id/observe → get state + screenshot      │ │
│  │  POST /sessions/:id/evaluate  → multi-signal scores       │ │
│  └─────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
                            │ HTTP
                            ▼
┌──────────────────────────────────────────────────────────────┐
│                    Agent (Claude API)                          │
│  Observes: DOM tree + screenshot + task description            │
│  Decides:  action (click, type, press_key, run_command, etc)  │
│  Loops:    until task complete or max steps reached            │
└──────────────────────────────────────────────────────────────┘
```

## Quick Start

### Local development

```bash
# 1. Install
python3 -m venv .venv && source .venv/bin/activate
make dev    # installs deps + Playwright chromium

# 2. Start the todo app (target for WebApp environment)
cd apps/todo-app && npm install && npm run build && node dist/server.js &

# 3. Start axiom server
uvicorn axiom.api.app:create_app --factory --port 8000

# 4. Run a demo
python scripts/run_demo.py --env json --task create_and_complete --agent random
python scripts/run_demo.py --env cli --task organize_files --agent random

# With Claude (requires ANTHROPIC_API_KEY):
export ANTHROPIC_API_KEY=sk-ant-...
python scripts/run_demo.py --env webapp --task add_three_todos --agent claude

# 5. Run benchmark
python scripts/benchmark.py --agent random
```

### Docker

```bash
docker-compose up --build
# axiom server: http://localhost:8000
# todo app:     http://localhost:3000

# Run agents against the dockerized stack:
python scripts/run_demo.py --env webapp --task add_three_todos --agent claude
```

## Environment Types

| Environment | Interface | Observation | Maps To |
|------------|-----------|-------------|---------|
| **WebApp** | Playwright (browser) | Screenshot + DOM tree | OSWorld |
| **CLI** | subprocess (shell) | Command output + file listing | Terminal-Bench |
| **JSON** | Pure Python (dict) | JSON state | Baseline |

## Evaluation Metrics

Every task is scored across four dimensions:

| Metric | Description | Range |
|--------|-------------|-------|
| **Completion** | Did the agent finish the task? | 0-1 |
| **Efficiency** | Steps taken vs optimal | 0-1 |
| **Accuracy** | How correct is the final state? | 0-1 |
| **Safety** | Did it avoid invalid/destructive actions? | 0-1 |

## Task Configuration

Tasks are defined as YAML files with goal specifications:

```yaml
name: "add_three_todos"
env: "webapp"
description: "Add three todos to the todo app."
app_url: "http://localhost:3000"
observation_mode: "hybrid"    # DOM + screenshots

goal:
  type: "element_count"
  selector: "[data-testid^='todo-item-']"
  count: 3

max_steps: 12
optimal_steps: 6
```

## Project Structure

```
axiom-ai/
├── axiom/                    # Python package
│   ├── models.py             # Pydantic data models (Action, Observation, StepResult, etc.)
│   ├── config.py             # pydantic-settings configuration
│   ├── exceptions.py         # Exception hierarchy
│   ├── logging.py            # structlog configuration
│   ├── core/                 # Framework machinery
│   │   ├── base_env.py       # BaseEnvironment ABC (async context manager)
│   │   ├── registry.py       # Environment registry (pluggable)
│   │   ├── session.py        # Session lifecycle manager
│   │   ├── trajectory.py     # Trajectory recording (screenshots as separate PNGs)
│   │   ├── evaluator.py      # Evaluator protocol
│   │   └── task_loader.py    # YAML task config loader
│   ├── envs/                 # Concrete environments
│   │   ├── json_env.py       # JSON state machine (baseline)
│   │   ├── webapp_env.py     # Playwright browser automation
│   │   └── cli_env.py        # Sandboxed shell execution
│   ├── api/                  # FastAPI REST API
│   │   ├── app.py            # Application factory + lifespan
│   │   ├── schemas.py        # API request/response models
│   │   ├── middleware.py      # Request ID + error handling
│   │   └── routes/           # Endpoint modules
│   └── utils/                # Utilities
│       ├── dom_parser.py     # HTML → simplified DOM (token-efficient)
│       └── screenshot.py     # Screenshot encode/decode
├── agents/                   # AI agents (interact via HTTP)
│   ├── claude_agent.py       # Claude with vision
│   └── random_agent.py       # Random baseline
├── apps/todo-app/            # TypeScript Express app (Docker target)
├── tasks/                    # YAML task definitions
├── scripts/                  # Demo and benchmark runners
├── tests/                    # pytest suite (79 unit + 8 integration)
├── Dockerfile                # axiom server (multi-stage, non-root)
└── docker-compose.yml        # Full orchestration
```

## Development

```bash
make lint        # ruff check
make format      # ruff format
make typecheck   # mypy --strict
make test        # pytest (unit tests only)
make check       # all three above
```

## Design Decisions

**Why async context managers on environments?** At scale (thousands of RL episodes), Playwright browser instances leak memory if not properly closed. `async with WebAppEnvironment(config) as env:` guarantees `cleanup()` runs even when exceptions occur.

**Why separate API schemas from domain models?** HTTP concerns (what the client sends) shouldn't leak into business logic (what the environment processes). `axiom/api/schemas.py` handles serialization; `axiom/models.py` handles validation.

**Why reset both client and server state?** A real web app has server-side state (database, in-memory store). Just refreshing the browser page doesn't clear it. `reset()` calls `/api/reset` on the target app — true reproducibility requires resetting everything.

**Why trajectory screenshots as separate files?** A single episode can generate 50+ screenshots. Storing them as inline base64 in a JSON file creates 100MB+ trajectories that are slow to parse. Separate PNGs keep the JSON small and the images accessible.

**Why full-command inspection for CLI sandboxing?** Checking only the first word (`cat` is allowed) misses `cat ../../etc/passwd`. The full command string is inspected for path traversal and dangerous patterns.

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Orchestrator | Python 3.11+ / FastAPI |
| Browser automation | Playwright |
| Target web app | TypeScript / Express |
| CLI sandbox | asyncio subprocess |
| Containers | Docker / docker-compose |
| AI agent | Anthropic Claude API (vision) |
| Data models | Pydantic v2 (strict) |
| Configuration | pydantic-settings |
| Logging | structlog |
| Testing | pytest / pytest-asyncio |
| Linting | ruff |
| Type checking | mypy (strict mode) |

## License

MIT
