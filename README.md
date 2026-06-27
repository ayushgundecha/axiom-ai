# axiom-ai

**Training gym for AI agents operating in real digital environments.**

> Real browsers. Real terminals. Real evaluation. Not simulated — agents interact with actual applications through Playwright, execute real shell commands in sandboxes, and are scored across four dimensions.

https://github.com/user-attachments/assets/9580ca2b-677b-4cd5-82cb-965bfc0d26e8

---

## The Problem

Training AI agents to use computers requires environments that are:

- **Real** — not mocked APIs or synthetic DOM trees, but actual applications running in browsers and terminals
- **Reproducible** — every episode starts from the exact same state (both client and server)
- **Measurable** — evaluation that goes beyond pass/fail to capture *how well* the agent performed
- **Scalable** — infrastructure that supports thousands of parallel episodes for RL training

Most agent frameworks give you one of these. axiom-ai gives you all four.

---

## What It Does

```
Agent observes environment → decides action → takes action → gets reward → repeat
                ↓                                              ↓
         screenshot + DOM                              trajectory recorded
         terminal output                               evaluation scored
         JSON state                                    training data saved
```

### Four Real Environments

| Environment | What's Happening | Agent Sees | Agent Does |
|------------|-----------------|------------|------------|
| **WebApp** | Playwright controls a real Chromium browser running a real web app | Screenshots + simplified DOM tree | Click, type, scroll, press keys |
| **AxiomChat** | Playwright drives **AxiomChat** — a deterministic, resettable mini-Slack (React SPA + Express) with a token-gated ground-truth oracle | Screenshots + simplified DOM tree (stable `data-testid`) | Post, reply in thread, react, pin, resolve, search |
| **CLI** | Async subprocess in a sandboxed temp directory with 24 allowlisted commands | Terminal output + file listing | Shell commands (`grep`, `mkdir`, `cat`, etc.) |
| **JSON** | Pure Python state machine — zero dependencies, instant execution | JSON state dict | API calls (`add_todo`, `complete_todo`) |

> **AxiomChat** is the substrate for the project's reward-robustness work: every workspace is generated from a seed (`POST /api/reset {seed,scale}`) so runs are byte-reproducible, and a privileged `GET /api/_oracle/state` (gated by `X-Oracle-Token`) exposes hidden ground-truth labels that a cheap proxy reward must not see. See [`apps/axiomchat/README.md`](apps/axiomchat/README.md).

![AxiomChat — a deterministic, resettable mini-Slack training environment with sidebar, threaded messages, reactions, and a thread pane](docs/images/axiomchat.png)

### Multi-Signal Evaluation

Every task is scored across four dimensions, not just pass/fail:

| Metric | What It Measures | Example |
|--------|-----------------|---------|
| **Completion** | Did it finish the task? | All 3 todos added → 1.0 |
| **Efficiency** | Steps vs optimal | 6 steps when optimal is 6 → 1.0 |
| **Accuracy** | Quality of the final state | 2/3 correct → 0.67 |
| **Safety** | Avoided destructive actions? | No invalid commands → 1.0 |

**Rule-based evaluation** for objective tasks (count DOM elements, check file existence).
**LLM-as-judge evaluation** for subjective tasks (Claude scores against rubrics — "Is this README professional?"). Combined via a weighted `CompositeEvaluator`.

### Trajectory Recording

Every episode produces training-ready data:

```
trajectories/
  a1b2c3d4e5f6/
    trajectory.json          # 5KB — metadata + actions + rewards + evaluation
    screenshots/
      step_1.png             # what the browser showed
      step_2.png
      ...
```

Screenshots are saved as separate PNGs — not inline base64. A 20-step episode would create a 27MB+ JSON file with base64. Separate files keep trajectories small and parseable.

### Parallel Execution

Run N episodes concurrently across M tasks with bounded concurrency:

```bash
python scripts/parallel_benchmark.py --concurrency 5 --agent claude
```

asyncio semaphore-based — each episode is isolated, failures don't cascade, results aggregated with per-agent statistics.

---

## Quick Start

```bash
# Install
python3 -m venv .venv && source .venv/bin/activate
make dev

# Start the target web app
cd apps/todo-app && npm install && npm run build && node dist/server.js &

# Start axiom server
uvicorn axiom.api.app:create_app --factory --port 8000

# Run an agent
python scripts/run_demo.py --env cli --task analyze_logs --agent claude
python scripts/run_demo.py --env webapp --task add_three_todos --agent claude

# Run parallel benchmark
python scripts/parallel_benchmark.py --concurrency 3 --agent random

# Open the mission control dashboard
open http://localhost:8000/static/demo.html

# Open the trajectory replay UI
open http://localhost:8000/static/replay.html
```

Docker:
```bash
docker-compose up --build
# axiom: http://localhost:8000 | todo-app: http://localhost:3000
```

---

## Architecture

![System overview](docs/images/system-overview.png)

```
┌──────────────────────────────────────────────────────────────────┐
│                        axiom-ai Server                            │
│                       (Python + FastAPI)                           │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                   Environment Registry                        │ │
│  │   Pluggable: register any class that implements BaseEnv       │ │
│  └────────┬───────────────┬───────────────────┬─────────────────┘ │
│           │               │                   │                    │
│  ┌────────▼──────┐ ┌──────▼────────┐ ┌───────▼───────────┐       │
│  │   JSON Env    │ │  WebApp Env   │ │    CLI Env         │       │
│  │  Pure Python  │ │  Playwright   │ │  Async subprocess  │       │
│  │  dict state   │ │  + Chromium   │ │  24 safe commands  │       │
│  └───────────────┘ │  screenshots  │ │  path traversal    │       │
│                    │  DOM parser   │ │  detection          │       │
│                    └───────────────┘ └────────────────────┘       │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                      Core Engine                               │ │
│  │  Session Manager · Trajectory Recorder · Task Loader (YAML)    │ │
│  │  DefaultEvaluator · LLMJudgeEvaluator · CompositeEvaluator     │ │
│  │  ParallelRunner (asyncio semaphore-bounded)                    │ │
│  └───────────────────────┬──────────────────────────────────────┘ │
│                          │                                         │
│  ┌───────────────────────▼──────────────────────────────────────┐ │
│  │                    REST API (FastAPI)                          │ │
│  │  POST /sessions             → create session + reset env      │ │
│  │  POST /sessions/:id/step    → execute action, record step     │ │
│  │  POST /sessions/:id/evaluate → multi-signal scores            │ │
│  │  GET  /trajectories/saved   → list saved trajectories         │ │
│  └──────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
                             │ HTTP
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                      AI Agent (Claude API)                        │
│  Observes: screenshot + DOM tree + task description               │
│  Decides:  next action via vision + reasoning                     │
│  Records:  full trajectory with evaluation scores                 │
└──────────────────────────────────────────────────────────────────┘
```

---

## Task Examples

**Log analysis** (CLI — 15 steps, Claude analyzing real server logs):
```yaml
name: "analyze_logs"
env: "cli"
description: "Analyze server log files. Count errors, extract them, produce a report."
initial_state:
  files:
    - path: "logs/api_server.log"
      content: "2024-01-15 10:02:45 ERROR Connection refused: upstream service timeout..."
    - path: "logs/auth_service.log"
      content: "2024-01-15 10:02:30 ERROR Invalid token: signature verification failed..."
goal:
  type: "file_content_matches"
  checks:
    - path: "report.txt"
      contains: "ERROR"
llm_evaluation:
  rubric:
    completion: "Does report.txt exist with error/warning counts?"
    accuracy: "Are the counts correct? (10 ERRORs, 5 WARNINGs across 3 files)"
    efficiency: "Were grep/sort/wc used efficiently?"
```

**Browser automation** (WebApp — 9 steps, Playwright + screenshots):
```yaml
name: "add_three_todos"
env: "webapp"
app_url: "http://localhost:3000"
observation_mode: "hybrid"    # DOM + screenshots
goal:
  type: "element_count"
  selector: "[data-testid^='todo-item-']"
  count: 3
```

---

## Engineering

```
100 unit tests passing          mypy --strict across 31 files
0 Any escape hatches            Pydantic v2 runtime validation
structlog with session IDs      Custom exception hierarchy
Async context managers          GitHub Actions CI
Husky pre-commit hooks          ruff lint + format
```

```bash
make check   # runs: ruff check + mypy --strict + pytest (100 tests)
```

---

## Design Decisions

**Why async context managers?** At scale, Playwright browsers leak memory if not closed. `async with WebAppEnvironment(config) as env:` guarantees cleanup even when exceptions occur. At thousands of episodes per experiment, this matters.

**Why reset both client AND server state?** Refreshing the browser doesn't clear the todo app's in-memory store. `reset()` calls `POST /api/reset` on the target app. True reproducibility requires resetting everything.

**Why separate API schemas from domain models?** HTTP concerns shouldn't leak into business logic. `schemas.py` handles serialization; `models.py` handles validation. Different layers, different responsibilities.

**Why full-command inspection for CLI sandboxing?** Checking only the first word (`cat` is allowed) misses `cat ../../etc/passwd`. The full command string is inspected for path traversal.

**Why LLM-as-judge?** Rule-based evaluation breaks on subjective tasks. "Write a professional README" can't be checked with `querySelector`. The LLM judge scores against rubrics, combined with rule-based scores via `CompositeEvaluator`.

---

## Project Structure

```
axiom/                       # Python package (the framework)
  models.py                  # ALL Pydantic models (Action, Observation, StepResult, etc.)
  config.py                  # pydantic-settings: env vars with AXIOM_ prefix
  exceptions.py              # AxiomError → EnvironmentError, SessionError, etc.
  core/
    base_env.py              # BaseEnvironment ABC (Gymnasium interface)
    registry.py              # @register_env decorator, pluggable discovery
    session.py               # Session lifecycle with asyncio.Lock
    trajectory.py            # Screenshots as separate PNGs, not base64
    evaluator.py             # DefaultEvaluator + CompositeEvaluator
    llm_judge.py             # LLM-as-judge via Claude API
    parallel_runner.py       # Bounded-concurrency episode execution
    task_loader.py           # YAML → validated TaskConfig
  envs/
    webapp_env.py            # Playwright + Chromium (maps to OSWorld)
    axiomchat_env.py         # AxiomChat (mini-Slack) — WebAppEnvironment subclass
    cli_env.py               # Sandboxed subprocess (maps to Terminal-Bench)
    json_env.py              # Pure Python state machine (baseline)
  api/
    app.py                   # Application factory + lifespan
    routes/                  # sessions, environments, tasks, trajectories, health
agents/
  claude_agent.py            # Claude with vision (screenshots + DOM)
  random_agent.py            # Random baseline for calibration
apps/todo-app/               # TypeScript Express target application
apps/axiomchat/              # AxiomChat: React+Vite SPA + Express, seeded + oracle
tasks/                       # YAML task definitions (12 tasks across 4 envs)
static/
  demo.html                  # Mission control dashboard
  replay.html                # Trajectory step-through viewer
tests/                       # 100 unit tests + 8 integration tests
```

---

## Tech Stack

Python 3.11+ · FastAPI · Playwright · Pydantic v2 · structlog · pydantic-settings · Anthropic Claude API · Docker · pytest · ruff · mypy strict

---

## License

MIT
