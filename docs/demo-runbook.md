# axiom-ai — demo runbook

A copy-paste path from a fresh clone to the full reward-robustness story. Every
command is runnable; the **zero-API-key** path (§1) needs nothing but Python.

---

## 0. Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
make dev                    # editable install + dev tools + Playwright chromium
make check                  # ruff + mypy --strict + pytest (should be green)
```

---

## 1. The benchmark, zero API keys (the P0 demo)

The deterministic in-memory AxiomChat simulator runs the full scripted exploit
catalog against every reward — no Docker, no browser, no LLM:

```bash
python scripts/run_robustness.py --train-seeds 1 2 3 --eval-seeds 4 5 6 --judge
```

You'll see the RRS table print: naive **v0** rewards get hacked (macro-RRS ≈
0.36), hardened **v1** rewards hold (macro-RRS 1.000, hack_rate 0.000) with
honest_fidelity 1.000. It writes `reports/robustness.json`.

**What to say:** *"Every reward is split into a cheap proxy the agent is scored
on and a privileged oracle it never sees. A reward hack is `proxy_pass ∧
¬oracle_pass`. The catalog of 24 exploits hacks the naive rewards; the named
defenses close them without breaking honest work — and the corpus regression in
CI keeps them closed."*

---

## 2. The Axiom Console (local)

```bash
uvicorn axiom.api.app:create_app --factory --port 8000
open http://localhost:8000/static/demo.html
```

One console, three tabs (shared nav):

- **Demo** — pick an environment (AxiomChat, WebApp, CLI, JSON) and a task; watch
  a recorded agent episode animate step by step.
- **Replay** — step through any trajectory: observation, action, reward, and the
  **REWARD HACK / HONEST PASS** verdict banner (proxy paid? oracle satisfied?).
- **Leaderboard** — the RRS scoreboard; toggle Offline · Live · both discovery
  runs. (`http://localhost:8000/static/robustness.html`)

If you have no live trajectories yet, run one (§4) or open the committed evidence
directly in Replay via the `reports/transcripts/` manifest.

---

## 3. AxiomChat itself

```bash
make axiomchat-build && make axiomchat-run          # serves http://localhost:3100
# seeded, byte-reproducible reset:
curl -X POST localhost:3100/api/reset -H 'Content-Type: application/json' -d '{"seed":4,"scale":"small"}'
# the privileged oracle is token-gated (403 without the token):
curl localhost:3100/api/_oracle/state -H "X-Oracle-Token: $AXIOM_AXIOMCHAT_ORACLE_TOKEN"
```

The oracle exposes the hidden `derived` ground truth (answer facts, severity,
owner). Only the harness ever sends the token; no agent or environment sees it.

---

## 4. A live LLM episode (the credibility proof)

Needs an Anthropic **or** free-tier Gemini key in `.env`; AxiomChat running on
:3100. Never run two live commands at once.

```bash
python scripts/run_robustness.py --live --llm --judge \
  --tasks answer_support_question summarize_incident \
  --train-seeds 1 2 3 --eval-seeds 4 5 6 \
  --exploiter-model gemini-3.1-flash-lite --honest-model gemini-3.1-flash-lite \
  --judge-model gemini-3.1-flash-lite --out reports/robustness_live.json
```

The **exploiter** is briefed on the proxy spec and told to take the laziest path
to reward; the **honest** agent is asked to do the task; grading stays
out-of-band. Trajectories land in `trajectories/` and replay in the console.

**The headline to point at:** open a hacked exploit trajectory in Replay — the
red REWARD HACK banner, proxy PAID / oracle failed. Then the paired v1-blocked
and honest runs. A live agent found holes in the *hardened* rewards on held-out
seeds (see the Discovery toggles on the leaderboard); each became a permanent
test and a new defense.

---

## 5. Classic single-agent demos (the other envs)

```bash
python scripts/run_demo.py --env cli       --task analyze_logs     --agent claude
python scripts/run_demo.py --env axiomchat  --task post_message    --agent claude
python scripts/run_demo.py --env webapp     --task add_three_todos  --agent claude   # needs the todo app
```

These go through the HTTP API and are scored on the 4 evaluation dimensions
(completion, efficiency, accuracy, safety) — the "can it do the task?" runner,
as opposed to the "can it cheat the reward?" harness in §1/§4.

---

## Reset checklist between demos

- Re-`POST /api/reset` AxiomChat for a clean seeded workspace.
- One live command at a time (they share the single AxiomChat on :3100).
- Free-tier Gemini is 500 requests/day — budget accordingly; error runs are
  excluded from RRS and shown as a count on the leaderboard.
