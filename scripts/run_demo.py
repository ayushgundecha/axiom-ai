#!/usr/bin/env python3
"""Demo runner for axiom-ai.

Runs the Claude agent (or random agent) against axiom environments
via the HTTP API and prints pretty evaluation results.

Usage:
    # Start the server first:
    #   1. Start todo app: cd apps/todo-app && node dist/server.js
    #   2. Start axiom:    uvicorn axiom.api.app:create_app --factory --port 8000

    # Run demos:
    python scripts/run_demo.py --env json --task create_and_complete
    python scripts/run_demo.py --env webapp --task add_three_todos
    python scripts/run_demo.py --env cli --task organize_files_by_extension
    python scripts/run_demo.py --env json --task create_and_complete --agent random
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def _load_dotenv(dotenv_path: Path) -> None:
    """Load simple KEY=VALUE pairs from a .env file into os.environ.

    We intentionally avoid extra dependencies (like python-dotenv) for the
    demo runner.
    """
    if not dotenv_path.exists():
        return

    for raw_line in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        # Split on the first '=' only (values may contain '=')
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            continue

        # Strip common quoting styles.
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


async def main() -> None:
    parser = argparse.ArgumentParser(description="axiom-ai demo runner")
    parser.add_argument("--env", required=True, help="Environment (json, webapp, cli)")
    parser.add_argument("--task", required=True, help="Task ID")
    parser.add_argument(
        "--agent",
        default="claude",
        choices=["claude", "random"],
        help="Agent type (default: claude)",
    )
    parser.add_argument("--base-url", default="http://localhost:8000", help="Axiom server URL")
    parser.add_argument("--model", default="claude-haiku-4-5-20251001", help="Claude model to use")
    args = parser.parse_args()

    # Ensure the Anthropic client can authenticate when using this demo runner.
    _load_dotenv(Path(__file__).parent.parent / ".env")

    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║                    axiom-ai Demo                        ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if args.agent == "claude":
        from agents.claude_agent import ClaudeAgent

        agent = ClaudeAgent(model=args.model)
    else:
        from agents.random_agent import RandomAgent

        agent = RandomAgent()

    try:
        await agent.run_episode(
            base_url=args.base_url,
            env_name=args.env,
            task_id=args.task,
            verbose=True,
        )
        print("Demo completed successfully.")
    except Exception as e:
        msg = str(e)
        print(f"\nError: {msg}")

        # Provide more targeted guidance for common failure modes.
        if "Could not resolve authentication method" in msg:
            print(
                "\nAnthropic authentication failed. Ensure `ANTHROPIC_API_KEY` is set "
                "in `.env` (or exported in your shell)."
            )
        elif "Connection error" in msg:
            print(
                "\nAnthropic connection failed. Check your network access to Anthropic "
                "(e.g., proxy/VPN/firewall), not the Axiom server."
            )
        else:
            print("\nMake sure the axiom server is running:")
            print("  uvicorn axiom.api.app:create_app --factory --port 8000")
        if args.env == "webapp":
            print("\nAnd the todo app:")
            print("  cd apps/todo-app && node dist/server.js")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
