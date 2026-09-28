"""
Interactive multi-turn chatbot REPL for the Support Desk Triage Agent.
Backed by the configured model provider (Groq/Gemini/fake) and the full
tool registry (app.tools.build_tool_registry).
"""

import argparse
import copy

from app.agent import run_agent_turn
from app.approval import TerminalApprovalInterface
from app.tools import build_tool_registry
from src.agent.system_prompt import SYSTEM_PROMPT
from src.core.compact_trace import CompactTraceView
from src.core.config import Config
from src.core.exceptions import (
    IterationLimitExceededError,
    TokenAccountingError,
    TokenLimitExceededError,
    ToolCallLimitExceededError,
)
from src.core.live_trace import LiveTraceView
from src.llm.factory import create_model_client

EXIT_COMMANDS = {"exit", "quit"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ABC Bookstore Support Desk Agent -- interactive chat"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Show the full step-by-step agent trace (every model call, full "
            "tool arguments, full tool results, token usage per step) "
            "instead of the default compact one-line-per-tool-call summary."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = create_model_client()
    registry = build_tool_registry(TerminalApprovalInterface())
    live_view = None
    if Config.LIVE_TRACE_VIEW:
        live_view = LiveTraceView() if args.verbose else CompactTraceView()

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    running_total_tokens = 0

    print(
        "ABC Bookstore Support Desk Agent -- interactive chat "
        "(blank line, 'exit'/'quit', or Ctrl-D/Ctrl-C to end)"
    )

    while True:
        try:
            user_input = input("\nYou: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        stripped = user_input.strip()
        if not stripped or stripped.lower() in EXIT_COMMANDS:
            print("Exiting.")
            break

        # Work on a deep copy so a mid-loop exception (execution-limit
        # errors, tool/model failures) never leaves partial model/tool
        # turns committed to the running conversation.
        candidate_messages = copy.deepcopy(messages)
        candidate_messages.append({"role": "user", "content": stripped})

        try:
            answer, usage, updated_messages = run_agent_turn(
                candidate_messages,
                client=client,
                registry=registry,
                live_view=live_view,
            )
        except (
            IterationLimitExceededError,
            TokenLimitExceededError,
            ToolCallLimitExceededError,
            TokenAccountingError,
        ) as exc:
            print(f"\n[Agent stopped: {exc}]")
            continue

        messages = updated_messages
        running_total_tokens += usage.total_tokens

        print(f"\nAgent: {answer}")
        print(
            f"[tokens this turn: {usage.total_tokens} "
            f"(in={usage.input_tokens}, out={usage.output_tokens}) | "
            f"running total: {running_total_tokens}]"
        )


if __name__ == "__main__":
    main()
