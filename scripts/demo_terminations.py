"""Demonstrate successful completion and iteration-cap termination.

Run with:
    python -m scripts.demo_terminations

No live API key or database connection is required.  Both scenarios use the
``fake`` provider so the output is fully deterministic and repeatable.

Scenario A — Happy-path completion:
    A representative billing ticket is processed by a scripted two-turn model
    client.  Turn 1 calls ``get_ticket``; Turn 2 returns a final triage answer.
    The run terminates with ``terminal_reason = "completed"``.

Scenario B — Impossible-goal / iteration cap:
    The same ticket is processed by a fake model that always requests a tool
    and therefore never produces a terminal text response.  With
    ``max_iterations=3`` the loop hits the cap and raises
    ``IterationLimitExceededError``.  The run terminates cleanly with
    ``terminal_reason = "execution_limit"``.

Both scenarios emit verbose diagnostics (iteration, message-count, call_tokens,
cumulative_tokens, requested_tool_name, terminal) to stdout.  Configuration
identifiers are printed; no secrets are included.
"""

from __future__ import annotations

import os

# Force the fake provider before project imports so this demo stays offline.
os.environ["LLM_PROVIDER"] = "fake"

import textwrap
from pathlib import Path

from app.agent import run_agent_turn
from app.tool_registry import ToolDefinition, ToolRegistry
from src.core.config import Config
from src.core.exceptions import IterationLimitExceededError
from src.llm.base import NormalizedResponse, TokenUsage, ToolRequest
from src.llm.fake import FakeModelClient, ScriptedModelClient

# ---------------------------------------------------------------------------
# Shared configuration
# ---------------------------------------------------------------------------
TICKET_ID = "T-SEED-001"
TICKET_CONTEXT = (
    "Customer: Ananya Rao <ananya.rao@example.com> | Plan: pro\n"
    "Subject: Incorrect charge on my last invoice\n"
    "Body: Hi, I was billed $99 this month but my pro plan should be $49. "
    "Please look into this and issue a correction."
)
SYSTEM_PROMPT = (
    "You are a support ticket triage agent. "
    "Gather evidence using tools, then provide a concise triage decision."
)

# Iteration cap used for Scenario B.
IMPOSSIBLE_CAP = 3

# Tokens reported by the fake model per call (deterministic).
_CALL_TOKENS = TokenUsage(
    input_tokens=15, output_tokens=10, total_tokens=25, reported=True
)

TRACE_PATH = Path("traces/demo_terminations.jsonl")


# ---------------------------------------------------------------------------
# Shared tool registry
# ---------------------------------------------------------------------------
def _build_registry() -> ToolRegistry:
    """Minimal registry with a read-only get_ticket stub."""

    def get_ticket(ticket_id: str) -> dict:
        return {
            "ticket_id": ticket_id,
            "status": "open",
            "category": "billing",
            "customer": "Ananya Rao",
            "plan": "pro",
            "subject": "Incorrect charge on my last invoice",
        }

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="get_ticket",
            description="Fetches full ticket details and customer context.",
            parameters={
                "type": "object",
                "properties": {"ticket_id": {"type": "string"}},
                "required": ["ticket_id"],
            },
            function=get_ticket,
        )
    )
    return registry


# ---------------------------------------------------------------------------
# Diagnostic helpers
# ---------------------------------------------------------------------------
def _print_header(title: str) -> None:
    bar = "=" * 60
    print(f"\n{bar}")
    print(f"  {title}")
    print(bar)


def _print_diagnostic(label: str, event: dict) -> None:
    """Print one structured diagnostic event in a human-readable line."""
    if event["event"] == "model_call":
        tool_info = (
            f"tool={event['requested_tool_name']!r}"
            if event["requested_tool_name"]
            else "no-tool"
        )
        print(
            f"  [{label}] iter={event['iteration']} "
            f"msgs={event['message_count']} "
            f"call_tokens={event['call_tokens']} "
            f"cumulative={event['cumulative_tokens']} "
            f"{tool_info} "
            f"terminal={event['terminal']}"
        )
    elif event["event"] == "termination":
        print(
            f"  [{label}] TERMINATION reason={event['terminal_reason']!r} "
            f"iter={event['iteration']} "
            f"cumulative_tokens={event['cumulative_tokens']}"
        )


def _print_config() -> None:
    """Print provider/model/trace identifiers shared by both scenarios.

    Per-scenario execution caps (max_iterations, max_total_tokens,
    max_calls_per_tool) differ between A and B and are printed inline
    with each scenario instead.
    """
    print(
        textwrap.dedent(f"""\
        Configuration (no secrets printed):
          provider   : {Config.PROVIDER}
          model      : {Config.MODEL_NAME}
          temperature: {Config.TEMPERATURE}
          trace_path : {TRACE_PATH}
        """)
    )


# ---------------------------------------------------------------------------
# Scenario A — Successful happy-path completion
# ---------------------------------------------------------------------------
def run_scenario_a() -> bool:
    """Two-turn scripted run: tool call then final answer.

    Returns True on expected outcome, False on failure.
    """
    _print_header("Scenario A — Successful completion")

    tool_request = ToolRequest(
        id="call-a1",
        name="get_ticket",
        arguments={"ticket_id": TICKET_ID},
    )

    # Turn 1: model requests get_ticket.
    turn1 = NormalizedResponse(
        tool_requests=[tool_request],
        usage=_CALL_TOKENS,
    )
    # Turn 2: model returns a terminal triage answer (no tool call).
    turn2 = NormalizedResponse(
        text=(
            "Triage decision: Billing — overcharge.\n"
            "The customer (pro plan, $49/mo) was billed $99. "
            "Recommend issuing a $50 credit and confirming the corrected amount. "
            "Escalate to billing team for invoice adjustment."
        ),
        usage=_CALL_TOKENS,
    )

    client = ScriptedModelClient([turn1, turn2])
    registry = _build_registry()
    diagnostics: list[dict] = []

    print(f"\nTicket : {TICKET_ID}")
    print(f"Cap    : max_iterations={Config.MAX_ITERATIONS} (default)")
    print()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"Please triage ticket {TICKET_ID}:\n\n{TICKET_CONTEXT}",
        },
    ]
    answer, usage, _messages = run_agent_turn(
        messages,
        client=client,
        registry=registry,
        trace_path=TRACE_PATH,
        run_id="demo-scenario-a",
        verbose_diagnostics=True,
        diagnostic_sink=diagnostics.append,
    )

    print("Diagnostics:")
    for event in diagnostics:
        _print_diagnostic("A", event)

    print(f"\nFinal answer ({len(answer)} chars):")
    for line in answer.splitlines():
        print(f"  {line}")

    termination_event = next(
        (e for e in diagnostics if e["event"] == "termination"), None
    )
    if termination_event and termination_event["terminal_reason"] == "completed":
        print("\n✓ Scenario A PASSED — terminal_reason='completed'")
        return True

    print("\n✗ Scenario A FAILED — unexpected termination")
    return False


# ---------------------------------------------------------------------------
# Scenario B — Impossible goal / iteration cap
# ---------------------------------------------------------------------------
def run_scenario_b() -> bool:
    """Fake model always requests a tool; loop must hit the iteration cap.

    Returns True on expected outcome (IterationLimitExceededError), False on failure.
    """
    _print_header("Scenario B — Impossible goal / iteration-cap termination")

    # Fake model always requests get_ticket — it can never return a terminal answer.
    canned = NormalizedResponse(
        text="Still checking…",
        tool_requests=[
            ToolRequest(
                id="call-b1",
                name="get_ticket",
                arguments={"ticket_id": TICKET_ID},
            )
        ],
        usage=_CALL_TOKENS,
    )

    client = FakeModelClient(canned_response=canned)
    registry = _build_registry()
    diagnostics: list[dict] = []

    # Print the exact caps passed to run_agent_turn below so the
    # reader can verify what is actually enforced, not the global defaults.
    print(f"\nTicket             : {TICKET_ID}")
    print(f"max_iterations     : {IMPOSSIBLE_CAP}  (deliberately low)")
    print(f"max_total_tokens   : {100_000}")
    print(f"max_calls_per_tool : {100}")
    print("Fake model         : always requests get_ticket (never terminates)")
    print()

    terminated_cleanly = False
    caught_exception: IterationLimitExceededError | None = None

    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Please triage ticket {TICKET_ID}:\n\n{TICKET_CONTEXT}",
            },
        ]
        run_agent_turn(
            messages,
            client=client,
            registry=registry,
            max_iterations=IMPOSSIBLE_CAP,
            max_total_tokens=100_000,
            max_calls_per_tool=100,
            trace_path=TRACE_PATH,
            run_id="demo-scenario-b",
            verbose_diagnostics=True,
            diagnostic_sink=diagnostics.append,
        )
    except IterationLimitExceededError as exc:
        caught_exception = exc
        terminated_cleanly = True

    print("Diagnostics:")
    for event in diagnostics:
        _print_diagnostic("B", event)

    if not terminated_cleanly:
        print("\n✗ Scenario B FAILED — no IterationLimitExceededError raised")
        return False

    termination_event = next(
        (e for e in diagnostics if e["event"] == "termination"), None
    )

    print(f"\nCaught exception : {type(caught_exception).__name__}")
    print(f"Exception message: {caught_exception}")

    model_call_events = [e for e in diagnostics if e["event"] == "model_call"]
    print(f"\nModel-call diagnostic events : {len(model_call_events)}")
    print(f"Configured cap               : {IMPOSSIBLE_CAP}")

    reason = termination_event["terminal_reason"] if termination_event else None
    if reason == "execution_limit" and len(model_call_events) == IMPOSSIBLE_CAP:
        print(
            "\n✓ Scenario B PASSED — terminal_reason='execution_limit', "
            f"stopped after {IMPOSSIBLE_CAP} iterations, no hang or overflow"
        )
        return True

    print("\n✗ Scenario B FAILED — unexpected state")
    return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> int:
    print("\nTermination demonstration")
    print("Two scenarios using the fake provider — no API key required.\n")
    _print_config()

    a_ok = run_scenario_a()
    b_ok = run_scenario_b()

    _print_header("Summary")
    print(f"  Scenario A (success)           : {'PASSED' if a_ok else 'FAILED'}")
    print(f"  Scenario B (iteration-cap stop): {'PASSED' if b_ok else 'FAILED'}")
    print(f"\nTrace appended to: {TRACE_PATH}")

    if a_ok and b_ok:
        print("\nAll scenarios passed. Exit 0.")
        return 0

    print("\nOne or more scenarios failed. Exit 1.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
