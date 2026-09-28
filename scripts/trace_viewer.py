"""Readable viewer for structured JSONL agent traces.

Usage:
    python -m scripts.trace_viewer traces/agent_runs.jsonl
    python -m scripts.trace_viewer traces/agent_runs.jsonl --run-id RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.core.pricing import estimate_cost
from src.core.trace_format import (
    display as _display,
    format_duration as _format_duration,
    format_tokens as _format_tokens,
    summarize_result as _summarize_result,
)


def _load_trace_records(path: Path) -> list[dict[str, Any]]:
    """Load one JSON object per line from a JSONL trace file."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"Could not read trace file: {exc}") from exc

    if not text.strip():
        raise ValueError("Trace file is empty.")

    records: list[dict[str, Any]] = []

    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue

        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON on line {line_number}: {exc.msg}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid trace on line {line_number}: expected a JSON object."
            )

        records.append(record)

    if not records:
        raise ValueError("Trace file contains no JSON records.")

    return records


def _select_record(records: list[dict[str, Any]], run_id: str | None) -> dict[str, Any]:
    """Select one trace record, optionally by run ID."""
    if run_id is None:
        if len(records) > 1:
            print(f"Trace file contains {len(records)} runs; displaying the first run.")
        return records[0]

    for record in records:
        if record.get("run_id") == run_id:
            return record

    raise ValueError(f"Run ID not found: {run_id}")


def _print_model_iteration(
    iteration: dict[str, Any], agent_loop_number: int | None = None
) -> None:
    """Print a model-call iteration."""
    number = _display(
        agent_loop_number if agent_loop_number is not None else iteration.get("number")
    )
    duration = _format_duration(iteration.get("model_duration_ms"))
    usage = _format_tokens(iteration.get("usage"))

    print(f"├── Agent loop {number}: model call")
    print(f"│   ├── Duration: {duration}")
    print(f"│   ├── Tokens: {usage}")

    requested_tools = iteration.get("requested_tools") or []

    if not requested_tools:
        print("│   └── Requested tools: none")
        return

    print("│   └── Requested tools:")

    for index, tool in enumerate(requested_tools):
        is_last = index == len(requested_tools) - 1
        branch = "    └──" if is_last else "    ├──"

        name = _display(tool.get("name"))
        arguments = _display(
            tool.get("validated_arguments", tool.get("arguments")),
            fallback="unknown",
        )
        validation_error = tool.get("validation_error")

        print(f"│       {branch} {name}")
        print(f"│           ├── Arguments: {arguments}")

        if validation_error:
            print(f"│           └── Validation: ERROR - {validation_error}")
        else:
            print("│           └── Validation: OK")


def _print_tool_iteration(
    iteration: dict[str, Any],
    is_last_iteration: bool,
    agent_loop_number: int | None = None,
    max_length: int | None = 160,
) -> None:
    """Print a tool-execution iteration."""
    number = _display(
        agent_loop_number if agent_loop_number is not None else iteration.get("number")
    )
    tools = iteration.get("executed_tools") or []

    prefix = "└──" if is_last_iteration else "├──"

    print(f"{prefix} Agent loop {number}: tool execution")

    if not tools:
        print("    └── Tools: none recorded")
        return

    for tool_index, tool in enumerate(tools):
        is_last_tool = tool_index == len(tools) - 1
        branch = "└──" if is_last_tool else "├──"

        name = _display(tool.get("name"))
        duration = _format_duration(tool.get("duration_ms"))
        success = _display(tool.get("success"))
        arguments = _display(
            tool.get("validated_arguments", tool.get("arguments")),
            fallback="unknown",
        )

        print(f"    {branch} Tool: {name}")
        print(f"        ├── Arguments: {arguments}")
        print(f"        ├── Duration: {duration}")
        print(f"        ├── Success: {success}")
        print(
            "        └── Result: "
            f"{_summarize_result(tool.get('result'), max_length=max_length)}"
        )


def _print_iterations(iterations: Any, max_length: int | None = 160) -> None:
    """Print all recorded iterations."""
    if not isinstance(iterations, list) or not iterations:
        print("├── Iterations: none recorded")
        return

    print("├── Agent loop iterations:")

    agent_loop_number = 0
    for index, iteration in enumerate(iterations):
        if not isinstance(iteration, dict):
            print(f"│   └── Iteration {index + 1}: invalid entry")
            continue

        iteration_type = iteration.get("iteration_type")

        if iteration_type == "model_call":
            agent_loop_number += 1
            _print_model_iteration(iteration, agent_loop_number)
        elif iteration_type == "tool_execution":
            _print_tool_iteration(
                iteration,
                is_last_iteration=index == len(iterations) - 1,
                agent_loop_number=agent_loop_number or None,
                max_length=max_length,
            )
        else:
            print(
                f"│   └── Iteration {_display(iteration.get('number'))}: unknown type"
            )


def _print_approvals(events: Any) -> None:
    """Print approval events, or a graceful placeholder."""
    approvals = []

    if isinstance(events, list):
        approvals = [
            event
            for event in events
            if isinstance(event, dict) and event.get("type") == "human_approval"
        ]

    if not approvals:
        print("├── Approvals: none recorded")
        return

    print("├── Approvals:")

    for index, approval in enumerate(approvals):
        outcome = _display(approval.get("outcome"))
        details = approval.get("reason") or approval.get("message")

        suffix = f" ({details})" if details else ""
        branch = "└──" if index == len(approvals) - 1 else "├──"

        print(f"│   {branch} {outcome}{suffix}")


def _print_costs(record: dict[str, Any]) -> None:
    """Print estimated and actual billed costs separately."""
    configuration = record.get("configuration")

    if not isinstance(configuration, dict):
        configuration = {}

    provider = configuration.get("provider")
    model = configuration.get("model")
    usage = record.get("usage")

    if not isinstance(usage, dict):
        usage = {}

    input_tokens = usage.get("input_tokens")
    output_tokens = usage.get("output_tokens")

    if isinstance(input_tokens, int) and isinstance(output_tokens, int):
        estimated = estimate_cost(
            provider,
            model,
            input_tokens,
            output_tokens,
        )
    else:
        estimated = None

    if estimated is None:
        print(
            "├── Estimated cost: unknown (pricing not configured or usage incomplete)"
        )
    else:
        print(f"├── Estimated cost: ${estimated:.8f} USD")

    actual_billed = record.get("actual_billed_cost")

    if actual_billed is None:
        print("└── Actual billed cost: unknown")
    else:
        print(f"└── Actual billed cost: ${actual_billed:.8f} USD")


def render_record(record: dict[str, Any], max_length: int | None = 160) -> None:
    """Print one trace record as a readable execution tree."""
    configuration = record.get("configuration")
    if not isinstance(configuration, dict):
        configuration = {}

    print(f"Run: {_display(record.get('run_id'))}")
    print(f"Provider: {_display(configuration.get('provider'))}")
    print(f"Model: {_display(configuration.get('model'))}")
    print(
        "Iteration limit: "
        f"{_display(configuration.get('max_iterations'))} agent-loop model calls"
    )
    print(f"Started: {_display(record.get('started_at'))}")
    print(f"Ended: {_display(record.get('ended_at'))}")
    print(f"Duration: {_format_duration(record.get('duration_ms'))}")
    print()

    _print_iterations(record.get("iterations"), max_length=max_length)
    print()

    _print_approvals(record.get("events"))
    print(f"├── Terminal reason: {_display(record.get('terminal_reason'))}")

    usage = record.get("usage")
    print(f"├── Tokens: {_format_tokens(usage)}")

    _print_costs(record)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Render a structured agent trace as a readable execution tree."
    )
    parser.add_argument(
        "trace_file",
        type=Path,
        help="Path to the JSONL trace file.",
    )
    parser.add_argument(
        "--run-id",
        help="Display only the trace record with this run ID.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Display complete tool execution results without truncation.",
    )

    args = parser.parse_args(argv)

    try:
        records = _load_trace_records(args.trace_file)
        record = _select_record(records, args.run_id)
        render_record(record, max_length=None if args.full else 160)
        return 0

    except ValueError as exc:
        print(f"Trace viewer error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
