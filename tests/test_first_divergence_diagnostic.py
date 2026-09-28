"""Regression checks for the documented first-divergence analysis."""

from __future__ import annotations

import json
import re
import tomllib
from pathlib import Path

from app.tools import build_tool_registry


ROOT = Path(__file__).resolve().parents[1]
NOTE_PATH = ROOT / "docs" / "first-divergence-diagnostic.md"
TERMINAL_STATUSES = {"escalated", "resolved", "closed"}
STATE_CHANGING_TOOLS = {"send_reply", "escalate"}


def _load_note() -> tuple[str, dict]:
    text = NOTE_PATH.read_text(encoding="utf-8")
    metadata_block = re.search(r"```toml\n(.*?)\n```", text, re.DOTALL)
    assert metadata_block is not None, "diagnostic note must contain TOML metadata"
    return text, tomllib.loads(metadata_block.group(1))


def _load_run(metadata: dict) -> dict:
    trace_path = ROOT / metadata["trace_file"]
    records = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    matching = [record for record in records if record["run_id"] == metadata["run_id"]]
    assert len(matching) == 1
    return matching[0]


def _iteration(run: dict, number: int, iteration_type: str) -> dict:
    matching = [
        iteration
        for iteration in run["iterations"]
        if iteration.get("number") == number
        and iteration.get("iteration_type") == iteration_type
    ]
    assert len(matching) == 1
    return matching[0]


def _first_state_change_after_terminal_status(run: dict) -> tuple[int, str] | None:
    """Return the first state-changing request made after terminal status evidence."""
    terminal_status_seen = False

    for iteration in run["iterations"]:
        if iteration.get("iteration_type") == "tool_execution":
            for tool in iteration.get("executed_tools", []):
                result = tool.get("result", {}).get("result", {})
                ticket = result.get("ticket") if isinstance(result, dict) else None
                if (
                    tool.get("name") == "get_ticket"
                    and isinstance(ticket, dict)
                    and ticket.get("status") in TERMINAL_STATUSES
                ):
                    terminal_status_seen = True
            continue

        if iteration.get("iteration_type") != "model_call" or not terminal_status_seen:
            continue

        for tool in iteration.get("requested_tools", []):
            if tool.get("name") in STATE_CHANGING_TOOLS:
                return iteration["number"], tool["name"]

    return None


def test_note_names_specific_real_trace_run_and_iteration():
    text, metadata = _load_note()
    trace_path = ROOT / metadata["trace_file"]

    assert metadata["source_trace_file"] == "traces/agent_runs.jsonl"
    assert trace_path.is_file()
    assert metadata["run_id"] == "f7a0c7af-1e26-4420-b917-436d4efd5cf4"
    assert metadata["first_off_course_iteration"] == 2

    run = _load_run(metadata)
    assert run["configuration"]["provider"] == "groq"
    _iteration(run, metadata["first_off_course_iteration"], "model_call")
    assert metadata["run_id"] in text


def test_note_contrasts_expected_and_observed_decisions():
    text, metadata = _load_note()
    run = _load_run(metadata)
    model_iteration = _iteration(
        run, metadata["first_off_course_iteration"], "model_call"
    )

    assert metadata["expected_decision"] == "stop_without_tool"
    assert [tool["name"] for tool in model_iteration["requested_tools"]] == [
        metadata["observed_decision"]
    ]
    # This note is a frozen record of analysis done against the prompt as it
    # existed when this trace was captured, which explicitly told the model to
    # stop on a terminal-status ticket. That prompt was later deliberately
    # simplified to leave tool sequencing to the model's judgment instead (see
    # src/agent/system_prompt.py), so this no longer asserts against the live
    # SYSTEM_PROMPT -- only against the note's own documented analysis.
    assert "terminal-status tickets must stop" in text.lower()
    assert "expected decision" in text.lower()
    assert "instead" in text.lower()


def test_note_identifies_origin_instead_of_later_consequences():
    text, metadata = _load_note()
    run = _load_run(metadata)

    evidence_model = _iteration(run, metadata["evidence_iteration"], "model_call")
    evidence_tool = _iteration(run, metadata["evidence_iteration"], "tool_execution")
    divergence_tool = _iteration(
        run, metadata["first_off_course_iteration"], "tool_execution"
    )
    recovery_model = _iteration(
        run, metadata["first_off_course_iteration"] + 1, "model_call"
    )

    assert [tool["name"] for tool in evidence_model["requested_tools"]] == [
        "get_ticket"
    ]
    ticket = evidence_tool["executed_tools"][0]["result"]["result"]["ticket"]
    assert ticket["status"] == "escalated"

    duplicate_result = divergence_tool["executed_tools"][0]["result"]["result"]
    assert duplicate_result["escalated"] is False
    assert "already has an open escalation" in duplicate_result["message"]
    assert recovery_model["requested_tools"] == []
    assert "consequences" in text.lower()
    assert "recovery" in text.lower()


def test_note_qualifies_prompt_tool_and_context_factors():
    text, metadata = _load_note()
    run = _load_run(metadata)
    evidence_tool = _iteration(run, metadata["evidence_iteration"], "tool_execution")
    nested_result = evidence_tool["executed_tools"][0]["result"]["result"]
    divergence_model = _iteration(
        run, metadata["first_off_course_iteration"], "model_call"
    )
    reason = divergence_model["requested_tools"][0]["arguments"]["reason"].lower()
    get_ticket_description = build_tool_registry().get("get_ticket").description.lower()

    assert nested_result["ticket"]["status"] == "escalated"
    assert "invoice" in reason
    assert "already escalated" not in reason
    assert "escalated" not in get_ticket_description
    assert "system prompt already says" in text.lower()
    assert "may indicate" in text.lower()
    assert "does not prove" in text.lower()
    assert "tool description" in text.lower()


def test_targeted_improvement_evaluation_check():
    text, metadata = _load_note()
    run = _load_run(metadata)

    assert _first_state_change_after_terminal_status(run) == (2, "escalate")

    # A compliant rerun stops on the next model decision after the same evidence.
    compliant_run = {
        "iterations": [
            _iteration(run, metadata["evidence_iteration"], "model_call"),
            _iteration(run, metadata["evidence_iteration"], "tool_execution"),
            {
                "iteration_type": "model_call",
                "number": metadata["first_off_course_iteration"],
                "requested_tools": [],
            },
        ]
    }
    assert _first_state_change_after_terminal_status(compliant_run) is None
    assert metadata["evaluation_check"] == (
        "tests/test_first_divergence_diagnostic.py::"
        "test_targeted_improvement_evaluation_check"
    )
    assert "plausible targeted improvement" in text.lower()
    assert "no such violation" in text.lower()
