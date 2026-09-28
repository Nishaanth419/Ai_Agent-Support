"""Tests for the structured trace viewer."""

import json

from scripts.trace_viewer import main


def _write_jsonl(tmp_path, record):
    path = tmp_path / "trace.jsonl"
    path.write_text(
        json.dumps(record, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return path


def _base_record():
    return {
        "run_id": "run-normal",
        "configuration": {
            "provider": "gemini",
            "model": "gemini-3.6-flash",
            "temperature": 0.0,
            "max_iterations": 10,
        },
        "started_at": "2026-09-09T10:00:00+00:00",
        "ended_at": "2026-09-09T10:00:03+00:00",
        "terminal_reason": "completed",
        "duration_ms": 2503,
        "usage": {
            "input_tokens": 20,
            "output_tokens": 40,
            "total_tokens": 60,
        },
        "iterations": [
            {
                "iteration_type": "model_call",
                "number": 1,
                "model_duration_ms": 1250,
                "usage": {
                    "input_tokens": 20,
                    "output_tokens": 40,
                    "total_tokens": 60,
                },
                "cumulative_usage": {
                    "input_tokens": 20,
                    "output_tokens": 40,
                    "total_tokens": 60,
                },
                "requested_tools": [
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"ticket_id": "T1"},
                        "validated_arguments": {"ticket_id": "T1"},
                        "validation_error": None,
                    }
                ],
                "elapsed_ms": 1251,
            },
            {
                "iteration_type": "tool_execution",
                "number": 1,
                "executed_tools": [
                    {
                        "id": "call-1",
                        "name": "lookup",
                        "arguments": {"ticket_id": "T1"},
                        "validated_arguments": {"ticket_id": "T1"},
                        "duration_ms": 450,
                        "success": True,
                        "result": {
                            "ok": True,
                            "result": {
                                "ticket_id": "T1",
                                "status": "open",
                            },
                        },
                    }
                ],
                "elapsed_ms": 1702,
            },
            {
                "iteration_type": "model_call",
                "number": 2,
                "model_duration_ms": 800,
                "usage": {
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "total_tokens": 0,
                },
                "requested_tools": [],
                "elapsed_ms": 2503,
            },
        ],
        "events": [
            {
                "type": "model_response_received",
                "duration_ms": 1250,
            },
            {
                "type": "tool_started",
                "tool_name": "lookup",
                "tool_call_id": "call-1",
            },
            {
                "type": "tool_completed",
                "tool_name": "lookup",
                "tool_call_id": "call-1",
                "duration_ms": 450,
                "success": True,
            },
            {
                "type": "final_answer",
                "answer": "Ticket reviewed successfully.",
            },
        ],
    }


def test_viewer_prints_normal_run(capsys, tmp_path):
    path = _write_jsonl(tmp_path, _base_record())

    assert main([str(path)]) == 0

    output = capsys.readouterr().out

    assert "Run: run-normal" in output
    assert "Iteration limit: 10 agent-loop model calls" in output
    assert "Agent loop 1: model call" in output
    assert "Agent loop 1: tool execution" in output
    assert "Agent loop 2: model call" in output
    assert "Tool: lookup" in output
    assert 'Arguments: {"ticket_id": "T1"}' in output
    assert "Result:" in output
    assert "Terminal reason: completed" in output
    assert "Tokens: input=20 output=40 total=60" in output
    assert "Estimated cost:" in output
    assert "Actual billed cost: unknown" in output


def test_viewer_handles_tool_failure(capsys, tmp_path):
    record = _base_record()
    record["iterations"][1]["executed_tools"][0] = {
        "id": "call-1",
        "name": "lookup",
        "duration_ms": 500,
        "success": False,
        "result": {
            "ok": False,
            "error": "service unavailable",
        },
        "error": "service unavailable",
    }

    path = _write_jsonl(tmp_path, record)

    assert main([str(path)]) == 0

    output = capsys.readouterr().out

    assert "Tool: lookup" in output
    assert "Success: False" in output
    assert "ERROR: service unavailable" in output


def test_viewer_handles_limit_termination(capsys, tmp_path):
    record = _base_record()
    record["terminal_reason"] = "execution_limit"
    record["events"].append(
        {
            "type": "tool_call_limit_hit",
            "tool_name": "lookup",
            "observed_calls": 3,
            "max_calls_per_tool": 2,
        }
    )

    path = _write_jsonl(tmp_path, record)

    assert main([str(path)]) == 0

    output = capsys.readouterr().out

    assert "Terminal reason: execution_limit" in output
    assert "lookup" in output


def test_viewer_handles_unknown_pricing(capsys, tmp_path):
    record = _base_record()
    record["configuration"]["model"] = "unknown-model"

    path = _write_jsonl(tmp_path, record)

    assert main([str(path)]) == 0

    output = capsys.readouterr().out

    assert "Estimated cost: unknown" in output
    assert "Actual billed cost: unknown" in output


def test_viewer_estimates_groq_gpt_oss_120b_cost(capsys, tmp_path):
    record = _base_record()
    record["configuration"].update({"provider": "groq", "model": "openai/gpt-oss-120b"})
    record["usage"] = {
        "input_tokens": 13_837,
        "output_tokens": 1_941,
        "total_tokens": 15_778,
    }
    path = _write_jsonl(tmp_path, record)

    assert main([str(path)]) == 0

    output = capsys.readouterr().out
    assert "Estimated cost: $0.00324015 USD" in output
    assert "Actual billed cost: unknown" in output


def test_viewer_handles_partial_trace(capsys, tmp_path):
    record = {
        "run_id": "partial-run",
        "terminal_reason": "execution_limit",
        "usage": {
            "input_tokens": 10,
        },
        "iterations": [
            {
                "iteration_type": "model_call",
                "number": 1,
                "requested_tools": [],
            }
        ],
    }

    path = _write_jsonl(tmp_path, record)

    assert main([str(path)]) == 0

    output = capsys.readouterr().out

    assert "Run: partial-run" in output
    assert "Provider: unknown" in output
    assert "Model: unknown" in output
    assert "Duration: unknown" in output
    assert "Terminal reason: execution_limit" in output
    assert "Estimated cost: unknown" in output


def test_viewer_rejects_malformed_json(tmp_path, capsys):
    path = tmp_path / "broken.jsonl"
    path.write_text(
        '{"run_id": "good"}\n{"run_id": "broken"\n',
        encoding="utf-8",
    )

    assert main([str(path)]) == 1

    error = capsys.readouterr().err
    assert "Malformed JSON on line 2" in error


def test_viewer_supports_run_id_filtering(capsys, tmp_path):
    path = tmp_path / "runs.jsonl"

    first = _base_record()
    first["run_id"] = "run-one"

    second = _base_record()
    second["run_id"] = "run-two"

    path.write_text(
        json.dumps(first) + "\n" + json.dumps(second) + "\n",
        encoding="utf-8",
    )

    assert main([str(path), "--run-id", "run-two"]) == 0

    output = capsys.readouterr().out

    assert "Run: run-two" in output
    assert "Run: run-one" not in output


def test_viewer_rejects_unknown_run_id(tmp_path, capsys):
    path = _write_jsonl(tmp_path, _base_record())

    assert main([str(path), "--run-id", "missing"]) == 1

    error = capsys.readouterr().err
    assert "Run ID not found: missing" in error


def test_viewer_supports_full_untruncated_output(tmp_path, capsys):
    record = _base_record()
    long_result = "x" * 200
    record["iterations"][1]["executed_tools"][0]["result"] = {
        "ok": True,
        "result": {"long_field": long_result},
    }

    path = _write_jsonl(tmp_path, record)

    # Without --full, should truncate
    assert main([str(path)]) == 0
    output_truncated = capsys.readouterr().out
    assert "…" in output_truncated
    assert long_result not in output_truncated

    # With --full, should not truncate
    assert main([str(path), "--full"]) == 0
    output_full = capsys.readouterr().out
    assert "…" not in output_full
    assert long_result in output_full
