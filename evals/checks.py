"""
Grading logic for evals/cases.py.

Deliberately independent of agent.py's internal message format: it
grades a flat, ordered list of {"name", "arguments", "result"} tool
dispatch records (see evals/harness.py's RecordingToolRegistry), not
agent.py's internal message history — even though run_agent_turn() does
return the message list, the recording registry stays the grading
source of truth so eval results don't depend on message-format details.
"""

from dataclasses import dataclass

from evals.cases import EvalCase


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    passed: bool
    failures: tuple[str, ...]


def _tool_names(calls: list[dict]) -> list[str]:
    return [c["name"] for c in calls]


def infer_decision(calls: list[dict]) -> str | None:
    """
    "reply" if send_reply was called, "escalate" if escalate was called,
    "conflicting" if both were (that's its own failure signal — a
    well-behaved trajectory should only ever hit one of the two
    terminal, state-changing tools), None if neither.
    """
    names = _tool_names(calls)
    saw_reply = "send_reply" in names
    saw_escalate = "escalate" in names
    if saw_reply and saw_escalate:
        return "conflicting"
    if saw_escalate:
        return "escalate"
    if saw_reply:
        return "reply"
    return None


def _reply_message(calls: list[dict]) -> str:
    for c in calls:
        if c["name"] == "send_reply":
            return str(c["arguments"].get("message", ""))
    return ""


def _escalate_reason(calls: list[dict]) -> str:
    for c in calls:
        if c["name"] == "escalate":
            return str(c["arguments"].get("reason", ""))
    return ""


def check_case(case: EvalCase, calls: list[dict], final_answer: str = "") -> CaseResult:
    failures: list[str] = []
    names = _tool_names(calls)

    decision = infer_decision(calls)
    expected_decision = (
        None if case.expected_decision == "answer" else case.expected_decision
    )
    if decision != expected_decision:
        failures.append(
            f"expected decision '{case.expected_decision}', observed '{decision}'"
        )

    for c in calls:
        if not c["result"].get("ok", False):
            failures.append(
                f"tool call '{c['name']}' failed: {c['result'].get('error')}"
            )

    if case.expected_decision == "reply":
        message = _reply_message(calls).lower()
        for concept in case.required_concepts:
            if concept.lower() not in message:
                failures.append(f"reply is missing required concept: {concept!r}")
        for claim in case.prohibited_claims:
            if claim.lower() in message:
                failures.append(f"reply contains a prohibited claim: {claim!r}")

        if case.trajectory.required_before_reply:
            reply_index = next(
                (i for i, c in enumerate(calls) if c["name"] == "send_reply"), None
            )
            if reply_index is None:
                failures.append("send_reply was never called; can't check ordering")
            else:
                before = names[:reply_index]
                for tool in case.trajectory.required_before_reply:
                    if tool not in before:
                        failures.append(
                            f"'{tool}' must be called before send_reply, but wasn't"
                        )

    if case.expected_decision == "answer":
        answer = final_answer.lower()
        for concept in case.required_concepts:
            if concept.lower() not in answer:
                failures.append(f"answer is missing required concept: {concept!r}")
        for claim in case.prohibited_claims:
            if claim.lower() in answer:
                failures.append(f"answer contains a prohibited claim: {claim!r}")

    if (
        case.expected_decision == "escalate"
        and case.trajectory.escalation_reason_contains
    ):
        reason = _escalate_reason(calls).lower()
        expected = case.trajectory.escalation_reason_contains.lower()
        if expected not in reason:
            failures.append(
                f"escalate reason is missing expected substring: {expected!r}"
            )

    for tool in case.trajectory.required_tools:
        if tool not in names:
            failures.append(f"required tool was never called: {tool!r}")

    for tool in case.trajectory.forbidden_tools:
        if tool in names:
            failures.append(f"forbidden tool was called: {tool!r}")

    return CaseResult(case_id=case.id, passed=not failures, failures=tuple(failures))
