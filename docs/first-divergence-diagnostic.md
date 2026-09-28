# First-divergence diagnostic: duplicate escalation

```toml
source_trace_file = "traces/agent_runs.jsonl"
trace_file = "tests/fixtures/first_divergence_trace.jsonl"
run_id = "f7a0c7af-1e26-4420-b917-436d4efd5cf4"
evidence_iteration = 1
first_off_course_iteration = 2
expected_decision = "stop_without_tool"
observed_decision = "escalate"
evaluation_check = "tests/test_first_divergence_diagnostic.py::test_targeted_improvement_evaluation_check"
```

Iteration 1 was on course: the initial `run_agent.py` context omitted ticket
status, so `get_ticket(ticket_id="2")` was a necessary lookup. Its result reported
`status="escalated"`. At iteration 2, the expected decision was therefore to call
no tool and return a brief status summary. Instead, the model called `escalate`.
The tool refused the duplicate operation with `escalated=false` and “ticket already
has an open escalation.”

Iteration 2 is the first divergence because iteration 1 gathered missing status
evidence correctly. The duplicate-escalation error and the correct no-tool summary
at iteration 3 occurred only after the off-course call, so they are consequences
and recovery rather than the originating mistake.

The system prompt already says terminal-status tickets must stop, so this trace
does not support claiming that the rule was absent. The status was nested inside a
larger tool result, while the iteration-2 escalation reason focused on the invoice
body. That may indicate an evidence-salience/context issue, but the trace alone
does not prove why the model overlooked status. The `get_ticket` tool description
also does not mention the stop rule; making terminal status operationally explicit
there is a plausible targeted improvement.

The linked evaluation check encodes the regression condition: once `get_ticket`
returns `escalated`, `resolved`, or `closed`, a later `send_reply` or `escalate`
request is a failure. A rerun after the tool-description improvement should finish
with a no-tool status summary and produce no such violation.
