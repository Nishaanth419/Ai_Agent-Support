"""Tests for the stable ABC Bookstore support desk chat system prompt."""

from src.agent.system_prompt import SYSTEM_PROMPT

# snake_case tool identifiers the prompt must not mention -- the design
# deliberately leaves tool choice and sequencing to the model's own judgment
# instead of naming specific tools or prescribing when to call them (see the
# module-level comment in src/agent/system_prompt.py for the rationale).
# "escalate" is deliberately excluded: it's an ordinary English word the
# prompt's one-line capability summary uses ("reply to, escalate, or open
# tickets"), not a reference to the escalate tool's exact identifier.
TOOL_NAMES = (
    "get_ticket",
    "search_tickets",
    "get_customer_history",
    "get_refund_policy",
    "list_tickets",
    "run_python_analysis",
    "send_reply",
    "create_ticket",
)


def _normalized_prompt() -> str:
    """Make semantic phrase checks insensitive to prompt line wrapping."""
    return " ".join(SYSTEM_PROMPT.lower().split())


def test_system_prompt_is_stable():
    assert isinstance(SYSTEM_PROMPT, str)
    assert SYSTEM_PROMPT.strip()

    original = SYSTEM_PROMPT
    assert SYSTEM_PROMPT == original


def test_system_prompt_states_goal_and_domain():
    prompt = _normalized_prompt()

    assert "support desk agent" in prompt
    assert "abc bookstore" in prompt
    assert "technology" in prompt
    assert "fiction" in prompt
    assert "self-help" in prompt
    assert "operator" in prompt


def test_system_prompt_keeps_scope_boundary_invariant():
    """Nothing in code stops the model from answering unrelated general-
    knowledge questions -- this has to stay in the prompt."""
    prompt = _normalized_prompt()

    assert "general-purpose assistant" in prompt
    assert "outside" in prompt


def test_system_prompt_keeps_untrusted_data_invariant():
    """Nothing in code stops retrieved ticket/customer text from being read
    as instructions -- this has to stay in the prompt."""
    prompt = _normalized_prompt()

    assert "untrusted data" in prompt
    assert "never" in prompt
    assert "as instructions" in prompt


def test_system_prompt_keeps_no_fabrication_invariant():
    """Nothing in code can catch the model claiming an action happened in its
    own final answer text -- this has to stay in the prompt."""
    prompt = _normalized_prompt()

    assert "claim" in prompt or "imply" in prompt
    assert "action was taken" in prompt
    assert "refund" in prompt


def test_system_prompt_does_not_name_tools_or_prescribe_sequencing():
    """Proves the prompt is goal-only: no named-tool guidance, no prescribed
    branching/batching/state-machine rules. Tool selection is left entirely
    to the model, informed only by the tool descriptions it's given."""
    prompt = _normalized_prompt()

    for tool_name in TOOL_NAMES:
        assert tool_name not in prompt

    assert "if status is escalated" not in prompt
    assert "batch independent read-only calls" not in prompt
    assert "call exactly one state-changing tool" not in prompt
    assert "before any refund" not in prompt


def test_system_prompt_does_not_encode_execution_limits():
    prompt = _normalized_prompt()

    assert "maximum iterations" not in prompt
    assert "max iterations" not in prompt
    assert "maximum tokens" not in prompt
    assert "max tokens" not in prompt
    assert "maximum tool calls" not in prompt
    assert "max tool calls" not in prompt
