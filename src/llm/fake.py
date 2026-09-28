from typing import Any, Dict, List, Optional
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage


class FakeModelClient(ModelClient):
    """A deterministically controllable model client for automated tests."""

    def __init__(self, canned_response: NormalizedResponse = None):
        self.canned_response = canned_response or NormalizedResponse(
            text="This is a deterministic fake response.",
            stop_reason="STOP",
            usage=TokenUsage(
                input_tokens=10,
                output_tokens=10,
                total_tokens=20,
                reported=True,
            ),
        )
        self.call_count = 0
        self.last_messages = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> NormalizedResponse:
        self.call_count += 1
        self.last_messages = messages
        return self.canned_response


class ScriptedModelClient(ModelClient):
    """
    Returns a fixed sequence of NormalizedResponse objects, one per call
    to generate(), in order.

    Added for the "define deterministic evaluation cases" card:
    FakeModelClient above always returns the same single canned
    response, which can't drive a multi-step tool-calling trajectory
    (e.g. get_refund_policy, then send_reply, then a final text answer).
    ScriptedModelClient lets an eval case author write that exact
    turn-by-turn script once, then run it through the real agent loop
    (app.agent.run_agent_turn) with no network call and no
    randomness — see evals/cases.py and evals/harness.py.

    Raises RuntimeError if generate() is called more times than the
    script has turns for, rather than silently repeating the last
    response or the first one: running out of script means either the
    case's script is too short for what the agent loop actually did, or
    the agent loop changed in a way the case wasn't written for. Either
    way, a loud failure here is more useful than a confusing extra tool
    call the case author never scripted.
    """

    def __init__(self, script: List[NormalizedResponse]):
        if not script:
            raise ValueError("ScriptedModelClient needs at least one scripted turn.")
        self._script = list(script)
        self.call_count = 0
        self.last_messages = []

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> NormalizedResponse:
        self.last_messages = messages
        if self.call_count >= len(self._script):
            raise RuntimeError(
                f"ScriptedModelClient exhausted: generate() called "
                f"{self.call_count + 1} time(s) but the script only has "
                f"{len(self._script)} turn(s). Either add another turn to "
                "the case's script, or the agent loop looped more than "
                "this case expected."
            )
        response = self._script[self.call_count]
        self.call_count += 1
        return response
