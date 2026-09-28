"""Human approval interfaces for state-changing support actions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class ApprovalDecision:
    """The human decision returned to a state-changing tool."""

    approved: bool
    feedback: str = ""


class ApprovalInterface(Protocol):
    """Boundary that production terminal input and test fakes both implement."""

    def request_approval(self, ticket_id: str, message: str) -> ApprovalDecision:
        """Display a proposed reply and return the human decision."""


class TerminalApprovalInterface:
    """Request an explicit approve/reject decision at the terminal."""

    def __init__(
        self,
        input_fn: Callable[[str], str] = input,
        output_fn: Callable[[str], None] = print,
    ) -> None:
        self._input = input_fn
        self._output = output_fn

    def request_approval(self, ticket_id: str, message: str) -> ApprovalDecision:
        self._output(f"Ticket ID: {ticket_id}")
        self._output("Proposed reply:")
        self._output(message)

        while True:
            decision = self._input("Decision [approve/reject]: ").strip().lower()
            if decision == "approve":
                return ApprovalDecision(approved=True)
            if decision == "reject":
                feedback = self._input("Feedback for the agent (optional): ").strip()
                return ApprovalDecision(approved=False, feedback=feedback)
            self._output("Please enter exactly 'approve' or 'reject'.")
