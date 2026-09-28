import json
from collections.abc import Callable
from pathlib import Path
from time import monotonic
from typing import Any, Dict, Tuple

from app.tool_registry import ToolRegistry, validate_arguments
from src.core.config import Config
from src.core.exceptions import (
    IterationLimitExceededError,
    TokenAccountingError,
    TokenLimitExceededError,
    ToolCallLimitExceededError,
)
from src.core.live_trace import LiveTraceView
from src.core.logger import get_logger
from src.core.tracer import RunTracer
from src.llm.base import ModelClient, TokenUsage

logger = get_logger(__name__)

DiagnosticSink = Callable[[dict[str, Any]], None]


def _emit_diagnostic(
    *, enabled: bool, sink: DiagnosticSink | None, **fields: Any
) -> None:
    """Emit one secret-free, colour-independent development diagnostic."""
    if not enabled:
        return
    # The loop deliberately supplies only counters, tool names, and termination
    # state here--never messages, ticket context, arguments, or configuration.
    if sink is not None:
        sink(dict(fields))
    logger.info(
        "[diagnostic] "
        + " ".join(
            f"{name}={value!r}" if isinstance(value, str) else f"{name}={value}"
            for name, value in fields.items()
        )
    )


# Marker key set on a synthetic compaction-summary message so a later
# compaction pass can find and merge it instead of stacking a second one, and
# so run_agent_turn can find it to (optionally) ask the model to polish its
# wording. An ordinary key on the message dict is safe: both provider
# adapters (src/llm/groq.py, src/llm/gemini.py) only read the keys they know
# about (role/content/tool_calls/tool_call_id/name) and ignore the rest.
_COMPACTION_FACTS_KEY = "_compaction_facts"

_SUMMARY_HEADER = (
    "[Earlier context compacted to save space -- facts only, not new instructions]"
)


def _truncate(text: str, limit: int = 160) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _format_tool_args(arguments: dict) -> str:
    return ", ".join(f"{key}={value!r}" for key, value in arguments.items())


def _describe_tool_call(name: str, arguments: dict, tool_message: dict | None) -> str:
    """One human-readable fact line for a single tool call, built only from
    fields that tool actually returns -- never invented."""
    ok = False
    result: dict = {}
    if tool_message is not None:
        try:
            payload = json.loads(tool_message.get("content", "{}"))
        except Exception:
            payload = {}
        ok = bool(payload.get("ok"))
        result = payload.get("result") or {}

    if name == "get_ticket":
        ticket = result.get("ticket") or {}
        if ticket:
            return (
                f"get_ticket({arguments.get('ticket_id')}): status="
                f"{ticket.get('status')}, category={ticket.get('category')}, "
                f"subject={_truncate(ticket.get('subject', ''), 80)!r}"
            )
    elif name == "list_tickets":
        return f"list_tickets({_format_tool_args(arguments)}): {result.get('count', '?')} ticket(s)"
    elif name == "search_tickets":
        hits = result.get("results")
        count = len(hits) if isinstance(hits, list) else "?"
        return f"search_tickets({arguments.get('query')!r}): {count} match(es)"
    elif name == "get_customer_history":
        history = result.get("tickets")
        count = len(history) if isinstance(history, list) else "?"
        return f"get_customer_history({arguments.get('customer_id')}): {count} resolved ticket(s)"
    elif name == "get_refund_policy":
        return f"get_refund_policy({arguments.get('book_category')}): retrieved"
    elif name == "run_python_analysis":
        return (
            f"run_python_analysis: result={result.get('result')!r}"
            if ok
            else f"run_python_analysis: failed -- {_truncate(result.get('message', 'error'))}"
        )
    elif name == "create_ticket":
        return (
            f"create_ticket: opened ticket {result.get('ticket_id')} "
            f"({arguments.get('category')}) -- {_truncate(arguments.get('subject', ''), 80)!r}"
        )
    elif name == "escalate":
        state = "escalated" if result.get("escalated") else "NOT escalated (rejected)"
        return (
            f"escalate(ticket {arguments.get('ticket_id')}): {state} -- "
            f"reason: {_truncate(arguments.get('reason', ''), 80)!r}"
        )
    elif name == "send_reply":
        status = result.get("status")
        if status == "rejected":
            return (
                f"send_reply(ticket {arguments.get('ticket_id')}): REJECTED by "
                f"human reviewer -- feedback: {result.get('human_feedback')!r}. "
                "Must be revised and retried, not abandoned."
            )
        if status == "approved":
            return f"send_reply(ticket {arguments.get('ticket_id')}): approved and sent"
        return f"send_reply(ticket {arguments.get('ticket_id')}): {status or ('ok' if ok else 'failed')}"

    return f"{name}({_format_tool_args(arguments)}): {'ok' if ok else 'failed'}"


def _extract_turn_facts(turn: list[dict]) -> list[str]:
    """Reduce one dropped turn to a short list of fact strings: one line per
    tool call (name, arguments, and outcome) plus any user/agent text in the
    turn. This is what a dropped turn becomes instead of disappearing."""
    facts: list[str] = []
    tool_by_id = {
        message["tool_call_id"]: message
        for message in turn
        if message["role"] == "tool" and "tool_call_id" in message
    }
    for message in turn:
        if message["role"] == "model" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                tool_message = tool_by_id.get(call.get("id"))
                facts.append(
                    _describe_tool_call(
                        call.get("name", "tool"),
                        call.get("arguments", {}),
                        tool_message,
                    )
                )
        elif message["role"] == "model" and message.get("content"):
            facts.append(f"Agent said: {_truncate(message['content'])}")
        elif message["role"] == "user":
            facts.append(f"Operator said: {_truncate(message['content'])}")
    return facts


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def _build_summary_message(facts: list[str], max_facts: int) -> dict:
    """The synthetic message that replaces every turn a compaction pass
    drops. Kept as a plain-text ``model`` message (a assistant-authored
    recap) rather than a ``user`` message, so it can never be mistaken for a
    new operator instruction, and never as a ``tool`` message, since it has
    no matching tool_call_id for a provider to pair it against.
    """
    facts = _dedupe_preserve_order(facts)[-max_facts:]
    content = _SUMMARY_HEADER + "\n" + "\n".join(f"- {fact}" for fact in facts)
    return {
        "role": "model",
        "content": content,
        _COMPACTION_FACTS_KEY: facts,
    }


def compact_history(
    messages: list[dict],
    threshold: int,
    max_summary_facts: int = Config.COMPACTION_SUMMARY_MAX_FACTS,
) -> tuple[list[dict], int]:
    """
    Compacts message history by replacing older, non-critical turns with a
    single summary message instead of deleting them outright.

    Policy:
      - Triggers when len(messages) > threshold.
      - Targets a length of (threshold - 4) to create a buffer. This hysteresis
        prevents compaction from running on every single iteration once crossed.
      - Preserves the System prompt and initial user context (verifies first 2 messages).
      - Preserves API protocol integrity (drops complete model + tool-result turns).
      - Preserves any turn containing human rejection feedback.
      - Preserves the most recent turn (unresolved evidence).
      - Preserves whichever turn holds the most recent user message, even
        when later tool round-trips have pushed it out of the last-turn slot.
      - Every turn that *is* dropped is reduced to fact lines (ticket ids,
        statuses, tool outcomes, human rejection feedback) and folded into
        one running summary message placed right after the preserved prefix,
        rather than discarded. A later compaction pass merges into that same
        summary instead of stacking a second one, so it never disappears and
        never grows without bound (capped at max_summary_facts).
    """
    if len(messages) <= threshold:
        return messages, 0

    # Target size creates a buffer so we don't compact every single iteration
    target_size = max(4, threshold - 4)

    # Defensive check: ensure the first two messages are exactly what we expect
    if (
        len(messages) >= 2
        and messages[0].get("role") == "system"
        and messages[1].get("role") == "user"
    ):
        kept_prefix = messages[:2]
        tail = messages[2:]
    else:
        # Fallback if structure changes unexpectedly
        logger.warning(
            "Message structure unexpected; preserving only the first message."
        )
        kept_prefix = messages[:1]
        tail = messages[1:]

    # Pull out any earlier compaction summary so it gets merged into the one
    # this pass produces instead of stacking a second summary message.
    carried_facts: list[str] = []
    filtered_tail = []
    for message in tail:
        if message["role"] == "model" and _COMPACTION_FACTS_KEY in message:
            carried_facts.extend(message[_COMPACTION_FACTS_KEY])
        else:
            filtered_tail.append(message)
    tail = filtered_tail

    # Group messages into complete "turns" (1 Model request + N Tool results)
    turns = []
    current_turn = []
    for msg in tail:
        if msg["role"] == "model":
            if current_turn:
                turns.append(current_turn)
            current_turn = [msg]
        else:
            current_turn.append(msg)
    if current_turn:
        turns.append(current_turn)

    def has_rejection(turn: list[dict]) -> bool:
        """Check if any tool result in this turn was rejected by a human."""
        for message in turn:
            if message["role"] == "tool":
                try:
                    data = json.loads(message.get("content", "{}"))
                    if data.get("status") == "rejected":
                        return True
                except Exception:
                    pass
        return False

    def has_user_message(turn: list[dict]) -> bool:
        """Check if this turn contains a user-authored message.

        A follow-up question has no ``model`` message of its own to start a
        turn, so grouping above appends it to whichever turn was still open
        (the *preceding* model/tool exchange). Dropping that turn would
        silently delete the user's current question from context while any
        data already fetched for it survives orphaned in a later turn --
        the model then falls back to the oldest still-visible question
        (the seed prompt) instead of the one it's actually meant to answer.
        """
        return any(message["role"] == "user" for message in turn)

    dropped_count = 0
    kept_turns = []
    dropped_facts: list[str] = list(carried_facts)
    current_len = len(kept_prefix) + len(tail)

    # Use enumerate to guarantee we safely identify the last turn without using `is` or `==`
    total_turns = len(turns)

    # Identify the turn holding the most recent user message so it's never
    # dropped, even when it isn't the structurally-last turn (e.g. several
    # tool round-trips have already been appended after it in this same run).
    latest_user_turn_index = -1
    for i, turn in enumerate(turns):
        if has_user_message(turn):
            latest_user_turn_index = i

    for i, turn in enumerate(turns):
        is_last_turn = i == total_turns - 1
        is_latest_user_turn = i == latest_user_turn_index

        # Drop the turn if we are over target, it's not the latest evidence,
        # it doesn't hold the current user question, and it has no rejections
        if (
            current_len > target_size
            and not is_last_turn
            and not is_latest_user_turn
            and not has_rejection(turn)
        ):
            dropped_count += len(turn)
            current_len -= len(turn)
            dropped_facts.extend(_extract_turn_facts(turn))
        else:
            kept_turns.append(turn)

    # A carried-forward summary that ends up empty this round (nothing new
    # dropped, nothing carried in) means there's nothing to insert; otherwise
    # a prior summary that shrank to zero new facts would still leave a
    # dangling near-empty message behind.
    new_messages = list(kept_prefix)
    if dropped_facts:
        new_messages.append(_build_summary_message(dropped_facts, max_summary_facts))
    for turn in kept_turns:
        new_messages.extend(turn)

    return new_messages, dropped_count


def _polish_compaction_summary(
    summary_message: dict, client: ModelClient, cumulative_tokens: TokenUsage
) -> bool:
    """Ask the model to rewrite compact_history's structured fact list as
    terser prose, in place on ``summary_message``.

    Best-effort: any failure -- a provider error, missing usage metadata, an
    empty reply -- leaves the structured bullet list (already correct,
    produced by compact_history itself) untouched, so a flaky summarization
    call degrades gracefully instead of losing history or crashing the run.
    Token usage from this extra call is folded into ``cumulative_tokens`` so
    it still counts against the run's token budget.
    """
    facts = summary_message.get(_COMPACTION_FACTS_KEY) or []
    if not facts:
        return False

    try:
        response = client.generate(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Rewrite the following facts about an in-progress "
                        "customer-support conversation as a short, dense "
                        "paragraph for the support agent's own working "
                        "memory. Preserve every ticket id, status, and human "
                        "reviewer feedback exactly as given. Do not add, "
                        "infer, omit, or soften any fact."
                    ),
                },
                {
                    "role": "user",
                    "content": "\n".join(f"- {fact}" for fact in facts),
                },
            ]
        )
    except Exception:
        logger.warning(
            "Compaction summary polish call failed; keeping structured facts.",
            exc_info=True,
        )
        return False

    if response.usage.reported:
        cumulative_tokens.input_tokens += response.usage.input_tokens
        cumulative_tokens.output_tokens += response.usage.output_tokens
        cumulative_tokens.total_tokens += response.usage.total_tokens

    if not response.text or not response.text.strip():
        return False

    summary_message["content"] = _SUMMARY_HEADER + "\n" + response.text.strip()
    return True


def run_agent_turn(
    messages: list[dict],
    client: ModelClient,
    registry: ToolRegistry,
    max_iterations: int = Config.MAX_ITERATIONS,
    max_total_tokens: int = Config.MAX_TOTAL_TOKENS,
    max_calls_per_tool: int = Config.MAX_CALLS_PER_TOOL,
    compaction_threshold: int = Config.COMPACTION_THRESHOLD,
    summarize_dropped_history: bool = Config.COMPACTION_USE_LLM_SUMMARY,
    trace_path: str | Path | None = None,
    run_id: str | None = None,
    verbose_diagnostics: bool = Config.VERBOSE_DIAGNOSTICS,
    diagnostic_sink: DiagnosticSink | None = None,
    live_view: LiveTraceView | None = None,
) -> Tuple[str, TokenUsage, list[dict]]:
    """
    The core framework-free agent loop. Takes an already-seeded message
    history (system prompt + at least one user turn, built by the caller --
    a chat REPL, an eval case, a demo script, or a test) and runs it until a
    terminal text reply is generated, an escalate call short-circuits the
    run, or an execution limit (iterations, tokens, tool calls) is reached.

    Returns the final answer, cumulative token usage, and the (possibly
    compacted) message list, so a caller can seed the next turn from it.
    """
    tracer = RunTracer(
        trace_path or Config.TRACE_PATH,
        configuration={
            "provider": Config.PROVIDER,
            "model": Config.MODEL_NAME,
            "temperature": Config.TEMPERATURE,
            "max_iterations": max_iterations,
            "max_total_tokens": max_total_tokens,
            "max_calls_per_tool": max_calls_per_tool,
            "compaction_threshold": compaction_threshold,
        },
        run_id=run_id,
    )

    # Snapshot the caller's seed messages so later compaction diagnostics can
    # report whether they're still intact, without this function needing to
    # know anything about "tickets" or how the caller built them.
    seed_system_message = (
        dict(messages[0]) if messages and messages[0].get("role") == "system" else None
    )
    seed_user_message = (
        dict(messages[1])
        if len(messages) >= 2 and messages[1].get("role") == "user"
        else None
    )

    # 2. Extract tools into a schema list for the client.
    # output_schema on each ToolDefinition is serialised into a "Returns:"
    # block and appended to the description at runtime, so the LLM knows
    # what every tool produces without any manual edits to tool descriptions.
    tools_schema = []
    for t in registry.definitions():
        description = t.description
        if t.output_schema:
            props = t.output_schema.get("properties", {})
            if props:
                lines = ["", "Returns:"]
                for field, meta in props.items():
                    ftype = meta.get("type", "any")
                    fdesc = meta.get("description", "")
                    lines.append(f"  {field} ({ftype}): {fdesc}" if fdesc else f"  {field} ({ftype})")
                description = description.rstrip() + "\n".join(lines)
        tools_schema.append(
            {"name": t.name, "description": description, "parameters": t.parameters}
        )

    cumulative_tokens = TokenUsage()
    tool_call_counts: Dict[str, int] = {}
    iteration = 0

    if live_view is not None:
        live_view.turn_started()

    try:
        while True:
            if len(messages) > compaction_threshold:
                old_count = len(messages)
                messages, dropped = compact_history(messages, compaction_threshold)
                if dropped > 0:
                    logger.info(
                        f"Context compacted: {old_count} messages -> {len(messages)} messages "
                        f"(dropped {dropped})"
                    )

                    # compact_history already replaced the dropped turns with
                    # a deterministic, structured summary message -- history
                    # is never silently deleted. This step is a pure
                    # nice-to-have on top of that: ask the model to rewrite
                    # those same facts as terser prose. Any failure here
                    # leaves the structured summary in place untouched.
                    llm_summary_applied = False
                    if summarize_dropped_history:
                        summary_message = next(
                            (m for m in messages if _COMPACTION_FACTS_KEY in m), None
                        )
                        if summary_message is not None:
                            llm_summary_applied = _polish_compaction_summary(
                                summary_message, client, cumulative_tokens
                            )
                            if cumulative_tokens.total_tokens > max_total_tokens:
                                tracer.add_event(
                                    "token_limit_hit",
                                    observed_total_tokens=cumulative_tokens.total_tokens,
                                    max_total_tokens=max_total_tokens,
                                )
                                raise TokenLimitExceededError(
                                    "Maximum token limit reached while summarizing "
                                    f"compacted history: observed {cumulative_tokens.total_tokens} "
                                    f"total tokens, cap is {max_total_tokens}."
                                )

                    tracer.add_event(
                        "history_compacted",
                        messages_before=old_count,
                        messages_after=len(messages),
                        dropped_messages=dropped,
                        threshold=compaction_threshold,
                        llm_summary_applied=llm_summary_applied,
                        system_prompt_preserved=(
                            len(messages) >= 1
                            and messages[0].get("role") == "system"
                            and messages[0] == seed_system_message
                        ),
                        seed_user_message_preserved=(
                            len(messages) >= 2
                            and messages[1].get("role") == "user"
                            and messages[1] == seed_user_message
                        ),
                        cumulative_usage=cumulative_tokens.model_dump(
                            exclude={"reported"}
                        ),
                        tool_call_counts=dict(tool_call_counts),
                    )
                    if live_view is not None:
                        live_view.compaction(
                            before=old_count,
                            after=len(messages),
                            dropped=dropped,
                            llm_summary_applied=llm_summary_applied,
                        )
                    _emit_diagnostic(
                        enabled=verbose_diagnostics,
                        sink=diagnostic_sink,
                        event="compaction",
                        iteration=iteration,
                        original_message_count=old_count,
                        compacted_message_count=len(messages),
                        dropped_count=dropped,
                    )

            iteration += 1

            # Check Point 1: Iteration limit check (before model call)
            if iteration > max_iterations:
                tracer.add_event(
                    "iteration_limit_hit",
                    completed_iterations=iteration - 1,
                    max_iterations=max_iterations,
                )
                raise IterationLimitExceededError(
                    f"Maximum iteration limit reached: observed {iteration - 1} completed iterations, cap is {max_iterations}."
                )

            # 3. Call the model
            model_started = monotonic()
            response = client.generate(messages=messages, tools=tools_schema)
            model_duration_ms = round((monotonic() - model_started) * 1000)
            tracer.add_event(
                "model_response_received",
                model_iteration=iteration,
                duration_ms=model_duration_ms,
                requested_tool_count=len(response.tool_requests),
            )

            # Check Point 2: Token metadata availability check
            if not response.usage.reported:
                tracer.add_event("token_accounting_failed", model_iteration=iteration)
                raise TokenAccountingError(
                    "LLM provider failed to return exact token usage metadata."
                )

            # Update token accounting (combining all calls)
            cumulative_tokens.input_tokens += response.usage.input_tokens
            cumulative_tokens.output_tokens += response.usage.output_tokens
            cumulative_tokens.total_tokens += response.usage.total_tokens

            requested_tool_names = [req.name for req in response.tool_requests]
            _emit_diagnostic(
                enabled=verbose_diagnostics,
                sink=diagnostic_sink,
                event="model_call",
                iteration=iteration,
                message_count=len(messages),
                call_tokens=response.usage.total_tokens,
                cumulative_tokens=cumulative_tokens.total_tokens,
                requested_tool_name=(
                    requested_tool_names[0] if requested_tool_names else None
                ),
                requested_tool_names=requested_tool_names,
                terminal=not response.tool_requests,
            )

            requested_tools = []
            validation_by_call_id = {}
            for req in response.tool_requests:
                # Some integrations provide the original minimal registry protocol
                # (``definitions`` + ``dispatch``) without ToolRegistry.get().
                # Resolve from definitions as a compatible fallback for trace-only
                # pre-validation.
                get_tool = getattr(registry, "get", None)
                definition = (
                    get_tool(req.name)
                    if callable(get_tool)
                    else next(
                        (
                            tool
                            for tool in registry.definitions()
                            if tool.name == req.name
                        ),
                        None,
                    )
                )
                validation = (
                    validate_arguments(definition, req.arguments)
                    if definition is not None
                    else {"ok": False, "error": f"Unknown tool: {req.name}"}
                )

                validation_by_call_id[req.id] = validation

                requested_tools.append(
                    {
                        "id": req.id,
                        "name": req.name,
                        "arguments": req.arguments,
                        "validated_arguments": validation.get("result")
                        if validation["ok"]
                        else None,
                        "validation_error": validation.get("error"),
                    }
                )
                if not validation["ok"]:
                    tracer.add_event(
                        "tool_validation_failed",
                        tool_name=req.name,
                        tool_call_id=req.id,
                        error=validation["error"],
                    )

            if live_view is not None:
                live_view.model_call(
                    iteration=iteration,
                    duration_ms=model_duration_ms,
                    usage=response.usage.model_dump(exclude={"reported"}),
                    requested_tools=requested_tools,
                )

            def trace_iteration() -> None:
                tracer.add_iteration(
                    # Trace phases use the same number as the agent loop. A
                    # model call and the resulting tool batch are two phases of
                    # one iteration, not two iterations for limit purposes.
                    number=iteration,
                    iteration_type="model_call",
                    model_duration_ms=model_duration_ms,
                    usage=response.usage.model_dump(exclude={"reported"}),
                    cumulative_usage=cumulative_tokens.model_dump(exclude={"reported"}),
                    requested_tools=requested_tools,
                )

            # Check Point 3: Total token limit check (immediately after token update)
            if cumulative_tokens.total_tokens > max_total_tokens:
                trace_iteration()
                tracer.add_event(
                    "token_limit_hit",
                    observed_total_tokens=cumulative_tokens.total_tokens,
                    max_total_tokens=max_total_tokens,
                )
                raise TokenLimitExceededError(
                    f"Maximum token limit reached: observed {cumulative_tokens.total_tokens} total tokens, cap is {max_total_tokens}."
                )

            # 4 & 5. Handle Model Response (Text and/or Tools combined into ONE message)
            model_message = {"role": "model"}
            has_content = False

            if response.text:
                model_message["content"] = response.text
                has_content = True

            if response.tool_requests:
                model_message["tool_calls"] = []
                for req in response.tool_requests:
                    tool_call = {
                        "id": req.id,
                        "name": req.name,
                        "arguments": req.arguments,
                    }
                    if req.provider_metadata:
                        tool_call["provider_metadata"] = req.provider_metadata
                    model_message["tool_calls"].append(tool_call)
                has_content = True

            # Append the single combined message
            if has_content:
                messages.append(model_message)

            # Dispatch tools if they were requested
            if response.tool_requests:
                executed_tools = []

                def trace_tool_execution() -> None:
                    tracer.add_iteration(
                        number=iteration,
                        iteration_type="tool_execution",
                        executed_tools=executed_tools,
                    )

                trace_iteration()
                for req in response.tool_requests:
                    validation = validation_by_call_id[req.id]
                    # Check Point 4: Per-tool call limit check (BEFORE executing the tool!)
                    tool_call_counts[req.name] = tool_call_counts.get(req.name, 0) + 1
                    if tool_call_counts[req.name] > max_calls_per_tool:
                        tracer.add_event(
                            "tool_call_limit_hit",
                            tool_name=req.name,
                            observed_calls=tool_call_counts[req.name],
                            max_calls_per_tool=max_calls_per_tool,
                        )
                        if executed_tools:
                            trace_tool_execution()
                        raise ToolCallLimitExceededError(
                            f"Maximum call limit reached for tool '{req.name}': "
                            f"observed {tool_call_counts[req.name]} calls, cap is {max_calls_per_tool}."
                        )

                    tracer.add_event(
                        "tool_started", tool_name=req.name, tool_call_id=req.id
                    )
                    if live_view is not None:
                        live_view.tool_started(name=req.name)
                    tool_started = monotonic()
                    result = registry.dispatch(req.name, req.arguments)
                    duration_ms = round((monotonic() - tool_started) * 1000)
                    success = result.get("ok") is True
                    if live_view is not None:
                        live_view.tool_completed(
                            name=req.name,
                            duration_ms=duration_ms,
                            success=success,
                            result=result,
                        )
                    tool_record = {
                        "id": req.id,
                        "name": req.name,
                        "arguments": req.arguments,
                        "validated_arguments": validation.get("result")
                        if validation["ok"]
                        else None,
                        "duration_ms": duration_ms,
                        "success": success,
                        "result": result,
                    }
                    if not success:
                        tool_record["error"] = result.get("error", "Tool failed.")
                    executed_tools.append(tool_record)
                    tracer.add_event(
                        "tool_completed",
                        tool_name=req.name,
                        tool_call_id=req.id,
                        duration_ms=duration_ms,
                        success=success,
                        **({"error": tool_record["error"]} if not success else {}),
                    )

                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": req.id,
                            "name": req.name,
                            "content": json.dumps(result),
                        }
                    )

                    # A confirmed escalation hands ownership to a human queue;
                    # do not give the model another turn in the same ticket run.
                    if (
                        req.name == "escalate"
                        and result.get("ok")
                        and result["result"].get("escalated") is True
                    ):
                        trace_tool_execution()
                        tracer.add_event(
                            "final_answer", answer=result["result"]["message"]
                        )
                        tracer.finalize(
                            "completed",
                            cumulative_tokens.model_dump(exclude={"reported"}),
                        )
                        if live_view is not None:
                            live_view.turn_finished(
                                terminal_reason="completed",
                                usage=cumulative_tokens.model_dump(
                                    exclude={"reported"}
                                ),
                            )
                        return result["result"]["message"], cumulative_tokens, messages

                trace_tool_execution()
                continue

            # 6. Loop Termination
            # If no tools were requested, the model's text is its final triage decision.
            final_answer = response.text or "Model terminated without text."
            trace_iteration()
            tracer.add_event("final_answer", answer=final_answer)
            tracer.finalize(
                "completed", cumulative_tokens.model_dump(exclude={"reported"})
            )
            if live_view is not None:
                live_view.turn_finished(
                    terminal_reason="completed",
                    usage=cumulative_tokens.model_dump(exclude={"reported"}),
                )
            _emit_diagnostic(
                enabled=verbose_diagnostics,
                sink=diagnostic_sink,
                event="termination",
                terminal_reason="completed",
                iteration=iteration,
                cumulative_tokens=cumulative_tokens.total_tokens,
            )
            return final_answer, cumulative_tokens, messages
    except Exception as exc:
        terminal_reason = (
            "execution_limit"
            if isinstance(
                exc,
                (
                    IterationLimitExceededError,
                    TokenLimitExceededError,
                    ToolCallLimitExceededError,
                ),
            )
            else "failed"
        )
        tracer.add_event(
            terminal_reason, error_type=type(exc).__name__, message=str(exc)
        )
        tracer.finalize(
            terminal_reason,
            cumulative_tokens.model_dump(exclude={"reported"}),
            error_type=type(exc).__name__,
        )
        if live_view is not None:
            live_view.run_failed(error_type=type(exc).__name__, message=str(exc))
        _emit_diagnostic(
            enabled=verbose_diagnostics,
            sink=diagnostic_sink,
            event="termination",
            terminal_reason=terminal_reason,
            iteration=iteration,
            cumulative_tokens=cumulative_tokens.total_tokens,
        )
        raise
