# Agent Design Note

## The Model Client
The client abstracts the underlying LLM provider. By injecting the `FakeModelClient` or `ScriptedModelClient` during `pytest` runs, we can simulate complex tool-use trajectories and iteration limits without consuming network requests or API quotas.

Logging is implemented natively. Set `LOG_LEVEL=DEBUG` in your `.env` to trace HTTP requests.

## Context History Compaction

To ensure long-running agent trajectories remain within context window constraints, `app/agent.py` implements a deterministic message history compaction mechanism (`compact_history`).

### 1. Configuration
- **Environment Variable**: `COMPACTION_THRESHOLD` (default: `12`).
- **Function Parameter**: `compaction_threshold` in `run_agent_turn()`.

### 2. Triggering and Hysteresis Buffer
Compaction evaluates at the beginning of each loop turn. When `len(messages) > compaction_threshold`, compaction triggers and condenses message history down to a target size of `max(4, threshold - 4)`. This 4-message hysteresis buffer prevents compaction from triggering repeatedly on every iteration once the threshold is crossed.

### 3. Policy & Retention Rules
- **Verbatim System Prompt & Ticket**: The system prompt (`messages[0]`) and original user ticket/goal (`messages[1]`) are always preserved verbatim.
- **Turn-based Protocol Integrity**: Messages in the message history tail are grouped into complete logical turns (a `model` request message and all corresponding `tool` result messages). Turns are dropped or retained as complete units so protocol integrity is preserved.
- **Human Rejection Feedback Retained**: Any turn containing human rejection feedback (`status == "rejected"`) is explicitly protected from compaction to maintain feedback context.
- **Recent Evidence Retained**: The most recent turn (unresolved evidence) is always retained.

### 4. Limit Independence
Compaction modifies only the active `messages` context payload sent to the LLM. It **never** alters or resets:
- Cumulative token accounting (`cumulative_tokens`)
- Completed iteration counts (`iteration`)
- Per-tool execution limits (`tool_call_counts`)

### 5. Diagnostics and Observability
- **Trace Event**: Emits a `history_compacted` event to `RunTracer` with fields `messages_before`, `messages_after`, `dropped_messages`, and `threshold`.
- **Diagnostic Event**: Emits an `event="compaction"` payload to standard diagnostics logging and any configured `diagnostic_sink`.