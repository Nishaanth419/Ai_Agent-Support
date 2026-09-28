# Support Desk Triage Agent

An autonomous AI support triage agent built from scratch in Python, using Groq by default and a provider‑neutral model boundary. It features manual agent loops, tool dispatch, context management, and security guardrails without external agent frameworks.

## Prerequisites

- **Python 3.11 to 3.13** (Python 3.14+ is currently incompatible with `pydantic‑core` Rust compilation bindings).

## Setup Instructions

For full setup instructions, see the [Setup Guide](docs/setup.md).

## Project Structure

```text
src/
  core/        # config, utilities
  db/          # schema, migrations
  llm/         # model clients (base, fake, groq, gemini, factory)
app/
  agent.py     # core agent loop (run_agent_turn)
scripts/
  demo_context_compaction.py  # compaction demo
  demo_prompt_injection.py    # injection demo
  demo_rejection_revision.py # rejection/revision demo
  demo_terminations.py        # termination demo
  demo_tool_cap.py            # per‑tool‑cap demo
  model_smoke.py              # model sanity check
  rebuild_db.py               # DB rebuild & seed
  trace_viewer.py             # trace visualizer
tests/
  *.py         # pytest suite
```

### Execution Flow

1. Clone the repository and follow the [Setup Guide](docs/setup.md).
2. Create a virtual environment and install dependencies.
3. Seed the development database (`python -m scripts.rebuild_db`).
4. Run a demo (`python -m scripts.<demo>`) or the interactive chat agent (`run_agent.py`).
5. `app.agent.run_agent_turn` invokes the model client via `src.llm.factory.create_model_client`.
6. The model client (real or `FakeModelClient`) generates responses.
7. Tools are dispatched; results are recorded in `traces/`.
8. Inspect execution with `python -m scripts.trace_viewer`.
9. Verify with the test suite (`python -m pytest`).

#### src Files

| File                     | Description |
|--------------------------|-------------|
| `src/core/config.py`     | load config |
| `src/db/schema.py`       | DB schema |
| `src/llm/base.py`        | abstract client |
| `src/llm/fake.py`        | deterministic fake |
| `src/llm/gemini.py`      | Gemini client |
| `src/llm/groq.py`        | Groq client |
| `src/llm/factory.py`     | client factory |

#### app Files

| File           | Description |
|----------------|-------------|
| `app/agent.py` | core agent loop (run_agent_turn) |

#### scripts Files

| File                                 | Description |
|--------------------------------------|-------------|
| `scripts/demo_context_compaction.py` | compaction demo |
| `scripts/demo_prompt_injection.py`   | injection demo |
| `scripts/demo_rejection_revision.py`| rejection/revision demo |
| `scripts/demo_terminations.py`       | termination demo |
| `scripts/demo_tool_cap.py`           | per‑tool‑cap demo |
| `scripts/model_smoke.py`             | model sanity check |
| `scripts/rebuild_db.py`              | DB rebuild & seed |
| `scripts/trace_viewer.py`            | trace visualizer |



## Rebuild Development Database

One command safely drops and recreates `support.db` (at the project root) with deterministic fixture data, then validates the result:

```bash
python -m scripts.rebuild_db
```

**What it does:**

1. Deletes `support.db` if it already exists.
2. Applies the schema (`src/db/schema.py`) from scratch.
3. Seeds 15 customers, 28 tickets (across all five themes), ticket history, and a planted prompt‑injection ticket.
4. Validates table counts and all fixture requirements, printing a summary.

**Exit codes:**

| Code | Meaning |
|------|---------|
| `0`  | Database rebuilt and all checks passed |
| `1`  | Schema error, seed failure, or validation failure — a readable message is printed to stderr |

> **Safety:** The command only rebuilds the named `support.db` at the project root. No user‑supplied database path is accepted, so it cannot silently overwrite an arbitrary file.

**Automated tests** always rebuild into temporary locations (`pytest`'s `tmp_path` fixture) and never touch the real `support.db`. See [`tests/test_rebuild.py`](tests/test_rebuild.py).

## Live trace: compact vs. verbose

While `run_agent.py` is running, each tool call the agent makes is printed live (`LIVE_TRACE_VIEW`, on by default). By default this is a **compact** view: one line per tool call showing just the tool name and the few arguments that matter (e.g. `ticket_id`, `book_category`), and one line for the outcome (`✓`/`✗` + a short reason on failure) — no raw JSON, no per-loop token bookkeeping.

```
  → get_ticket(ticket_id=482)
    ✓ 41 ms
  → get_refund_policy(book_category=technology)
    ✓ 6 ms
```

Pass `-v`/`--verbose` to see the full step-by-step trace instead — every model call, full tool arguments, full tool results, tokens per step (the same detail `scripts/trace_viewer.py` renders after the fact from `traces/agent_runs.jsonl`, but live):

```bash
python3 -m run_agent --verbose
```

Set `LIVE_TRACE_VIEW=false` to turn off live trace output entirely, in either mode.

## Trace viewer

Render a structured JSONL agent trace as a readable execution tree:

```bash
python -m scripts.trace_viewer traces/agent_runs.jsonl
```

To view a specific run:

```bash
python -m scripts.trace_viewer traces/agent_runs.jsonl --run-id RUN_ID
```

**Optional helper – list all run IDs** (macOS):

```bash
# Install jq (JSON processor) via Homebrew
brew install jq

# Print every run_id in the trace file
jq -r '.run_id' traces/agent_runs.jsonl
```

You can pipe the IDs to the viewer, e.g.:

```bash
run_id=$(jq -r '.run_id' traces/agent_runs.jsonl | head -n 1)
python -m scripts.trace_viewer traces/agent_runs.jsonl --run-id "$run_id"
```

## Audit traces

Every `run_agent_turn` invocation appends one JSON object to `traces/agent_runs.jsonl` by default. Set `TRACE_PATH` to use another JSONL file, or pass `trace_path` when invoking the loop in tests or an integration. Records contain configuration identifiers, normalized reported usage, ordered `model_call` and `tool_execution` iterations, per‑tool timing and success/failure details, and milestone events for model responses, validation failures, tool activity, limits, and terminal outcomes. Trace values named like credentials and email addresses are redacted; ticket context and message history are intentionally not written.

An agent‑loop iteration is one model call plus the tool batch requested by that response. Both trace phases share the same iteration number. `MAX_ITERATIONS` therefore limits model calls, not individual tools or trace phases.

## Development diagnostics

Set `VERBOSE_DIAGNOSTICS=true` to emit concise, plain‑text loop diagnostics. Each model‑call line is prefixed with `[diagnostic]` and includes the iteration, message count, call token total, cumulative token total, requested tool name(s), and whether that response terminates the loop. Termination is logged separately.

These diagnostics never include message content, tool arguments, ticket context, or configuration values, so configured secrets are not printed. They are disabled by default and do not affect agent execution. Tests and integrations can pass `verbose_diagnostics=True` with a `diagnostic_sink` callback to assert the same structured fields without parsing terminal output or colours.

## Context‑compaction demonstration

Run the deterministic, offline scenario with:

```bash
python -m scripts.demo_context_compaction
```

It deliberately sets the compaction threshold to four messages, prints each before/after message count, proves that the system prompt and initial ticket are still in the next model context, and reports the valid iterations, cumulative tokens, and tool‑call count after compaction. The JSONL trace is appended to `traces/demo_context_compaction.jsonl`.

## Model smoke test

After setting `GROQ_API_KEY` in your shell or local `.env` file, verify provider access before developing the agent:

```bash
python -m scripts.model_smoke
```

This makes exactly one minimal model request. It prints the configured provider and model, a non‑empty response, and provider‑reported token usage when available. It never prints the API key. If it fails, follow the corrective guidance in the error output; do not add keys or response data to this README.

Gemini remains available as a rollback provider: set `LLM_PROVIDER=gemini`, choose a Gemini model in `LLM_MODEL_NAME`, and provide `GEMINI_API_KEY`.

## Per‑tool call cap demonstration

Run the deterministic demonstration with:

```bash
python -m scripts.demo_tool_cap
```

## Termination demonstration

Two deterministic scenarios — no API key or database required:

```bash
python -m scripts.demo_terminations
```

**Scenario A — Successful completion**

A seeded billing ticket is processed by a scripted two‑turn fake model. Turn 1 calls `get_ticket`; Turn 2 returns a final triage answer. The run terminates with `terminal_reason = "completed"`.

**Scenario B — Impossible goal / iteration‑cap termination**

The same ticket is processed by a fake model that always requests a tool and therefore never produces a terminal text response. With `max_iterations=3` the loop hits the configured cap, raises `IterationLimitExceededError`, and stops cleanly — no hang, no recursion overflow, no unhandled exception. The run terminates with `terminal_reason = "execution_limit"`.

Both scenarios print per‑iteration diagnostics (iteration number, message count, call tokens, cumulative tokens, requested tool name) to stdout and append a structured trace record to `traces/demo_terminations.jsonl`. Configuration identifiers (provider, model, temperature, caps) are printed; no secrets are included.

The same terminal‑reason assertions are verified automatically by:

```bash
python -m pytest tests/test_termination_demos.py -v
```

## Running Tests

Execute the automated test suite:

```bash
python -m pytest
```

## Code Quality & Formatting

Linting and code formatting are managed using [Ruff](https://docs.astral.sh/ruff/):

```bash
# Check code for linting issues
ruff check .

# Check formatting compliance
ruff format --check .

# Automatically apply formatting fixes
ruff format .
```

## CI/CD Pipeline

The project uses GitHub Actions for continuous integration, with workflows located in `.github/workflows/`. Every push and pull request triggers linting, formatting checks, and the full test suite.
