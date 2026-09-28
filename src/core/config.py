import os
from dotenv import load_dotenv
from src.core.exceptions import ConfigurationError

load_dotenv()


def _environment_flag(name: str, default: bool = False) -> bool:
    """Read a conventional boolean environment flag without raising on typos."""
    return os.getenv(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


class Config:
    # Provider Settings
    PROVIDER = os.getenv("LLM_PROVIDER", "groq").lower()
    MODEL_NAME = os.getenv("LLM_MODEL_NAME", "openai/gpt-oss-120b")
    TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.0"))

    # Credentials
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # Execution Limits
    MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", "10"))
    MAX_TOTAL_TOKENS = int(os.getenv("MAX_TOTAL_TOKENS", "50000"))
    MAX_CALLS_PER_TOOL = int(os.getenv("MAX_CALLS_PER_TOOL", "5"))

    # Trigger compaction when message history exceeds this length. 24 means
    # the 25th message triggers it (the loop checks len(messages) > threshold).
    COMPACTION_THRESHOLD = int(os.getenv("COMPACTION_THRESHOLD", "24"))

    # When compaction drops older turns, replace them with a summary message
    # instead of deleting them outright (see compact_history in app/agent.py).
    # The summary is always built deterministically from structured facts
    # extracted from the dropped tool calls; when this is on, the agent loop
    # additionally asks the model itself to rewrite those facts into a terser
    # paragraph, at the cost of one extra model call per compaction. Off by
    # default in anything driven by a scripted/fake client (evals, most
    # tests), since those assert exact call counts.
    COMPACTION_USE_LLM_SUMMARY = _environment_flag(
        "COMPACTION_USE_LLM_SUMMARY", default=True
    )

    # Hard cap on how many summary bullet facts compact_history carries
    # forward across repeated compactions in one long-running conversation,
    # so the summary itself can't grow without bound.
    COMPACTION_SUMMARY_MAX_FACTS = int(os.getenv("COMPACTION_SUMMARY_MAX_FACTS", "40"))

    # Development-only loop diagnostics. This never changes loop decisions.
    VERBOSE_DIAGNOSTICS = _environment_flag("VERBOSE_DIAGNOSTICS")

    # Live, human-readable trace of each agent-loop step (model calls, tool
    # calls/results, compaction) printed to the terminal as a turn runs --
    # the same detail scripts/trace_viewer.py renders after the fact, but
    # live. On by default so a customer-care operator can see what the
    # agent is doing without re-running the trace viewer.
    LIVE_TRACE_VIEW = _environment_flag("LIVE_TRACE_VIEW", default=True)

    # Audit tracing. This is a JSONL stream: one complete run record per line.
    TRACE_PATH = os.getenv("TRACE_PATH", "traces/agent_runs.jsonl")

    # Hybrid ticket search (Qdrant + local sentence-transformers embeddings).
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "support_tickets")
    EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    # all-MiniLM-L6-v2's known output size. Kept as a fixed constant (rather
    # than read from a loaded model) so constructing the default vector
    # store never has to load the embedding model first -- a Qdrant
    # reachability check can fail fast without paying model-load cost.
    EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))

    # Weights blending vector similarity with the keyword-overlap signal into
    # one relevance score; SEARCH_RELEVANCE_FLOOR drops weak matches so
    # search_tickets can still say "no matching past tickets." These are a
    # reasoned starting point, not empirically tuned -- revisit via evals/
    # once real embeddings are exercised end-to-end.
    SEARCH_VECTOR_WEIGHT = float(os.getenv("SEARCH_VECTOR_WEIGHT", "0.6"))
    SEARCH_KEYWORD_WEIGHT = float(os.getenv("SEARCH_KEYWORD_WEIGHT", "0.4"))
    SEARCH_RELEVANCE_FLOOR = float(os.getenv("SEARCH_RELEVANCE_FLOOR", "0.35"))
    # Candidates fetched from Qdrant before keyword re-ranking narrows down
    # to SEARCH_RESULT_LIMIT; larger than the final cap so re-ranking has
    # room to reorder.
    SEARCH_CANDIDATE_LIMIT = int(os.getenv("SEARCH_CANDIDATE_LIMIT", "30"))

    # list_tickets: hard cap on rows returned per call, regardless of what
    # the model requests, so one analysis query can't pull the whole table.
    LIST_TICKETS_MAX_LIMIT = int(os.getenv("LIST_TICKETS_MAX_LIMIT", "200"))

    # run_python_analysis sandbox limits (see src/sandbox/executor.py).
    # SANDBOX_TIMEOUT_SECONDS bounds wall-clock time (backstop even when the
    # platform doesn't support the RLIMIT_CPU/RLIMIT_AS best-effort limits).
    SANDBOX_TIMEOUT_SECONDS = int(os.getenv("SANDBOX_TIMEOUT_SECONDS", "5"))
    SANDBOX_MEMORY_LIMIT_MB = int(os.getenv("SANDBOX_MEMORY_LIMIT_MB", "256"))
    SANDBOX_MAX_CODE_LENGTH = int(os.getenv("SANDBOX_MAX_CODE_LENGTH", "4000"))
    SANDBOX_MAX_OUTPUT_CHARS = int(os.getenv("SANDBOX_MAX_OUTPUT_CHARS", "4000"))

    # Groq occasionally emits a tool call whose arguments aren't valid JSON
    # (its own parser rejects them server-side with code "tool_use_failed"
    # before a response ever reaches us) -- a transient generation glitch,
    # not a real request problem, and simply retrying the same request
    # usually succeeds. This bounds how many times we retry before giving up.
    GROQ_TOOL_USE_FAILED_MAX_RETRIES = int(
        os.getenv("GROQ_TOOL_USE_FAILED_MAX_RETRIES", "2")
    )

    @classmethod
    def validate(cls):
        if cls.PROVIDER == "groq" and not cls.GROQ_API_KEY:
            raise ConfigurationError(
                "GROQ_API_KEY is missing. Please check your .env file."
            )

        if cls.PROVIDER == "gemini" and not cls.GEMINI_API_KEY:
            raise ConfigurationError(
                "GEMINI_API_KEY is missing. Please check your .env file."
            )

        if cls.PROVIDER not in ["groq", "gemini", "fake"]:
            raise ConfigurationError(f"Unsupported provider: {cls.PROVIDER}")
