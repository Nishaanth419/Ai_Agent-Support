class AgentError(Exception):
    """Base exception for all agent-related errors."""

    pass


class ConfigurationError(AgentError):
    """Raised when environment variables are missing or invalid."""

    pass


class ProviderAPIError(AgentError):
    """Raised when the LLM provider returns an API error (auth, rate limits, malformed)."""

    pass


class TokenAccountingError(AgentError):
    """Raised when exact token usage metadata is unavailable from provider."""

    pass


class ExecutionLimitExceededError(AgentError):
    """Base class for execution cap limit violations."""

    pass


class IterationLimitExceededError(ExecutionLimitExceededError):
    """Raised when agent exceeds maximum allowed iterations."""

    pass


class TokenLimitExceededError(ExecutionLimitExceededError):
    """Raised when agent exceeds maximum allowed total tokens."""

    pass


class ToolCallLimitExceededError(ExecutionLimitExceededError):
    """Raised when a specific tool exceeds maximum allowed calls per run."""

    pass
