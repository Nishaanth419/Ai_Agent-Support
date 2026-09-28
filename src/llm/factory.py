"""Construct the configured model client without exposing provider details to callers."""

from src.core.config import Config
from src.llm.base import ModelClient
from src.llm.fake import FakeModelClient
from src.llm.gemini import GeminiClient
from src.llm.groq import GroqClient


def create_model_client() -> ModelClient:
    """Return a client for the provider selected by ``LLM_PROVIDER``."""
    # Keep configuration imports safe for offline tools and unit tests. A real
    # provider credential is required only when an application asks to create
    # the configured model client.
    Config.validate()

    if Config.PROVIDER == "groq":
        return GroqClient()
    if Config.PROVIDER == "gemini":
        return GeminiClient()
    if Config.PROVIDER == "fake":
        return FakeModelClient()

    # Config.validate() prevents this; retain the guard for type narrowing and
    # callers that replace Config.validate() during tests.
    raise ValueError(f"Unsupported provider: {Config.PROVIDER}")
