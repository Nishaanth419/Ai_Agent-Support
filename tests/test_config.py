from unittest.mock import patch

import pytest
from src.core.config import Config
from src.core.exceptions import ConfigurationError


def test_missing_api_key():
    with (
        patch.object(Config, "GEMINI_API_KEY", None),
        patch.object(Config, "PROVIDER", "gemini"),
    ):
        with pytest.raises(ConfigurationError, match="GEMINI_API_KEY is missing"):
            Config.validate()


def test_missing_groq_api_key():
    with (
        patch.object(Config, "GROQ_API_KEY", None),
        patch.object(Config, "PROVIDER", "groq"),
    ):
        with pytest.raises(ConfigurationError, match="GROQ_API_KEY is missing"):
            Config.validate()


def test_unsupported_provider():
    with patch.object(Config, "PROVIDER", "unsupported_provider"):
        with pytest.raises(ConfigurationError, match="Unsupported provider"):
            Config.validate()


def test_valid_config():
    with (
        patch.object(Config, "GEMINI_API_KEY", "test-key"),
        patch.object(Config, "PROVIDER", "gemini"),
    ):
        # Validation completes without raising ConfigurationError
        Config.validate()


def test_valid_groq_config():
    with (
        patch.object(Config, "GROQ_API_KEY", "test-key"),
        patch.object(Config, "PROVIDER", "groq"),
    ):
        Config.validate()
