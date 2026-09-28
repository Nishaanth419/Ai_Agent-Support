from src.core.exceptions import ConfigurationError, ProviderAPIError


PROMPT = "Reply with the single word: ready"


def main() -> int:
    # Delayed imports let configuration errors be rendered as actionable output.
    try:
        from src.core.config import Config
        from src.llm.factory import create_model_client
    except ConfigurationError as error:
        print(f"Configuration failed: {error}")
        print(
            "Set the required environment variables (or copy .env.example to .env) "
            "and run this command again.",
        )
        return 1

    print(f"Provider: {Config.PROVIDER}")
    print(f"Model: {Config.MODEL_NAME}")

    try:
        response = create_model_client().generate([{"role": "user", "content": PROMPT}])
    except ConfigurationError as error:
        print(f"Configuration failed: {error}")
        print(
            "Set the required environment variables (or copy .env.example to .env) "
            "and run this command again.",
        )
        return 1
    except ProviderAPIError as error:
        print(f"Model request failed: {error}")
        print(
            "Check that the configured API key is valid, the selected model is "
            "available to the provider account, and the account has network/API access.",
        )
        return 1
    except Exception:
        print("Model client could not be initialized or contacted.")
        print(
            "Verify LLM_PROVIDER, LLM_MODEL_NAME, and the provider-specific API "
            "key environment variable, then try again.",
        )
        return 1

    text = (response.text or "").strip()
    if not text:
        print("Model request returned no text response.")
        print(
            "Confirm the selected model can generate text and that provider access "
            "is enabled for this account.",
        )
        return 1

    print(f"Response: {text}")
    if response.usage.reported:
        print(f"Input tokens: {response.usage.input_tokens}")
        print(f"Output tokens: {response.usage.output_tokens}")
        print(f"Total tokens: {response.usage.total_tokens}")
    else:
        print("Token usage: not supplied by provider")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
