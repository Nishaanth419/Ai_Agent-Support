"""Provider/model pricing used by the trace viewer.

Prices are USD per 1 million tokens.

Sources:
https://ai.google.dev/gemini-api/docs/pricing
https://console.groq.com/docs/model/openai/gpt-oss-120b

The viewer uses these rates only to estimate what the recorded token usage
would cost at the configured paid-tier list price. Actual billed cost is
reported separately and remains unknown unless the trace explicitly contains it.
"""

PRICING = {
    ("groq", "openai/gpt-oss-120b"): {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
    },
    ("gemini", "gemini-3.6-flash"): {
        "input_per_million": 0.75,
        "output_per_million": 3.75,
    },
}


def get_pricing(provider: str | None, model: str | None) -> dict[str, float] | None:
    """Return pricing for a provider/model pair, or None when unknown."""
    if not provider or not model:
        return None

    return PRICING.get((provider.lower(), model))


def estimate_cost(
    provider: str | None,
    model: str | None,
    input_tokens: int,
    output_tokens: int,
) -> float | None:
    """Estimate USD cost from provider/model pricing and token usage."""
    pricing = get_pricing(provider, model)

    if pricing is None:
        return None

    input_cost = (input_tokens / 1_000_000) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000) * pricing["output_per_million"]

    return input_cost + output_cost
