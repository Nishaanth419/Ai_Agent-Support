import json
from typing import Any, Dict, List, Optional

from groq import APIConnectionError, APIStatusError, Groq

from src.core.config import Config
from src.core.exceptions import ProviderAPIError
from src.core.logger import get_logger
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest

logger = get_logger(__name__)


def _is_tool_use_failed(error: APIStatusError) -> bool:
    """True for Groq's "tool_use_failed" 400: the model's own tool-call
    arguments didn't parse as JSON server-side. Distinct from a malformed
    response we successfully receive and then fail to decode ourselves."""
    body = error.body
    return (
        error.status_code == 400
        and isinstance(body, dict)
        and isinstance(body.get("error"), dict)
        and body["error"].get("code") == "tool_use_failed"
    )


class GroqClient(ModelClient):
    """Groq Chat Completions adapter for the provider-neutral agent loop."""

    def __init__(self):
        try:
            self.client = Groq(api_key=Config.GROQ_API_KEY)
            self.model = Config.MODEL_NAME
        except Exception as error:
            raise ProviderAPIError(
                f"Failed to initialize Groq client: {error}"
            ) from error

    @staticmethod
    def _convert_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Translate the loop's normalized history to Groq chat messages."""
        formatted: List[Dict[str, Any]] = []

        for message in messages:
            role = message["role"]

            if role in {"system", "user"}:
                formatted.append({"role": role, "content": message["content"]})
                continue

            if role == "model":
                assistant_message: Dict[str, Any] = {
                    "role": "assistant",
                    "content": message.get("content"),
                }
                if message.get("tool_calls"):
                    assistant_message["tool_calls"] = [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"]),
                            },
                        }
                        for call in message["tool_calls"]
                    ]
                formatted.append(assistant_message)
                continue

            if role == "tool":
                formatted.append(
                    {
                        "role": "tool",
                        "tool_call_id": message["tool_call_id"],
                        "name": message["name"],
                        "content": message["content"],
                    }
                )

        return formatted

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> NormalizedResponse:
        request: Dict[str, Any] = {
            "model": self.model,
            "messages": self._convert_messages(messages),
            "temperature": Config.TEMPERATURE,
            # Allow the model to return several independent tool calls in one
            # response. The agent loop validates and dispatches the full batch.
            "parallel_tool_calls": True,
        }
        if tools:
            request["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool["description"],
                        "parameters": tool["parameters"],
                    },
                }
                for tool in tools
            ]

        max_attempts = Config.GROQ_TOOL_USE_FAILED_MAX_RETRIES + 1
        response = None
        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(f"Calling Groq model {self.model}")
                response = self.client.chat.completions.create(**request)
                break
            except APIStatusError as error:
                if error.status_code in {401, 403}:
                    raise ProviderAPIError(
                        "Authentication failed. Check your Groq API key."
                    ) from error
                if error.status_code == 429:
                    raise ProviderAPIError(
                        "Groq rate limit exceeded. Please try again later."
                    ) from error
                if _is_tool_use_failed(error) and attempt < max_attempts:
                    logger.warning(
                        f"Groq failed to parse tool call arguments as JSON "
                        f"(attempt {attempt}/{max_attempts}); retrying."
                    )
                    continue
                raise ProviderAPIError(
                    f"Groq API error [{error.status_code}]: {error}"
                ) from error
            except APIConnectionError as error:
                raise ProviderAPIError(f"Could not connect to Groq: {error}") from error
            except Exception as error:
                raise ProviderAPIError(
                    f"Unexpected error communicating with Groq: {error}"
                ) from error

        usage = TokenUsage()
        if response.usage is not None:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
                reported=True,
            )

        text = None
        tool_requests = []
        stop_reason = None

        if response.choices:
            choice = response.choices[0]
            stop_reason = choice.finish_reason
            text = choice.message.content or None

            for call in choice.message.tool_calls or []:
                try:
                    arguments = json.loads(call.function.arguments)
                except (json.JSONDecodeError, TypeError) as error:
                    raise ProviderAPIError(
                        f"Groq returned invalid JSON arguments for tool "
                        f"'{call.function.name}'."
                    ) from error

                if not isinstance(arguments, dict):
                    raise ProviderAPIError(
                        f"Groq returned non-object arguments for tool "
                        f"'{call.function.name}'."
                    )

                tool_requests.append(
                    ToolRequest(
                        id=call.id,
                        name=call.function.name,
                        arguments=arguments,
                    )
                )

        return NormalizedResponse(
            text=text,
            tool_requests=tool_requests,
            stop_reason=stop_reason,
            usage=usage,
        )
