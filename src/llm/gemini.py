import json
import uuid
from typing import Any, Dict, List, Optional
from google import genai
from google.genai import types
from google.genai.errors import APIError

from src.core.config import Config
from src.core.exceptions import ProviderAPIError
from src.core.logger import get_logger
from src.llm.base import ModelClient, NormalizedResponse, TokenUsage, ToolRequest

logger = get_logger(__name__)


def _gemini_compatible_schema(schema: Any) -> Any:
    """Copy a JSON schema while removing keywords unsupported by Gemini.

    The registry intentionally uses ``additionalProperties: false`` for strict
    local argument validation. Older Gemini schema models reject that JSON
    Schema keyword, so it is omitted only when building the provider request.
    Recursive copying preserves the registry's original schema and also handles
    object schemas nested inside arrays or properties.

    A few id-like fields declare ``"type": ["string", "integer"]`` so a
    provider's own strict tool-call validation accepts either JSON form for a
    numeric-looking id. Gemini's ``types.Schema`` only accepts one type
    value, so a list is collapsed to ``"string"`` (or the sole remaining
    type) here; the tool functions already accept both forms regardless of
    what the provider sends.
    """
    if isinstance(schema, dict):
        result = {}
        for key, value in schema.items():
            if key == "additionalProperties":
                continue
            if key == "type" and isinstance(value, list):
                value = "string" if "string" in value else value[0]
            result[key] = _gemini_compatible_schema(value)
        return result
    if isinstance(schema, list):
        return [_gemini_compatible_schema(value) for value in schema]
    return schema


class GeminiClient(ModelClient):
    def __init__(self):
        # Initializes using the exact SDK pattern. API key is pulled from env by the SDK or explicitly passed.
        try:
            self.client = genai.Client(api_key=Config.GEMINI_API_KEY)
            self.model = Config.MODEL_NAME
        except Exception as e:
            raise ProviderAPIError(f"Failed to initialize Gemini Client: {str(e)}")

    def _convert_messages(self, messages: List[Dict[str, Any]]):
        """Translates standard message dicts into Gemini's specific Content objects."""
        system_instruction = None
        formatted = []

        for msg in messages:
            # 1. System Prompt
            if msg["role"] == "system":
                system_instruction = msg["content"]
                continue

            # 2. User input
            if msg["role"] == "user":
                formatted.append(
                    types.Content(
                        role="user", parts=[types.Part.from_text(text=msg["content"])]
                    )
                )

            # 3. Model Output (Text and/or Tool Calls)
            elif msg["role"] == "model":
                parts = []

                # Add text part if present
                if "content" in msg and msg["content"]:
                    parts.append(types.Part.from_text(text=msg["content"]))

                # Add function call parts if present
                if "tool_calls" in msg:
                    for c in msg["tool_calls"]:
                        provider_metadata = c.get("provider_metadata", {})
                        parts.append(
                            types.Part(
                                function_call=types.FunctionCall(
                                    name=c["name"], args=c["arguments"]
                                ),
                                thought_signature=provider_metadata.get(
                                    "thought_signature"
                                ),
                            )
                        )

                if parts:
                    formatted.append(types.Content(role="model", parts=parts))

            # 4. Tool Results
            elif msg["role"] == "tool":
                # Gemini expects tool results wrapped in a 'user' role with a function_response part
                formatted.append(
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_function_response(
                                name=msg["name"],
                                # SDK expects a dict here, so we load the JSON string back
                                response=json.loads(msg["content"]),
                            )
                        ],
                    )
                )

        return system_instruction, formatted

    def generate(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
    ) -> NormalizedResponse:
        system_instruction, gemini_messages = self._convert_messages(messages)

        # Map our agnostic tool schemas to Gemini's FunctionDeclaration
        gemini_tools = None
        if tools:
            declarations = [
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t["description"],
                    parameters=_gemini_compatible_schema(t["parameters"]),
                )
                for t in tools
            ]
            gemini_tools = [types.Tool(function_declarations=declarations)]

        config = types.GenerateContentConfig(
            temperature=Config.TEMPERATURE,
            system_instruction=system_instruction,
            tools=gemini_tools,
        )

        try:
            logger.debug(f"Calling Gemini model {self.model}")
            response = self.client.models.generate_content(
                model=self.model, contents=gemini_messages, config=config
            )
        except APIError as e:
            # Normalizing provider errors into readable application errors
            if e.code in [401, 403]:
                raise ProviderAPIError(
                    f"Authentication failed. Check your API key. Details: {e.message}"
                )
            elif e.code == 429:
                raise ProviderAPIError("Rate limit exceeded. Please try again later.")
            else:
                raise ProviderAPIError(f"Provider API Error [{e.code}]: {e.message}")
        except Exception as e:
            raise ProviderAPIError(
                f"Unexpected error communicating with Gemini: {str(e)}"
            )

        # 1. Extract Usage
        usage = TokenUsage()
        if response.usage_metadata:
            usage.input_tokens = response.usage_metadata.prompt_token_count or 0
            usage.output_tokens = response.usage_metadata.candidates_token_count or 0
            usage.total_tokens = response.usage_metadata.total_token_count or 0
            usage.reported = True

        # 2. Extract Response / Tools / Stop Reason
        text = None
        tool_requests = []
        stop_reason = None

        if response.candidates:
            candidate = response.candidates[0]
            stop_reason = (
                candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
            )

            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if part.text:
                        text = (text or "") + part.text
                    elif part.function_call:
                        provider_metadata = {}
                        if part.thought_signature is not None:
                            provider_metadata["thought_signature"] = (
                                part.thought_signature
                            )
                        tool_requests.append(
                            ToolRequest(
                                id=str(
                                    uuid.uuid4()
                                ),  # Gemini doesn't always provide tool IDs, so we mint one
                                name=part.function_call.name,
                                arguments=dict(part.function_call.args),
                                provider_metadata=provider_metadata,
                            )
                        )

        return NormalizedResponse(
            text=text, tool_requests=tool_requests, stop_reason=stop_reason, usage=usage
        )


# trigger CI refresh
