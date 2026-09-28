"""Provider-boundary tests for Gemini function declaration schemas."""

from google.genai import types

from app.tools import build_tool_registry
from src.llm.gemini import _gemini_compatible_schema


def test_registered_tool_schemas_can_build_gemini_function_declarations():
    for tool in build_tool_registry().definitions():
        declaration = types.FunctionDeclaration(
            name=tool.name,
            description=tool.description,
            parameters=_gemini_compatible_schema(tool.parameters),
        )

        assert declaration.name == tool.name
        assert declaration.parameters is not None


def test_gemini_conversion_preserves_strict_registry_schema():
    registry_schema = {
        "type": "object",
        "properties": {
            "payload": {
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "additionalProperties": False,
            }
        },
        "additionalProperties": False,
    }

    provider_schema = _gemini_compatible_schema(registry_schema)

    assert "additionalProperties" not in provider_schema
    assert "additionalProperties" not in provider_schema["properties"]["payload"]
    assert registry_schema["additionalProperties"] is False
    assert registry_schema["properties"]["payload"]["additionalProperties"] is False
