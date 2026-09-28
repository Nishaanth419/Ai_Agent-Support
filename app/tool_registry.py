from dataclasses import dataclass
from typing import Any, Callable

import json
from jsonschema import Draft202012Validator


@dataclass(frozen=True)
class ToolDefinition:
    name: str
    description: str
    parameters: dict[str, Any]
    function: Callable[..., Any]
    state_changing: bool = False
    post_dispatch_hook: Callable[[Any, dict[str, Any]], Any] | None = None
    output_schema: dict[str, Any] | None = None


def success_result(result: Any) -> dict[str, Any]:
    return {
        "ok": True,
        "result": result,
    }


def error_result(message: str) -> dict[str, Any]:
    return {
        "ok": False,
        "error": message,
    }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool already registered: {tool.name}")

        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDefinition | None:
        return self._tools.get(name)

    def definitions(self) -> list[ToolDefinition]:
        return list(self._tools.values())

    def dispatch(
        self,
        name: str,
        arguments: str | dict[str, Any],
    ) -> dict[str, Any]:
        tool = self.get(name)

        if tool is None:
            return error_result(f"Unknown tool: {name}")

        validation = validate_arguments(tool, arguments)

        if not validation["ok"]:
            return validation

        validated_arguments = validation["result"]

        try:
            result = tool.function(**validated_arguments)
            if tool.post_dispatch_hook is not None:
                result = tool.post_dispatch_hook(result, validated_arguments)
            return success_result(result)
        except Exception as exc:
            return error_result(f"Tool execution failed: {type(exc).__name__}: {exc}")


def validate_arguments(
    tool: ToolDefinition,
    arguments: str | dict[str, Any],
) -> dict[str, Any]:
    try:
        if isinstance(arguments, str):
            parsed = json.loads(arguments)
        elif isinstance(arguments, dict):
            parsed = arguments
        else:
            return error_result("Malformed tool arguments: expected a JSON object.")
    except json.JSONDecodeError:
        return error_result("Malformed tool arguments: expected valid JSON.")

    if not isinstance(parsed, dict):
        return error_result("Malformed tool arguments: expected a JSON object.")

    validator = Draft202012Validator(tool.parameters)
    errors = sorted(
        validator.iter_errors(parsed),
        key=lambda error: list(error.path),
    )

    if errors:
        error = errors[0]

        path = ".".join(str(part) for part in error.path)

        if error.validator == "required":
            missing = error.message.split("'")[1]
            return error_result(f"Missing required argument: {missing}")

        if error.validator == "additionalProperties":
            unexpected = error.message.split("'")[1]
            return error_result(f"Unexpected argument: {unexpected}")

        if error.validator == "enum":
            field = path or "value"
            allowed = ", ".join(repr(value) for value in error.validator_value)
            return error_result(
                f"Invalid value for {field}. Expected one of: {allowed}"
            )

        if error.validator == "type":
            field = path or "value"
            expected = error.validator_value
            return error_result(f"Invalid type for {field}: expected {expected}")

        return error_result(f"Invalid arguments: {error.message}")

    return success_result(parsed)
