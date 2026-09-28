from app.tool_registry import ToolDefinition, ToolRegistry
from app.tools import (
    GET_REFUND_POLICY_SCHEMA,
    get_refund_policy,
)
import pytest
from app.tools import build_tool_registry


def echo(message: str) -> dict:
    return {"echo": message}


ECHO_SCHEMA = {
    "type": "object",
    "properties": {
        "message": {
            "type": "string",
        }
    },
    "required": ["message"],
    "additionalProperties": False,
}


def make_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo the supplied message for testing.",
            parameters=ECHO_SCHEMA,
            function=echo,
        )
    )

    return registry


def test_successful_dispatch():
    registry = make_registry()

    result = registry.dispatch(
        "echo",
        '{"message": "hello"}',
    )

    assert result["ok"] is True
    assert result["result"] == {"echo": "hello"}


def test_unknown_tool():
    registry = make_registry()

    result = registry.dispatch(
        "delete_database",
        "{}",
    )

    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


def test_malformed_json():
    registry = make_registry()

    result = registry.dispatch(
        "echo",
        '{"message": ',
    )

    assert result["ok"] is False
    assert "Malformed tool arguments" in result["error"]


def test_missing_required_argument():
    registry = make_registry()

    result = registry.dispatch(
        "echo",
        "{}",
    )

    assert result["ok"] is False
    assert "Missing required argument" in result["error"]


def test_unexpected_argument():
    registry = make_registry()

    result = registry.dispatch(
        "echo",
        '{"message": "hello", "hack": true}',
    )

    assert result["ok"] is False
    assert "Unexpected argument" in result["error"]


def test_wrong_argument_type():
    registry = make_registry()

    result = registry.dispatch(
        "echo",
        '{"message": 123}',
    )

    assert result["ok"] is False
    assert "Invalid type" in result["error"]


def test_invalid_plan_enum():
    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="get_refund_policy",
            description="Retrieve the refund policy.",
            parameters=GET_REFUND_POLICY_SCHEMA,
            function=get_refund_policy,
        )
    )

    result = registry.dispatch(
        "get_refund_policy",
        '{"book_category": "premium"}',
    )

    assert result["ok"] is False
    assert "Invalid value" in result["error"]


def test_invalid_arguments_do_not_execute_function():
    called = False

    def tool(message: str):
        nonlocal called
        called = True
        return {"message": message}

    registry = ToolRegistry()

    registry.register(
        ToolDefinition(
            name="test",
            description="Test tool.",
            parameters={
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                },
                "required": ["message"],
                "additionalProperties": False,
            },
            function=tool,
        )
    )

    result = registry.dispatch(
        "test",
        '{"message": 123}',
    )

    assert result["ok"] is False
    assert called is False


def test_post_dispatch_hook_transforms_a_successful_result():
    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo the supplied message for testing.",
            parameters=ECHO_SCHEMA,
            function=echo,
            post_dispatch_hook=lambda result, arguments: {
                **result,
                "shouted": arguments["message"].upper(),
            },
        )
    )

    result = registry.dispatch("echo", '{"message": "hello"}')

    assert result["ok"] is True
    assert result["result"] == {"echo": "hello", "shouted": "HELLO"}


def test_post_dispatch_hook_is_not_applied_when_validation_fails():
    called = False

    def hook(result, arguments):
        nonlocal called
        called = True
        return result

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo the supplied message for testing.",
            parameters=ECHO_SCHEMA,
            function=echo,
            post_dispatch_hook=hook,
        )
    )

    result = registry.dispatch("echo", "{}")

    assert result["ok"] is False
    assert called is False


def test_post_dispatch_hook_failure_degrades_like_a_failing_tool_function():
    def failing_hook(result, arguments):
        raise RuntimeError("hook exploded")

    registry = ToolRegistry()
    registry.register(
        ToolDefinition(
            name="echo",
            description="Echo the supplied message for testing.",
            parameters=ECHO_SCHEMA,
            function=echo,
            post_dispatch_hook=failing_hook,
        )
    )

    result = registry.dispatch("echo", '{"message": "hello"}')

    assert result["ok"] is False
    assert "Tool execution failed" in result["error"]
    assert "RuntimeError" in result["error"]


def test_duplicate_tool_registration_fails():
    registry = make_registry()

    with pytest.raises(ValueError):
        registry.register(
            ToolDefinition(
                name="echo",
                description="Duplicate.",
                parameters=ECHO_SCHEMA,
                function=echo,
            )
        )


def test_all_expected_tools_are_registered():
    registry = build_tool_registry()

    names = {tool.name for tool in registry.definitions()}

    assert names == {
        "get_ticket",
        "search_tickets",
        "get_customer_history",
        "get_refund_policy",
        "list_tickets",
        "run_python_analysis",
        "send_reply",
        "escalate",
        "create_ticket",
    }


def test_tools_have_descriptions_and_schemas():
    registry = build_tool_registry()

    for tool in registry.definitions():
        assert tool.name
        assert tool.description
        assert isinstance(tool.parameters, dict)
        assert tool.function


def test_tool_schemas_mark_required_arguments():
    registry = build_tool_registry()

    tools = {tool.name: tool for tool in registry.definitions()}

    assert tools["get_ticket"].parameters["required"] == ["ticket_id"]

    assert tools["search_tickets"].parameters["required"] == ["query"]

    assert tools["get_customer_history"].parameters["required"] == ["customer_id"]

    assert tools["get_refund_policy"].parameters["required"] == ["book_category"]

    assert tools["send_reply"].parameters["required"] == [
        "ticket_id",
        "message",
    ]

    assert tools["escalate"].parameters["required"] == [
        "ticket_id",
        "reason",
    ]

    assert tools["create_ticket"].parameters["required"] == [
        "customer_id",
        "category",
        "subject",
        "body",
    ]


def test_category_enum():
    registry = build_tool_registry()

    tool = registry.get("search_tickets")

    assert tool is not None

    assert tool.parameters["properties"]["category"]["enum"] == [
        "billing",
        "technical",
        "account",
        "general",
        "feature_request",
        "other",
    ]


def test_book_category_enum():
    registry = build_tool_registry()

    tool = registry.get("get_refund_policy")

    assert tool is not None

    assert tool.parameters["properties"]["book_category"]["enum"] == [
        "technology",
        "fiction",
        "self-help",
    ]
