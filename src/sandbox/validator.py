"""Static (AST-based) pre-check for agent-authored Python analysis code.

This is the fast, cheap check the parent process runs before ever spawning a
sandboxed subprocess (see src.sandbox.executor). The spawned child re-checks
an equivalent rule set on its own (src/sandbox/runner.py) so the guarantee
still holds even if this module is ever bypassed -- see that file's
docstring for why it duplicates rather than imports this one.

The rule set exists to prevent the classic Python sandbox-escape gadget
chain (e.g. ``().__class__.__base__.__subclasses__()``) and any reach for
file/network/process primitives: no import statements (the sandbox binds
its allowed modules directly, without an import), no dunder attribute
access, and no calls to a fixed list of builtins that would otherwise let
code reach the file system, re-enter the interpreter, or introspect its
way around the restricted globals.
"""

import ast

FORBIDDEN_NAMES = frozenset(
    {
        "eval",
        "exec",
        "compile",
        "open",
        "__import__",
        "globals",
        "locals",
        "vars",
        "input",
        "breakpoint",
        "exit",
        "quit",
        "getattr",
        "setattr",
        "delattr",
        "memoryview",
    }
)


class SandboxViolation(ValueError):
    """Raised when agent-authored analysis code fails static validation."""


def validate_code(code: str, max_length: int) -> None:
    if not isinstance(code, str) or not code.strip():
        raise SandboxViolation("code must be a non-empty string.")
    if len(code) > max_length:
        raise SandboxViolation(
            f"code exceeds the maximum length of {max_length} characters."
        )

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise SandboxViolation(f"code is not valid Python: {exc}") from None

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise SandboxViolation(
                "import statements are not allowed; use the pre-provided "
                "collections, statistics, itertools, math, json, datetime, "
                "and re names directly."
            )
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise SandboxViolation(f"use of '{node.id}' is not allowed.")
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            raise SandboxViolation(
                f"access to dunder attribute '{node.attr}' is not allowed."
            )
