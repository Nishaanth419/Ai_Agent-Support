"""Trusted, self-contained entry point for the analysis sandbox subprocess.

Invoked by src.sandbox.executor as ``python -I -S runner.py``, with a JSON
payload ``{"code": ..., "data": ..., "max_code_length": ...}`` on stdin, and
a single JSON result line on stdout.

Deliberately imports nothing from this repository -- only the standard
library. That means the validation logic below is a duplicate of
src/sandbox/validator.py rather than an import of it: importing this
repo's own package would put its modules (including the database
connection helpers in src.db.schema) on this process's sys.path/sys.modules,
which is exactly what a sandboxed subprocess should not have available,
even in the event of a bug elsewhere in this file. Keep the two rule sets
in sync if either changes.

Isolation notes:
  - No import statement is allowed in the executed code; the modules it is
    allowed to use are bound directly into its globals below.
  - The executed code's __builtins__ is a small explicit allowlist -- no
    open, exec, eval, compile, getattr/setattr, or __import__.
  - `io`, `sys`, `contextlib`, `builtins`, and `ast` are used by this
    trusted wrapper only; none of them are exposed to the executed code.
  - CPU/memory/time limits and process isolation are applied by the parent
    (src.sandbox.executor) before this script is exec'd, not here.
"""

import ast
import builtins
import collections
import contextlib
import datetime
import io
import itertools
import json
import math
import re
import statistics
import sys

_FORBIDDEN_NAMES = frozenset(
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

_SAFE_BUILTIN_NAMES = (
    "len",
    "range",
    "sum",
    "min",
    "max",
    "sorted",
    "dict",
    "list",
    "tuple",
    "set",
    "frozenset",
    "str",
    "int",
    "float",
    "bool",
    "enumerate",
    "zip",
    "map",
    "filter",
    "print",
    "round",
    "abs",
    "any",
    "all",
    "reversed",
    "isinstance",
    "True",
    "False",
    "None",
)


class _SandboxViolation(ValueError):
    pass


def _validate(code: str, max_length: int) -> None:
    if not isinstance(code, str) or not code.strip():
        raise _SandboxViolation("code must be a non-empty string.")
    if len(code) > max_length:
        raise _SandboxViolation(
            f"code exceeds the maximum length of {max_length} characters."
        )
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise _SandboxViolation(f"code is not valid Python: {exc}") from None
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise _SandboxViolation("import statements are not allowed.")
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            raise _SandboxViolation(f"use of '{node.id}' is not allowed.")
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr.endswith("__")
        ):
            raise _SandboxViolation(
                f"access to dunder attribute '{node.attr}' is not allowed."
            )


def _build_globals(data) -> dict:
    safe_builtins = {name: getattr(builtins, name) for name in _SAFE_BUILTIN_NAMES}
    return {
        "__builtins__": safe_builtins,
        "collections": collections,
        "statistics": statistics,
        "itertools": itertools,
        "math": math,
        "json": json,
        "datetime": datetime,
        "re": re,
        "data": data,
        "result": None,
    }


def main() -> int:
    try:
        payload = json.loads(sys.stdin.read())
    except json.JSONDecodeError:
        print(json.dumps({"ok": False, "error": "Malformed sandbox payload."}))
        return 1

    code = payload.get("code", "")
    data = payload.get("data")
    max_code_length = payload.get("max_code_length", 20_000)

    try:
        _validate(code, max_code_length)
    except _SandboxViolation as exc:
        print(json.dumps({"ok": False, "error": str(exc)}))
        return 1

    sandbox_globals = _build_globals(data)
    stdout_buffer = io.StringIO()
    try:
        with contextlib.redirect_stdout(stdout_buffer):
            exec(compile(code, "<agent_analysis>", "exec"), sandbox_globals)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "stdout": stdout_buffer.getvalue(),
                }
            )
        )
        return 1

    result = sandbox_globals.get("result")
    try:
        json.dumps(result)
    except TypeError:
        print(
            json.dumps(
                {
                    "ok": False,
                    "error": (
                        "result must be JSON-serializable (dict, list, str, "
                        "number, bool, or None)."
                    ),
                    "stdout": stdout_buffer.getvalue(),
                }
            )
        )
        return 1

    print(
        json.dumps(
            {
                "ok": True,
                "result": result,
                "stdout": stdout_buffer.getvalue(),
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
