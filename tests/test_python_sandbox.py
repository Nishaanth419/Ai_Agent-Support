"""Tests for the src.sandbox package used by run_python_analysis.

validate_code tests cover the static (AST) pre-check run before a subprocess
is ever spawned. run_sandboxed tests exercise the real subprocess end to
end -- they are slower (each spawns a real Python interpreter) but they are
what actually proves timeout/memory/import isolation, not just the static
checker.
"""

import pytest

from src.sandbox.validator import SandboxViolation, validate_code
from src.sandbox.executor import run_sandboxed

MAX_LEN = 4000


# ---------------------------------------------------------------------
# validate_code
# ---------------------------------------------------------------------


def test_valid_stdlib_only_code_passes():
    validate_code(
        "counts = collections.Counter(x['status'] for x in data)\n"
        "result = dict(counts)\n",
        MAX_LEN,
    )


@pytest.mark.parametrize(
    "snippet",
    [
        "import os\nresult = 1",
        "from os import path\nresult = 1",
        "import collections\nresult = 1",  # even an allowed module: no imports at all
    ],
)
def test_import_statements_are_rejected(snippet):
    with pytest.raises(SandboxViolation, match="import statements are not allowed"):
        validate_code(snippet, MAX_LEN)


@pytest.mark.parametrize(
    "snippet",
    [
        "result = open('/etc/passwd').read()",
        "result = eval('1+1')",
        "result = exec('x=1')",
        "result = compile('1', '<s>', 'eval')",
        "result = __import__('os')",
        "result = globals()",
        "result = locals()",
        "result = vars()",
        "result = getattr(data, 'items')",
    ],
)
def test_forbidden_builtin_calls_are_rejected(snippet):
    with pytest.raises(SandboxViolation, match="not allowed"):
        validate_code(snippet, MAX_LEN)


@pytest.mark.parametrize(
    "snippet",
    [
        "result = (1).__class__",
        "result = data.__class__.__bases__",
        "result = ().__class__.__base__.__subclasses__()",
    ],
)
def test_dunder_attribute_access_is_rejected(snippet):
    with pytest.raises(SandboxViolation, match="dunder attribute"):
        validate_code(snippet, MAX_LEN)


def test_empty_code_is_rejected():
    with pytest.raises(SandboxViolation, match="non-empty"):
        validate_code("   ", MAX_LEN)


def test_oversized_code_is_rejected():
    with pytest.raises(SandboxViolation, match="maximum length"):
        validate_code("result = 1\n" * 10_000, 100)


def test_syntax_error_is_rejected():
    with pytest.raises(SandboxViolation, match="not valid Python"):
        validate_code("result = (", MAX_LEN)


# ---------------------------------------------------------------------
# run_sandboxed (spawns a real subprocess)
# ---------------------------------------------------------------------


def test_simple_group_by_count():
    data = [
        {"status": "open"},
        {"status": "open"},
        {"status": "resolved"},
    ]
    code = (
        "counts = collections.Counter(t['status'] for t in data)\n"
        "result = dict(counts)\n"
    )
    outcome = run_sandboxed(
        code,
        data,
        timeout_s=5,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )

    assert outcome["ok"] is True
    assert outcome["result"] == {"open": 2, "resolved": 1}


def test_stdout_is_captured():
    code = "print('hello from sandbox')\nresult = 1\n"
    outcome = run_sandboxed(
        code,
        None,
        timeout_s=5,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )

    assert outcome["ok"] is True
    assert "hello from sandbox" in outcome["stdout"]


def test_infinite_loop_times_out():
    code = "while True:\n    pass\n"
    outcome = run_sandboxed(
        code,
        None,
        timeout_s=1,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )

    assert outcome["ok"] is False
    assert "timed out" in outcome["error"].lower()


def test_non_serializable_result_is_reported():
    code = "result = {1, 2, 3}\n"  # a set is not JSON-serializable
    outcome = run_sandboxed(
        code,
        None,
        timeout_s=5,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )

    assert outcome["ok"] is False
    assert "JSON-serializable" in outcome["error"]


def test_runtime_exception_in_code_is_reported_not_raised():
    code = "result = 1 / 0\n"
    outcome = run_sandboxed(
        code,
        None,
        timeout_s=5,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )

    assert outcome["ok"] is False
    assert "ZeroDivisionError" in outcome["error"]


def test_data_is_available_and_isolated_per_call():
    outcome_a = run_sandboxed(
        "result = len(data)",
        [1, 2, 3],
        timeout_s=5,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )
    outcome_b = run_sandboxed(
        "result = len(data)",
        [1, 2],
        timeout_s=5,
        memory_mb=256,
        max_output_chars=4000,
        max_code_length=MAX_LEN,
    )

    assert outcome_a["result"] == 3
    assert outcome_b["result"] == 2


def test_output_is_truncated_to_configured_limit():
    code = "print('x' * 1000)\nresult = 1\n"
    outcome = run_sandboxed(
        code,
        None,
        timeout_s=5,
        memory_mb=256,
        max_output_chars=50,
        max_code_length=MAX_LEN,
    )

    assert outcome["ok"] is True
    assert len(outcome["stdout"]) <= 51  # 50 chars + the truncation marker
