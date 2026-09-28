import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
# SCRIPT = PROJECT_ROOT / "scripts" / "model_smoke.py"


def test_smoke_test_reports_safe_model_metadata_with_fake_provider():
    environment = os.environ.copy()
    environment["LLM_PROVIDER"] = "fake"
    environment["LLM_MODEL_NAME"] = "test-model"
    environment.pop("GEMINI_API_KEY", None)

    result = subprocess.run(
        [sys.executable, "-m", "scripts.model_smoke"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "Provider: fake" in result.stdout
    assert "Model: test-model" in result.stdout
    assert "Response: This is a deterministic fake response." in result.stdout
    assert "Input tokens: 10" in result.stdout
    assert "Output tokens: 10" in result.stdout
    assert "Total tokens: 20" in result.stdout
    assert "GEMINI_API_KEY" not in result.stdout
