"""
Integration-level regression coverage for the model-catalogue wiring in
each suite. Unit tests on core/registry.py alone would not catch the
class of bug this guards against: a suite silently ignoring a catalogued
override because of a wiring mistake in main(), not a bug in the
registry module itself. These run the real CLI entry points as
subprocesses with --dry-run (no model, no network) and --registry-file
pointed at a throwaway fixture, and check the resolved payload.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"


def _run_dry(module, extra_args, registry_path):
    env_pythonpath = str(SRC)
    result = subprocess.run(
        [sys.executable, "-m", module, *extra_args, "--dry-run",
         "--registry-file", str(registry_path)],
        capture_output=True, text=True, timeout=30,
        env={**__import__("os").environ, "PYTHONPATH": env_pythonpath},
    )
    return result.stdout + result.stderr


@pytest.fixture
def fixture_registry(tmp_path):
    registry = {
        "models": {
            "catalogue-test-model:1b": {
                "config_overrides": {"num_ctx": 9999, "num_predict": 4321, "timeout": 111, "max_turns": 3, "max_task_turns": 3},
                "caveats": ["fixture caveat marker"],
            },
            "catalogue-effort-model:1b": {
                "thinking_mode": "effort",
                "config_overrides": {},
                "caveats": [],
            },
            "catalogue-ignores-think-model:1b": {
                "thinking_mode": "ignores_think",
                "config_overrides": {},
                "caveats": [],
            },
        }
    }
    path = tmp_path / "models.json"
    path.write_text(json.dumps(registry), encoding="utf-8")
    return path


def test_containment_applies_catalogued_num_ctx_and_num_predict(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.containment",
        ["--model", "catalogue-test-model:1b", "--task", "fx_lookup", "--boundary", "stated"],
        fixture_registry,
    )
    assert '"num_ctx": 9999' in out
    assert '"num_predict": 4321' in out
    assert "fixture caveat marker" in out


def test_containment_explicit_flag_wins_over_catalogue(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.containment",
        ["--model", "catalogue-test-model:1b", "--task", "fx_lookup", "--boundary", "stated",
         "--num-ctx", "555"],
        fixture_registry,
    )
    assert '"num_ctx": 555' in out
    assert '"num_predict": 4321' in out  # untouched override still applies


def test_containment_no_catalogue_flag_ignores_registry(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.containment",
        ["--model", "catalogue-test-model:1b", "--task", "fx_lookup", "--boundary", "stated",
         "--no-catalogue"],
        fixture_registry,
    )
    assert '"num_ctx": 9999' not in out
    assert "fixture caveat marker" not in out


def test_channel_applies_catalogued_num_predict(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.channel",
        ["--model", "catalogue-test-model:1b", "--think", "true"],
        fixture_registry,
    )
    assert '"num_predict": 4321' in out


def test_channel_thinking_mode_effort_auto_selects_effort_sweep(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.channel",
        ["--model", "catalogue-effort-model:1b"],
        fixture_registry,
    )
    assert "think=low" in out or '"think": "low"' in out


def test_channel_thinking_mode_ignores_think_auto_selects_off_only(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.channel",
        ["--model", "catalogue-ignores-think-model:1b"],
        fixture_registry,
    )
    assert "think=off" in out


def test_persistence_applies_catalogued_overrides(fixture_registry):
    out = _run_dry(
        "openllm_cbench.suites.persistence",
        ["--model", "catalogue-test-model:1b"],
        fixture_registry,
    )
    assert '"num_ctx": 9999' in out
    assert '"num_predict": 4321' in out
