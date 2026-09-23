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
            "catalogue-reasoning-model:1b": {
                "thinking": True,
                "config_overrides": {},
                "caveats": [],
            },
            "catalogue-plain-model:1b": {
                "thinking": False,
                "config_overrides": {},
                "caveats": [],
            },
            "catalogue-reasoning-off-model:1b": {
                "thinking": True,
                "config_overrides": {"think": False},
                "caveats": [],
            },
            "catalogue-inline-reasoner:1b": {
                "thinking": True,
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


# --- automatic generation budget (core/budget.py) ------------------------

SUITE_ARGS = {
    "openllm_cbench.suites.containment": ["--task", "fx_lookup", "--boundary", "stated"],
    "openllm_cbench.suites.channel": ["--think", "true"],
    "openllm_cbench.suites.persistence": [],
}


@pytest.mark.parametrize("module", sorted(SUITE_ARGS))
def test_a_reasoning_model_gets_the_reasoning_budget_in_every_suite(fixture_registry, module):
    """At 2,048 reply tokens qwen3:4b wrote no S3 log in four rows of four;
    at 8,192 every row finished. Nothing set that for a reasoning model:
    the catalogue's per-model config_overrides were designed for it and
    left empty, so the fix was a flag the user had to know about."""
    out = _run_dry(module, ["--model", "catalogue-reasoning-model:1b", *SUITE_ARGS[module]],
                   fixture_registry)
    assert '"num_predict": 8192' in out
    assert '"num_ctx": 16384' in out
    assert "automatic, sized for a reasoning model" in out


@pytest.mark.parametrize("module", sorted(SUITE_ARGS))
def test_a_model_that_does_not_reason_keeps_the_suite_default(fixture_registry, module):
    out = _run_dry(module, ["--model", "catalogue-plain-model:1b", *SUITE_ARGS[module]],
                   fixture_registry)
    assert '"num_predict": 2048' in out
    assert "automatic, this suite's default" in out


def test_reasoning_switched_off_needs_no_reasoning_budget(fixture_registry):
    out = _run_dry("openllm_cbench.suites.persistence",
                   ["--model", "catalogue-reasoning-off-model:1b"], fixture_registry)
    assert '"num_predict": 2048' in out


def test_a_flag_still_beats_the_automatic_budget(fixture_registry):
    out = _run_dry("openllm_cbench.suites.persistence",
                   ["--model", "catalogue-reasoning-model:1b", "--num-predict", "3000"],
                   fixture_registry)
    assert '"num_predict": 3000' in out
    assert '"num_ctx": 16384' in out
    assert "context automatic, sized for a reasoning model; reply set on the command line" in out


def test_s2_budgets_for_reasoning_the_catalogue_switched_off_for_tools(fixture_registry):
    """{"think": false} is a tool-calling setting S1 and S3 apply. S2 runs
    with reasoning on regardless, so its budget must be the reasoning one;
    it was the non-reasoning default, 2,048 reply tokens for a reasoning pass."""
    out = _run_dry("openllm_cbench.suites.channel",
                   ["--model", "catalogue-reasoning-off-model:1b"], fixture_registry)
    assert '"num_predict": 8192' in out


def test_s2_budgets_for_a_model_that_reasons_whatever_think_says(fixture_registry):
    """S2 sends --think false to a model that ignores it, and the model
    reasons inline anyway."""
    out = _run_dry("openllm_cbench.suites.channel",
                   ["--model", "catalogue-inline-reasoner:1b"], fixture_registry)
    assert '"num_predict": 8192' in out


def test_s2_with_reasoning_switched_off_by_the_user_keeps_the_default(fixture_registry):
    out = _run_dry("openllm_cbench.suites.channel",
                   ["--model", "catalogue-reasoning-model:1b", "--think", "false"],
                   fixture_registry)
    assert '"num_predict": 2048' in out
