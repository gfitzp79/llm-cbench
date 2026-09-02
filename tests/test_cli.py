"""
Coverage for cli.py's own dispatch logic -- distinct from each suite's own
--help/--dry-run tests, which cover the suite modules themselves once
cli.py has already handed off to them.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"


def _run(args, timeout=15):
    import os
    return subprocess.run(
        [sys.executable, "-m", "openllm_cbench.cli", *args],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )


def test_bare_help_exits_immediately():
    result = _run([])
    assert result.returncode == 0
    assert "cbench" in result.stdout


def test_gate_help_exits_immediately_without_a_model():
    result = _run(["gate", "--help"])
    assert result.returncode == 0
    assert "--model" in result.stdout


def test_tui_help_exits_immediately_without_launching_the_app():
    """Regression test: found live (2026-09-02) that `cbench tui --help`
    hung indefinitely -- _cmd_tui() never inspected its own argv, so
    --help fell straight through to launching the real interactive
    Textual app, which then blocked waiting on a terminal it didn't have.
    A short subprocess timeout here makes that failure mode fail fast
    (an explicit assertion) instead of hanging the test run the same way
    it hung the original CLI invocation."""
    try:
        result = _run(["tui", "--help"], timeout=10)
    except subprocess.TimeoutExpired:
        pytest.fail(
            "`cbench tui --help` did not exit within 10s -- it almost certainly "
            "launched the interactive app instead of printing help. This is "
            "exactly the regression this test exists to catch."
        )
    assert result.returncode == 0
    assert "usage: cbench tui" in result.stdout


def test_unknown_subcommand_exits_immediately_with_nonzero():
    result = _run(["not-a-real-subcommand"])
    assert result.returncode == 2
    assert "Unknown subcommand" in result.stderr
