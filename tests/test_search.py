"""
Coverage for cbench search / core/pull.py:check_model_availability().

check_model_availability() itself needs a live endpoint (not tested here,
network-dependent, same as core/discover.py:list_local_models() -- see
that module's test file for the identical precedent) -- live-verified
manually against both a real tag (found, correct size, confirmed via
`ollama list` that nothing was actually downloaded) and a nonexistent one
(not found, correct error) before this was considered done. These tests
cover cli.py's argument handling and help text, which need no network.
"""

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
SRC = REPO_ROOT / "src"


def _run(args, timeout=15):
    return subprocess.run(
        [sys.executable, "-m", "openllm_cbench.cli", *args],
        capture_output=True, text=True, timeout=timeout,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )


def test_search_help_exits_immediately_no_model_needed():
    result = _run(["search", "--help"])
    assert result.returncode == 0
    assert "--model" in result.stdout
    assert "without downloading" in result.stdout


def test_search_requires_a_model_argument():
    result = _run(["search"])
    assert result.returncode == 2  # argparse's own missing-required-arg exit


def test_search_registered_as_a_native_subcommand():
    from openllm_cbench.cli import _NATIVE
    assert "search" in _NATIVE
