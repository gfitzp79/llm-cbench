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


def test_discover_help_exits_immediately_no_model_needed():
    result = _run(["discover", "--help"])
    assert result.returncode == 0
    assert "--gate-all" in result.stdout
    assert "--limit" in result.stdout


def test_tui_help_exits_immediately_without_launching_the_app():
    """Regression test: found live that `cbench tui --help`
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


def test_tui_without_textual_names_an_install_that_works(monkeypatch, capsys):
    """The hint used to be `pip install "openllm-cbench[tui]"`, which fails
    while the package is not on PyPI, and a bare `pip` can install into a
    different environment from the one cbench runs in. The hint must name
    the package and this interpreter."""
    from openllm_cbench import cli
    # A None entry in sys.modules makes the import raise ImportError.
    monkeypatch.setitem(sys.modules, "openllm_cbench.tui.app", None)
    assert cli._cmd_tui([]) == 1
    err = capsys.readouterr().err
    assert f'"{sys.executable}" -m pip install textual' in err
    assert "openllm-cbench[" not in err


def test_version_prints_the_package_version():
    from openllm_cbench import __version__
    for flag in ("--version", "-V"):
        result = _run([flag])
        assert result.returncode == 0
        assert result.stdout.strip() == f"cbench {__version__}"


def test_a_submission_records_the_version_cbench_version_prints(monkeypatch):
    """The submission template documents cbench_version as what `cbench
    --version` prints. It was read from the installed metadata instead,
    which an editable install freezes at install time: a checkout at
    0.1.2 stamped its submissions 0.1.0."""
    from importlib import metadata
    from openllm_cbench import __version__
    from openllm_cbench.core import community
    monkeypatch.setattr(metadata, "version", lambda name: "0.0.0-stale")
    assert community.detect_cbench_version() == __version__


def test_a_gate_that_never_reached_the_model_refuses_and_saves_nothing(tmp_path):
    """It exited 1, like a check that ran and found caveats, and --save
    catalogued `"tools": false` for a model nobody had reached."""
    import os
    registry = tmp_path / "models.json"
    result = subprocess.run(
        [sys.executable, "-m", "openllm_cbench.cli", "gate", "--model", "nowhere:1b",
         "--endpoint", "http://127.0.0.1:9", "--save", "--registry-file", str(registry)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert result.returncode == 2, result.stderr
    assert "nothing was saved" in result.stderr
    assert not registry.exists()


def test_unknown_subcommand_exits_immediately_with_nonzero():
    result = _run(["not-a-real-subcommand"])
    assert result.returncode == 2
    assert "Unknown subcommand" in result.stderr


def test_quick_score_gives_s3_the_trials_it_needs_and_says_so(monkeypatch, capsys):
    from openllm_cbench import cli
    from openllm_cbench.scoring.scorecard import depth_trials
    seen = {}
    monkeypatch.setattr(cli, "_assess_body",
                        lambda args, suites, info: seen.update(args.trials_by_suite) or 0)
    monkeypatch.setattr(sys, "argv", ["cbench", "score", "--model", "x:1b",
                                      "--depth", "quick", "--skip-preflight"])
    cli.main()
    assert seen == {"s1": 1, "s2": 1, "s3": depth_trials("quick", "s3")}
    assert seen["s3"] > 1
    assert "for S3 so it has enough rows to rate" in capsys.readouterr().out
