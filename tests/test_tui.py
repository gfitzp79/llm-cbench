"""
Coverage for the TUI. Two layers:

  - jobs.py: pure functions (argv construction), tested directly, no
    Textual/event loop involved.
  - app.py: screen composition and navigation, driven headlessly via
    Textual's own App.run_test() pilot -- no model, no network, and
    crucially no REAL subprocess is ever launched by these tests (the
    Run/Gate buttons are never pressed here; that would need a live
    endpoint the same way a suite's own --dry-run tests don't need one
    either).

Skipped entirely if `textual` isn't installed -- it's an optional
dependency (`pip install "openllm-cbench[tui]"`), and the rest of the
suite must be runnable without it.
"""

import asyncio
import sys

import pytest

textual = pytest.importorskip("textual")

from openllm_cbench.tui.jobs import build_args, cbench_command, RUNNABLE_SUITES  # noqa: E402
from openllm_cbench.tui.app import (  # noqa: E402
    CBenchTUI, DashboardScreen, RunScreen, GateScreen, ReportsScreen,
    ModelsScreen, PullScreen,
)


# --- jobs.py: pure argv construction -------------------------------------

def test_cbench_command_uses_the_real_cli_module():
    argv = cbench_command("containment", ["--model", "x:1b", "--dry-run"])
    assert argv[:4] == [sys.executable, "-u", "-m", "openllm_cbench.cli"]
    assert argv[4] == "containment"
    assert "--model" in argv and "x:1b" in argv
    assert "--dry-run" in argv


def test_build_args_model_and_dry_run():
    args = build_args("gemma3:12b", True, "")
    assert args == ["--model", "gemma3:12b", "--dry-run"]


def test_build_args_no_dry_run_omits_flag():
    args = build_args("gemma3:12b", False, "")
    assert "--dry-run" not in args


def test_build_args_extra_flags_are_shlex_split():
    args = build_args("m:1b", True, "--boundary both --sandbox extended")
    assert args == ["--model", "m:1b", "--dry-run", "--boundary", "both", "--sandbox", "extended"]


def test_build_args_no_model_omits_model_flag():
    args = build_args("", False, "")
    assert args == []


def test_build_args_raises_on_unbalanced_quotes():
    with pytest.raises(ValueError):
        build_args("m:1b", False, "--task 'unterminated")


def test_runnable_suites_map_to_real_passthrough_subcommands():
    # Every suite the TUI offers a form for must be a real cbench
    # subcommand -- catches the TUI form drifting out of sync with cli.py.
    from openllm_cbench.cli import _PASSTHROUGH
    for _label, subcommand in RUNNABLE_SUITES:
        assert subcommand in _PASSTHROUGH


# --- app.py: headless screen composition ---------------------------------

def test_dashboard_composes_and_navigates_to_run_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)
            await pilot.click("#goto-run")
            await pilot.pause()
            assert isinstance(app.screen, RunScreen)
    asyncio.run(scenario())


def test_dashboard_navigates_to_gate_and_reports_screens():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-gate")
            await pilot.pause()
            assert isinstance(app.screen, GateScreen)
            app.pop_screen()
            await pilot.pause()
            await pilot.click("#goto-reports")
            await pilot.pause()
            assert isinstance(app.screen, ReportsScreen)
    asyncio.run(scenario())


def test_run_screen_requires_a_model_before_running():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-run")
            await pilot.pause()
            # No model typed -- pressing Run must not attempt to launch a
            # subprocess (checked by asserting the log shows the local
            # validation message, not a "$ ..." command line).
            await pilot.click("#run-button")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#run-log").lines]
            assert any("model tag is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_invariant_bar_present_on_every_pushed_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            text = app.screen.query_one("#invariant-text").content
            assert "ATTEMPT" in str(text)
            await pilot.click("#goto-run")
            await pilot.pause()
            text = app.screen.query_one("#invariant-text").content
            assert "ATTEMPT" in str(text)
    asyncio.run(scenario())


def test_dashboard_doctor_caption_explains_what_the_button_does():
    # Regression: a user reported "Refresh doctor" was unclear. The
    # button is now labelled "Check environment" and a permanent caption
    # (not just log-widget text a user might miss) explains it.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            caption = str(app.screen.query_one("#doctor-caption").content)
            assert "cbench doctor" in caption
            assert "endpoint" in caption.lower()
    asyncio.run(scenario())


def test_dashboard_navigates_to_models_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            assert isinstance(app.screen, ModelsScreen)
    asyncio.run(scenario())


def test_models_screen_navigates_to_pull_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            await pilot.click("#goto-pull")
            await pilot.pause()
            assert isinstance(app.screen, PullScreen)
    asyncio.run(scenario())


def test_pull_screen_requires_a_model_before_pulling():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            await pilot.click("#goto-pull")
            await pilot.pause()
            await pilot.click("#pull-button")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#pull-log").lines]
            assert any("model tag is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_run_screen_has_a_model_select_alongside_the_free_text_input():
    # Both must exist: the picker for convenience, the Input as the
    # actual source of truth (still works for a tag not pulled yet).
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-run")
            await pilot.pause()
            from textual.widgets import Select, Input
            assert app.screen.query_one("#model-select", Select) is not None
            assert app.screen.query_one("#model-input", Input) is not None
    asyncio.run(scenario())


def test_gate_screen_has_a_model_select_alongside_the_free_text_input():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-gate")
            await pilot.pause()
            from textual.widgets import Select, Input
            assert app.screen.query_one("#gate-model-select", Select) is not None
            assert app.screen.query_one("#gate-model-input", Input) is not None
    asyncio.run(scenario())
