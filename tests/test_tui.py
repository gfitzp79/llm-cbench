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
    ModelsScreen, PullScreen, AssessmentScreen, ScoreScreen, AboutScreen,
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


def test_pull_screen_requires_a_model_before_searching():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            await pilot.click("#goto-pull")
            await pilot.pause()
            await pilot.click("#search-button")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#pull-log").lines]
            assert any("model tag is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_pull_screen_search_button_builds_the_correct_search_command():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            await pilot.click("#goto-pull")
            await pilot.pause()
            await pilot.click("#pull-model-input")
            await pilot.press(*list("ornith-1.5:9b"))
            await pilot.click("#search-button")
            await pilot.pause()
            # .text (not str(Strip), which reprs each Rich segment
            # separately) -- a wrapped value like "ornith-1.5:9b" can
            # land split across segments/lines, so join the real text
            # before checking rather than searching one wrapped line at
            # a time (see doctor-caption's identical fix earlier).
            log_text = " ".join(x.text for x in app.screen.query_one("#pull-log").lines)
            assert "search" in log_text
            assert "ornith-1.5:9b" in log_text.replace("\n", "")
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


def test_dashboard_navigates_to_assessment_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-assess")
            await pilot.pause()
            assert isinstance(app.screen, AssessmentScreen)
    asyncio.run(scenario())


def test_dashboard_navigates_to_about_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-about")
            await pilot.pause()
            assert isinstance(app.screen, AboutScreen)
            await pilot.click("#about-back")
            await pilot.pause()
            assert isinstance(app.screen, DashboardScreen)
    asyncio.run(scenario())


def test_about_screen_mentions_both_ai_tool_families_with_no_vendor_lock_in():
    # This screen exists specifically because the user asked for "encourage
    # using Claude Code/Cowork or Codex" -- guidance-only, no runtime
    # dependency on either. Guard both halves of that decision.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-about")
            await pilot.pause()
            text = str(app.screen.query_one("#about-body Static").content)
            assert "Claude Code" in text or "Claude" in text
            assert "Codex" in text
            assert "CONTRIBUTING.md" in text
    asyncio.run(scenario())


def test_assessment_screen_requires_a_model_before_starting():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-assess")
            await pilot.pause()
            await pilot.click("#assess-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#assess-log").lines]
            assert any("model tag is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_assessment_screen_requires_at_least_one_suite():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-assess")
            await pilot.pause()
            await pilot.click("#assess-model-input")
            await pilot.press(*list("x:1b"))
            for cb_id in ("#assess-s1", "#assess-s2", "#assess-s3"):
                await pilot.click(cb_id)  # uncheck all three (default is checked)
            await pilot.click("#assess-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#assess-log").lines]
            assert any("select at least one suite" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_assessment_screen_rejects_a_non_numeric_trial_count():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-assess")
            await pilot.pause()
            await pilot.click("#assess-model-input")
            await pilot.press(*list("x:1b"))
            trials_input = app.screen.query_one("#assess-trials-input")
            trials_input.value = "not-a-number"
            await pilot.click("#assess-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#assess-log").lines]
            assert any("positive whole number" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_dashboard_navigates_to_score_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            assert isinstance(app.screen, ScoreScreen)
    asyncio.run(scenario())


def test_score_screen_requires_a_model_before_starting():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#score-log").lines]
            assert any("model tag is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_score_screen_requires_at_least_one_suite():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            for cb_id in ("#score-s1", "#score-s2", "#score-s3"):
                await pilot.click(cb_id)  # uncheck all three (default is checked)
            await pilot.click("#score-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#score-log").lines]
            assert any("select at least one suite" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_score_screen_dry_run_builds_the_correct_score_command():
    # Confirms the real `cbench score` argv is built correctly from the form
    # fields -- suites joined, depth passed through, --dry-run only when
    # checked (checked by default here).
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-s2")  # uncheck S2, leaving S1+S3
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "score" in preview
            assert "--model x:1b" in preview
            assert "--suites s1,s3" in preview
            assert "--depth standard" in preview
            assert "--dry-run" in preview
            assert "--from-existing" not in preview
    asyncio.run(scenario())


def test_score_screen_from_existing_ignores_depth_and_dry_run():
    # --from-existing makes no model call and depth has no meaning for it --
    # the built command must reflect that, not silently include both.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-from-existing")
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "--from-existing" in preview
            assert "--depth" not in preview
            assert "--dry-run" not in preview
    asyncio.run(scenario())


def test_assessment_screen_dry_run_builds_the_correct_assess_command():
    # Confirms the real `cbench assess` argv is built correctly from the
    # form fields -- suites joined, trials passed through, --dry-run
    # only when checked (checked by default here).
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-assess")
            await pilot.pause()
            await pilot.click("#assess-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#assess-s2")  # uncheck S2, leaving S1+S3
            await pilot.click("#assess-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#assess-preview").content)
            assert "assess" in preview
            assert "--model x:1b" in preview
            assert "--suites s1,s3" in preview
            assert "--trials 3" in preview
            assert "--dry-run" in preview
    asyncio.run(scenario())
