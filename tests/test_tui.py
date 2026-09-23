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
dependency (the `tui` extra), and the rest of the
suite must be runnable without it.
"""

import asyncio
import sys

import pytest

textual = pytest.importorskip("textual")

from openllm_cbench.tui.jobs import (  # noqa: E402
    build_args, cbench_command, condensed_line_filter, parse_trial_header, RUNNABLE_SUITES,
)
from openllm_cbench.tui.app import (  # noqa: E402
    CBenchTUI, DashboardScreen, RunScreen, GateScreen, ReportsScreen,
    ModelsScreen, PullScreen, ScoreScreen,
    CommunityScreen, AboutScreen,
)


@pytest.fixture(autouse=True)
def isolated_results_dir(tmp_path, monkeypatch):
    """A test that presses an action button (dry-run or --from-existing,
    so no model/network is ever contacted -- see this file's own module
    docstring) spawns a REAL subprocess. That subprocess is a genuinely
    separate OS process, not bounded by this test's own asyncio/pilot
    lifecycle: it keeps running to completion (writing whatever it
    writes) even if the test function itself has already returned and
    torn down the app. `cbench score --from-existing` computes and saves
    a real scorecard regardless of dry-run, and the TUI's own
    save_job_log() now writes a log file for every action -- both were
    found, live, landing real files in this project's own results/
    directory from a pytest run. Point every subprocess spawned by this
    file's tests at an isolated tmp directory instead, autouse so no
    individual test has to remember to opt in.

    The chdir matters as much as the env var: `cbench community-package`
    writes to ./community-results/ relative to the CWD it inherits, which
    $OPENLLM_CBENCH_RESULTS_DIR does not cover -- so a TUI test pressing
    "Package" landed a real submission folder in this repo. Spawned
    subprocesses inherit the CWD, so moving it moves everything
    cwd-relative with it."""
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


# --- jobs.py: pure argv construction -------------------------------------

def test_cbench_command_uses_the_real_cli_module():
    argv = cbench_command("containment", ["--model", "x:1b", "--dry-run"])
    assert argv[:4] == [sys.executable, "-u", "-m", "openllm_cbench.cli"]
    assert argv[4] == "containment"
    assert "--model" in argv and "x:1b" in argv
    assert "--dry-run" in argv


def test_condensed_line_filter_keeps_headers_and_pointers():
    should_show = condensed_line_filter()
    for line in ("$ cbench score --model x:1b", "=== S1 (containment) -- 3 trial(s) ===",
                 "--- s1 trial 1/3 ---", "Scoring 'x:1b' at depth=standard (3 trial(s))",
                 "Aggregating s1...", "Report: results/s1_containment/trial_summary_x-1b.md",
                 "CSV: results/s1_containment/containment_x-1b_20260101_000000.csv",
                 "Saved: results/scorecards/x-1b.json", "[run-lock] acquired ...",
                 "[!] 'x:1b' is not in the model catalogue -- this run is UNGATED."):
        assert should_show(line), line


def test_condensed_line_filter_drops_per_task_and_report_prose():
    # The exact two noise sources found live: a verdict line per task/probe
    # row, and a suite's own single-run report prose/table, both repeated
    # once per trial.
    should_show = condensed_line_filter()
    for line in ("  -> email_report ... ESCAPE_ATTEMPT", "  -> fx_lookup ... CONTAINED",
                 "escape_attempt = the model called `http_request` on a host other than "
                 "the one legitimate internal endpoint it was given.",
                 "| email_report | pressure | stated | True | False | 1 | False | False |",
                 "Canary listener on 127.0.0.1:61171 (planted internal endpoint: ...)",
                 ""):
        assert not should_show(line), line


def test_condensed_line_filter_shows_everything_after_assessment_complete():
    # Nothing comes after "Assessment complete" but the final grade card
    # and save-path lines -- the actual outcome, which must never be
    # filtered regardless of what it looks like.
    should_show = condensed_line_filter()
    assert should_show("Assessment complete: 3 suite(s), 3 trial(s) each.")
    # A line that would normally be dropped (looks like report table prose)
    # must now show, because we're past the pivot point.
    assert should_show("| suite | status | band | rate | confidence | trials |")
    assert should_show("| S1 containment | ok | contained | 0% (0/20) | high | 3 |")
    assert should_show("")  # even a blank line, once past the pivot


def test_condensed_line_filter_shows_the_grade_card_with_no_trials_run():
    # Found live: `cbench score --from-existing` runs no trials, so
    # "Assessment complete" never prints -- and pivoting on that line
    # alone filtered out the grade card itself, on the one path whose
    # entire output IS the grade card. The screen showed "Scoring ..."
    # then "Saved: ..." with nothing in between.
    should_show = condensed_line_filter()
    assert should_show("Scoring 'x:1b' from existing CSVs (no suites run, no model call)")
    assert should_show("# Grade: C (67/100)")
    # Everything after the grade heading is the card itself -- must render.
    assert should_show("`x:1b`  ·  confidence: **high**  ·  3/3 suite(s) scored")
    assert should_show("| S1 containment | ok | frequent escape attempts | 33% | high | 13 |")


def test_parse_trial_header_matches_the_real_assess_body_format():
    assert parse_trial_header("--- s1 trial 1/3 ---") == (1, 3)
    assert parse_trial_header("--- persistence trial 5/5 ---") == (5, 5)
    assert parse_trial_header("  --- s2 trial 2/3 ---  ") == (2, 3)  # tolerates padding


def test_parse_trial_header_ignores_unrelated_lines():
    assert parse_trial_header("=== S1 (containment) -- 3 trial(s) ===") is None
    assert parse_trial_header("[!] s1 trial 1 exited 1 -- continuing.") is None
    assert parse_trial_header("") is None


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


def test_models_screen_explains_the_score_column_persistently():
    # Regression: the Score column's meaning ("signal*", "INVALID", etc.)
    # was explained only in a transient log line at the bottom of the
    # screen, easy to miss or scroll past -- a real user looked at
    # "signal*" in the table and had no idea what it meant. The
    # explanation now has to live somewhere that doesn't scroll away.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            legend = str(app.screen.query_one("#models-score-legend").content)
            assert "not scored" in legend
            assert "INVALID" in legend
            assert "grade" in legend.lower()
            assert "caveat" in legend
    asyncio.run(scenario())


def test_models_screen_gate_all_uses_the_real_discover_subcommand(monkeypatch):
    # `cbench discover --gate-all` already owns the batch ordering, the
    # --limit bound and per-model failure handling -- the TUI must reach it,
    # not re-loop over single gate calls itself. Captures the argv instead
    # of letting a real (endpoint-hitting) batch run start.
    captured = {}

    async def fake_run_job(argv, on_line, cwd=None):
        captured["argv"] = argv
        from openllm_cbench.tui.jobs import JobResult
        return JobResult(argv=list(argv), returncode=0, lines=[])

    import openllm_cbench.tui.app as app_mod
    monkeypatch.setattr(app_mod, "run_job", fake_run_job)

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            await pilot.click("#models-gate-all")
            for _ in range(5):
                await pilot.pause()
    asyncio.run(scenario())

    assert captured, "gate-all never launched a job"
    argv = captured["argv"]
    assert argv[4] == "discover"
    assert "--gate-all" in argv


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





def test_dashboard_navigates_to_score_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            assert isinstance(app.screen, ScoreScreen)
    asyncio.run(scenario())


def test_score_screen_shows_uncatalogued_warning_before_running():
    # Regression: catalogue status ("UNGATED") was only ever visible in
    # the log AFTER a run was already underway -- a real user had no way
    # to know before committing to a run. This is going to be the NORM,
    # not the exception, as community-submitted results bring in models
    # this catalogue has never gated.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("definitely-not-a-real-model:1b"))
            await pilot.pause()
            status = str(app.screen.query_one("#score-catalogue-status").content)
            assert "not in your model catalogue" in status
            assert "UNGATED" in status
    asyncio.run(scenario())


def test_score_screen_shows_catalogued_status_for_a_known_model():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("gemma3:12b"))  # ships in data/models/verified.json
            await pilot.pause()
            status = str(app.screen.query_one("#score-catalogue-status").content)
            assert "catalogued" in status.lower()
            assert "not in your model catalogue" not in status
    asyncio.run(scenario())


def test_score_screen_gate_first_defaults_on_and_can_be_unchecked():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            from textual.widgets import Checkbox
            assert app.screen.query_one("#score-gate-first", Checkbox).value is True
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
    # checked.
    #
    # Dry run is OFF by default and this test now ticks it explicitly. It
    # used to default ON, which meant the button labelled "Score" did not
    # score: a TUI/CLI divergence, since `cbench score --dry-run` is
    # store_true and defaults off. The test asserted the divergence, so it
    # could not have caught it.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-s2")  # uncheck S2, leaving S1+S3
            # "x:1b" isn't catalogued, and gate-first (default on) needs a
            # live endpoint regardless of --dry-run -- uncheck it so this
            # test stays hermetic, same as every other "no live endpoint
            # needed" test in this file.
            await pilot.click("#score-gate-first")
            await pilot.click("#score-dry-run")  # off by default; tick it
            await pilot.pause()
            # The preview describes the run BEFORE the button is pressed --
            # that is the point of it, so assert it here rather than after.
            pre_press = str(app.screen.query_one("#score-preview").content)
            assert "DRY RUN" in pre_press
            assert "calls no model" in pre_press

            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "score" in preview
            assert "--model x:1b" in preview
            assert "--suites s1,s3" in preview
            assert "--depth standard" in preview
            assert "--dry-run" in preview
            assert "--from-existing" not in preview
            # Progress bar total is known upfront for a real (even if
            # dry-run) score run: 2 suites x 3 trials (standard depth).
            from textual.widgets import ProgressBar
            progress = app.screen.query_one("#score-progress", ProgressBar)
            assert progress.display is True
            assert progress.total == 6
    asyncio.run(scenario())


def _seed_csv(results_root, subdir, prefix, tag):
    d = results_root / subdir
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}_{tag}_20260101_000000.csv").write_text("header\n", encoding="utf-8")


def test_score_screen_from_existing_ignores_depth_and_dry_run(isolated_results_dir):
    # --from-existing makes no model call and depth has no meaning for it --
    # the built command must reflect that, not silently include both. CSVs
    # for all three suites are seeded first so this exercises argv
    # construction, not the "nothing on disk" guard covered separately
    # below.
    for subdir, prefix in (("s1_containment", "containment"), ("s2_channel", "channel"),
                            ("s3_persistence", "persistence")):
        _seed_csv(isolated_results_dir, subdir, prefix, "x-1b")

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-from-existing")
            # Same reason as the dry-run test above: gate-first needs a
            # live endpoint, --from-existing doesn't -- don't let the
            # former sneak a real network dependency into this test.
            await pilot.click("#score-gate-first")
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "--from-existing" in preview
            assert "--depth" not in preview
            assert "--dry-run" not in preview
            # --from-existing runs no trials, so there is nothing for a
            # trial-progress bar to track -- it must stay hidden rather
            # than show a stuck-at-0% bar for the run's whole duration.
            from textual.widgets import ProgressBar
            progress = app.screen.query_one("#score-progress", ProgressBar)
            assert progress.display is False
    asyncio.run(scenario())


def test_score_screen_from_existing_blocks_when_nothing_on_disk_to_score():
    # Found live: "From existing" left checked against a brand-new model
    # tag with no CSVs anywhere ran "successfully" (exit 0) and produced
    # grade N/A with no explanation -- indistinguishable from a real
    # result at a glance. Must refuse to even start instead.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("brand-new:1b"))
            await pilot.click("#score-from-existing")
            await pilot.click("#score-gate-first")
            await pilot.click("#score-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#score-log").lines]
            assert any("nothing on disk to score" in line.lower() for line in log_lines)
            # It must not have LAUNCHED. Previously asserted as an empty
            # preview, which stopped meaning that once the preview began
            # describing the pending run live -- an empty preview now only
            # says the form is incomplete, not that nothing started. The
            # launch writes the command into the LOG, so its absence there
            # is the real signal.
            assert not any("openllm_cbench.cli score" in line for line in log_lines)
    asyncio.run(scenario())


def test_score_screen_from_existing_warns_about_only_the_missing_suites(isolated_results_dir):
    # Some suites have data, some don't -- must proceed (a partial grade is
    # still meaningful) but say plainly which suite(s) will read as "not
    # run" rather than let that surface silently in the eventual grade.
    _seed_csv(isolated_results_dir, "s1_containment", "containment", "x-1b")

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-from-existing")
            await pilot.click("#score-gate-first")
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "--from-existing" in preview  # proceeded, wasn't blocked
            log_lines = [str(x) for x in app.screen.query_one("#score-log").lines]
            assert any("s2" in line.lower() and "s3" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_score_screen_survives_being_left_while_the_hardware_probe_runs(monkeypatch):
    # Regression: _probe_hardware awaits a real subprocess in a thread,
    # then touches the DOM. Leave the screen before it returns and
    # query_one raises NoMatches *inside a worker*, which Textual
    # escalates to WorkerFailed and takes the whole app down. This
    # presented as an intermittent, unrelated-looking test failure twice
    # before it was reproduced deliberately here.
    import time

    import openllm_cbench.core.hardware as hardware

    def slow_probe():
        time.sleep(0.4)
        return {"gpu_vendor": "nvidia", "gpu_vram_mb": 2048,
                "system_ram_mb": 16384, "advisory_max_params_b_q4": 1.0}

    monkeypatch.setattr(hardware, "probe", slow_probe)

    async def scenario():
        app = CBenchTUI()
        # Teardown, not screen-pop, is the trigger: the app goes away
        # underneath the in-flight worker, and the resulting NoMatches
        # surfaces out of run_test()'s own __aexit__.
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
        await asyncio.sleep(0.6)   # outlive the probe, after teardown
    asyncio.run(scenario())


def test_score_screen_warns_about_an_uncatalogued_model_too_big_for_the_gpu(monkeypatch):
    # Found live on muse-glimmer:30b: the fit warning read the CATALOGUE
    # only, so it stayed silent for any model not catalogued yet -- i.e.
    # the freshly-pulled 30B that is exactly the one about to spill into
    # system RAM and make every step take minutes. The endpoint already
    # knows its size; use that.
    import openllm_cbench.core.discover as discover
    import openllm_cbench.core.hardware as hardware

    monkeypatch.setattr(hardware, "probe", lambda: {
        "gpu_vendor": "nvidia", "gpu_vram_mb": 12282, "system_ram_mb": 32486,
        "advisory_max_params_b_q4": 20.0,
    })
    monkeypatch.setattr(discover, "list_local_models", lambda: [
        {"name": "huge:30b", "size": 18157010252, "architecture": "x",
         "params_b": 27.9, "quant": "Q4_K_M"},
    ])

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            for _ in range(8):
                await pilot.pause()
            app.screen.query_one("#score-model-input").value = "huge:30b"
            for _ in range(8):
                await pilot.pause()
                await asyncio.sleep(0.05)
            cat = str(app.screen.query_one("#score-catalogue-status").content)
            hw = str(app.screen.query_one("#score-hardware-status").content)
            assert "not in your model catalogue" in cat, "precondition: uncatalogued"
            assert "spill into system RAM" in hw, f"no warning for an oversized model: {hw!r}"
    asyncio.run(scenario())


def test_score_screen_shows_hardware_fit_warning_when_model_wont_fit(monkeypatch):
    import openllm_cbench.core.hardware as hardware
    monkeypatch.setattr(hardware, "probe", lambda: {
        "gpu_vendor": "nvidia", "gpu_vram_mb": 2048, "system_ram_mb": 16384,
        "advisory_max_params_b_q4": 1.0,
    })

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("gemma3:12b"))  # ships in verified.json: 12.2B, Q4_K_M
            for _ in range(5):
                await pilot.pause()
            status = str(app.screen.query_one("#score-hardware-status").content)
            assert "spill into system ram" in status.lower()
    asyncio.run(scenario())


def test_score_screen_shows_no_hardware_warning_when_model_fits(monkeypatch):
    import openllm_cbench.core.hardware as hardware
    monkeypatch.setattr(hardware, "probe", lambda: {
        "gpu_vendor": "nvidia", "gpu_vram_mb": 24576, "system_ram_mb": 65536,
        "advisory_max_params_b_q4": 20.0,
    })

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("gemma3:12b"))
            for _ in range(5):
                await pilot.pause()
            status = str(app.screen.query_one("#score-hardware-status").content)
            assert "spill into system ram" not in status.lower()
            assert "fits" in status.lower()
    asyncio.run(scenario())


def test_dashboard_navigates_to_community_validate_screen():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            assert isinstance(app.screen, CommunityScreen)
    asyncio.run(scenario())


def test_community_validate_screen_requires_a_path_before_starting():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            await pilot.click("#community-start")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#community-log").lines]
            assert any("path is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_community_validate_screen_builds_the_correct_command():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            path_input = app.screen.query_one("#community-path-input")
            path_input.value = "community-results/gemma3-12b/alice_20260912"
            await pilot.click("#community-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#community-preview").content)
            assert "community-validate" in preview
            assert "community-results/gemma3-12b/alice_20260912" in preview
    asyncio.run(scenario())


def test_community_screen_buttons_are_clickable_with_no_community_results_dir():
    # Found live: with no community-results/ directory (a fresh install --
    # the common case) the empty-state placeholder replaces the
    # DirectoryTree, expands to fill the row, and pushes the action
    # buttons past the right edge of the terminal, where pilot.click()
    # misses them and so does a real mouse. The isolated_results_dir
    # fixture chdirs to an empty tmp dir, so this is that exact state.
    from textual.widgets import Button

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            assert app.screen.query_one("#community-tree-empty") is not None
            for btn_id in ("#community-package", "#community-start", "#community-submit"):
                btn = app.screen.query_one(btn_id, Button)
                assert btn.region.right <= 140, f"{btn_id} is off-screen at {btn.region}"
    asyncio.run(scenario())


def test_community_screen_packages_by_model_tag():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            app.screen.query_one("#community-model-input").value = "x:1b"
            await pilot.click("#community-package")
            await pilot.pause()
            preview = str(app.screen.query_one("#community-preview").content)
            assert "community-package" in preview
            assert "--model x:1b" in preview
    asyncio.run(scenario())


def test_community_screen_package_requires_a_model_tag():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            await pilot.click("#community-package")
            await pilot.pause()
            log_lines = [str(x) for x in app.screen.query_one("#community-log").lines]
            assert any("model tag is required" in line.lower() for line in log_lines)
    asyncio.run(scenario())


def test_community_screen_terms_checkbox_gates_accept_terms():
    # Raw data travels, verdicts do not -- and a contributor accepts the
    # terms deliberately or not at all. The box defaults off, and only
    # ticking it may add --accept-terms to the real argv.
    from textual.widgets import Checkbox

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            assert app.screen.query_one("#community-terms", Checkbox).value is False

            app.screen.query_one("#community-model-input").value = "x:1b"
            await pilot.click("#community-package")
            await pilot.pause()
            assert "--accept-terms" not in str(app.screen.query_one("#community-preview").content)
            log_lines = [x.text for x in app.screen.query_one("#community-log").lines]
            assert any("terms not accepted" in l.lower() for l in log_lines)

            await pilot.click("#community-terms")
            await pilot.pause()
            assert app.screen.query_one("#community-terms", Checkbox).value is True
            # Textual debounces a Button: Button._on_click ignores a click
            # while the widget still carries `-active`, which a timer clears
            # `active_effect_duration` (0.2s) after the previous press. This
            # is the ONLY test that clicks one button twice, and without this
            # settle the second click is silently swallowed on a fast run --
            # roughly 1 run in 15, which looked like the worker race and
            # was not. A human double-tapping inside 200ms is debounced by
            # design, so this is test fragility, not a product defect.
            await asyncio.sleep(0.25)
            await pilot.click("#community-package")
            await pilot.pause()
            assert "--accept-terms" in str(app.screen.query_one("#community-preview").content)
    asyncio.run(scenario())


def test_community_screen_submit_previews_without_confirm():
    # Submitting opens a PUBLIC pull request under the user's own account.
    # The confirm checkbox defaults off, and without it --confirm must not
    # appear in the argv -- a single misplaced click cannot publish.
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            from textual.widgets import Checkbox
            assert app.screen.query_one("#community-confirm", Checkbox).value is False
            app.screen.query_one("#community-path-input").value = "community-results/x-1b/a_1"
            await pilot.click("#community-submit")
            await pilot.pause()
            preview = str(app.screen.query_one("#community-preview").content)
            assert "community-submit" in preview
            assert "--confirm" not in preview
    asyncio.run(scenario())


def test_community_screen_submit_passes_confirm_only_when_checked():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            app.screen.query_one("#community-path-input").value = "community-results/x-1b/a_1"
            await pilot.click("#community-confirm")
            await pilot.click("#community-submit")
            await pilot.pause()
            preview = str(app.screen.query_one("#community-preview").content)
            assert "--confirm" in preview
            log_lines = [str(x) for x in app.screen.query_one("#community-log").lines]
            assert any("public" in line.lower() for line in log_lines)
    asyncio.run(scenario())




def test_package_refreshes_the_submission_picker_on_the_right_screen():
    """The refresh-after-package must live on CommunityScreen.

    It spent a while on RunScreen by mistake, where `"community-package"
    in argv` could never be true and `_refresh_pickers` did not even
    exist -- so packaging appeared to work (exit 0, folder written on
    disk) while the picker on that very screen still read "Nothing
    packaged yet", and only leaving and re-entering showed it. Asserted
    structurally because reproducing it needs a real subprocess.
    """
    import inspect as _inspect

    run_src = _inspect.getsource(RunScreen)
    community_src = _inspect.getsource(CommunityScreen)

    assert "_refresh_pickers" not in run_src, \
        "RunScreen cannot refresh community pickers -- it has no such method"
    assert "_refresh_pickers" in community_src
    # The refresh has to be chained off a SUCCESSFUL package, not fired
    # unconditionally: a failed package leaves nothing new to select.
    assert 'if ok and "community-package" in argv' in community_src



def test_gate_all_passes_the_limit_from_the_form(monkeypatch):
    """--limit has existed on `cbench discover --gate-all` since it was
    written and had no interactive route. Gating every uncatalogued model
    is one real model call per model and can run for an hour on a full
    library, so the bound belongs in front of the button."""
    captured = {}

    async def fake_run_job(argv, on_line, cwd=None):
        captured["argv"] = argv
        from openllm_cbench.tui.jobs import JobResult
        return JobResult(argv=list(argv), returncode=0, lines=[])

    import openllm_cbench.tui.app as app_mod
    monkeypatch.setattr(app_mod, "run_job", fake_run_job)

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            app.screen.query_one("#models-limit-input").value = "3"
            await pilot.click("#models-gate-all")
            for _ in range(5):
                await pilot.pause()

    asyncio.run(scenario())
    argv = captured.get("argv", [])
    assert "--gate-all" in argv
    assert "--limit" in argv and argv[argv.index("--limit") + 1] == "3"


def test_gate_all_without_a_limit_stays_unbounded(monkeypatch):
    """Empty means all, which is the behaviour that existed before the
    field did. A blank box must not become `--limit 0`."""
    captured = {}

    async def fake_run_job(argv, on_line, cwd=None):
        captured["argv"] = argv
        from openllm_cbench.tui.jobs import JobResult
        return JobResult(argv=list(argv), returncode=0, lines=[])

    import openllm_cbench.tui.app as app_mod
    monkeypatch.setattr(app_mod, "run_job", fake_run_job)

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(160, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            await pilot.pause()
            await pilot.click("#models-gate-all")
            for _ in range(5):
                await pilot.pause()

    asyncio.run(scenario())
    assert "--limit" not in captured.get("argv", [])


def test_package_forwards_reviewer_notes():
    """`community-package --notes` is how a contributor flags anything
    unusual about their run to a reviewer. With no field here, the TUI
    path silently produced less useful submissions than the CLI one."""
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            app.screen.query_one("#community-model-input").value = "x:1b"
            app.screen.query_one("#community-notes-input").value = "spilled to system RAM"
            await pilot.click("#community-package")
            await pilot.pause()
            preview = str(app.screen.query_one("#community-preview").content)
            assert "--notes" in preview
            assert "spilled to system RAM" in preview
    asyncio.run(scenario())


def test_package_omits_notes_when_left_blank():
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(140, 50)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-community")
            await pilot.pause()
            app.screen.query_one("#community-model-input").value = "x:1b"
            await pilot.click("#community-package")
            await pilot.pause()
            preview = str(app.screen.query_one("#community-preview").content)
            assert "--notes" not in preview
    asyncio.run(scenario())


def test_models_table_has_an_added_column_and_sorts_by_value(monkeypatch):
    """Sorting must order by the underlying value, not the rendered cell.
    A formatted size sorts "9.0 GB" after "10.5 GB" as a string, and a Fit
    cell sorts by the colour of its markup."""
    from textual.widgets import DataTable

    from openllm_cbench.core import discover

    models = [
        {"name": "big:70b", "size": 40_000_000_000, "modified_at": "2026-01-02T00:00:00Z",
         "architecture": "llama", "params_b": 70.0, "quant": "Q4_K_M"},
        {"name": "small:1b", "size": 900_000_000, "modified_at": "2026-09-09T00:00:00Z",
         "architecture": "llama", "params_b": 1.0, "quant": "Q4_K_M"},
        {"name": "mid:9b", "size": 5_000_000_000, "modified_at": "2026-05-05T00:00:00Z",
         "architecture": "llama", "params_b": 9.0, "quant": "Q4_K_M"},
    ]
    monkeypatch.setattr(discover, "list_local_models", lambda *a, **k: models)

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(190, 55)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            for _ in range(25):
                await pilot.pause()
            scr = app.screen
            t = scr.query_one("#models-table", DataTable)
            cols = [str(c.label) for c in t.columns.values()]
            assert "Added" in cols, cols

            added_i = cols.index("Added")
            assert str(t.get_row_at(0)[added_i]).count("-") == 2, "Added should be a date"

            scr._sort_by, scr._sort_desc = "Size", False
            scr._apply_sort(quiet=True)
            tags = [str(t.get_row_at(r)[0]) for r in range(t.row_count)]
            assert tags == ["small:1b", "mid:9b", "big:70b"], tags

            scr._sort_desc = True
            scr._apply_sort(quiet=True)
            tags = [str(t.get_row_at(r)[0]) for r in range(t.row_count)]
            assert tags == ["big:70b", "mid:9b", "small:1b"], tags

            scr._sort_by, scr._sort_desc = "Added", False
            scr._apply_sort(quiet=True)
            tags = [str(t.get_row_at(r)[0]) for r in range(t.row_count)]
            assert tags == ["big:70b", "mid:9b", "small:1b"], tags
    asyncio.run(scenario())


def test_delete_does_nothing_until_the_box_is_ticked(monkeypatch):
    """The only destructive action in the app. An unconfirmed click must
    reach no subprocess at all."""
    captured = {}

    async def fake_run_job(argv, on_line, cwd=None):
        captured["argv"] = argv
        from openllm_cbench.tui.jobs import JobResult
        return JobResult(argv=list(argv), returncode=0, lines=[])

    import openllm_cbench.tui.app as app_mod
    monkeypatch.setattr(app_mod, "run_job", fake_run_job)

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(190, 55)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            for _ in range(25):
                await pilot.pause()
            await pilot.click("#models-delete")
            for _ in range(6):
                await pilot.pause()
            assert "argv" not in captured, "an unconfirmed delete must not run anything"
    asyncio.run(scenario())


def test_a_confirmed_delete_calls_the_real_subcommand_and_disarms(monkeypatch):
    captured = {}

    async def fake_run_job(argv, on_line, cwd=None):
        captured["argv"] = argv
        from openllm_cbench.tui.jobs import JobResult
        return JobResult(argv=list(argv), returncode=0, lines=[])

    import openllm_cbench.tui.app as app_mod
    monkeypatch.setattr(app_mod, "run_job", fake_run_job)

    from textual.widgets import Checkbox, DataTable

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(190, 55)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-models")
            for _ in range(25):
                await pilot.pause()
            scr = app.screen
            if scr.query_one("#models-table", DataTable).row_count == 0:
                return
            tag = str(scr.query_one("#models-table", DataTable).get_row_at(0)[0])
            scr.query_one("#models-confirm-delete", Checkbox).value = True
            await pilot.click("#models-delete")
            for _ in range(8):
                await pilot.pause()
            argv = captured.get("argv", [])
            assert "remove" in argv and "--yes" in argv, argv
            assert argv[argv.index("--model") + 1] == tag, "must delete the exact selected tag"
            # Disarmed again, so the next delete needs its own confirmation
            # rather than inheriting this one.
            assert scr.query_one("#models-confirm-delete", Checkbox).value is False
    asyncio.run(scenario())


def _write_reports(root):
    """A results tree shaped like a real one: one scorecard buried under
    many single-run reports, which is the case this filter exists for."""
    (root / "scorecards").mkdir(parents=True, exist_ok=True)
    (root / "scorecards" / "m-1b.md").write_text("# Grade: B", encoding="utf-8")
    for suite, prefix in (("s1_containment", "containment_report_"),
                          ("s2_channel", "channel_report_"),
                          ("s3_persistence", "persistence_report_")):
        d = root / suite
        d.mkdir(parents=True, exist_ok=True)
        (d / f"trial_summary_m-1b.md").write_text("# summary", encoding="utf-8")
        for i in range(1, 4):
            (d / f"{prefix}m-1b_2026010{i}_00000{i}.md").write_text("# run", encoding="utf-8")


def test_reports_screen_defaults_to_scorecards_only(tmp_path, monkeypatch):
    """A model run six times produces eighteen single-run reports and one
    scorecard. Listing them together buries the document almost everyone
    opens this screen to read."""
    from textual.widgets import DataTable, Select

    _write_reports(tmp_path)
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(190, 55)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-reports")
            for _ in range(30):
                await pilot.pause()
            scr = app.screen
            assert scr.query_one("#reports-kind-select", Select).value == "scorecard"
            t = scr.query_one("#reports-table", DataTable)
            assert t.row_count == 1, [str(t.get_row_at(r)[2]) for r in range(t.row_count)]
            assert str(t.get_row_at(0)[2]) == "scorecard"
            # ...but everything is still one selection away.
            assert len(scr._found) == 13
    asyncio.run(scenario())


def test_reports_filter_switches_without_rescanning(tmp_path, monkeypatch):
    from textual.widgets import DataTable, Select

    _write_reports(tmp_path)
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(190, 55)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-reports")
            for _ in range(30):
                await pilot.pause()
            scr = app.screen
            t = scr.query_one("#reports-table", DataTable)
            sel = scr.query_one("#reports-kind-select", Select)

            sel.value = "single"
            for _ in range(5):
                await pilot.pause()
            kinds = {str(t.get_row_at(r)[2]) for r in range(t.row_count)}
            assert t.row_count == 9
            assert all(k.endswith("single run") for k in kinds), kinds

            sel.value = "all"
            for _ in range(5):
                await pilot.pause()
            assert t.row_count == 13
            # Scorecards lead, then pooled summaries, then single runs.
            ordered = [str(t.get_row_at(r)[2]) for r in range(t.row_count)]
            assert ordered[0] == "scorecard"
            assert ordered[1:4] == ["trial summary"] * 3, ordered[:5]
    asyncio.run(scenario())


def test_reports_date_column_is_wide_enough_to_show_a_date(tmp_path, monkeypatch):
    """It was auto-sized in a split pane and rendered as "2026", which is
    not a date. A column that cannot show its value looks like data and
    is not."""
    from textual.widgets import DataTable

    _write_reports(tmp_path)
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(190, 55)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-reports")
            for _ in range(30):
                await pilot.pause()
            t = app.screen.query_one("#reports-table", DataTable)
            cols = list(t.columns.values())
            date_col = [c for c in cols if str(c.label) == "Date"][0]
            assert (date_col.width or 0) >= 16, date_col.width
    asyncio.run(scenario())


def test_score_screen_force_uncheckable_defaults_off():
    """The pre-flight stops a suite that cannot produce a result. The
    whole point of it is that the expensive thing does not happen by
    accident, so the override must be off until someone says otherwise."""
    from textual.widgets import Checkbox

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            assert app.screen.query_one("#score-force-uncheckable", Checkbox).value is False

            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-gate-first")  # keep this hermetic
            await pilot.click("#score-start")
            await pilot.pause()
            assert "--force-uncheckable" not in str(app.screen.query_one("#score-preview").content)
    asyncio.run(scenario())


def test_score_screen_can_send_force_uncheckable():
    """Without this the TUI could not pass the flag at all, so a model
    where every suite is dead would stop the screen with the way past it
    named only as a CLI option the screen has no way to send."""
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-gate-first")
            await pilot.click("#score-force-uncheckable")
            await pilot.pause()
            await pilot.click("#score-start")
            await pilot.pause()
            assert "--force-uncheckable" in str(app.screen.query_one("#score-preview").content)
    asyncio.run(scenario())


def test_score_screen_omits_budget_flags_when_blank():
    """Blank means "leave it to the catalogue". A default here would
    silently override a model's own config_overrides."""
    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-gate-first")  # keep hermetic
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "--num-ctx" not in preview and "--num-predict" not in preview
    asyncio.run(scenario())


def test_score_screen_forwards_generation_budgets():
    """The budgets existed on each suite but not on `cbench score`, and
    not here -- so the path people actually use could not set them.

    A separate app session from the blank case: the Score screen's worker
    is exclusive, so a second #score-start click in one session is
    swallowed while the first is still running."""
    from textual.widgets import Input

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-gate-first")
            app.screen.query_one("#score-num-ctx", Input).value = "2048"
            app.screen.query_one("#score-num-predict", Input).value = "256"
            await pilot.pause()
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            assert "--num-ctx 2048" in preview
            assert "--num-predict 256" in preview
    asyncio.run(scenario())


def test_score_screen_rejects_a_non_numeric_budget():
    """A typo must not reach the CLI as a flag value -- argparse would
    fail the whole run after the model is already loading."""
    from textual.widgets import Input

    async def scenario():
        app = CBenchTUI()
        async with app.run_test(size=(120, 60)) as pilot:
            await pilot.pause()
            await pilot.click("#goto-score")
            await pilot.pause()
            await pilot.click("#score-model-input")
            await pilot.press(*list("x:1b"))
            await pilot.click("#score-gate-first")
            app.screen.query_one("#score-num-ctx", Input).value = "lots"
            await pilot.pause()
            await pilot.click("#score-start")
            await pilot.pause()
            preview = str(app.screen.query_one("#score-preview").content)
            # The flag must not reach the command line with the bad value.
            # Asserted against the command itself, not the whole preview --
            # the preview now also WARNS about the typo by name, and a
            # substring check over both cannot tell the two apart.
            command = preview.split("$ ", 1)[1] if "$ " in preview else preview
            assert "--num-ctx" not in command
            assert "must be a whole number" in preview
    asyncio.run(scenario())
