"""
`--plan` and the row-based progress estimate built on it (core/plan.py).

Measured on a real quick score: the four trials took 2:39, 11:54, 0:33 and
0:29. Progress counted in trials gave the TUI no ETA for the whole first
trial, then a third of the real time, then a countdown frozen through the
longest trial. Rows are the unit that moves steadily, and the row count has
to come from each suite's own start-up, not a copy of it.
"""

import asyncio
import json
import sys

import pytest

from openllm_cbench import cli
from openllm_cbench.core import plan
from openllm_cbench.core.paths import data_file


def _no_model_calls(monkeypatch):
    """--plan must stop before anything that reaches a model or a socket."""
    import openllm_cbench.suites.channel as s2
    import openllm_cbench.suites.containment as s1
    import openllm_cbench.suites.persistence as s3

    def boom(*a, **k):
        raise AssertionError("--plan went past planning")
    monkeypatch.setattr(s1, "run_task", boom)
    monkeypatch.setattr(s1, "start_canary", boom)
    monkeypatch.setattr(s2, "call_model", boom)
    monkeypatch.setattr(s3, "run_scenario", boom)


def _plan(module, argv, capsys):
    assert cli._dispatch_passthrough(module, ["--model", "x:1b", *argv, "--plan"]) == 0
    return plan.parse_plan(capsys.readouterr().out)


# --- the line formats ---------------------------------------------------------

def test_plan_and_total_lines_round_trip():
    assert plan.parse_plan("banner\n" + plan.plan_line(24) + "\n") == 24
    assert plan.parse_plan("no plan here") is None
    line = plan.total_line([("s1", 24, 1), ("s2", 200, 1), ("s3", 2, 2)], 228)
    assert line == ("Plan: S1 24 rows x 1 trial(s), S2 200 rows x 1 trial(s), "
                    "S3 2 rows x 2 trial(s); 228 rows in total.")
    assert plan.parse_total(line) == 228
    assert plan.parse_total("Planned rows: 24") is None


def test_row_lines_are_every_suite_s_per_row_line_and_nothing_else():
    # The exact shapes the three suites print as a row finishes.
    assert plan.is_row_line("  -> email_report ... CONTAINED")
    assert plan.is_row_line("  -> prompt_injection_1 ... CHANNEL_LEAK")
    assert plan.is_row_line("  -> dedup_customer_records ... no signal (clean)")
    # A row whose verdict landed on a later line still counts once.
    assert plan.is_row_line("  -> fx_lookup ...")
    for other in ("--- fx_lookup / stated -> POST http://localhost:11434/api/chat ---",
                  "=== boundary=stated ===", "--- s1 trial 1/3 ---",
                  "    -> nested detail ...", "-> x ..."):
        assert not plan.is_row_line(other), other


# --- each suite's own --plan --------------------------------------------------

def test_s1_plans_every_task_at_both_boundaries(monkeypatch, capsys, tmp_path):
    _no_model_calls(monkeypatch)
    tasks = json.loads(data_file("tasks", "containment_tasks.json").read_text(encoding="utf-8"))
    mod = "openllm_cbench.suites.containment"
    assert _plan(mod, [], capsys) == 2 * len(tasks)
    assert _plan(mod, ["--boundary", "stated"], capsys) == len(tasks)
    assert not (tmp_path / "results").exists()


def test_s2_plans_after_the_capability_check_drops_variants(monkeypatch, capsys, tmp_path):
    import openllm_cbench.suites.channel as s2
    _no_model_calls(monkeypatch)
    probes = len(s2.load_prompts(None))
    mod = "openllm_cbench.suites.channel"
    monkeypatch.setattr(s2, "supports_thinking", lambda *a, **k: True)
    assert _plan(mod, [], capsys) == 2 * probes
    # No thinking capability: the 'on' rows are never asked, so not planned.
    monkeypatch.setattr(s2, "supports_thinking", lambda *a, **k: False)
    assert _plan(mod, [], capsys) == probes
    monkeypatch.setattr(s2, "supports_thinking", lambda *a, **k: True)
    assert _plan(mod, ["--effort", "all"], capsys) == 3 * probes
    assert not (tmp_path / "results").exists()


def test_s3_plans_one_row_per_scenario(monkeypatch, capsys, tmp_path):
    _no_model_calls(monkeypatch)
    scenarios = json.loads(
        data_file("scenarios", "persistence_scenarios.json").read_text(encoding="utf-8"))
    mod = "openllm_cbench.suites.persistence"
    assert _plan(mod, [], capsys) == len(scenarios)
    assert _plan(mod, ["--scenario", scenarios[0]["id"]], capsys) == 1
    assert not (tmp_path / "results").exists()


# --- the run-level plan -------------------------------------------------------

class _Args:
    model = "x:1b"
    trials = 3
    temperature = top_p = top_k = seed = None


def test_the_run_plan_multiplies_rows_by_each_suite_s_trials(monkeypatch):
    rows = {"m1": 24, "m2": 200, "m3": 2}
    monkeypatch.setattr(cli, "_plan_rows", lambda module, argv: rows[module])
    info = {"s1": (None, "m1"), "s2": (None, "m2"), "s3": (None, "m3")}
    line = cli._plan_line(_Args(), ["s1", "s2", "s3"], info, {"s1": 1, "s2": 1, "s3": 2})
    assert plan.parse_total(line) == 24 + 200 + 2 * 2


def test_no_plan_when_any_suite_cannot_say(monkeypatch):
    # A total missing one suite would make the estimate confidently wrong.
    monkeypatch.setattr(cli, "_plan_rows", lambda module, argv: None if module == "m2" else 5)
    info = {"s1": (None, "m1"), "s2": (None, "m2")}
    assert cli._plan_line(_Args(), ["s1", "s2"], info, {}) is None


def test_a_suite_that_exits_while_planning_gives_no_count(monkeypatch):
    def exits(module, argv):
        raise SystemExit("no think=off variant was requested")
    monkeypatch.setattr(cli, "_dispatch_passthrough", exits)
    assert cli._plan_rows("openllm_cbench.suites.channel", ["--model", "x:1b"]) is None


# --- the live log -------------------------------------------------------------

def test_a_job_s_log_is_on_disk_while_it_runs(tmp_path):
    # A run stopped part-way (a fanless laptop too hot to continue) left no
    # log at all, because the log was written only when the job exited.
    from openllm_cbench.tui.jobs import run_job, save_job_log

    code = "print('first row'); import sys; sys.stdout.flush(); print('second row')"
    argv = [sys.executable, "-c", code, "openllm_cbench.cli", "score"]
    seen_on_disk = []

    def on_line(line):
        if line == "second row":
            [path] = (tmp_path / "results" / "tui-logs").glob("score_*.log")
            seen_on_disk.append(path.read_text(encoding="utf-8"))

    result = asyncio.run(run_job(argv, on_line, live_log=True))
    assert "first row" in seen_on_disk[0]
    path = save_job_log(result)
    assert path == result.log_path
    text = path.read_text(encoding="utf-8")
    assert text.startswith("$ ") and "first row\nsecond row\n[" in text
    assert len(list(path.parent.glob("score_*.log"))) == 1


@pytest.mark.parametrize("live", [False, True])
def test_live_and_end_of_job_logs_have_the_same_shape(tmp_path, live):
    from openllm_cbench.tui.jobs import run_job, save_job_log

    argv = [sys.executable, "-c", "print('a'); print('b')", "openllm_cbench.cli", "gate"]
    result = asyncio.run(run_job(argv, lambda line: None, live_log=live))
    text = save_job_log(result).read_text(encoding="utf-8")
    assert text.split("\n")[2:4] == ["a", "b"]
    assert text.split("\n")[4].startswith("[")
