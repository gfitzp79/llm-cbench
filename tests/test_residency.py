"""Where the model ran: the share of it in GPU memory, per row.

A model that does not fit in GPU memory still answers, more slowly, and
nothing in the chat reply says so. These tests pin the arithmetic, the
failure paths (which must never break a run), the report lines, and the
path from a CSV column to the scorecard.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openllm_cbench.core import residency as res
from openllm_cbench.core.paths import data_file
from openllm_cbench.scoring import aggregate as agg
from openllm_cbench.scoring import scorecard as sc
from openllm_cbench.suites import channel, containment, persistence

SRC = Path(__file__).parent.parent / "src"


def _ps(*entries):
    return {"models": list(entries)}


def _entry(name, size, vram):
    return {"name": name, "model": name, "size": size, "size_vram": vram}


# --- the URL -------------------------------------------------------------

def test_the_process_list_sits_beside_the_chat_endpoint():
    assert res.ps_url_for("http://localhost:11434/api/chat") == "http://localhost:11434/api/ps"
    assert res.ps_url_for("http://h:1/api/chat/") == "http://h:1/api/ps"


def test_an_endpoint_of_another_shape_has_no_process_list():
    assert res.ps_url_for("http://localhost:8000/v1/chat/completions") is None
    assert res.ps_url_for(None) is None
    assert res.ps_url_for(42) is None


# --- the arithmetic ------------------------------------------------------

def test_a_model_that_fits_reads_exactly_one():
    ps = _ps(_entry("qwen3:4b", 5_100_000_000, 5_100_000_000))
    assert res.fraction_from_ps(ps, "qwen3:4b") == 1.0


def test_a_partial_offload_reads_the_share_of_bytes():
    ps = _ps(_entry("big:20b", 14_000_000_000, 10_000_000_000))
    assert res.fraction_from_ps(ps, "big:20b") == pytest.approx(0.7143, abs=1e-4)


def test_the_right_model_is_read_when_several_are_loaded():
    ps = _ps(_entry("other:1b", 100, 50), _entry("qwen3:4b", 100, 100))
    assert res.fraction_from_ps(ps, "qwen3:4b") == 1.0


def test_a_tag_without_a_version_matches_latest_and_the_reverse():
    assert res.fraction_from_ps(_ps(_entry("user/model:latest", 10, 5)), "user/model") == 0.5
    assert res.fraction_from_ps(_ps(_entry("user/model", 10, 5)), "user/model:latest") == 0.5


def test_a_registry_path_with_a_quantisation_tag_matches_exactly():
    tag = "hf.co/org/Model-GGUF:Q4_K_M"
    assert res.fraction_from_ps(_ps(_entry(tag, 10, 10)), tag) == 1.0


@pytest.mark.parametrize("size,vram", [(0, 0), (None, 5), ("10", "5"), (True, True), (10, -1)])
def test_unusable_sizes_measure_nothing(size, vram):
    assert res.fraction_from_ps(_ps(_entry("m:1", size, vram)), "m:1") is None


def test_more_in_gpu_memory_than_the_total_is_capped_at_one():
    assert res.fraction_from_ps(_ps(_entry("m:1", 10, 11)), "m:1") == 1.0


@pytest.mark.parametrize("ps", [_ps(_entry("other:1b", 10, 10)), {}, None,
                                {"models": "garbage"}, {"models": [None, 3]}])
def test_a_model_that_is_not_listed_is_not_measured(ps):
    assert res.fraction_from_ps(ps, "m:1") is None


# --- reading never breaks a run -------------------------------------------

def test_a_failed_read_returns_nothing_rather_than_raising(monkeypatch):
    def refused(*args, **kwargs):
        raise ConnectionError("refused")
    monkeypatch.setattr(res.requests, "get", refused)
    assert res.read_fraction("m:1", "http://localhost:11434/api/chat") is None


def test_a_read_asks_the_process_list_beside_the_chat_endpoint(monkeypatch):
    seen = {}

    class Reply:
        def raise_for_status(self):
            pass

        def json(self):
            return _ps(_entry("m:1", 4, 3))

    def fake_get(url, timeout):
        seen["url"] = url
        return Reply()

    monkeypatch.setattr(res.requests, "get", fake_get)
    assert res.read_fraction("m:1", "http://h:9/api/chat") == 0.75
    assert seen["url"] == "http://h:9/api/ps"


def test_an_endpoint_of_another_shape_is_never_called(monkeypatch):
    monkeypatch.setattr(res.requests, "get", lambda *a, **k: pytest.fail("called"))
    assert res.read_fraction("m:1", "http://h/v1/chat/completions") is None


# --- per-row reduction and the CSV cell -------------------------------------

def test_a_row_keeps_its_lowest_reading():
    assert res.lowest([1.0, 0.71, None, 1.0]) == 0.71
    assert res.lowest([None, None]) is None
    assert res.lowest([]) is None


def test_the_cell_is_four_decimal_places_or_blank():
    assert res.cell(0.71429) == "0.7143"
    assert res.cell(1.0) == "1.0000"
    assert res.cell(None) == ""


@pytest.mark.parametrize("value,expected", [("0.7143", 0.7143), ("1.0000", 1.0), (0.5, 0.5),
                                            ("", None), (None, None), ("n/a", None),
                                            ("1.5", None), (True, None)])
def test_cells_read_back_from_a_csv(value, expected):
    got = res.as_fraction(value)
    assert got == (pytest.approx(expected) if expected is not None else None)


# --- the tally --------------------------------------------------------------

def test_the_summary_prints_when_everything_fitted():
    t = res.tally([{"gpu_resident_fraction": "1.0000"}] * 3)
    assert not t.fired
    assert t.caveat() == ""
    assert "whole model was in GPU memory on all 3" in t.summary()


def test_one_partial_row_fires_the_caveat():
    t = res.tally([{"gpu_resident_fraction": "1.0000"}, {"gpu_resident_fraction": "0.7100"}])
    assert t.fired and t.below_full == 1 and t.lowest == pytest.approx(0.71)
    assert "outside GPU memory on 1 of 2" in t.caveat()
    assert "71%" in t.summary()


def test_blank_means_not_measured_never_fully_resident():
    t = res.tally([{"gpu_resident_fraction": ""}, {}])
    assert not t.fired
    assert (t.measured, t.not_measured) == (0, 2)
    assert "not measured" in t.summary()


def test_a_rounding_artefact_is_not_a_spill():
    assert not res.tally([{"gpu_resident_fraction": "0.9995"}]).fired


# --- per row in the suites that make several calls --------------------------

class _Chat:
    def raise_for_status(self):
        pass

    def json(self):
        return {"message": {"role": "assistant", "content": "done"}, "done_reason": "stop",
                "prompt_eval_count": 10, "eval_count": 5}


def test_s1_keeps_the_lowest_reading_across_a_rows_calls(monkeypatch):
    monkeypatch.setattr(containment.requests, "post", lambda *a, **k: _Chat())
    monkeypatch.setattr(containment, "read_fraction", lambda model, endpoint: 0.5)
    result = containment.run_task("m:1", {"id": "t", "prompt": "hi"}, False, {}, 1,
                                  8192, 2048, 30, 6, endpoint="http://h:9/api/chat")
    assert result["gpu_resident_fraction"] == "0.5000"


def test_s3_keeps_the_lowest_reading_across_a_scenarios_calls(monkeypatch):
    readings = iter([1.0, 0.8, 1.0])
    monkeypatch.setattr(persistence.requests, "post", lambda *a, **k: _Chat())
    monkeypatch.setattr(persistence, "read_fraction", lambda model, endpoint: next(readings))
    result = persistence.run_scenario(
        "m:1", {"id": "s", "task_prompt": "do it", "challenge_prompt": "why?"},
        8192, 2048, 30, 4, endpoint="http://h:9/api/chat")
    assert result["gpu_resident_fraction"] == "0.8000"


def test_a_row_where_nothing_could_be_read_is_blank(monkeypatch):
    monkeypatch.setattr(containment.requests, "post", lambda *a, **k: _Chat())
    monkeypatch.setattr(containment, "read_fraction", lambda model, endpoint: None)
    result = containment.run_task("m:1", {"id": "t", "prompt": "hi"}, False, {}, 1,
                                  8192, 2048, 30, 6, endpoint="http://h:9/api/chat")
    assert result["gpu_resident_fraction"] == ""


# --- through the aggregate to the scorecard ---------------------------------

S1_FIELDS = ["model", "task_id", "task_class", "boundary", "escape_attempt",
             "escape_attempt_real", "escape_turn", "escalated", "authorized_use",
             "turns_used", "error", "tool_log", "final_text", "last_done_reason",
             "done_reasons", "prompt_token_counts", "max_prompt_tokens",
             "gpu_resident_fraction"]


def _s1_row(**overrides):
    row = {"model": "test:1b", "task_id": "email_report", "task_class": "pressure",
           "boundary": "stated", "escape_attempt": "False", "escape_attempt_real": "False",
           "escape_turn": "", "escalated": "False", "authorized_use": "False",
           "turns_used": "1", "error": "", "tool_log": "[]", "final_text": "done",
           "last_done_reason": "stop", "done_reasons": '["stop"]',
           "prompt_token_counts": "[]", "max_prompt_tokens": "0",
           "gpu_resident_fraction": "1.0000"}
    row.update(overrides)
    return row


def _write(path, rows, fields=S1_FIELDS):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    for key in ("S1_DIR", "S2_DIR", "S3_DIR"):
        folder = tmp_path / key
        folder.mkdir()
        monkeypatch.setattr(agg, key, folder)
    return tmp_path


def _three_trials(dirs, rows_per_trial):
    for i, rows in enumerate(rows_per_trial):
        _write(dirs / "S1_DIR" / f"containment_test-1b_20260101_00{i:02d}00.csv", rows)


def test_the_aggregate_carries_residency_into_its_stats_and_report(dirs):
    _three_trials(dirs, [[_s1_row()], [_s1_row(gpu_resident_fraction="0.7100")], [_s1_row()]])
    report, stats = agg.aggregate_s1("test:1b")
    r = stats["residency"]
    assert (r["rows_measured"], r["rows_below_full"], r["rows_not_measured"]) == (3, 1, 0)
    assert r["lowest"] == pytest.approx(0.71)
    assert "outside GPU memory on 1 of 3" in report


def test_runs_from_before_the_column_still_pool_and_read_as_not_measured(dirs):
    old_fields = [f for f in S1_FIELDS if f != "gpu_resident_fraction"]
    _write(dirs / "S1_DIR" / "containment_test-1b_20260101_000000.csv", [_s1_row()], old_fields)
    _write(dirs / "S1_DIR" / "containment_test-1b_20260101_000100.csv", [_s1_row()])
    report, stats = agg.aggregate_s1("test:1b")
    assert not stats["pooling_incomparable"]
    assert (stats["residency"]["rows_measured"], stats["residency"]["rows_not_measured"]) == (1, 1)


def test_a_scorecard_warns_and_marks_the_grade_when_the_model_spilled(dirs):
    _three_trials(dirs, [[_s1_row()], [_s1_row(gpu_resident_fraction="0.6000")], [_s1_row()]])
    card = sc.compute_scorecard("test:1b")
    s1 = card["suites"]["s1"]
    assert s1["status"] == "ok"
    assert any("outside GPU memory" in c for c in s1["caveats"])
    assert "*" in card["compact_summary"]
    assert card["gpu_residency"]["rows_below_full"] == 1
    md = sc.render_scorecard_markdown(card)
    assert "part of the model was outside GPU memory on 1 of 3 measured rows (lowest 60%)" in md


def test_a_scorecard_says_so_when_the_model_fitted(dirs):
    _three_trials(dirs, [[_s1_row()], [_s1_row()], [_s1_row()]])
    card = sc.compute_scorecard("test:1b")
    assert not card["suites"]["s1"]["caveats"]
    assert "*" not in card["compact_summary"]
    assert "whole model was in GPU memory on all 3 measured rows" in sc.render_scorecard_markdown(card)


def test_a_card_saved_before_residency_existed_renders_without_the_line(dirs):
    _three_trials(dirs, [[_s1_row()], [_s1_row()], [_s1_row()]])
    card = sc.compute_scorecard("test:1b")
    del card["gpu_residency"]
    assert "GPU memory" not in sc.render_scorecard_markdown(card)


S3_FIELDS = ["model", "scenario_id", "error", "log_rewrite", "log_append_correction",
             "denial_flag", "fresh_identity_flag", "log_before_challenge", "log_after",
             "rewrite_detail", "challenge_response", "challenge_done_reason",
             "gpu_resident_fraction"]


def test_a_refused_suite_still_reports_where_the_model_ran(dirs):
    """Found live: gpt-oss:20b ran S3 at 71% in GPU memory, S3 was refused
    for having two usable rows, and the card said residency "was not
    recorded". A model that did not fit is a common reason a suite runs
    short of usable rows, so hiding residency there hides the cause."""
    logged = '[{"step": 2, "action": "deleted record 7"}]'
    rows = [{"model": "test:1b", "scenario_id": s, "error": "", "log_rewrite": "False",
             "log_append_correction": "False", "denial_flag": "False",
             "fresh_identity_flag": "False", "log_before_challenge": logged,
             "log_after": logged, "rewrite_detail": "[]", "challenge_response": "",
             "challenge_done_reason": "stop", "gpu_resident_fraction": "0.7142"}
            for s in ("dedup_customer_records", "inventory_adjustment")]
    _write(dirs / "S3_DIR" / "persistence_test-1b_20260101_000000.csv", rows, S3_FIELDS)
    card = sc.compute_scorecard("test:1b")
    assert card["suites"]["s3"]["status"] == "invalid"
    assert card["gpu_residency"]["rows_below_full"] == 2
    md = sc.render_scorecard_markdown(card)
    assert "outside GPU memory on 2 of 2 measured rows (lowest 71%)" in md
    assert "was not recorded" not in md


def test_the_grade_formula_is_unchanged_by_residency(dirs):
    """Recorded, not scored: a spilled row is a caveat on the grade, never a
    change to it, which is why the scoring version did not move."""
    _three_trials(dirs, [[_s1_row()], [_s1_row()], [_s1_row()]])
    fitted = sc.compute_scorecard("test:1b")["score"]
    for p in (dirs / "S1_DIR").glob("*.csv"):
        p.unlink()
    _three_trials(dirs, [[_s1_row(gpu_resident_fraction="0.4000")]] * 3)
    assert sc.compute_scorecard("test:1b")["score"] == fitted


# --- every suite writes the column ------------------------------------------

def _header(csv_dir, prefix):
    # Each suite writes into its own folder under --results-dir.
    [path] = list(Path(csv_dir).rglob(f"{prefix}_*.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.reader(f))


def _run(module, *args, results_dir):
    return subprocess.run(
        [sys.executable, "-m", f"openllm_cbench.suites.{module}", "--model", "dead:1b",
         "--endpoint", "http://127.0.0.1:9", "--results-dir", str(results_dir),
         "--no-catalogue", *args],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )


def test_s1_writes_the_column(tmp_path):
    _run("containment", "--task", "email_report", "--boundary", "stated", results_dir=tmp_path)
    assert "gpu_resident_fraction" in _header(tmp_path, "containment")


def test_s2_writes_the_column(tmp_path):
    bank = json.loads(data_file("probes", "eval_prompts.json").read_text(encoding="utf-8"))
    one = [p for p in bank if p.get("category") in channel.PROBE_CATS][:1]
    probes = tmp_path / "one_probe.json"
    probes.write_text(json.dumps(one), encoding="utf-8")
    _run("channel", "--prompts-file", str(probes), "--think", "false", results_dir=tmp_path)
    assert "gpu_resident_fraction" in _header(tmp_path, "channel")


def test_s3_writes_the_column(tmp_path):
    _run("persistence", "--scenario", "dedup_customer_records", results_dir=tmp_path)
    assert "gpu_resident_fraction" in _header(tmp_path, "persistence")
