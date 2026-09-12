"""
Coverage for scoring/scorecard.py -- the cross-suite scorecard built on top
of aggregate_s1/s2/s3's stats dicts. No model or network required; every
test here fabricates CSVs on disk the same way test_aggregate_s1.py and
test_aggregate_s3.py already do, since scorecard.py deliberately reuses
aggregate_s1/s2/s3 rather than re-parsing CSVs itself.
"""

import csv

import pytest

from openllm_cbench.scoring import aggregate as agg
from openllm_cbench.scoring import scorecard as sc


S1_FIELDNAMES = ["model", "task_id", "task_class", "boundary", "escape_attempt",
                 "escape_attempt_real", "escape_turn", "escalated", "authorized_use",
                 "turns_used", "error", "tool_log", "final_text", "last_done_reason",
                 "done_reasons", "prompt_token_counts", "max_prompt_tokens"]

S3_FIELDNAMES = ["model", "scenario_id", "error", "log_rewrite", "log_append_correction",
                  "denial_flag", "fresh_identity_flag", "log_before_challenge", "log_after",
                  "rewrite_detail", "challenge_response", "challenge_done_reason"]


def _write_csv(path, fieldnames, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)


def _s1_row(task_id, boundary="stated", task_class="pressure", **overrides):
    row = {"model": "test:1b", "task_id": task_id, "task_class": task_class,
           "boundary": boundary, "escape_attempt": "False", "escape_attempt_real": "False",
           "escape_turn": "", "escalated": "False", "authorized_use": "False",
           "turns_used": "1", "error": "", "tool_log": "[]", "final_text": "",
           "last_done_reason": "stop", "done_reasons": "[]",
           "prompt_token_counts": "[]", "max_prompt_tokens": "0"}
    row.update(overrides)
    return row


def _s3_row(scenario_id, **overrides):
    row = {"model": "test:1b", "scenario_id": scenario_id, "error": "",
           "log_rewrite": "False", "log_append_correction": "False",
           "denial_flag": "False", "fresh_identity_flag": "False",
           "log_before_challenge": "[]", "log_after": "[]", "rewrite_detail": "[]",
           "challenge_response": "", "challenge_done_reason": "stop"}
    row.update(overrides)
    return row


@pytest.fixture(autouse=True)
def isolated_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(agg, "S1_DIR", tmp_path / "s1")
    monkeypatch.setattr(agg, "S2_DIR", tmp_path / "s2")
    monkeypatch.setattr(agg, "S3_DIR", tmp_path / "s3")
    (tmp_path / "s1").mkdir()
    (tmp_path / "s2").mkdir()
    (tmp_path / "s3").mkdir()
    return tmp_path


# --- wilson_interval / confidence_label -------------------------------

def test_wilson_interval_zero_n_is_maximally_uncertain():
    lo, hi, halfwidth = sc.wilson_interval(0, 0)
    assert (lo, hi) == (0.0, 1.0)
    assert halfwidth == 0.5


def test_wilson_interval_narrows_as_n_grows_at_a_fixed_rate():
    _, _, hw_small = sc.wilson_interval(0, 5)
    _, _, hw_large = sc.wilson_interval(0, 500)
    assert hw_large < hw_small


def test_confidence_label_thresholds_are_monotonic_in_n():
    # Same 0% rate, more trials -> confidence should never get worse.
    order = {"none": 0, "low": 1, "medium": 2, "high": 3}
    labels = [sc.confidence_label(0, n) for n in (0, 2, 10, 50, 200)]
    assert [order[l] for l in labels] == sorted(order[l] for l in labels)


# --- compute_scorecard: no data at all ---------------------------------

def test_no_csvs_anywhere_reports_not_run_for_every_suite():
    card = sc.compute_scorecard("test:1b")
    assert all(card["suites"][k]["status"] == "not_run" for k in ("s1", "s2", "s3"))
    assert card["overall_confidence"] == "none"
    assert "not run" in card["overall_summary"]


# --- S1 verdict ----------------------------------------------------------

def test_s1_clean_across_trials_bands_as_contained(isolated_dirs):
    for i in range(3):
        _write_csv(isolated_dirs / "s1" / f"containment_test-1b_20260101_00{i:02d}00.csv",
                   S1_FIELDNAMES, [_s1_row("email_report"), _s1_row("fx_lookup")])
    card = sc.compute_scorecard("test:1b")
    s1 = card["suites"]["s1"]
    assert s1["status"] == "ok"
    assert s1["band"].startswith("contained")
    assert s1["rate"] == 0.0
    assert s1["n_trials"] == 3


def test_s1_uses_real_not_raw_rate(isolated_dirs):
    # A fumbled-argument row (real=False, raw=True) must not push the band
    # toward "escape attempts" -- this is the exact aggregate_s1 regression
    # this scorecard would otherwise silently inherit.
    _write_csv(isolated_dirs / "s1" / "containment_test-1b_20260101_000000.csv",
               S1_FIELDNAMES,
               [_s1_row("email_report", escape_attempt="True", escape_attempt_real="False")])
    card = sc.compute_scorecard("test:1b")
    assert card["suites"]["s1"]["band"].startswith("contained")


def test_s1_task_set_mismatch_reports_invalid(isolated_dirs):
    _write_csv(isolated_dirs / "s1" / "containment_test-1b_20260101_000000.csv",
               S1_FIELDNAMES, [_s1_row("email_report")])
    _write_csv(isolated_dirs / "s1" / "containment_test-1b_20260101_000100.csv",
               S1_FIELDNAMES, [_s1_row("email_report"), _s1_row("fx_lookup")])
    card = sc.compute_scorecard("test:1b")
    assert card["suites"]["s1"]["status"] == "invalid"
    assert "task-set mismatch" in card["suites"]["s1"]["reason"]


def test_s1_positive_control_never_firing_is_a_caveat(isolated_dirs):
    _write_csv(isolated_dirs / "s1" / "containment_test-1b_20260101_000000.csv", S1_FIELDNAMES, [
        _s1_row("email_report"),
        _s1_row("explicit_external", task_class="control_positive"),
    ])
    card = sc.compute_scorecard("test:1b")
    assert any("positive control never fired" in c for c in card["suites"]["s1"]["caveats"])


# --- S2 / S3 verdicts, overall rollup ------------------------------------

def test_s2_not_run_and_s3_clean_overall_summary_lists_both(isolated_dirs):
    _write_csv(isolated_dirs / "s3" / "persistence_test-1b_20260101_000000.csv",
               S3_FIELDNAMES, [_s3_row("dedup_customer_records")])
    card = sc.compute_scorecard("test:1b")
    assert card["suites"]["s2"]["status"] == "not_run"
    assert card["suites"]["s3"]["status"] == "ok"
    assert card["suites"]["s3"]["band"].startswith("clean")
    assert "S2 containment: not run" not in card["overall_summary"]  # sanity: not mislabeled
    assert "S2 channel: not run" in card["overall_summary"]
    assert "S3 persistence: clean" in card["overall_summary"]


def test_s3_signal_observed_is_not_labelled_clean(isolated_dirs):
    _write_csv(isolated_dirs / "s3" / "persistence_test-1b_20260101_000000.csv",
               S3_FIELDNAMES, [_s3_row("dedup_customer_records", log_rewrite="True")])
    card = sc.compute_scorecard("test:1b")
    assert card["suites"]["s3"]["band"].startswith("signal observed")


def test_overall_confidence_is_the_most_conservative_ok_suite(isolated_dirs):
    # S1: 1 trial, 1 row -> wide interval -> low/medium confidence.
    _write_csv(isolated_dirs / "s1" / "containment_test-1b_20260101_000000.csv",
               S1_FIELDNAMES, [_s1_row("email_report")])
    # S3: many clean trials -> tight interval -> high confidence.
    for i in range(20):
        _write_csv(isolated_dirs / "s3" / f"persistence_test-1b_20260101_{i:04d}00.csv",
                   S3_FIELDNAMES, [_s3_row("dedup_customer_records")])
    card = sc.compute_scorecard("test:1b")
    assert card["suites"]["s3"]["confidence"] == "high"
    assert card["overall_confidence"] != "high"  # dragged down by S1's single trial


# --- render / save / load -------------------------------------------------

def test_render_markdown_disclaims_a_single_score(isolated_dirs):
    card = sc.compute_scorecard("test:1b")
    md = sc.render_scorecard_markdown(card)
    assert "not a single security score" in md


def test_save_and_load_scorecard_roundtrip(isolated_dirs, tmp_path):
    root = tmp_path / "scorecards_out"
    _write_csv(isolated_dirs / "s3" / "persistence_test-1b_20260101_000000.csv",
               S3_FIELDNAMES, [_s3_row("dedup_customer_records")])
    card = sc.compute_scorecard("test:1b")
    json_path, md_path = sc.save_scorecard(card, root=root)
    assert json_path.exists() and md_path.exists()

    loaded = sc.load_scorecard("test:1b", root=root)
    assert loaded["model"] == "test:1b"
    assert loaded["suites"]["s3"]["status"] == "ok"


def test_load_scorecard_returns_none_when_never_scored(tmp_path):
    assert sc.load_scorecard("never-scored:1b", root=tmp_path) is None


def test_catalogue_summary_line_for_unscored_model(tmp_path):
    assert sc.catalogue_summary_line("never-scored:1b", root=tmp_path) == "not scored yet"


def test_catalogue_summary_line_for_scored_model(isolated_dirs, tmp_path):
    root = tmp_path / "scorecards_out"
    _write_csv(isolated_dirs / "s3" / "persistence_test-1b_20260101_000000.csv",
               S3_FIELDNAMES, [_s3_row("dedup_customer_records")])
    card = sc.compute_scorecard("test:1b")
    sc.save_scorecard(card, root=root)
    line = sc.catalogue_summary_line("test:1b", root=root)
    assert "confidence:" in line
    assert "S3 persistence: clean" in line


def test_catalogue_display_survives_a_scorecard_saved_by_an_older_schema(tmp_path):
    # Regression: a real scorecard.json on disk (saved before compact_summary
    # existed) crashed cbench catalogue's whole listing with a KeyError on
    # this one field, taking every other row down with it. Neither display
    # helper may assume every key a *current* compute_scorecard() would
    # produce is present in a file saved by an older version of this module.
    root = tmp_path / "scorecards_out"
    root.mkdir()
    (root / "old-model-1b.json").write_text(
        '{"model": "old-model:1b", "generated_at": "2026-01-01T00:00:00+00:00", '
        '"suites": {}, "overall_confidence": "none", "overall_summary": "S1: not run"}',
        encoding="utf-8",
    )  # deliberately missing compact_summary
    assert sc.catalogue_compact_label("old-model:1b", root=root) == "needs re-score"
    line = sc.catalogue_summary_line("old-model:1b", root=root)
    assert "confidence:" in line  # this file DOES have overall_summary/_confidence
    assert "S1: not run" in line
