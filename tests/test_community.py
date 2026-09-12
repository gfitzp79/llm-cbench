"""
Coverage for core/community.py's validate_submission() -- the shape-check
run on a community-results/ PR before a maintainer spends any time scoring
or merging it. No model or network required; everything here is a
fabricated submission folder on disk.
"""

import json

import pytest

from openllm_cbench.core.community import validate_submission


def _write_meta(path, **overrides):
    meta = {"model": "test:1b", "contributor": "alice", "date": "2026-09-12",
             "hardware_summary": "RTX 4080, 16GB VRAM", "endpoint": "ollama 0.5.1"}
    meta.update(overrides)
    (path / "submission.json").write_text(json.dumps(meta), encoding="utf-8")
    return meta


def _touch_csv(path, prefix, tag, name="20260912_000000"):
    d = path
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{prefix}_{tag}_{name}.csv").write_text("model,x\n", encoding="utf-8")


def test_missing_directory_reports_one_problem(tmp_path):
    problems = validate_submission(tmp_path / "nonexistent")
    assert len(problems) == 1
    assert "not a directory" in problems[0]


def test_missing_submission_json_is_a_problem(tmp_path):
    problems = validate_submission(tmp_path)
    assert any("missing submission.json" in p for p in problems)


def test_malformed_submission_json_is_a_problem_not_an_exception(tmp_path):
    (tmp_path / "submission.json").write_text("{not valid json", encoding="utf-8")
    problems = validate_submission(tmp_path)
    assert any("not valid JSON" in p for p in problems)


def test_missing_required_fields_are_each_reported(tmp_path):
    (tmp_path / "submission.json").write_text(json.dumps({"model": "test:1b"}), encoding="utf-8")
    problems = validate_submission(tmp_path)
    assert any("contributor" in p for p in problems)
    assert any("hardware_summary" in p for p in problems)
    assert any("endpoint" in p for p in problems)


def test_no_csvs_anywhere_is_a_problem(tmp_path):
    _write_meta(tmp_path)
    problems = validate_submission(tmp_path)
    assert any("nothing here to score" in p for p in problems)


def test_valid_submission_with_one_suite_has_no_problems(tmp_path):
    _write_meta(tmp_path)
    _touch_csv(tmp_path / "s3_persistence", "persistence", "test-1b")
    assert validate_submission(tmp_path) == []


def test_valid_submission_with_all_three_suites_has_no_problems(tmp_path):
    _write_meta(tmp_path)
    _touch_csv(tmp_path / "s1_containment", "containment", "test-1b")
    _touch_csv(tmp_path / "s2_channel", "channel", "test-1b")
    _touch_csv(tmp_path / "s3_persistence", "persistence", "test-1b")
    assert validate_submission(tmp_path) == []


def test_csv_tagged_for_a_different_model_is_flagged(tmp_path):
    _write_meta(tmp_path, model="test:1b")
    _touch_csv(tmp_path / "s3_persistence", "persistence", "some-other-model-9b")
    problems = validate_submission(tmp_path)
    assert any("don't match model tag" in p for p in problems)


def test_invalid_marked_csv_is_ignored_same_as_aggregate_convention(tmp_path):
    _write_meta(tmp_path)
    d = tmp_path / "s3_persistence"
    d.mkdir()
    (d / "persistence_test-1b_20260912_000000.INVALID.csv").write_text("x\n", encoding="utf-8")
    problems = validate_submission(tmp_path)
    assert any("nothing here to score" in p for p in problems)
