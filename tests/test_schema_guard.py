"""Coverage for the S2 schema-version guard.

Written because documenting the rule in METHODOLOGY_TECHNICAL section 3.4
("test a gate against data that should PASS, not only data that should
fail") exposed that this particular gate had no test in either
direction. A rule written into the methodology that the codebase does
not follow is the drift this project keeps having to correct.

The guard fires when some pooled CSVs carry the `truncation_suspected`
column and some predate it, because a leak rate pooled across the two
mixes a different generation-budget and TRUNCATED-aware scoring version
into one number.
"""

import csv

import pytest

BASE_FIELDS = [
    "model", "prompt_id", "category", "think_label", "content_verdict",
    "thinking_verdict", "combined_verdict", "content_note", "thinking_note",
    "content_full", "thinking_full", "done_reason", "error",
    "merged_channel_suspected",
]


def _write_s2(path, with_truncation_column):
    fields = list(BASE_FIELDS)
    if with_truncation_column:
        fields.append("truncation_suspected")
    rows = []
    for prompt_id in ("p_one", "p_two"):
        row = {k: "" for k in fields}
        row.update(model="m:1b", prompt_id=prompt_id, category="prompt_injection",
                   think_label="off", content_verdict="PASS",
                   thinking_verdict="PASS", combined_verdict="CLEAN",
                   content_full="an answer", thinking_full="",
                   done_reason="stop", merged_channel_suspected="False")
        if with_truncation_column:
            row["truncation_suspected"] = "False"
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def s2_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))
    d = tmp_path / "s2_channel"
    d.mkdir(parents=True)
    import importlib
    from openllm_cbench.scoring import aggregate
    # S2_DIR is resolved at import time.
    importlib.reload(aggregate)
    return d, aggregate


def test_mixed_schema_versions_are_caught(s2_env):
    d, aggregate = s2_env
    _write_s2(d / "channel_m-1b_20260101_000001.csv", with_truncation_column=True)
    _write_s2(d / "channel_m-1b_20260101_000002.csv", with_truncation_column=False)
    md, stats = aggregate.aggregate_s2("m:1b")
    assert stats["schema_mismatch"] is True
    assert "SCHEMA-VERSION MISMATCH" in md


def test_a_uniformly_modern_corpus_stays_quiet(s2_env):
    """THE case that was missing. A gate is trivially satisfiable by
    always returning 'invalid', and a test suite containing only positive
    cases cannot tell that apart from a working gate."""
    d, aggregate = s2_env
    _write_s2(d / "channel_m-1b_20260101_000001.csv", with_truncation_column=True)
    _write_s2(d / "channel_m-1b_20260101_000002.csv", with_truncation_column=True)
    md, stats = aggregate.aggregate_s2("m:1b")
    assert stats["schema_mismatch"] is False
    assert "SCHEMA-VERSION MISMATCH" not in md


def test_a_uniformly_legacy_corpus_stays_quiet(s2_env):
    """Old CSVs are all missing the column in the SAME way, so they remain
    comparable with each other. Invalidating a corpus for being uniformly
    old would be a gate firing on correct data."""
    d, aggregate = s2_env
    _write_s2(d / "channel_m-1b_20260101_000001.csv", with_truncation_column=False)
    _write_s2(d / "channel_m-1b_20260101_000002.csv", with_truncation_column=False)
    md, stats = aggregate.aggregate_s2("m:1b")
    assert stats["schema_mismatch"] is False
    assert "SCHEMA-VERSION MISMATCH" not in md


def test_a_single_file_is_never_a_schema_mismatch(s2_env):
    d, aggregate = s2_env
    _write_s2(d / "channel_m-1b_20260101_000001.csv", with_truncation_column=True)
    _, stats = aggregate.aggregate_s2("m:1b")
    assert stats["schema_mismatch"] is False
