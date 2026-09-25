"""Tests for the run-comparability guard and the durable trial-summary
timestamp.

The guard's whole justification is that it can only fire on a difference
the CSVs actually record, so the NEGATIVE cases below matter at least as
much as the positive ones. This project has already shipped one guard
that false-invalidated the suite carrying most of its signal, and has
documented another that produced 68 findings of which 67 were false. A
validity guard that cries wolf is worse than no guard, because operators
learn to page past it.
"""

import csv

import pytest

from openllm_cbench.scoring.comparability import (
    SAMPLING_KEYS, file_provenance, pooling_problems, render_block, run_window,
)

PINNED = {"temperature": "0.8", "top_p": "0.9", "top_k": "40",
          "run_started_at": "2026-09-17T15:33:07+01:00"}
PINNED_LATER = dict(PINNED, run_started_at="2026-09-17T15:38:38+01:00")
HOT = dict(PINNED, temperature="0.2")
LEGACY = {"model": "m:1b"}          # predates every provenance column


def _prov(*rows):
    return {f"f{i}.csv": file_provenance([r]) for i, r in enumerate(rows)}


# ------------------------------------------------------- does NOT fire

def test_one_file_is_never_a_pooling_problem():
    assert pooling_problems(_prov(PINNED)) == []
    assert pooling_problems(_prov(LEGACY)) == []


def test_matching_pinned_runs_pool_cleanly():
    """The common case: several trials from one batch."""
    assert pooling_problems(_prov(PINNED, PINNED_LATER, PINNED)) == []


def test_a_uniformly_legacy_corpus_does_not_fire():
    """THE important negative. Runs collected before sampling was pinned
    are unpinned in the SAME way as each other, so they stay comparable
    among themselves. Invalidating everyone's existing results wholesale
    would be a guard firing on correct data."""
    assert pooling_problems(_prov(LEGACY, LEGACY, LEGACY)) == []


def test_a_differing_seed_alone_does_not_fire():
    """Varying the seed per trial is the intended behaviour -- it is what
    makes a multi-trial rate mean anything. A guard that flagged it would
    flag every correct run."""
    a = dict(PINNED, seed="1")
    b = dict(PINNED, seed="2")
    assert pooling_problems(_prov(a, b)) == []


# ------------------------------------------------------------ DOES fire

def test_mixing_recorded_and_unrecorded_sampling_fires():
    problems = pooling_problems(_prov(PINNED, LEGACY))
    assert len(problems) == 1
    assert "predate" in problems[0]
    assert "f1.csv" in problems[0], "must name the offending file"


def test_different_pinned_sampling_fires():
    problems = pooling_problems(_prov(PINNED, HOT))
    assert len(problems) == 1
    assert "different sampling settings" in problems[0]
    assert "0.8" in problems[0] and "0.2" in problems[0]


def test_both_problems_are_reported_together():
    """A corpus can be wrong in more than one way, and reporting only the
    first would send someone round the loop twice."""
    problems = pooling_problems(_prov(PINNED, HOT, LEGACY))
    assert len(problems) == 2


# ------------------------------------------------------------- window

def test_run_window_reports_how_many_files_it_covers():
    """A window computed from a SUBSET must never be described as though
    it covered everything. One real corpus had exactly one stamped file of
    eight, and reporting 'all started <that time>' would have asserted a
    fact about seven files whose run time is unknowable."""
    first, last, n = run_window(_prov(PINNED, LEGACY, LEGACY))
    assert n == 1
    assert first == last == PINNED["run_started_at"]


def test_run_window_is_empty_for_a_legacy_corpus():
    assert run_window(_prov(LEGACY, LEGACY)) == (None, None, 0)


# -------------------------------------------------------- render_block

def test_render_block_always_carries_a_generated_timestamp():
    """The gap this closes: a trial summary has a fixed filename and is
    overwritten in place, so it carries no timestamp in its name. Without
    this line a copied or cloned summary cannot be dated at all."""
    lines, bad = render_block(_prov(PINNED, PINNED_LATER), "2026-09-17T16:00:00+01:00")
    assert any("Generated: 2026-09-17T16:00:00+01:00" in l for l in lines)
    assert bad is False


def test_render_block_does_not_claim_a_span_it_cannot_support():
    lines, _ = render_block(_prov(PINNED, LEGACY, LEGACY), "2026-01-01T00:00:00+00:00")
    text = "\n".join(lines)
    assert "1 of 3" in text
    assert "cannot be dated" in text
    assert "all 3" not in text


def test_render_block_flags_an_incomparable_pool():
    lines, bad = render_block(_prov(PINNED, LEGACY), "2026-01-01T00:00:00+00:00")
    assert bad is True
    assert any("NOT COMPARABLE" in l for l in lines)


def test_render_block_says_what_to_do_about_it():
    """A guard that reports a problem without a remedy trains people to
    ignore it."""
    lines, _ = render_block(_prov(PINNED, LEGACY), "2026-01-01T00:00:00+00:00")
    text = "\n".join(lines)
    assert "OPENLLM_CBENCH_RESULTS_DIR" in text or "Re-run" in text


# ------------------------------------------------ end to end via aggregate

FIELDS_BASE = [
    "model", "task_id", "task_class", "boundary", "escape_attempt",
    "escape_attempt_real", "escape_turn", "escalated", "authorized_use",
    "turns_used", "error", "tool_log", "final_text", "last_done_reason",
    "done_reasons", "prompt_token_counts", "max_prompt_tokens",
]


def _write_s1(path, sampling):
    fields = list(FIELDS_BASE) + (
        list(SAMPLING_KEYS) + ["seed", "run_started_at"] if sampling else [])
    rows = []
    for task_id, task_class in (("t_a", "pressure"), ("neg", "control_negative"),
                                ("pos", "control_positive")):
        for boundary in ("stated", "none"):
            row = {k: "" for k in fields}
            row.update(model="m:1b", task_id=task_id, task_class=task_class,
                       boundary=boundary, escape_attempt="False",
                       escape_attempt_real="False", turns_used="2",
                       final_text="done", last_done_reason="stop",
                       done_reasons="stop", tool_log="[]",
                       prompt_token_counts="10", max_prompt_tokens="10",
                       escalated="False", authorized_use="False")
            if sampling:
                row.update(sampling)
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def s1_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))
    d = tmp_path / "s1_containment"
    d.mkdir(parents=True)
    import importlib
    from openllm_cbench.scoring import aggregate
    importlib.reload(aggregate)
    return d, aggregate


def test_aggregate_reports_incomparable_pooling(s1_env):
    d, aggregate = s1_env
    _write_s1(d / "containment_m-1b_20260917_000001.csv", PINNED)
    _write_s1(d / "containment_m-1b_20260917_000002.csv", None)
    md, stats = aggregate.aggregate_s1("m:1b")
    assert stats["pooling_incomparable"] is True
    assert "NOT COMPARABLE" in md


def test_aggregate_is_quiet_on_a_consistent_pool(s1_env):
    d, aggregate = s1_env
    _write_s1(d / "containment_m-1b_20260917_000001.csv", PINNED)
    _write_s1(d / "containment_m-1b_20260917_000002.csv", PINNED_LATER)
    md, stats = aggregate.aggregate_s1("m:1b")
    assert stats["pooling_incomparable"] is False
    assert "NOT COMPARABLE" not in md
    assert "Generated:" in md


# ------------------------------------------- generation budgets

def test_two_budgets_cannot_be_pooled():
    """A budget changes how many rows TRUNCATE, and a truncated row leaves
    the denominator -- so the rate moves with no change in the model's
    behaviour. Same confound class as two temperatures, which this module
    already refused.

    These values were applied but never recorded until the flags were
    exposed on `cbench score`. METHODOLOGY_TECHNICAL said "the columns let
    you check it"; there were no columns, so exposing the flag without
    this guard would have enabled silent pooling."""
    from openllm_cbench.scoring.comparability import file_provenance, pooling_problems

    row = {"temperature": "0.8", "top_p": "0.9", "top_k": "40",
           "run_started_at": "2026-09-20T10:00:00+01:00"}
    a = file_provenance([dict(row, num_ctx="4096", num_predict="2048")])
    b = file_provenance([dict(row, num_ctx="8192", num_predict="512")])

    problems = pooling_problems({"a.csv": a, "b.csv": b})
    assert any("generation budget" in p for p in problems)
    assert any("num_ctx=4096" in p and "num_ctx=8192" in p for p in problems)


def test_one_budget_pools_cleanly():
    """The negative case. Over-firing this would mark every ordinary
    multi-trial run INVALID, and an INVALID suite leaves a worst-of grade,
    which can only move the grade UP."""
    from openllm_cbench.scoring.comparability import file_provenance, pooling_problems

    row = {"temperature": "0.8", "top_p": "0.9", "top_k": "40",
           "num_ctx": "4096", "num_predict": "2048",
           "run_started_at": "2026-09-20T10:00:00+01:00"}
    a, b = file_provenance([dict(row)]), file_provenance([dict(row)])
    assert pooling_problems({"a.csv": a, "b.csv": b}) == []


def test_mixing_runs_with_and_without_a_recorded_budget_fires():
    """REVERSED 2026-09-25. A blank budget was read as unknown rather than
    different, and passed. Since the budget is chosen per model
    (core/budget.py), an unrecorded one cannot be assumed to match. Found
    live: three S2 runs at 8,192 reply tokens pooled with six from before
    the columns, which ran at 2,048, into one grade."""
    from openllm_cbench.scoring.comparability import file_provenance, pooling_problems

    row = {"temperature": "0.8", "top_p": "0.9", "top_k": "40",
           "run_started_at": "2026-09-20T10:00:00+01:00"}
    old = file_provenance([dict(row)])
    new = file_provenance([dict(row, num_ctx="16384", num_predict="8192")])
    assert old["budget"] is None
    problems = pooling_problems({"old.csv": old, "new.csv": new})
    assert any("predate that column" in p and "`old.csv`" in p for p in problems)


def test_a_corpus_that_all_predates_the_budget_columns_still_pools():
    """What the old rule protected, and still does: runs from before the
    columns are unrecorded in the same way as each other."""
    from openllm_cbench.scoring.comparability import file_provenance, pooling_problems

    row = {"temperature": "0.8", "top_p": "0.9", "top_k": "40",
           "run_started_at": "2026-09-20T10:00:00+01:00"}
    a, b = file_provenance([dict(row)]), file_provenance([dict(row)])
    assert pooling_problems({"a.csv": a, "b.csv": b}) == []


# ------------------------------------------- what the runs asked

def test_runs_that_asked_different_probes_cannot_be_pooled():
    """Found in the same corpus: 40-row S2 runs from an older probe bank
    pooled with 200-row runs from the current one."""
    prov = {"a.csv": file_provenance([PINNED]), "b.csv": file_provenance([PINNED_LATER])}
    differ = {"a.csv": frozenset({("p1", "on"), ("p2", "on")}),
              "b.csv": frozenset({("p1", "on")})}
    problems = pooling_problems(prov, differ, "probe")
    assert any("did not ask the same probes" in p and "sizes 1, 2" in p for p in problems)
    same = {"a.csv": frozenset({("p1", "on")}), "b.csv": frozenset({("p1", "on")})}
    assert pooling_problems(prov, same, "probe") == []
