"""Tests for the second port pass: run-time provenance (runclock), the
shared reasoning-delimiter guard, catalogued per-model delimiters, and
sampling forwarding through assess/score.

Several of these are STRUCTURAL rather than behavioural -- they assert
that one definition is used in two places rather than that a function
returns the right value. That is deliberate. Both bugs this pass fixed
were two copies of one predicate drifting apart, and a behavioural test
of each copy passes happily while they disagree.
"""

import argparse
import csv
import json
from datetime import datetime

import pytest

from openllm_cbench.core import gate as gate_mod
from openllm_cbench.core.delimiters import (
    MERGE_DELIMITERS, merge_evidence, merge_suspected, parse_catalogued_delimiters,
)
from openllm_cbench.core.registry import delimiters_for, save_entry
from openllm_cbench.core.runclock import (
    RUN_TIME_FIELDS, run_started_now, run_time_row_fields, time_from_filename,
)


# --------------------------------------------------------------- runclock

def test_run_started_now_is_parseable_and_carries_an_offset():
    """A bare local timestamp is only interpretable by the person who
    produced it, which defeats the point of recording it in a file other
    people read."""
    value = run_started_now()
    parsed = datetime.fromisoformat(value)
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() is not None


def test_time_from_filename_reads_the_suites_own_stamp():
    assert time_from_filename("containment_qwen3-8b_20260917_143012.csv") == \
        datetime(2026, 9, 17, 14, 30, 12)
    assert time_from_filename("trial_summary_x_20260901_090000.md") == \
        datetime(2026, 9, 1, 9, 0, 0)


def test_time_from_filename_returns_none_rather_than_guessing():
    """A caller that cannot recover a real time must be able to tell.
    Returning "now", or the epoch, would be a made-up value indistinguishable
    from a measured one at the point of use."""
    assert time_from_filename("scorecard_no_stamp.md") is None
    assert time_from_filename("") is None
    # Digits in the right shape but not a real date.
    assert time_from_filename("x_20261345_996060.csv") is None


def test_run_time_row_fields_matches_the_declared_fieldnames():
    row = run_time_row_fields("2026-09-17T14:30:12+01:00")
    assert set(row) == set(RUN_TIME_FIELDS)


@pytest.mark.parametrize("suite", ["containment", "channel", "persistence"])
def test_every_suite_declares_the_run_time_column(suite):
    """The whole point is that the column is in the DATA. A suite that
    computes the value and forgets to write it is the same as not having
    it."""
    import importlib
    mod = importlib.import_module(f"openllm_cbench.suites.{suite}")
    src = open(mod.__file__, encoding="utf-8").read()
    assert "*RUN_TIME_FIELDS]" in src, f"{suite} does not write the run-time column"
    assert "run_time_row_fields(run_started_at)" in src, \
        f"{suite} declares the column but never stamps it"


# ------------------------------------------------------------- delimiters

def test_catalogued_delimiters_are_reported_under_their_own_label():
    """A catalogued delimiter is something an operator gate-checked and
    wrote down for this model. Attributing it to whichever built-in family
    happens to overlap loses the fact that a human confirmed it."""
    assert merge_evidence("<think>x</think>", "") == "think_tag"
    assert merge_evidence("<think>x</think>", "", ("<think>",)) == "catalogued"


def test_a_real_thinking_trace_is_never_evidence_of_a_merge():
    for _, markers in MERGE_DELIMITERS:
        for m in markers:
            assert merge_evidence(f"answer {m} more", "a real trace") == ""
            assert merge_suspected(f"answer {m} more", "a real trace") is False


def test_parse_catalogued_delimiters_splits_the_documented_pair_form():
    """The catalogue writes a delimiter the way a human reads it. Both
    halves have to be matchable, because a model emits the opening marker
    and then runs out of budget far more often than it emits the exact
    joined string, which is never."""
    got = parse_catalogued_delimiters(
        ["<think>...</think>", "[BEGIN FINAL RESPONSE]...[END FINAL RESPONSE]"])
    assert "<think>" in got and "</think>" in got
    assert "[begin final response]" in got and "[end final response]" in got
    assert "<think>...</think>" not in got


@pytest.mark.parametrize("junk", [None, 42, [1, 2], {"a": 1}, [None]])
def test_malformed_catalogue_delimiters_never_raise(junk):
    """models.json is user-editable and read at run time. A typo in it
    must not stop a run that would otherwise have produced data."""
    assert parse_catalogued_delimiters(junk) == ()


def test_gate_and_channel_suite_share_one_merge_guard():
    """STRUCTURAL. The gate used to carry its own hard-coded `<think>`
    test whose docstring claimed it matched the suite's guard. It did,
    until the suite's grew to four delimiters and the gate's did not, so
    a model the suite would flag on every row could be gate-checked CLEAN
    and catalogued that way first.

    Asserting the two agree on a value is not enough -- that is exactly
    what passed while they disagreed on every other value."""
    from openllm_cbench.suites import channel
    assert channel.channel_merge_evidence is merge_evidence
    assert channel.channel_merge_suspected is merge_suspected
    assert gate_mod.merge_evidence is merge_evidence

    gate_src = open(gate_mod.__file__, encoding="utf-8").read()
    body = gate_src[gate_src.index("def check_channel_at"):]
    body = body[:body.index("\ndef ")]
    assert '"<think>" in content' not in body, "the gate grew a second copy again"


def test_gate_channel_check_accepts_catalogued_markers():
    """A re-gate that ignored what an operator had already written down
    would keep reporting clean on the one model they recorded as needing
    special handling."""
    import inspect
    sig = inspect.signature(gate_mod.check_channel_at)
    assert "extra_markers" in sig.parameters


# --------------------------------------------------------------- registry

def test_delimiters_for_accepts_both_spellings_of_the_field(tmp_path, monkeypatch):
    """`gate --save` writes a flat key; a hand-written entry copied from
    the schema documentation uses the nested one. Supporting only one
    makes the documented format and the generated format disagree."""
    monkeypatch.chdir(tmp_path)
    overlay = tmp_path / "models.json"
    overlay.write_text(json.dumps({"models": {
        "flat:1b": {"delimiters": ["<odd>...</odd>"]},
        "nested:1b": {"reasoning": {"delimiters": ["[MARK]"]}},
        "neither:1b": {"params_b": 1},
    }}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))

    assert "<odd>" in delimiters_for("flat:1b")
    assert "[mark]" in delimiters_for("nested:1b")
    assert delimiters_for("neither:1b") == ()
    assert delimiters_for("not-in-catalogue:1b") == ()


# ------------------------------------------------- sampling forwarding

def _ns(**kw):
    base = dict(temperature=None, top_p=None, top_k=None, seed=None)
    base.update(kw)
    return argparse.Namespace(**base)


def test_sampling_flags_are_forwarded_to_each_trial():
    from openllm_cbench.cli import _sampling_argv
    argv = _sampling_argv(_ns(temperature=0.2, top_p=0.5, top_k=10), 1)
    assert argv == ["--temperature", "0.2", "--top-p", "0.5", "--top-k", "10"]


def test_an_explicit_seed_is_offset_per_trial():
    """THE important one. A single seed handed to every trial makes all N
    trials byte-identical, so `--depth standard` costs three times as long
    and reports a confidence interval computed over three copies of one
    sample. That looks like evidence and is not."""
    from openllm_cbench.cli import _sampling_argv
    seeds = [_sampling_argv(_ns(seed=7), t)[-1] for t in (1, 2, 3)]
    assert seeds == ["7", "8", "9"]
    assert len(set(seeds)) == 3


def test_unset_sampling_forwards_nothing():
    """So the suite's own resolve_sampling() applies the pinned default
    and generates a fresh per-trial seed, which is the existing behaviour
    and the right one."""
    from openllm_cbench.cli import _sampling_argv
    assert _sampling_argv(_ns(), 2) == []


@pytest.mark.parametrize("cmd", ["assess", "score"])
def test_both_entry_points_expose_the_sampling_flags(cmd, capsys, monkeypatch):
    """The gap this closes: the flags existed on the three suites and on
    neither of the two commands most people actually run."""
    from openllm_cbench import cli
    monkeypatch.setattr("sys.argv", ["cbench", cmd, "--help"])
    with pytest.raises(SystemExit):
        cli.main()
    out = capsys.readouterr().out
    for flag in ("--temperature", "--top-p", "--top-k", "--seed"):
        assert flag in out, f"{cmd} does not expose {flag}"
