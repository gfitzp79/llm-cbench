"""Tests for the context-window headroom detector.

The detector exists because `num_ctx` overflow is silent: the server drops
tokens off the front of the conversation and answers anyway.

THE MEASUREMENT THESE TESTS ENCODE
==================================
The first version of this detector warned when the token count approached
`num_ctx`, and was wrong, because `prompt_eval_count` reports the count
AFTER truncation. One 128-token prompt, shrinking windows, qwen3:0.6b:

    num_ctx   prompt_eval   eval   truncated?
      4096        128         32      no
       512        128         32      no
       256        128         32      no
       128         66         32     YES
        96         50         32     YES
        64         34         32     YES

Truncation pulls the count DOWN to about half the window, so an evicted
run looks comfortable. `test_clamped_count_does_not_read_as_healthy`
below is that exact case, and it is the test that would have caught the
first version.
"""
import csv
import json
from pathlib import Path

import pytest

from openllm_cbench.core.context_window import (
    AT_RISK, AT_RISK_FRACTION, CONTEXT_FIELDS, EVICTED, OK, UNKNOWN,
    HeadroomTally, call_occupancy, headroom_caveat, headroom_summary,
    headroom_verdict, peak, prompt_tokens, row_verdict, series_is_cumulative,
)

S1_DIR = Path(__file__).resolve().parents[1] / "results" / "s1_containment"


# --- reading the two counts off a response -----------------------------

def test_prompt_tokens_reads_the_evaluated_prompt():
    assert prompt_tokens({"prompt_eval_count": 128, "eval_count": 32}) == 128
    assert prompt_tokens({}) is None


def test_occupancy_is_prompt_plus_generated():
    # Prompt alone cannot see an eviction that happened DURING generation.
    assert call_occupancy({"prompt_eval_count": 128, "eval_count": 32}) == 160


def test_occupancy_tolerates_one_missing_count():
    assert call_occupancy({"prompt_eval_count": 900}) == 900
    assert call_occupancy({"eval_count": 120}) == 120


def test_occupancy_is_none_when_the_server_reported_neither():
    # An endpoint that reports no counts must yield UNKNOWN downstream, not
    # a confident zero.
    assert call_occupancy({}) is None
    assert call_occupancy({"message": {"content": "hi"}}) is None


def test_peak_ignores_calls_that_reported_nothing():
    assert peak([400, None, 1200, None, 900]) == 1200
    assert peak([None, None]) is None
    assert peak([]) is None


# --- THE regression: a clamped count must not read as healthy ----------

def test_clamped_count_does_not_read_as_healthy():
    """The measured `num_ctx` 64 row, which the first version passed.

    Real numbers from the table in this module's docstring: a 128-token
    prompt against a 64-token window reported 34 prompt tokens and 32
    generated. The naive check saw "34 of 64, 53% used" and said fine,
    about a prompt that had lost three quarters of its content."""
    assert headroom_verdict(34, 34 + 32, 64) == EVICTED


def test_mild_truncation_is_flagged_even_without_eviction():
    """The measured `num_ctx` 128 row.

    66 + 32 = 98 never reaches the 128-token window, so the eviction test
    alone would miss it -- but the prompt WAS truncated, from 128 down to
    66. Caught by the half-window rule instead."""
    assert headroom_verdict(66, 66 + 32, 128) == AT_RISK


@pytest.mark.parametrize("num_ctx,prompt,generated,expected", [
    (4096, 128, 32, OK),        # measured: untruncated
    (512, 128, 32, OK),         # measured: untruncated
    (256, 128, 32, AT_RISK),    # measured: untruncated, but exactly half -- flagged
    (128, 66, 32, AT_RISK),     # measured: TRUNCATED
    (96, 50, 32, AT_RISK),      # measured: TRUNCATED
    (64, 34, 32, EVICTED),      # measured: TRUNCATED
])
def test_verdicts_match_the_measured_table(num_ctx, prompt, generated, expected):
    """Every row of the measurement, asserted.

    The three genuinely truncated rows all fire. The one false alarm
    (`num_ctx` 256) is a prompt filling exactly half the window, which is
    the deliberate cost of a test that is sound in the direction that
    matters: it never reports fine about a run that was truncated."""
    assert headroom_verdict(prompt, prompt + generated, num_ctx) == expected


# --- the boundaries ----------------------------------------------------

def test_ok_only_below_half_the_window():
    # Below half is the one thing a single observation can prove: a
    # truncated prompt would have been pushed UP to about half.
    assert headroom_verdict(4095, 4200, 8192) == OK
    assert headroom_verdict(4096, 4200, 8192) == AT_RISK


def test_eviction_outranks_at_risk():
    # A prompt over half the window AND an occupancy that reached it: the
    # stronger finding is the fact, not the uncertainty.
    assert headroom_verdict(5000, 8192, 8192) == EVICTED


def test_verdict_unknown_when_the_window_is_missing():
    assert headroom_verdict(977, 1100, None) == UNKNOWN
    assert headroom_verdict(977, 1100, "") == UNKNOWN


def test_verdict_unknown_when_no_count_was_recorded():
    # UNKNOWN is not OK. A row that cannot be checked must not be reported
    # as one that passed.
    assert headroom_verdict(None, None, 8192) == UNKNOWN


def test_occupancy_alone_cannot_answer_the_truncation_question():
    # Prompt-plus-generation conflates a small prompt with a long answer
    # and a truncated prompt with a long answer. Without the prompt count
    # the honest verdict is UNKNOWN, not OK.
    assert headroom_verdict(None, 5000, 8192) == UNKNOWN


def test_verdict_reads_csv_strings_as_well_as_ints():
    assert headroom_verdict("34", "66", "64") == EVICTED
    assert headroom_verdict("100", "200", "8192") == OK
    assert headroom_verdict("not-a-number", "", "8192") == UNKNOWN


def test_row_verdict_reads_the_columns_the_suites_write():
    row = {"max_prompt_tokens": 34, "peak_context_tokens": 66, "num_ctx": 64}
    assert row_verdict(row) == EVICTED
    assert row_verdict({"num_ctx": 8192}) == UNKNOWN


# --- the multi-turn assumption -----------------------------------------

def test_series_is_cumulative_recognises_a_growing_series():
    assert series_is_cumulative([477, 573, 671, 765, 875, 977]) is True
    assert series_is_cumulative([477]) is True
    assert series_is_cumulative([]) is True


def test_series_is_cumulative_rejects_a_cache_miss_shape():
    # What it would look like if the server reported only newly evaluated
    # tokens: a large first turn, then small ones. The peak would then be
    # meaningless on exactly the multi-turn runs this exists for.
    assert series_is_cumulative([477, 96, 98, 94]) is False


@pytest.mark.skipif(not S1_DIR.is_dir(), reason="no S1 corpus in this checkout")
def test_real_corpus_confirms_cumulative_accounting():
    """The claim in the module docstring, asserted against the data.

    If a future runtime switches to reporting only the cache miss, this
    fails and the detector's soundness is revisited -- rather than the
    docstring quietly going stale while the check goes blind."""
    examined = 0
    for path in sorted(S1_DIR.glob("*.csv")):
        with path.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                raw = (row.get("prompt_token_counts") or "").strip()
                if not raw:
                    continue
                try:
                    counts = [c for c in json.loads(raw) if isinstance(c, int)]
                except (ValueError, TypeError):
                    continue
                if len(counts) < 2:
                    continue
                examined += 1
                assert series_is_cumulative(counts), f"{path.name}: {counts}"
    if not examined:
        pytest.skip("no multi-turn rows with a token series")


# --- the tally ---------------------------------------------------------

def test_tally_counts_each_verdict():
    t = HeadroomTally()
    t.add({"max_prompt_tokens": 900, "peak_context_tokens": 1000, "num_ctx": 8192})
    t.add({"max_prompt_tokens": 5000, "peak_context_tokens": 6000, "num_ctx": 8192})
    t.add({"max_prompt_tokens": 5000, "peak_context_tokens": 8192, "num_ctx": 8192})
    t.add({"num_ctx": 8192})
    assert (t.ok, t.at_risk, t.evicted, t.unknown) == (1, 1, 1, 1)
    assert t.observed_peak == 8192


def test_tally_reports_the_window_only_when_rows_agree():
    t = HeadroomTally()
    t.add({"max_prompt_tokens": 900, "peak_context_tokens": 1000, "num_ctx": 8192})
    t.add({"max_prompt_tokens": 900, "peak_context_tokens": 1000, "num_ctx": 8192})
    assert t.window == 8192

    mixed = HeadroomTally()
    mixed.add({"max_prompt_tokens": 900, "peak_context_tokens": 1000, "num_ctx": 8192})
    mixed.add({"max_prompt_tokens": 900, "peak_context_tokens": 1000, "num_ctx": 4096})
    # Two windows pooled: no single denominator, so no percentage.
    assert mixed.window is None
    assert "%" not in mixed.summary()
    assert "4096" in mixed.summary() and "8192" in mixed.summary()


def test_summary_distinguishes_no_window_from_conflicting_windows():
    # Not the same problem. Rows that simply predate the column have not
    # drifted from anything, and saying they did invents a fault.
    absent = headroom_summary([{"max_prompt_tokens": 900}])
    assert "no row records the window" in absent

    conflicting = headroom_summary([
        {"max_prompt_tokens": 900, "num_ctx": 8192},
        {"max_prompt_tokens": 900, "num_ctx": 4096},
    ])
    assert "different windows" in conflicting


# --- what it says, and when ---------------------------------------------

def test_caveat_is_silent_when_nothing_fired():
    # Mirrors the TRUNCATED caveat: a warning in a report has to mean
    # something happened, or operators learn to skip past it.
    rows = [{"max_prompt_tokens": 977, "peak_context_tokens": 1100, "num_ctx": 8192}] * 5
    assert headroom_caveat(rows) == ""


def test_caveat_is_silent_on_unknown_rows():
    # Unknown is not a finding. It must not manufacture a warning either.
    assert headroom_caveat([{"model": "m"}] * 3) == ""


def test_at_risk_caveat_states_that_it_cannot_tell():
    # The honest claim is "cannot be ruled out", not "was truncated". The
    # difference is the whole design.
    rows = [{"max_prompt_tokens": 5000, "peak_context_tokens": 6000, "num_ctx": 8192}]
    caveat = headroom_caveat(rows)
    assert "cannot tell them apart" in caveat
    assert "proves it cannot be ruled out" in caveat


def test_at_risk_caveat_gives_the_disambiguating_action():
    # A warning an operator cannot act on is noise. Re-running at a bigger
    # window is what actually separates the two cases.
    rows = [{"max_prompt_tokens": 5000, "peak_context_tokens": 6000, "num_ctx": 8192}]
    assert "larger `--num-ctx`" in headroom_caveat(rows)


def test_evicted_caveat_calls_the_rows_unusable():
    rows = [{"max_prompt_tokens": 5000, "peak_context_tokens": 8192, "num_ctx": 8192}]
    caveat = headroom_caveat(rows)
    assert "bound on 1 row(s)" in caveat
    assert "unusable" in caveat


def test_eviction_outranks_at_risk_in_the_caveat():
    # One evicted row is the more serious finding and must not be buried
    # under a count of merely-uncertain ones.
    rows = [{"max_prompt_tokens": 5000, "peak_context_tokens": 6000, "num_ctx": 8192}] * 9
    rows.append({"max_prompt_tokens": 5000, "peak_context_tokens": 8192, "num_ctx": 8192})
    assert "unusable" in headroom_caveat(rows)


def test_summary_speaks_even_when_the_check_passes():
    # The reason this is unconditional: this project has audited detectors
    # that had fired zero times in their lifetime and nobody noticed,
    # because silence and absence look identical. A printed peak is how an
    # operator can see the check is alive.
    summary = headroom_summary(
        [{"max_prompt_tokens": 900, "peak_context_tokens": 977, "num_ctx": 8192}])
    assert "977" in summary and "8192" in summary and "12%" in summary


def test_summary_says_so_when_nothing_was_measured():
    assert "not measured" in headroom_summary([{"model": "m"}])


def test_at_risk_fraction_is_the_documented_half():
    # The soundness argument rests on this specific value, not on a
    # comfort margin. Raising it would make the test unsound rather than
    # merely less sensitive.
    assert AT_RISK_FRACTION == 0.5


# --- the columns actually reach the CSVs --------------------------------

@pytest.mark.parametrize("module_name", [
    "openllm_cbench.suites.containment",
    "openllm_cbench.suites.channel",
    "openllm_cbench.suites.persistence",
])
def test_every_suite_declares_both_context_columns(module_name):
    """A detector that reads a column no suite writes is worse than none.

    Both columns, not one: a suite recording only the occupancy can never
    return anything but UNKNOWN from the truncation test."""
    import importlib
    module = importlib.import_module(module_name)
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "CONTEXT_FIELDS" in source, f"{module_name} does not stamp the context columns"
    for field in CONTEXT_FIELDS:
        assert field in source, f"{module_name} never sets {field}"
