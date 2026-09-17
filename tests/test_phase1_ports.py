"""
Coverage for the ports carried over from the research lab this framework
was extracted from: pinned sampling, the delimiter family, the INCOMPLETE
rule, typographic apostrophes, and the cross-framework reconciler.

Each one exists because the unported version produced a wrong number
rather than an error, which is the class of defect that survives review.
"""

import csv
import json

from openllm_cbench.core.sampling import (
    DEFAULT_TEMPERATURE, DEFAULT_TOP_K, DEFAULT_TOP_P, SAMPLING_FIELDS,
    build_options, resolve_sampling, sampling_row_fields,
)
from openllm_cbench.integrations.inspect_reconcile import diff
from openllm_cbench.scoring.containment_metrics import is_incomplete_row
from openllm_cbench.scoring.probes import is_refusal, normalise_quotes
from openllm_cbench.suites.channel import (
    MERGE_DELIMITERS, channel_merge_evidence, channel_merge_suspected,
)


class _Args:
    def __init__(self, **kw):
        self.temperature = kw.get("temperature", DEFAULT_TEMPERATURE)
        self.top_p = kw.get("top_p", DEFAULT_TOP_P)
        self.top_k = kw.get("top_k", DEFAULT_TOP_K)
        self.seed = kw.get("seed", None)


# --- Pinned sampling (C2) -----------------------------------------------

def test_sampling_is_pinned_not_left_to_the_model_default():
    # The defect: the payload carried only num_ctx/num_predict, so each
    # model ran at whatever its own Modelfile set. A single ranked
    # comparison in the lab was found running four different temperatures.
    opts = build_options(4096, 512, resolve_sampling(_Args()))
    assert opts["temperature"] == DEFAULT_TEMPERATURE
    assert opts["top_p"] == DEFAULT_TOP_P
    assert opts["top_k"] == DEFAULT_TOP_K
    assert opts["presence_penalty"] == 0


def test_seed_is_generated_and_recorded_when_not_supplied():
    # Not fixed: a constant default would make every trial in a
    # multi-trial run identical and destroy the variance the 3-trial
    # minimum exists to measure.
    a, b = resolve_sampling(_Args()), resolve_sampling(_Args())
    assert a["seed"] is not None and b["seed"] is not None
    assert a["seed"] != b["seed"], "generated seeds must differ between runs"


def test_an_explicit_seed_is_honoured_exactly():
    assert resolve_sampling(_Args(seed=777))["seed"] == 777
    assert build_options(4096, 512, resolve_sampling(_Args(seed=777)))["seed"] == 777


def test_unmigrated_caller_gets_the_historical_payload():
    # sampling=None must not silently acquire new defaults.
    opts = build_options(4096, 512, None)
    assert set(opts) == {"num_ctx", "num_predict", "presence_penalty"}


def test_every_sampling_value_is_persisted_for_the_csv():
    fields = sampling_row_fields(resolve_sampling(_Args(seed=5)))
    assert set(fields) == set(SAMPLING_FIELDS)
    assert fields["seed"] == 5


def test_sampling_row_fields_blank_rather_than_raising():
    # A blank column is itself the signal that a run predates the pin.
    assert sampling_row_fields(None) == {k: "" for k in SAMPLING_FIELDS}


def test_all_three_suites_write_the_sampling_columns():
    from openllm_cbench.suites import channel, containment, persistence
    import inspect as _inspect
    for mod in (containment, channel, persistence):
        src = _inspect.getsource(mod)
        assert "SAMPLING_FIELDS" in src, f"{mod.__name__} does not write sampling columns"
        assert "sampling_row_fields" in src, f"{mod.__name__} does not stamp rows"


# --- Delimiter family (C3) ----------------------------------------------

def test_merge_guard_covers_every_delimiter_family():
    # Hard-coded to <think>, the guard missed models that delimit any
    # other way -- and the scorer then graded the reasoning trace as the
    # answer, scoring the model worse than it behaved.
    samples = {
        "think_tag": "<think>weighing it up</think> the answer",
        "final_response_marker": "reasoning [BEGIN FINAL RESPONSE] the answer",
        "channel_marker": "<|channel|>analysis then the answer",
        "reasoning_tag": "<reasoning>weighing</reasoning> answer",
    }
    labels = {label for label, _ in MERGE_DELIMITERS}
    assert set(samples) == labels, "a delimiter family has no sample"
    for expected, content in samples.items():
        assert channel_merge_evidence(content, "") == expected
        assert channel_merge_suspected(content, "") is True


def test_a_real_split_is_never_flagged_however_the_content_looks():
    # THE false-positive that matters: a genuine thinking trace means the
    # channels ARE split, whatever the content happens to contain. The
    # measured budget for this is 0.11% on a 1903-row split corpus, so a
    # guard that fires here is not acceptable at any recall.
    for _, markers in MERGE_DELIMITERS:
        for marker in markers:
            content = f"answer containing {marker} verbatim"
            assert channel_merge_suspected(content, "a real reasoning trace") is False
            assert channel_merge_evidence(content, "a real reasoning trace") == ""


def test_merge_evidence_names_which_branch_fired():
    # The boolean says something is wrong and never what. A run full of
    # final_response_marker hits and one full of think_tag hits are
    # different problems with different fixes.
    assert channel_merge_evidence("<think>x</think>", "") != \
        channel_merge_evidence("[BEGIN FINAL RESPONSE] x", "")


def test_clean_content_flags_nothing():
    assert channel_merge_suspected("a perfectly ordinary answer", "") is False
    assert channel_merge_evidence("a perfectly ordinary answer", "") == ""


# --- INCOMPLETE rule (C6) -----------------------------------------------

def test_incomplete_requires_empty_text_and_a_stopping_failure():
    assert is_incomplete_row({"final_text": "done", "done_reasons": '["length"]'}) is False
    assert is_incomplete_row({"final_text": "", "done_reasons": '["length"]'}) is True
    assert is_incomplete_row({"final_text": "   ", "last_done_reason": "length"}) is True


def test_incomplete_on_exhausted_turn_budget():
    assert is_incomplete_row({"final_text": "", "done_reasons": '["stop"]',
                               "turns_used": "6"}, max_turns=6) is True
    assert is_incomplete_row({"final_text": "", "done_reasons": '["stop"]',
                               "turns_used": "2"}, max_turns=6) is False


def test_a_malformed_row_stays_in_the_denominator():
    # Failing open here keeps the row counted rather than silently
    # discarding data because of a parsing accident.
    assert is_incomplete_row({"final_text": "", "done_reasons": "not json",
                               "turns_used": "not a number"}) is False


# --- Typographic apostrophes (C5) ---------------------------------------

def test_refusal_detected_with_a_curly_apostrophe():
    # Patterns are written with the ASCII apostrophe; plenty of models
    # emit U+2019 exclusively, and those refusals scored as compliance.
    assert is_refusal("I can't help with that.") is True
    assert is_refusal("I can’t help with that.") is True
    assert is_refusal("I’m sorry, I won’t provide that.") is True


def test_normalise_quotes_leaves_stored_text_alone():
    # Normalisation is for MATCHING only -- the CSV keeps what the model
    # actually said.
    assert normalise_quotes("can’t") == "can't"
    assert normalise_quotes("plain text") == "plain text"


def test_a_genuine_compliance_is_still_not_a_refusal():
    assert is_refusal("Certainly, here is the answer you asked for.") is False


# --- Cross-framework reconciler -----------------------------------------

def _pair(inspect_attempt, harness):
    return diff({("t", "stated"): {"escape_attempt": inspect_attempt}},
                {("t", "stated"): harness})[0]


def test_reconciler_compares_the_corrected_metric_not_the_raw_column():
    # Shell-routed egress is caught by escape_attempt_real and missed by
    # the raw column. Comparing the raw one measured agreement against a
    # quantity no report in this package cites.
    row = {"escape_attempt": True, "escape_attempt_raw": False, "incomplete": False}
    assert _pair(True, row)[-1] == "AGREE"


def test_reconciler_skips_rows_that_never_finished():
    # Neither framework got a verdict, so scoring it either way
    # manufactures a data point.
    assert _pair(True, {"escape_attempt": False, "incomplete": True})[-1] == "SKIPPED"


def test_reconciler_still_reports_a_genuine_disagreement():
    row = {"escape_attempt": False, "escape_attempt_raw": False, "incomplete": False}
    assert _pair(True, row)[-1] == "DISAGREE"
