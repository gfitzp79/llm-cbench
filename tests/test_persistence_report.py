"""
Regression coverage for suites/persistence.py:render_report()'s "did any
signal fire" summary text.

Found live (end-to-end smoke test against a real model, 2026-08-31): a
row with log_append_correction=True rendered a table showing that flag
True, immediately followed by prose declaring "No signal fired on any
scenario" -- self-contradictory in the same generated report. Root
cause: `any_fired` was computed from only 3 of the 4 scored flags
(log_rewrite, denial_flag, fresh_identity_flag), silently dropping
log_append_correction even though it's a real column in the same table
and is treated as a genuine signal by this suite's own live per-row
console output (see main()'s `flags` list, which includes
"APPEND_CORRECTION"). No test caught this because render_report() had
zero test coverage before this file.
"""

from openllm_cbench.suites.persistence import render_report

ALL_FALSE = {
    "scenario_id": "s", "error": "",
    "log_rewrite": False, "log_append_correction": False,
    "denial_flag": False, "fresh_identity_flag": False,
}


def _row(**overrides):
    row = dict(ALL_FALSE)
    row.update(overrides)
    return row


def test_all_flags_false_is_a_clean_null_result():
    report = render_report("model:1b", [_row()])
    assert "No signal fired on any scenario" in report
    assert "At least one signal fired" not in report


def test_append_correction_alone_counts_as_a_fired_signal():
    # The bug this guards: append_correction=True with the other three
    # False must NOT render as "no signal fired" -- that's a table saying
    # True right next to prose saying nothing happened.
    report = render_report("model:1b", [_row(log_append_correction=True)])
    assert "At least one signal fired" in report
    assert "No signal fired on any scenario" not in report


def test_log_rewrite_alone_counts_as_a_fired_signal():
    report = render_report("model:1b", [_row(log_rewrite=True)])
    assert "At least one signal fired" in report


def test_denial_alone_counts_as_a_fired_signal():
    report = render_report("model:1b", [_row(denial_flag=True)])
    assert "At least one signal fired" in report


def test_fresh_identity_alone_counts_as_a_fired_signal():
    report = render_report("model:1b", [_row(fresh_identity_flag=True)])
    assert "At least one signal fired" in report


def test_table_shows_the_true_value_it_claims_fired():
    report = render_report("model:1b", [_row(log_append_correction=True)])
    assert "| s | False | True | False | False |" in report
