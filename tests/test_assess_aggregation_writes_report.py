"""
Regression test for a crash found running a real end-to-end `cbench assess`
against a live model: `_assess_body` called `_results_dir(...)` to place the
trial-summary report, but that name was only ever imported inside the
*caller*, `_cmd_assess` (a local, lazy import) -- not in `_assess_body`
itself, which is a separate module-level function and does not see it.

Every real (non-dry-run) `cbench assess` invocation hit this: all N trials
of the first suite would run to completion (real model calls, real wall
time spent), then the "Aggregating s1..." step raised NameError and the
whole command crashed before writing a trial-summary report or moving on
to the next suite. --dry-run never exercised this path, since dry-run
skips aggregation entirely -- which is exactly how it shipped unnoticed.

This test calls `_assess_body` directly (no model, no subprocess) with a
stubbed-out trial runner and aggregate function, and asserts the
trial-summary file actually lands on disk. It fails against the code as
written before this fix (NameError) and passes after.
"""

from unittest.mock import patch

from openllm_cbench import cli


def test_assess_body_writes_trial_summary_without_nameerror(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))

    class Args:
        model = "fake-model:1b"
        trials = 1
        dry_run = False

    suite_info = {
        "s1": ("containment", "openllm_cbench.suites.containment",
               lambda model: ("# fake aggregate report\n", {"n_trials": 1}), "s1_containment"),
    }

    # A real trial run creates results/<suite>/ itself when it writes CSVs,
    # before aggregation is ever reached -- mocked out here since no suite
    # actually runs, so create it the same way the real trial would.
    (tmp_path / "s1_containment").mkdir()

    with patch.object(cli, "_dispatch_passthrough", return_value=0) as mock_dispatch:
        rc = cli._assess_body(Args(), ["s1"], suite_info)

    assert mock_dispatch.call_count == 1  # one trial, as configured
    assert rc == 0

    out_path = tmp_path / "s1_containment" / "trial_summary_fake-model-1b.md"
    assert out_path.exists(), (
        "trial-summary report was not written -- _assess_body's aggregation "
        "step didn't complete (this is the NameError regression if it fails)"
    )
    assert out_path.read_text(encoding="utf-8") == "# fake aggregate report\n"
