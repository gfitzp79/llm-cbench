"""
Reconciliation between the Inspect-framework persistence task and this
project's own persistence suite.

Same discipline as inspect_reconcile.py (containment): every disagreement
is a calibration signal about the hand-rolled harness's own scoring, not a
judgment about the model. The persistence suite has four boolean flags per
scenario instead of the containment suite's single escape_attempt, so a
scenario's verdict is AGREE only if every flag matches; a DISAGREE row says
exactly which flag(s) split.

**Before trusting a reconciliation run this script writes, confirm the
`--persistence-csv` actually belongs to the same model as `--inspect-log`.**
Nothing here can verify that itself -- scenario_id keys are identical
across models, so a mismatched pair produces a plausible-looking report
with no error. A mismatched pairing here produces output that looks
correct and isn't -- double-check the pairing by hand every time, not just
on suspicion something went wrong.

Usage (per model -- run the Inspect task first, from
inspect_persistence.py; path is relative to the project root, see that
module's docstring for why it's a file path, not `-m ...`):

    inspect eval src/openllm_cbench/integrations/inspect_persistence.py@persistence \\
        --model ollama/<model-tag> --log-dir results/s5_inspect

    python -m openllm_cbench.suites.persistence --model <model-tag>
    # note the CSV path it prints, then:

    python -m openllm_cbench.integrations.inspect_reconcile_persistence \\
        --model <model-tag> \\
        --inspect-log results/s5_inspect/<run>.eval \\
        --persistence-csv results/s3_persistence/persistence_<model-tag>_<ts>.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from inspect_ai.log import read_eval_log

from openllm_cbench.core.paths import results_dir as _results_dir
from openllm_cbench.core.provenance import provenance_note
from openllm_cbench.core.console import ensure_utf8_stdio
RESULTS_DIR = _results_dir("s5_inspect")

FLAGS = ("log_rewrite", "log_append_correction", "denial_flag", "fresh_identity_flag")


def load_inspect_results(log_paths):
    """Returns {scenario_id: {flag: bool, ...}} pulled from one or more
    Inspect .eval log files."""
    results = {}
    for path in log_paths:
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            meta = sample.metadata or {}
            scenario_id = meta.get("scenario_id")
            if not scenario_id:
                continue
            score = None
            if sample.scores:
                score = next(iter(sample.scores.values()), None)
            score_meta = (score.metadata if score and score.metadata else {})
            results[scenario_id] = {flag: bool(score_meta.get(flag)) for flag in FLAGS}
            results[scenario_id]["source"] = "inspect"
    return results


def load_persistence_csv(csv_path):
    """Returns {scenario_id: {flag: bool, ...}} from the persistence suite's CSV."""
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            scenario_id = row.get("scenario_id")
            if not scenario_id:
                continue

            def as_bool(v):
                return str(v).strip().lower() == "true"

            results[scenario_id] = {flag: as_bool(row.get(flag, "")) for flag in FLAGS}
            results[scenario_id]["source"] = "persistence_suite"
    return results


def diff(inspect_results, harness_results):
    keys = sorted(set(inspect_results) | set(harness_results))
    rows = []
    for scenario_id in keys:
        i = inspect_results.get(scenario_id)
        h = harness_results.get(scenario_id)
        if i is None:
            rows.append((scenario_id, "MISSING (inspect)", None, [], "n/a"))
            continue
        if h is None:
            rows.append((scenario_id, i, "MISSING (harness)", [], "n/a"))
            continue
        mismatched = [flag for flag in FLAGS if bool(i[flag]) != bool(h[flag])]
        rows.append((scenario_id, i, h, mismatched, "AGREE" if not mismatched else "DISAGREE"))
    return rows


def render_report(model, rows):
    n_disagree = sum(1 for r in rows if r[4] == "DISAGREE")
    n_total = len(rows)
    L = [
        f"# Inspect reconciliation (S3 -- persistence) -- `{model}`",
        "",
    ]
    note = provenance_note(model)
    if note:
        L += [note.rstrip("\n"), ""]
    L += [
        f"{n_total - n_disagree}/{n_total} scenarios agree on all four flags "
        f"(log_rewrite, log_append_correction, denial_flag, fresh_identity_flag) "
        f"between Inspect and the persistence suite.",
        "",
        "| scenario | inspect flags | harness flags | mismatched | verdict |",
        "|---|---|---|---|---|",
    ]
    for scenario_id, i_val, h_val, mismatched, verdict in rows:
        if verdict == "n/a":
            L.append(f"| {scenario_id} | {i_val} | {h_val} | - | {verdict} |")
            continue
        i_str = ", ".join(f"{f}={i_val[f]}" for f in FLAGS)
        h_str = ", ".join(f"{f}={h_val[f]}" for f in FLAGS)
        L.append(f"| {scenario_id} | {i_str} | {h_str} | {', '.join(mismatched) or '-'} | {verdict} |")

    L += ["", "## Reading disagreements", ""]
    if n_disagree:
        L.append(
            f"**{n_disagree} scenario(s) disagree.** `log_rewrite`/`log_append_correction` splits "
            "usually mean the two phase-gating implementations classified a tool call's phase "
            "differently -- read the underlying transcript (harness `rewrite_detail` column, or "
            "`inspect view` on the .eval file). `denial_flag`/`fresh_identity_flag` splits mean the "
            "keyword heuristic matched different final-response text -- check whether Inspect's "
            "challenge-phase `generate()` produced a different response than the harness's, not just "
            "a different verdict on the same text. Both scorers use the identical keyword list, so a "
            "split there is a real behavioral difference in the transcript, not a scoring bug."
        )
    else:
        L.append(
            "No disagreements. This is a real calibration result, not just a clean run -- "
            "state it plainly as \"the hand-rolled harness and Inspect agree on every scenario "
            "tested,\" not silently. If every flag is False on both sides, remember the "
            "null-result framing still applies: 'did not manifest,' not 'is safe from this "
            "failure mode.'"
        )
    return "\n".join(L) + "\n"


def main():
    ensure_utf8_stdio()
    p = argparse.ArgumentParser(description="Reconcile Inspect's persistence task against this "
                                              "project's own persistence suite.")
    p.add_argument("--model", required=True, help="Model tag this reconciliation covers (label only).")
    p.add_argument("--inspect-log", action="append", required=True,
                    help="Path to an Inspect .eval log file.")
    p.add_argument("--persistence-csv", required=True,
                    help="Path to the persistence suite's CSV output for the same model.")
    args = p.parse_args()

    inspect_results = load_inspect_results(args.inspect_log)
    harness_results = load_persistence_csv(args.persistence_csv)
    if not inspect_results:
        raise SystemExit("No samples found in the given Inspect log(s). Check the --inspect-log path(s).")
    if not harness_results:
        raise SystemExit(f"No rows found in {args.persistence_csv}.")

    rows = diff(inspect_results, harness_results)
    md = render_report(args.model, rows)
    print(md)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace(":", "-").replace("/", "-")
    out_path = RESULTS_DIR / f"reconcile_persistence_{model_tag}_{ts}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
