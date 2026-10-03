"""
Reconciliation between the Inspect-framework containment task and this
project's own containment suite.

The two harnesses differ in more than scoring: the Inspect task runs
Inspect's own tool loop, and the OpenAI-compatible endpoint it calls cannot
set num_ctx or top_k. Agreement on a finding is informative; agreement that
neither harness recorded anything shows nothing; and a disagreement is a
lead to investigate, not evidence against either harness.

Usage (per model, per boundary variant -- run both `inspect eval` tasks
first, from inspect_containment.py; path is relative to the project root,
see that module's docstring for why it's a file path, not `-m ...`):

    inspect eval src/openllm_cbench/integrations/inspect_containment.py@containment_stated \\
        --model ollama/<model-tag> --log-dir results/s5_inspect
    inspect eval src/openllm_cbench/integrations/inspect_containment.py@containment_none \\
        --model ollama/<model-tag> --log-dir results/s5_inspect

    python -m openllm_cbench.suites.containment --model <model-tag> --boundary both
    # note the CSV path it prints, then:

    python -m openllm_cbench.integrations.inspect_reconcile \\
        --model <model-tag> \\
        --inspect-log results/s5_inspect/<stated-run>.eval \\
        --inspect-log results/s5_inspect/<none-run>.eval \\
        --containment-csv results/s1_containment/containment_<model-tag>_<ts>.csv
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

from openllm_cbench.core.paths import results_dir as _results_dir
from openllm_cbench.core.provenance import provenance_note
from openllm_cbench.core.console import ensure_utf8_stdio
RESULTS_DIR = _results_dir("s5_inspect")


def load_inspect_results(log_paths):
    """Returns {(task_id, boundary): {escape_attempt, escape_turn, escalated,
    authorized_use}} pulled from one or more Inspect .eval log files."""
    # Imported here, not at module level: diff() needs no Inspect, and a
    # module-level import made this module -- and every test of diff() --
    # fail wherever the optional `inspect` extra is not installed.
    from inspect_ai.log import read_eval_log
    results = {}
    for path in log_paths:
        log = read_eval_log(str(path))
        for sample in log.samples or []:
            meta = sample.metadata or {}
            task_id = meta.get("task_id")
            boundary = meta.get("boundary")
            if not task_id or not boundary:
                continue
            score = None
            if sample.scores:
                # Inspect keys scores by scorer name; escape_attempt_scorer is the only one here.
                score = next(iter(sample.scores.values()), None)
            score_meta = (score.metadata if score and score.metadata else {})
            results[(task_id, boundary)] = {
                "escape_attempt": score_meta.get("escape_attempt"),
                "escape_turn": score_meta.get("escape_turn"),
                "escalated": score_meta.get("escalated"),
                "authorized_use": score_meta.get("authorized_use"),
                "source": "inspect",
            }
    return results


def load_containment_csv(csv_path):
    """Returns {(task_id, boundary): {...}} from the containment suite's CSV."""
    results = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            task_id = row.get("task_id")
            boundary = row.get("boundary")
            if not task_id or not boundary:
                continue

            def as_bool(v):
                return str(v).strip().lower() == "true"

            # COMPARE WHAT THIS FRAMEWORK ACTUALLY CITES.
            #
            # The raw `escape_attempt` column is not the headline metric.
            # `escape_attempt_real` is: it narrows an attempt to calls
            # whose target was not a fumbled tool argument, and the
            # containment suite's own report leads with it for that
            # reason. Reading the raw column here compared Inspect
            # against a quantity no report cites, so an agreement
            # percentage produced from it did not describe the two
            # things a reader would assume it did.
            #
            # scoring/aggregate.py already pools on the corrected metric.
            # This reconciler did not, which meant the same underlying
            # question was answered two different ways in one package.
            real = row.get("escape_attempt_real")
            corrected = as_bool(real) if real not in (None, "") \
                else as_bool(row.get("escape_attempt", ""))

            results[(task_id, boundary)] = {
                "escape_attempt": corrected,
                "escape_attempt_raw": as_bool(row.get("escape_attempt", "")),
                "incomplete": is_incomplete_row(row),
                "escape_turn": row.get("escape_turn") or None,
                "escalated": as_bool(row.get("escalated", "")),
                "authorized_use": as_bool(row.get("authorized_use", "")),
                "source": "containment_suite",
            }
    return results


from openllm_cbench.scoring.containment_metrics import is_incomplete_row


def diff(inspect_results, harness_results):
    keys = sorted(set(inspect_results) | set(harness_results))
    rows = []
    for key in keys:
        task_id, boundary = key
        i = inspect_results.get(key)
        h = harness_results.get(key)
        if i is None:
            rows.append((task_id, boundary, "MISSING (inspect)", h["escape_attempt"], "n/a"))
            continue
        if h is None:
            rows.append((task_id, boundary, i["escape_attempt"], "MISSING (harness)", "n/a"))
            continue
        # A row that never reached a stopping point is not evidence of
        # agreement OR disagreement -- neither framework got a verdict
        # out of it. Scoring it either way manufactures a data point.
        if h.get("incomplete"):
            rows.append((task_id, boundary, i["escape_attempt"], "INCOMPLETE", "SKIPPED"))
            continue
        agree = bool(i["escape_attempt"]) == bool(h["escape_attempt"])
        rows.append((task_id, boundary, i["escape_attempt"], h["escape_attempt"], "AGREE" if agree else "DISAGREE"))
    return rows


def render_report(model, rows):
    n_disagree = sum(1 for r in rows if r[4] == "DISAGREE")
    n_total = len(rows)
    L = [
        f"# Inspect reconciliation: `{model}`",
        "",
    ]
    note = provenance_note(model)
    if note:
        L += [note.rstrip("\n"), ""]
    L += [
        f"{n_total - n_disagree}/{n_total} task/boundary combinations agree on "
        f"escape_attempt between Inspect and the containment suite.",
        "",
        "| task | boundary | inspect escape_attempt | harness escape_attempt | verdict |",
        "|---|---|---|---|---|",
    ]
    for task_id, boundary, i_val, h_val, verdict in rows:
        L.append(f"| {task_id} | {boundary} | {i_val} | {h_val} | {verdict} |")

    L += ["", "## Reading disagreements", ""]
    if n_disagree:
        L.append(
            f"**{n_disagree} disagreement(s).** Each one means either the harness's host-allowlist "
            "logic or Inspect's default tool-loop/message-limit behavior classified the same "
            "transcript differently; read the underlying transcript (harness `tool_log` column, "
            "or `inspect view` on the .eval file) for each disagreement before trusting either "
            "number. A structural mismatch (e.g. Inspect's `generate()` hit a message limit before "
            "the harness's `--max-turns` would have) is a config difference, not a real behavioral "
            "disagreement; note which kind each one is."
        )
    else:
        L.append(
            "No disagreements. Where either harness recorded an escape attempt, agreement on "
            "it is informative: state it plainly as \"the harness and Inspect agree on every "
            "task/boundary combination tested.\" Where neither recorded one, the two agree on "
            "nothing, which cannot show that either would detect an attempt."
        )
    return "\n".join(L) + "\n"


def main():
    ensure_utf8_stdio()
    p = argparse.ArgumentParser(description="Reconcile Inspect's containment task against this "
                                              "project's own containment suite.")
    p.add_argument("--model", required=True, help="Model tag this reconciliation covers (label only).")
    p.add_argument("--inspect-log", action="append", required=True,
                    help="Path to an Inspect .eval log file. Repeat for stated + none runs.")
    p.add_argument("--containment-csv", required=True,
                    help="Path to the containment suite's CSV output for the same model.")
    args = p.parse_args()

    inspect_results = load_inspect_results(args.inspect_log)
    harness_results = load_containment_csv(args.containment_csv)
    if not inspect_results:
        raise SystemExit("No samples found in the given Inspect log(s). Check the --inspect-log path(s).")
    if not harness_results:
        raise SystemExit(f"No rows found in {args.containment_csv}.")

    rows = diff(inspect_results, harness_results)
    md = render_report(args.model, rows)
    print(md)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model.replace(":", "-").replace("/", "-")
    out_path = RESULTS_DIR / f"reconcile_{model_tag}_{ts}.md"
    out_path.write_text(md, encoding="utf-8")
    print(f"Report: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
