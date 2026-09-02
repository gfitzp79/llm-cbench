"""
Reconciliation between the Inspect-framework containment task and this
project's own containment suite.

Every disagreement between the two is a calibration signal about the
hand-rolled harness's own scoring -- cross-framework agreement is a
stronger validity claim than either tool's numbers on their own.

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

from inspect_ai.log import read_eval_log

from openllm_cbench.core.paths import results_dir as _results_dir
from openllm_cbench.core.provenance import provenance_note
from openllm_cbench.core.console import ensure_utf8_stdio
RESULTS_DIR = _results_dir("s5_inspect")


def load_inspect_results(log_paths):
    """Returns {(task_id, boundary): {escape_attempt, escape_turn, escalated,
    authorized_use}} pulled from one or more Inspect .eval log files."""
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

            results[(task_id, boundary)] = {
                "escape_attempt": as_bool(row.get("escape_attempt", "")),
                "escape_turn": row.get("escape_turn") or None,
                "escalated": as_bool(row.get("escalated", "")),
                "authorized_use": as_bool(row.get("authorized_use", "")),
                "source": "containment_suite",
            }
    return results


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
        agree = bool(i["escape_attempt"]) == bool(h["escape_attempt"])
        rows.append((task_id, boundary, i["escape_attempt"], h["escape_attempt"], "AGREE" if agree else "DISAGREE"))
    return rows


def render_report(model, rows):
    n_disagree = sum(1 for r in rows if r[4] == "DISAGREE")
    n_total = len(rows)
    L = [
        f"# Inspect reconciliation -- `{model}`",
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
            "transcript differently -- read the underlying transcript (harness `tool_log` column, "
            "or `inspect view` on the .eval file) for each disagreement before trusting either "
            "number. A structural mismatch (e.g. Inspect's `generate()` hit a message limit before "
            "the harness's `--max-turns` would have) is a config difference, not a real behavioral "
            "disagreement -- note which kind each one is."
        )
    else:
        L.append(
            "No disagreements. This is a real calibration result, not just a clean run -- "
            "state it plainly as \"the hand-rolled harness and Inspect agree on every "
            "task/boundary combination tested,\" not silently."
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
