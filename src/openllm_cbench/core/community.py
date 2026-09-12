"""
Validation for a community-submitted results folder under
community-results/ -- checked by a maintainer (or a future CI step)
before trusting or merging a PR, per community-results/README.md's own
submission convention.

Deliberately does NOT run a suite, aggregate anything, or compute a
scorecard -- this only checks that a submission folder is *shaped*
correctly (the right metadata, CSVs that plausibly belong to the model
claimed) before anyone spends time on the next step. Scoring a validated
submission is `cbench score --model <tag> --from-existing` with
$OPENLLM_CBENCH_RESULTS_DIR pointed at the submission's folder -- see
cli.py's own _cmd_score docstring for why that has to be an environment
variable set before the process starts, not a flag this module or that
command could apply after the fact.

Never raises on a malformed submission: a missing or unreadable
submission.json is itself a reported problem, the same as any other
"why is this row of the batch bad" model already used across this project
(core/hardware.py's probe(), the discover/gate-all batch loop, etc).
"""

import json
from pathlib import Path

REQUIRED_SUBMISSION_FIELDS = ("model", "contributor", "date", "hardware_summary", "endpoint")

# suite results subdirectory -> the CSV filename prefix that suite writes
# (same prefixes scoring/aggregate.py's own find_csvs() globs on).
SUITE_DIRS = {
    "s1_containment": "containment",
    "s2_channel": "channel",
    "s3_persistence": "persistence",
}


def validate_submission(path):
    """Returns a list of problems with the submission folder at `path` --
    empty list means valid. Read-only, makes no model or network call."""
    path = Path(path)
    problems = []
    if not path.is_dir():
        return [f"{path} is not a directory"]

    sub_path = path / "submission.json"
    meta = {}
    if not sub_path.exists():
        problems.append("missing submission.json -- see SUBMISSION_TEMPLATE.json")
    else:
        try:
            meta = json.loads(sub_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            problems.append(f"submission.json is not valid JSON: {e}")

    for field in REQUIRED_SUBMISSION_FIELDS:
        if not meta.get(field):
            problems.append(f"submission.json missing required field: '{field}'")

    model = meta.get("model")
    tag = None
    if model:
        from openllm_cbench.scoring.aggregate import model_tag
        tag = model_tag(model)

    found_any_csv = False
    for suite_dir, prefix in SUITE_DIRS.items():
        d = path / suite_dir
        if not d.is_dir():
            continue
        csvs = [p for p in d.glob(f"{prefix}_*.csv") if ".INVALID" not in p.name]
        if not csvs:
            problems.append(f"{suite_dir}/ exists but has no {prefix}_*.csv files in it")
            continue
        found_any_csv = True
        if tag:
            mismatched = sorted(p.name for p in csvs if f"_{tag}_" not in p.name)
            if mismatched:
                problems.append(
                    f"{suite_dir}/ has file(s) that don't match model tag '{tag}' from "
                    f"submission.json: {', '.join(mismatched)}"
                )

    if not found_any_csv:
        problems.append("no S1/S2/S3 CSVs found in s1_containment/, s2_channel/, or "
                         "s3_persistence/ -- nothing here to score")

    return problems
