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

import datetime
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

REQUIRED_SUBMISSION_FIELDS = ("model", "contributor", "date", "hardware_summary", "endpoint")

# THE PUBLICATION RULE THIS MODULE ENFORCES
# =========================================
# Raw measurements travel. Verdicts do not.
#
# A submission carries per-row CSVs and nothing else. It never carries a
# scorecard, a grade, or a claimed rate -- anyone who wants a grade
# computes it themselves from the CSVs with `cbench score --from-existing`,
# on their own machine, under their own name.
#
# This is a deliberate publication policy, not a file-format preference:
#
#   - A grade is a CONCLUSION about a named commercial product. A CSV row
#     is a MEASUREMENT taken on one machine. The first is an editorial
#     claim someone can be wrong about in public; the second is a fact
#     about what happened, qualified by the configuration recorded beside
#     it. Only the second is safe for a stranger to hand a maintainer.
#   - Results are hardware-dependent in ways that can invert a verdict.
#     Found live: a 27.9B model on a 12GB card had every gate check time
#     out, and the report read "Tool call check failed" -- which sounds
#     like "this model cannot call tools". It could; the machine was too
#     small. A submission built from that run would have published a
#     false claim about a named product in good faith. See
#     `gate_evidence()`, which is the guard that now blocks exactly that.
#   - Grades collapse the confidence intervals, caveats and validity
#     guards this framework works hard to produce. The nuance is the
#     defensible part; the letter is the part that travels badly.
#
# The practical consequence: this repo hosts DATA, and each reader
# computes their own verdict. It is not a leaderboard and must not become
# one by accident -- which is what validate_submission() refusing a
# scorecard file is actually protecting.

# Filename fragments that indicate a verdict rather than a measurement.
VERDICT_MARKERS = ("scorecard", "trial_summary")

# The version identifies the exact wording a contributor accepted, so any
# change to the text bumps it. 2: wording only (a dash became ", and"); the
# terms themselves are unchanged. Bumped before any submission existed.
ATTESTATION_VERSION = 2
ATTESTATION_TEXT = (
    "I confirm that: (1) I have the right to share these files and they are "
    "mine to submit; (2) they contain no confidential, personal or "
    "proprietary material, and I have read the raw CSVs, not just this "
    "summary; (3) the hardware, runtime and model tag recorded here are "
    "accurate and describe the machine that actually produced these rows; "
    "(4) I grant this project a perpetual, irrevocable licence to publish "
    "and redistribute these files under the repository's own licence "
    "(Apache-2.0); and (5) I understand this submission is raw data, that "
    "no score or grade is being claimed on my behalf, and that it will be "
    "published permanently in public git history where deletion does not "
    "remove it."
)

# Shown before a submission is packaged, and again before it is sent.
#
# This is not boilerplate. A trial CSV contains the model's RAW OUTPUT,
# including whatever it said while reaching for a tool -- and the
# containment suite deliberately hands the model a fabricated sandbox
# whose contents a contributor may well have customised with real
# material. A pull request puts that in public git history permanently,
# where deleting the file later does not remove it. Contributors will not
# think of this on their own; the tool that packages the data is the last
# place it can be said in time to matter.
PRIVACY_NOTICE = (
    "Before you submit: these CSVs contain the model's raw output, including "
    "anything it produced while reaching for a tool. If you customised the "
    "sandbox fixtures or task files with real internal material, that material "
    "is in these rows. A pull request publishes it permanently: git history "
    "keeps it even if the file is deleted later. Open the CSVs and read them "
    "before sending."
)

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
        problems.append("missing submission.json (see SUBMISSION_TEMPLATE.json)")
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
                         "s3_persistence/: nothing here to score")

    problems += _checksum_problems(path, meta)
    problems += _policy_problems(path, meta)
    return problems


# --- Publication policy --------------------------------------------------


def _verdict_files(folder):
    """Any file in a submission that carries a conclusion rather than a
    measurement -- a saved scorecard, an aggregated trial summary. See
    this module's publication-rule comment for why these must not
    travel."""
    folder = Path(folder)
    # Every part of the path, not just the file name: `cbench score` run in
    # place writes `scorecards/<tag>.json`, whose file name carries no
    # marker, and a name-only check let that grade through validation.
    return sorted(
        str(p.relative_to(folder)).replace("\\", "/")
        for p in folder.rglob("*")
        if p.is_file() and any(m in part.lower()
                               for part in p.relative_to(folder).parts
                               for m in VERDICT_MARKERS)
    )


def _policy_problems(folder, meta):
    problems = []

    for rel in _verdict_files(folder):
        problems.append(
            f"'{rel}' is a verdict, not a measurement: submissions carry raw CSVs only. "
            f"Anyone who wants a grade runs `cbench score --from-existing` against these "
            f"rows themselves. Delete it and re-validate."
        )

    for key in ("grade", "score", "rate", "band"):
        if meta.get(key) not in (None, ""):
            problems.append(
                f"submission.json claims '{key}'. A submission must not assert a result. "
                f"Remove it; the CSVs are the claim."
            )

    att = meta.get("attestation") or {}
    if not att.get("accepted"):
        problems.append(
            "contributor attestation not accepted: re-run `cbench community-package` "
            "with --accept-terms once you have read the terms and the raw CSVs."
        )
    elif att.get("version") != ATTESTATION_VERSION:
        problems.append(
            f"attestation is version {att.get('version')}, current terms are version "
            f"{ATTESTATION_VERSION}: re-package to accept the current terms."
        )

    gate = meta.get("gate_check") or {}
    if not gate:
        problems.append(
            "no gate_check record: run `cbench gate --model <tag> --save` and re-package. "
            "A submission from a machine that could not complete the gate checks cannot be "
            "distinguished from one where the model genuinely failed them."
        )
    elif gate.get("unverified"):
        problems.append(
            "gate check did not complete on this machine: "
            + "; ".join(gate.get("unverified", []))
            + ". This usually means the model is too large for available VRAM, NOT that it "
              "failed, which is exactly why the resulting rows must not be published as "
              "though they measured the model. Re-run the gate where it can complete."
        )

    return problems


def gate_evidence(model):
    """What the local catalogue records about this model's gate check, in
    the shape a submission stores it.

    The `unverified` list is the load-bearing part. A gate check that
    TIMED OUT is not a failed check -- it is an absent one, and rows
    produced on that machine may reflect the machine rather than the
    model. Publishing them as a measurement of the model is the specific
    mistake this guard exists to prevent (see the publication-rule
    comment at the top of this module for the live incident).

    Slowness alone is deliberately NOT disqualifying: a model that spills
    into system RAM still produces valid rows, it just takes longer.
    Unverified is the problem; slow is a scheduling inconvenience."""
    from openllm_cbench.core.registry import load_registry, lookup

    entry = lookup(model, load_registry())
    if entry is None:
        return {}

    caveats = list(entry.get("caveats") or [])
    unverified = [c for c in caveats if "could NOT be verified" in c or "did not finish in time" in c]

    separation = entry.get("channel_separation") or {}
    for state, label in sorted(separation.items()):
        if isinstance(label, str) and label.startswith("error:"):
            unverified.append(f"channel separation at {state} ({label})")

    return {
        "catalogued": True,
        "caveats": caveats,
        "unverified": unverified,
        "channel_separation": separation or None,
    }


# --- Integrity -----------------------------------------------------------
#
# A transport can't make data trustworthy. What it can do is make
# corruption and post-review tampering DETECTABLE, which is a different
# and achievable goal: a maintainer who merges a submission should be able
# to show later that what's in the repo is byte-identical to what they
# reviewed, and a contributor should be able to prove a mangled upload
# wasn't what they sent.
#
# Absent checksums are not an error. The submission convention predates
# this, and a hand-assembled submission is still a valid one -- a missing
# manifest just can't be verified, which is a weaker claim than a wrong
# one and is reported as such.


def file_sha256(path, _chunk=1024 * 1024):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


def compute_checksums(folder):
    """{relative posix path -> sha256} for every CSV in a submission
    folder. Relative and posix-style on purpose: the manifest has to
    compare equal after travelling between a Windows contributor and a
    Linux maintainer."""
    folder = Path(folder)
    out = {}
    for suite_dir in SUITE_DIRS:
        d = folder / suite_dir
        if not d.is_dir():
            continue
        for p in sorted(d.glob("*.csv")):
            out[f"{suite_dir}/{p.name}"] = file_sha256(p)
    return out


def _checksum_problems(folder, meta):
    recorded = (meta or {}).get("checksums")
    if not recorded:
        return []
    actual = compute_checksums(folder)
    problems = []
    for rel, digest in sorted(recorded.items()):
        if rel not in actual:
            problems.append(f"checksums list '{rel}' but that file isn't in the submission")
        elif actual[rel] != digest:
            problems.append(
                f"checksum mismatch for '{rel}': the file on disk is not the one that "
                f"was packaged (corrupted in transit, or edited after packaging)"
            )
    for rel in sorted(set(actual) - set(recorded)):
        problems.append(f"'{rel}' is in the submission but missing from checksums "
                         f"(added after packaging?)")
    return problems


# --- Packaging -----------------------------------------------------------
#
# Builds a submission folder from results this machine already produced.
# Makes no network call except one optional, best-effort read of the local
# endpoint's own /api/version -- the same endpoint every suite already
# talks to -- purely to fill in a runtime field a human would otherwise
# type by hand and often get wrong.
#
# Nothing here uploads anything. Packaging and submitting are separate
# commands on purpose: a contributor should be able to produce a
# submission, read exactly what's in it (see PRIVACY_NOTICE), and decide
# about sending it as a second, deliberate act rather than discovering
# both happened at once.


def os_label():
    """OS name and version for a submission's hardware_summary.

    Does NOT use platform.release() alone on Windows: it reports the NT
    release, so Windows 11 comes back as the literal string "10" -- a
    wrong OS version recorded in a provenance field that exists precisely
    so someone can tell two runs apart later. The build number
    disambiguates (11 is build 22000+), and is included regardless so the
    record is specific rather than merely correct."""
    import platform

    system = platform.system()
    if system != "Windows":
        return f"{system} {platform.release()}".strip()
    build = 0
    try:
        import sys
        build = getattr(sys.getwindowsversion(), "build", 0)
    except Exception:
        pass
    name = "Windows 11" if build >= 22000 else f"Windows {platform.release()}"
    return f"{name} (build {build})" if build else name


def _git_user_name():
    try:
        out = subprocess.run(["git", "config", "user.name"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def detect_endpoint_runtime(base_url=None, timeout=5):
    """Best-effort '<runtime> <version>' for the submission's `endpoint`
    field, read from the local endpoint's own /api/version. Returns None
    rather than raising or guessing -- an unreachable endpoint at
    packaging time isn't a reason to refuse to package results that were
    produced when it was reachable."""
    try:
        import requests

        from openllm_cbench.core.endpoint import resolve_base_url
        base = base_url or resolve_base_url()
        resp = requests.get(f"{base.rstrip('/')}/api/version", timeout=timeout)
        resp.raise_for_status()
        version = (resp.json() or {}).get("version")
        return f"ollama {version}" if version else None
    except Exception:
        return None


def detect_cbench_version():
    """The version `cbench --version` prints: the running package's own
    __version__. Not the installed metadata, which an editable install
    freezes when it is installed, so a checkout bumped since then stamped
    its results with an older release. Results that cannot be tied to a
    harness version are much harder to compare later; ARCHITECTURE.md's
    schema-mismatch warnings are about exactly this."""
    from openllm_cbench import __version__
    return __version__


def build_submission_metadata(model, contributor=None, notes="", base_url=None,
                               accept_terms=False):
    """Assembles submission.json's contents from what this machine can
    determine for itself. Every value is either discovered or left empty
    for the contributor to fill -- never invented. An empty required field
    is what makes validate_submission() refuse, which is the intended
    outcome for anything that genuinely couldn't be detected: this
    returns the metadata, it doesn't decide the metadata is good enough."""
    from openllm_cbench.core.hardware import probe
    from openllm_cbench.core.registry import load_registry, lookup

    hw = probe()
    bits = []
    if hw.get("gpu_vendor") and hw.get("gpu_vram_mb"):
        bits.append(f"{hw['gpu_vendor']} {hw['gpu_vram_mb']} MB VRAM")
    if hw.get("system_ram_mb"):
        bits.append(f"{hw['system_ram_mb']} MB system RAM")

    bits.append(os_label())

    entry = lookup(model, load_registry()) or {}
    return {
        "model": model,
        "contributor": contributor or _git_user_name() or "",
        "date": datetime.date.today().isoformat(),
        "hardware_summary": ", ".join(bits),
        "endpoint": detect_endpoint_runtime(base_url) or "",
        "quant": entry.get("quant") or "",
        "cbench_version": detect_cbench_version() or "",
        "notes": notes,
        # Recorded, never asserted: what the gate check found on THIS
        # machine, so a reviewer can tell "the model did that" from "this
        # machine couldn't check". validate_submission() refuses a
        # submission whose gate checks never completed.
        "gate_check": gate_evidence(model),
        "attestation": {
            "accepted": bool(accept_terms),
            "version": ATTESTATION_VERSION,
            "text": ATTESTATION_TEXT,
        },
    }


def package_submission(model, contributor=None, notes="", out_root=None,
                        results_root=None, base_url=None, accept_terms=False):
    """Builds a ready-to-submit folder from CSVs already on disk for
    `model`, and returns (folder_path, info).

    Packages even an invalid submission (e.g. nothing detected for
    `endpoint`) and reports the problems in `info`, rather than refusing:
    a contributor who can see the folder and the specific missing field
    fixes it in one edit, where a hard refusal with no artifact tells them
    much less.

    Copies CSVs -- never moves or rewrites them. The originals stay where
    the suites wrote them, and the copies are byte-identical, which is the
    entire point of submitting raw CSVs rather than a number. The checksum
    manifest is computed from the copies after they land, so it certifies
    what is actually in the folder."""
    from openllm_cbench.core.paths import results_dir
    from openllm_cbench.scoring.aggregate import find_csvs, model_tag

    tag = model_tag(model)
    meta = build_submission_metadata(model, contributor, notes, base_url, accept_terms)
    handle = (meta["contributor"] or "anon").strip().replace(" ", "-").replace("/", "-")

    root = Path(out_root) if out_root else Path.cwd() / "community-results"
    folder = root / tag / f"{handle}_{datetime.date.today().strftime('%Y%m%d')}"
    folder.mkdir(parents=True, exist_ok=True)

    copied = {}
    for suite_dir, prefix in SUITE_DIRS.items():
        src_dir = (Path(results_root) / suite_dir) if results_root else results_dir(suite_dir)
        csvs = find_csvs(src_dir, prefix, tag)
        if not csvs:
            continue
        # find_csvs() already globs only this suite's own per-trial CSVs,
        # but filter again explicitly: a trial_summary or scorecard must
        # never be copied into a submission, and relying on a glob pattern
        # to enforce a publication policy is how policies quietly stop
        # being enforced.
        csvs = [p for p in csvs if not any(m in p.name.lower() for m in VERDICT_MARKERS)]
        if not csvs:
            continue
        dest = folder / suite_dir
        dest.mkdir(parents=True, exist_ok=True)
        for p in csvs:
            shutil.copy2(p, dest / p.name)
        copied[suite_dir] = [p.name for p in csvs]

    meta["checksums"] = compute_checksums(folder)
    (folder / "submission.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    return folder, {
        "copied": copied,
        "metadata": meta,
        "problems": validate_submission(folder),
    }


def zip_submission(folder):
    """Zips a packaged folder next to itself and returns the archive path.
    Exists for the no-git route: a zip can be dragged onto a GitHub issue
    by anyone with an account -- no fork, clone, or command line."""
    folder = Path(folder)
    archive = shutil.make_archive(str(folder), "zip", root_dir=str(folder.parent),
                                   base_dir=folder.name)
    return Path(archive)


def models_with_local_results(results_root=None, known_tags=None):
    """Model tags that actually have S1/S2/S3 CSVs on disk -- i.e. the
    exact set that `cbench community-package` could package right now.

    Exists so a user picks a model from what exists rather than typing a
    tag and finding out afterwards that nothing was produced for it. The
    filename tag is lossy (model_tag() flattens ':' and '/'), so when the
    caller supplies `known_tags` (the endpoint's own model list) each
    filename tag is mapped back to the real tag it came from; anything
    unmatched is returned as-is rather than dropped, since a result for a
    model that has since been deleted locally is still a real result.

    Returns [(model_tag, {suite_dir: csv_count})], sorted."""
    from openllm_cbench.core.paths import results_dir
    from openllm_cbench.scoring.aggregate import model_tag

    found = {}
    for suite_dir, prefix in SUITE_DIRS.items():
        d = (Path(results_root) / suite_dir) if results_root else results_dir(suite_dir)
        if not d.is_dir():
            continue
        for p in d.glob(f"{prefix}_*.csv"):
            if ".INVALID" in p.name:
                continue
            stem = p.name[len(prefix) + 1:]
            # <tag>_<YYYYmmdd>_<HHMMSS>.csv -- strip the two timestamp parts.
            parts = stem.rsplit("_", 2)
            tag = parts[0] if len(parts) == 3 else stem.rsplit(".", 1)[0]
            found.setdefault(tag, {}).setdefault(suite_dir, 0)
            found[tag][suite_dir] += 1

    reverse = {model_tag(t): t for t in (known_tags or [])}
    return sorted(((reverse.get(tag, tag), suites) for tag, suites in found.items()),
                   key=lambda x: x[0])


def list_packaged_submissions(root=None):
    """Submission folders already built under community-results/, newest
    first, each with whatever submission.json says about it.

    So "Submit" can offer a list of things that exist instead of asking
    for a path -- a typed path is a class of error with no upside here."""
    root = Path(root) if root else Path.cwd() / "community-results"
    out = []
    if not root.is_dir():
        return out
    for sub in sorted(root.glob("*/*/submission.json")):
        try:
            meta = json.loads(sub.read_text(encoding="utf-8"))
        except Exception:
            meta = {}
        out.append({
            "path": sub.parent,
            "model": meta.get("model") or sub.parent.parent.name,
            "date": meta.get("date") or "",
            "contributor": meta.get("contributor") or "",
            "accepted": bool((meta.get("attestation") or {}).get("accepted")),
        })
    return sorted(out, key=lambda d: d["date"], reverse=True)
