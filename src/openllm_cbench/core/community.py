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
    "is in these rows. A pull request publishes it permanently -- git history "
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

    problems += _checksum_problems(path, meta)
    return problems


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
                f"checksum mismatch for '{rel}' -- the file on disk is not the one that "
                f"was packaged (corrupted in transit, or edited after packaging)"
            )
    for rel in sorted(set(actual) - set(recorded)):
        problems.append(f"'{rel}' is in the submission but missing from checksums -- "
                         f"added after packaging?")
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
    """Installed package version if there is one, else the git commit this
    is running from, else None. Results that can't be tied to a harness
    version are much harder to compare later -- ARCHITECTURE.md's own
    schema-mismatch warnings are exactly about this -- so it tries hard
    before giving up."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        try:
            return version("openllm-cbench")
        except PackageNotFoundError:
            pass
    except Exception:
        pass
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5,
                             cwd=str(Path(__file__).resolve().parent))
        return out.stdout.strip() or None
    except Exception:
        return None


def build_submission_metadata(model, contributor=None, notes="", base_url=None):
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
    }


def package_submission(model, contributor=None, notes="", out_root=None,
                        results_root=None, base_url=None):
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
    meta = build_submission_metadata(model, contributor, notes, base_url)
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
