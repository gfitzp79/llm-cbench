"""
Opening a community submission as a real pull request, without this
framework ever touching a credential.

WHY THIS SHELLS OUT TO `gh` AND NOTHING ELSE
============================================
Uploading a submission means authenticating as the contributor to a
service this project doesn't control. There are three ways to do that and
only one of them is acceptable here:

  1. Ask for a personal access token and push with it. Rejected: it means
     this tool handles a credential that grants write access to every
     repository that user can reach, for the sake of adding one folder.
     No amount of care makes that a good trade.
  2. Run a hosted upload endpoint of our own. Rejected: it contradicts
     this framework's entire posture (see the safety invariant -- the only
     host a suite can reach is a loopback canary), makes the maintainer a
     custodian of other people's data, and replaces a reviewable pull
     request with a trust-us pipeline.
  3. Delegate to `gh`, GitHub's own CLI, which the contributor has already
     authenticated themselves, out of band, under their own control.
     This is what we do. This module never sees, stores, or transmits a
     token; it builds a list of `gh`/`git` commands and runs them.

If `gh` isn't installed or isn't authenticated, this does NOT degrade into
some lesser upload -- it prints the exact manual steps instead, including
a prefilled issue URL for the no-git route. A contributor who can't or
won't install `gh` is not a contributor we drop.

WHY THE PLAN IS BUILT SEPARATELY FROM RUNNING IT
================================================
`build_submit_plan()` is pure: it returns the exact argv of every command
that would run, and nothing executes until `execute_plan()` is called with
them. That's the same discipline the TUI already follows (show the real
`cbench ...` invocation before launching it) applied to an action with
much higher stakes -- this one creates public content under someone's own
name. It also means the whole sequence is testable without a network, a
GitHub account, or `gh` installed.

WHY IT WORKS IN A TEMPORARY CLONE
=================================
Never in the contributor's own working tree. Their checkout may have
uncommitted work, be on a branch mid-rebase, or be a fork three months
behind -- none of which should be able to turn into a surprise commit or
push. A shallow clone into a temp directory means the worst case for a
failed submission is a leftover temp folder.
"""

import json
import shutil
import subprocess
import urllib.parse
from pathlib import Path

# The project a submission goes to. Overridable per-invocation (a fork of
# this framework should point at its own), but defaulted so a contributor
# doesn't have to know it.
UPSTREAM_REPO = "gfitzp79/llm-cbench"


def detect_gh():
    """Returns (available, detail). `available` is True only if `gh` is
    both installed AND authenticated -- an installed-but-logged-out `gh`
    is not usable here and saying so plainly beats failing four commands
    later with something cryptic."""
    if not shutil.which("gh"):
        return False, "the GitHub CLI (`gh`) is not installed"
    try:
        out = subprocess.run(["gh", "auth", "status"], capture_output=True,
                             text=True, timeout=15)
    except Exception as e:
        return False, f"could not run `gh auth status`: {e}"
    if out.returncode != 0:
        return False, "`gh` is installed but not logged in (run `gh auth login`)"
    return True, "`gh` is installed and authenticated"


def _branch_name(meta):
    from openllm_cbench.scoring.aggregate import model_tag
    tag = model_tag(meta.get("model", "unknown"))
    handle = (meta.get("contributor") or "anon").strip().replace(" ", "-").replace("/", "-")
    date = (meta.get("date") or "").replace("-", "")
    return f"submission/{tag}-{handle}-{date}"


def suites_in_folder(folder):
    """{suite_dir -> [csv names]} actually present in a packaged folder.

    Exists because `community-submit` is given a path, not the packaging
    step's own return value -- it has to read what's there rather than be
    told. Without this the generated PR body claimed "Suites included:
    (none)" for a submission carrying 39 CSVs, which is exactly the kind
    of wrong-but-plausible metadata a reviewer would have to catch by
    hand."""
    from openllm_cbench.core.community import SUITE_DIRS

    folder = Path(folder)
    out = {}
    for suite_dir in SUITE_DIRS:
        d = folder / suite_dir
        if d.is_dir():
            names = sorted(p.name for p in d.glob("*.csv") if ".INVALID" not in p.name)
            if names:
                out[suite_dir] = names
    return out


def pr_title_and_body(meta, copied):
    if copied:
        suites = ", ".join(f"{k} ({len(v)} CSV{'s' if len(v) != 1 else ''})"
                            for k, v in sorted(copied.items()))
    else:
        suites = "(none)"
    title = f"Community results: {meta.get('model', 'unknown')}"
    body = (
        f"Raw trial CSVs for `{meta.get('model')}`, submitted per "
        f"`community-results/README.md`.\n\n"
        f"- Suites included: {suites}\n"
        f"- Hardware: {meta.get('hardware_summary') or 'not stated'}\n"
        f"- Runtime: {meta.get('endpoint') or 'not stated'}\n"
        f"- Harness version: {meta.get('cbench_version') or 'not stated'}\n\n"
        f"Notes: {meta.get('notes') or '(none)'}\n\n"
        f"Validated locally with `cbench community-validate` before opening. "
        f"No score is claimed here: these are the raw CSVs so the scorecard "
        f"can be regenerated independently with "
        f"`cbench score --model {meta.get('model')} --from-existing`."
    )
    return title, body


def build_submit_plan(folder, meta, copied, repo=UPSTREAM_REPO, workdir="<tmp>"):
    """The exact command sequence a submission would run, as a list of
    {argv, why} steps. Pure -- builds strings, touches nothing. Callers
    print this for confirmation before anything executes."""
    branch = _branch_name(meta)
    title, body = pr_title_and_body(meta, copied)
    folder = Path(folder)
    return [
        {"argv": ["gh", "repo", "fork", repo, "--clone=false", "--remote=false"],
         "why": f"ensure you have a fork of {repo} to push to (no-op if you already do, "
                f"and skipped automatically if you own the repo)"},
        {"argv": ["gh", "repo", "clone", repo, workdir, "--", "--depth", "1"],
         "why": "shallow-clone into a temp directory; your own checkout is never touched"},
        {"argv": ["git", "-C", workdir, "checkout", "-b", branch],
         "why": f"branch for this submission"},
        {"argv": ["<copy>", str(folder), f"{workdir}/community-results/..."],
         "why": "copy the packaged submission folder into the clone"},
        {"argv": ["git", "-C", workdir, "add", "community-results"],
         "why": "stage only the submission folder, nothing else"},
        {"argv": ["git", "-C", workdir, "commit", "-m", title],
         "why": "commit it"},
        {"argv": ["git", "-C", workdir, "push", "-u", "origin", branch],
         "why": "push the branch to your fork"},
        {"argv": ["gh", "pr", "create", "--repo", repo, "--title", title,
                  "--body", body],
         "why": f"open the pull request against {repo}"},
    ]


def manual_instructions(folder, meta, repo=UPSTREAM_REPO, zip_path=None, copied=None):
    """What to tell someone who has no `gh`. Two routes, because they suit
    different people: a real fork-and-PR for anyone comfortable with git,
    and an issue-with-an-attachment for anyone who isn't -- that second
    one needs nothing but a GitHub account and a browser, and a maintainer
    can move the files into the repo from there."""
    # Read the folder when the caller didn't say what's in it -- the issue
    # body is the only description a maintainer gets on this route, so
    # "(none)" for a submission full of CSVs is worse than useless.
    if copied is None:
        copied = suites_in_folder(folder)
    title, body = pr_title_and_body(meta, copied)
    issue_url = (f"https://github.com/{repo}/issues/new?"
                 + urllib.parse.urlencode({"title": title, "body": body}))
    lines = [
        "Route A: fork and pull request (needs git and a GitHub account)",
        f"  1. Fork https://github.com/{repo} in your browser.",
        "  2. Clone your fork, then copy this folder into it, preserving the path:",
        f"       {folder}",
        "     -> community-results/<model-tag>/<handle>_<date>/",
        "  3. git add community-results && git commit && git push",
        "  4. Open the PR against the upstream repo.",
        "",
        "Route B: attach to an issue (needs only a GitHub account and a browser)",
    ]
    if zip_path:
        lines.append(f"  1. This zip is ready to attach: {zip_path}")
    else:
        lines.append("  1. Re-run `cbench community-package` with --zip to get an "
                     "attachable archive.")
    lines += [
        "  2. Open this prefilled issue and drag the zip onto it:",
        f"       {issue_url}",
        "",
        f"Either way a maintainer re-runs `cbench score --model {meta.get('model')} "
        f"--from-existing` against your CSVs before anything is trusted (see "
        f"community-results/README.md).",
    ]
    return "\n".join(lines)


def execute_plan(folder, meta, copied, repo=UPSTREAM_REPO, runner=None, echo=print):
    """Actually submits. Only ever called after an explicit --confirm.

    `runner` exists so tests can assert the whole sequence without a
    network, a GitHub account, or `gh` installed. Returns (ok, message).
    """
    import tempfile

    runner = runner or (lambda argv, **kw: subprocess.run(
        argv, capture_output=True, text=True, timeout=300, **kw))
    branch = _branch_name(meta)
    title, body = pr_title_and_body(meta, copied)
    folder = Path(folder)

    tmp = Path(tempfile.mkdtemp(prefix="cbench-submit-"))
    clone = tmp / "repo"
    try:
        # A fork is only needed when you don't own the repo; gh fails
        # harmlessly if you do, so its failure is never fatal here.
        echo(f"$ gh repo fork {repo} --clone=false --remote=false")
        runner(["gh", "repo", "fork", repo, "--clone=false", "--remote=false"])

        for argv in (
            ["gh", "repo", "clone", repo, str(clone), "--", "--depth", "1"],
            ["git", "-C", str(clone), "checkout", "-b", branch],
        ):
            echo(f"$ {' '.join(argv)}")
            out = runner(argv)
            if out.returncode != 0:
                return False, f"failed: {' '.join(argv)}\n{out.stderr or out.stdout}"

        dest = clone / "community-results" / folder.parent.name / folder.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(folder, dest)
        echo(f"copied submission into {dest.relative_to(clone)}")

        for argv in (
            ["git", "-C", str(clone), "add", "community-results"],
            ["git", "-C", str(clone), "commit", "-m", title],
            ["git", "-C", str(clone), "push", "-u", "origin", branch],
            ["gh", "pr", "create", "--repo", repo, "--title", title, "--body", body,
             "--head", branch],
        ):
            echo(f"$ {' '.join(argv[:6])}{' ...' if len(argv) > 6 else ''}")
            out = runner(argv, cwd=str(clone) if argv[0] == "gh" else None)
            if out.returncode != 0:
                return False, f"failed: {' '.join(argv[:4])}\n{out.stderr or out.stdout}"
            if argv[0] == "gh" and "pr" in argv:
                return True, (out.stdout or "").strip()
        return True, "submitted"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def load_submission_metadata(folder):
    try:
        return json.loads((Path(folder) / "submission.json").read_text(encoding="utf-8"))
    except Exception:
        return {}
