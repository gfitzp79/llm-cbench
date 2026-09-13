"""
Coverage for packaging and submitting a community result -- no network, no
model, no `gh`, no GitHub account.

The submit path is tested through build_submit_plan()/execute_plan()'s own
injected runner rather than by running anything: the whole point of
building the command sequence as data first is that it can be asserted
without side effects.
"""

import json

import pytest

from openllm_cbench.core.community import (
    PRIVACY_NOTICE, compute_checksums, package_submission, validate_submission,
    zip_submission,
)
from openllm_cbench.core.community_submit import (
    build_submit_plan, load_submission_metadata, manual_instructions, pr_title_and_body,
)


@pytest.fixture
def results_tree(tmp_path):
    """A fake results/ tree with one real-looking CSV per suite for
    'x:1b', shaped the way find_csvs() globs for them."""
    for subdir, prefix in (("s1_containment", "containment"),
                            ("s2_channel", "channel"),
                            ("s3_persistence", "persistence")):
        d = tmp_path / subdir
        d.mkdir(parents=True)
        (d / f"{prefix}_x-1b_20260101_000000.csv").write_text(
            "model,task_id,escape_attempt\nx:1b,fx_lookup,False\n", encoding="utf-8")
    return tmp_path


def test_package_copies_every_suites_csvs(tmp_path, results_tree):
    folder, info = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    assert folder.is_dir()
    assert set(info["copied"]) == {"s1_containment", "s2_channel", "s3_persistence"}
    for suite_dir in info["copied"]:
        assert (folder / suite_dir).is_dir()
        assert list((folder / suite_dir).glob("*.csv"))


def test_package_leaves_the_original_csvs_untouched(tmp_path, results_tree):
    original = (results_tree / "s1_containment" / "containment_x-1b_20260101_000000.csv")
    before = original.read_bytes()
    package_submission("x:1b", contributor="alice", out_root=tmp_path / "out",
                        results_root=results_tree)
    assert original.exists(), "packaging must copy, never move"
    assert original.read_bytes() == before


def test_package_records_a_checksum_per_csv(tmp_path, results_tree):
    folder, info = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    checksums = info["metadata"]["checksums"]
    assert len(checksums) == 3
    assert all(len(v) == 64 for v in checksums.values()), "sha256 hex digests"
    # Paths are relative and posix-style so the manifest survives travelling
    # between a Windows contributor and a Linux maintainer.
    assert all(k.startswith(("s1_", "s2_", "s3_")) and "\\" not in k for k in checksums)
    assert checksums == compute_checksums(folder)


def test_validate_catches_a_csv_edited_after_packaging(tmp_path, results_tree):
    folder, _ = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    assert validate_submission(folder) == []

    victim = next((folder / "s1_containment").glob("*.csv"))
    victim.write_text("model,task_id,escape_attempt\nx:1b,fx_lookup,True\n", encoding="utf-8")

    problems = validate_submission(folder)
    assert any("checksum mismatch" in p for p in problems), problems


def test_validate_catches_a_csv_added_after_packaging(tmp_path, results_tree):
    folder, _ = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    (folder / "s1_containment" / "containment_x-1b_20260102_000000.csv").write_text(
        "model,task_id,escape_attempt\nx:1b,other,False\n", encoding="utf-8")

    problems = validate_submission(folder)
    assert any("missing from checksums" in p for p in problems), problems


def test_validate_still_accepts_a_submission_with_no_checksums(tmp_path, results_tree):
    # The submission convention predates the manifest, and a hand-assembled
    # folder is still valid -- unverifiable is a weaker claim than wrong.
    folder, _ = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    meta = json.loads((folder / "submission.json").read_text(encoding="utf-8"))
    del meta["checksums"]
    (folder / "submission.json").write_text(json.dumps(meta), encoding="utf-8")

    assert validate_submission(folder) == []


def test_package_never_invents_a_value_it_could_not_detect(tmp_path, results_tree, monkeypatch):
    # An undetectable endpoint must come back empty so validate_submission()
    # refuses it -- guessing would produce a confidently-wrong provenance
    # field, which is worse than an obviously-missing one.
    import openllm_cbench.core.community as community
    monkeypatch.setattr(community, "detect_endpoint_runtime", lambda *a, **k: None)

    folder, info = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    assert info["metadata"]["endpoint"] == ""
    assert any("endpoint" in p for p in info["problems"])


def test_package_with_no_csvs_reports_rather_than_crashing(tmp_path):
    folder, info = package_submission(
        "nothing:1b", contributor="alice", out_root=tmp_path / "out", results_root=tmp_path)
    assert info["copied"] == {}
    assert any("nothing here to score" in p for p in info["problems"])


def test_zip_submission_produces_an_attachable_archive(tmp_path, results_tree):
    folder, _ = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)
    archive = zip_submission(folder)
    assert archive.exists() and archive.suffix == ".zip"
    assert archive.stat().st_size > 0


def test_privacy_notice_names_the_actual_hazard():
    # Not boilerplate: it has to say what's in the data and that publishing
    # is permanent, or it doesn't do its job.
    lower = PRIVACY_NOTICE.lower()
    assert "raw output" in lower
    assert "permanently" in lower or "permanent" in lower
    assert "git history" in lower


# --- submit (plan only -- nothing executes) ------------------------------

def _meta():
    return {"model": "x:1b", "contributor": "alice", "date": "2026-09-13",
            "hardware_summary": "nvidia 8192 MB VRAM", "endpoint": "ollama 0.5.1",
            "cbench_version": "abc1234", "notes": ""}


def test_submit_plan_never_contains_a_credential_flag():
    # The whole reason this shells out to `gh` is that authentication stays
    # entirely outside this project. A plan step carrying a token would
    # mean that decision had been quietly reversed.
    plan = build_submit_plan("/tmp/sub", _meta(), {})
    flat = " ".join(" ".join(str(a) for a in step["argv"]) for step in plan).lower()
    for forbidden in ("--token", "-p ", "password", "GITHUB_TOKEN".lower(), "ghp_"):
        assert forbidden not in flat


def test_submit_plan_pushes_to_a_fork_and_targets_upstream():
    plan = build_submit_plan("/tmp/sub", _meta(), {}, repo="owner/repo")
    argvs = [step["argv"] for step in plan]
    assert ["gh", "repo", "fork", "owner/repo", "--clone=false", "--remote=false"] in argvs
    assert any(a[:3] == ["gh", "pr", "create"] and "owner/repo" in a for a in argvs)


def test_submit_plan_works_in_a_temp_clone_not_the_users_tree():
    # A contributor's own checkout may have uncommitted work or be
    # mid-rebase; a submission must never be able to commit into it.
    plan = build_submit_plan("/tmp/sub", _meta(), {}, workdir="/tmp/clone")
    git_steps = [s["argv"] for s in plan if s["argv"][0] == "git"]
    assert git_steps, "expected git steps"
    for argv in git_steps:
        assert argv[1] == "-C" and argv[2] == "/tmp/clone"


def test_pr_body_claims_no_score_and_points_at_recomputation():
    _, body = pr_title_and_body(_meta(), {"s1_containment": ["a.csv"]})
    assert "No score is claimed" in body
    assert "--from-existing" in body


def test_manual_instructions_offer_a_route_that_needs_no_git():
    text = manual_instructions("/tmp/sub", _meta(), repo="owner/repo", zip_path="/tmp/sub.zip")
    assert "Route A" in text and "Route B" in text
    assert "issues/new" in text
    assert "/tmp/sub.zip" in text


def test_load_submission_metadata_degrades_on_a_missing_file(tmp_path):
    assert load_submission_metadata(tmp_path) == {}


def test_submit_describes_the_suites_actually_in_the_folder(tmp_path, results_tree):
    # Found live: `community-submit` is handed a path, not the packaging
    # step's return value, so it described a 39-CSV submission as
    # "Suites included: (none)" in the PR/issue body -- wrong-but-plausible
    # metadata a reviewer would have had to catch by hand.
    from openllm_cbench.core.community_submit import suites_in_folder

    folder, _ = package_submission(
        "x:1b", contributor="alice", out_root=tmp_path / "out", results_root=results_tree)

    found = suites_in_folder(folder)
    assert set(found) == {"s1_containment", "s2_channel", "s3_persistence"}

    _, body = pr_title_and_body(_meta(), found)
    suites_line = next(l for l in body.splitlines() if l.startswith("- Suites included:"))
    assert "(none)" not in suites_line
    assert "s1_containment" in suites_line

    # The no-`gh` route reads the folder for itself when not told.
    text = manual_instructions(folder, _meta(), repo="owner/repo")
    assert "s1_containment" in text or "s1_containment" in urlunquote(text)


def urlunquote(s):
    import urllib.parse
    return urllib.parse.unquote_plus(s)
