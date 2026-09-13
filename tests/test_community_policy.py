"""
Coverage for the publication rule: raw measurements travel, verdicts do
not, and a submission must record a completed gate check plus accepted
contributor terms.

These are policy guards, not formatting checks. Each one exists because
the alternative is publishing an editorial claim about a named commercial
product on evidence that can't support it -- see core/community.py's
publication-rule comment for the live incident behind the gate guard.
"""

import json

import pytest

from openllm_cbench.core.community import (
    ATTESTATION_TEXT, ATTESTATION_VERSION, gate_evidence, package_submission,
    validate_submission,
)


@pytest.fixture
def results_tree(tmp_path):
    for subdir, prefix in (("s1_containment", "containment"),
                            ("s2_channel", "channel"),
                            ("s3_persistence", "persistence")):
        d = tmp_path / subdir
        d.mkdir(parents=True)
        (d / f"{prefix}_x-1b_20260101_000000.csv").write_text(
            "model,task_id,escape_attempt\nx:1b,fx_lookup,False\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def catalogued_model(tmp_path, monkeypatch):
    """A local catalogue where 'x:1b' has a completed, clean gate check."""
    overlay = tmp_path / "models.json"
    overlay.write_text(json.dumps({"models": {
        "x:1b": {"architecture": "llama", "params_b": 1, "quant": "Q4_K_M",
                 "tools": True, "thinking": True,
                 "channel_separation": {"think_on": "clean", "think_off": "clean"},
                 "config_overrides": {}, "caveats": []},
    }}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))
    return overlay


def _package(tmp_path, results_tree, **kw):
    kw.setdefault("accept_terms", True)
    return package_submission("x:1b", contributor="alice", out_root=tmp_path / "out",
                               results_root=results_tree, **kw)


# --- Guardrail: no verdicts travel --------------------------------------

def test_a_scorecard_in_a_submission_is_refused(tmp_path, results_tree, catalogued_model):
    folder, _ = _package(tmp_path, results_tree)
    assert validate_submission(folder) == []

    (folder / "s1_containment" / "trial_summary_x-1b.md").write_text("# Grade: A", encoding="utf-8")
    problems = validate_submission(folder)
    assert any("verdict, not a measurement" in p for p in problems), problems


def test_a_claimed_grade_in_submission_json_is_refused(tmp_path, results_tree, catalogued_model):
    folder, _ = _package(tmp_path, results_tree)
    meta = json.loads((folder / "submission.json").read_text(encoding="utf-8"))
    meta["grade"] = "A"
    (folder / "submission.json").write_text(json.dumps(meta), encoding="utf-8")

    problems = validate_submission(folder)
    assert any("must not assert a result" in p for p in problems), problems


def test_packaging_never_copies_a_verdict_file(tmp_path, results_tree, catalogued_model):
    # A trial summary sitting in the same results directory as the CSVs
    # must not be swept into the submission.
    (results_tree / "s1_containment" / "trial_summary_x-1b.md").write_text("# Grade: A",
                                                                            encoding="utf-8")
    (results_tree / "s1_containment" / "scorecard_x-1b.csv").write_text("grade\nA\n",
                                                                         encoding="utf-8")
    folder, info = _package(tmp_path, results_tree)

    copied = [n for names in info["copied"].values() for n in names]
    assert not any("trial_summary" in n or "scorecard" in n for n in copied), copied
    assert validate_submission(folder) == []


# --- Guardrail: contributor terms ---------------------------------------

def test_a_submission_without_accepted_terms_is_refused(tmp_path, results_tree, catalogued_model):
    folder, info = _package(tmp_path, results_tree, accept_terms=False)
    assert info["metadata"]["attestation"]["accepted"] is False
    assert any("attestation not accepted" in p for p in info["problems"])


def test_accepted_terms_record_the_text_and_version(tmp_path, results_tree, catalogued_model):
    # The terms a contributor agreed to must be stored with the
    # submission, not merely referenced -- the wording can change later.
    _, info = _package(tmp_path, results_tree)
    att = info["metadata"]["attestation"]
    assert att["accepted"] is True
    assert att["version"] == ATTESTATION_VERSION
    assert att["text"] == ATTESTATION_TEXT


def test_terms_accepted_under_older_wording_are_refused(tmp_path, results_tree, catalogued_model):
    folder, _ = _package(tmp_path, results_tree)
    meta = json.loads((folder / "submission.json").read_text(encoding="utf-8"))
    meta["attestation"]["version"] = ATTESTATION_VERSION - 1
    (folder / "submission.json").write_text(json.dumps(meta), encoding="utf-8")

    assert any("current terms are version" in p for p in validate_submission(folder))


def test_the_terms_cover_what_they_need_to():
    lower = ATTESTATION_TEXT.lower()
    assert "right to share" in lower
    assert "confidential" in lower
    assert "accurate" in lower
    assert "apache-2.0" in lower
    assert "permanently" in lower


# --- Guardrail: a completed gate check ----------------------------------

def test_a_model_with_no_gate_check_cannot_be_submitted(tmp_path, results_tree, monkeypatch):
    # No catalogue entry at all -> nothing establishes the checks ran.
    overlay = tmp_path / "empty.json"
    overlay.write_text(json.dumps({"models": {}}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))

    _, info = _package(tmp_path, results_tree)
    assert any("no gate_check record" in p for p in info["problems"]), info["problems"]


def test_a_timed_out_gate_check_blocks_submission(tmp_path, results_tree, monkeypatch):
    # THE incident this guard exists for: a model too large for the GPU
    # times out, the report reads like a capability failure, and the rows
    # describe the machine rather than the model.
    overlay = tmp_path / "models.json"
    overlay.write_text(json.dumps({"models": {
        "x:1b": {"architecture": "llama", "params_b": 1, "quant": "Q4_K_M",
                 "tools": True, "thinking": True,
                 "channel_separation": {"think_on": "error: Read timed out",
                                         "think_off": "clean"},
                 "config_overrides": {}, "caveats": []},
    }}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))

    _, info = _package(tmp_path, results_tree)
    assert any("did not complete on this machine" in p for p in info["problems"]), info["problems"]
    assert any("think_on" in p for p in info["problems"])


def test_slowness_alone_does_not_block_submission(tmp_path, results_tree, monkeypatch):
    # A model that spills into system RAM still produces valid rows. Only
    # an UNVERIFIED check is disqualifying; slow is a scheduling problem.
    overlay = tmp_path / "models.json"
    overlay.write_text(json.dumps({"models": {
        "x:1b": {"architecture": "llama", "params_b": 1, "quant": "Q4_K_M",
                 "tools": True, "thinking": True,
                 "channel_separation": {"think_on": "clean", "think_off": "clean"},
                 "config_overrides": {},
                 "caveats": ["SLOW: a one-token generation took 17.5s, which usually means "
                              "this model doesn't fit in available VRAM."]},
    }}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))

    folder, info = _package(tmp_path, results_tree)
    assert info["problems"] == [], info["problems"]
    # ...but the slowness is still recorded for a reviewer to see.
    assert any("SLOW" in c for c in info["metadata"]["gate_check"]["caveats"])


def test_gate_evidence_reports_nothing_for_an_unknown_model(monkeypatch, tmp_path):
    overlay = tmp_path / "empty.json"
    overlay.write_text(json.dumps({"models": {}}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))
    assert gate_evidence("never-seen:1b") == {}


# --- The gh-available path ----------------------------------------------

def test_submit_runs_end_to_end_when_gh_is_available(tmp_path, results_tree,
                                                      catalogued_model, monkeypatch, capsys):
    """Covers `cbench community-submit` on a machine that HAS `gh`.

    This path shipped with a NameError -- `copied` was used but never
    assigned -- and nothing caught it, because every test and every manual
    run happened on a machine without `gh` installed, so the "no gh"
    branch returned first every time. The broken path was the one that
    only runs for people who actually have the tool: i.e. every real
    contributor.
    """
    import openllm_cbench.cli as cli
    import openllm_cbench.core.community_submit as cs

    folder, _ = _package(tmp_path, results_tree)
    monkeypatch.setattr(cs, "detect_gh", lambda: (True, "`gh` is installed and authenticated"))

    # Preview only: no --confirm, so nothing may execute.
    def explode(*a, **k):
        raise AssertionError("execute_plan must not run without --confirm")
    monkeypatch.setattr(cs, "execute_plan", explode)

    rc = cli._cmd_community_submit([str(folder)])
    out = capsys.readouterr().out

    assert rc == 0
    assert "gh repo fork" in out
    assert "Nothing sent" in out
    assert "raw CSVs only" in out
    # The plan must describe what's actually in the folder, not "(none)".
    assert "s1_containment" in out or "Suites included" not in out


def test_submit_refuses_a_submission_that_fails_policy(tmp_path, results_tree,
                                                        catalogued_model, capsys):
    import openllm_cbench.cli as cli

    folder, _ = _package(tmp_path, results_tree, accept_terms=False)
    rc = cli._cmd_community_submit([str(folder)])
    out = capsys.readouterr().out

    assert rc == 1
    assert "Not submitting" in out
    assert "attestation not accepted" in out
