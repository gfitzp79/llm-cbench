"""Tests for the local progress panel.

The panel's whole justification is that it tiers on rigour rather than
volume, so the tests that matter most are the ones asserting a pile of
shallow work does NOT advance a tier. A board that can be climbed by
running many single-trial scans would fill a crowd-sourced corpus with
results the framework's own rules say are not citable, which is worse
than a smaller corpus because the noise looks like data.
"""

import json

import pytest

from openllm_cbench.core.progress import (
    MIN_CITABLE_TRIALS, compute_progress, is_citable, render_line,
)


def _card(model, statuses=("ok", "ok", "ok"), trials=5):
    s1, s2, s3 = statuses
    def suite(status):
        if status == "ok":
            return {"status": "ok", "rate": 0.1, "n_trials": trials}
        return {"status": status}
    return {"model": model, "grade": "B", "score": 80,
            "suites": {"s1": suite(s1), "s2": suite(s2), "s3": suite(s3)}}


def _registry(*tags):
    return {"models": {t: {} for t in tags}}


def _write_cards(root, cards):
    root.mkdir(parents=True, exist_ok=True)
    for c in cards:
        name = c["model"].replace(":", "-").replace("/", "-")
        (root / f"{name}.json").write_text(json.dumps(c), encoding="utf-8")


# --------------------------------------------------------- is_citable

def test_a_properly_run_model_is_citable():
    assert is_citable(_card("m:1b"), {"m:1b"}) is True


def test_an_uncatalogued_model_is_not_citable():
    """No gate check means the tool-calling and channel gaps that would
    invalidate the run were never looked for."""
    assert is_citable(_card("m:1b"), set()) is False


def test_a_single_trial_run_is_not_citable():
    """THE central case. `quick` depth is deliberately below this
    framework's own pre-registered minimum, and a board that counted it
    would reward running many shallow scans."""
    assert is_citable(_card("m:1b", trials=1), {"m:1b"}) is False
    assert is_citable(_card("m:1b", trials=MIN_CITABLE_TRIALS - 1), {"m:1b"}) is False
    assert is_citable(_card("m:1b", trials=MIN_CITABLE_TRIALS), {"m:1b"}) is True


def test_a_refused_suite_makes_the_whole_result_uncitable():
    """Covers every validity guard at once -- task-set, comparability,
    schema-version and the positive control all surface as an invalid
    suite, so none of them needs its own clause here."""
    assert is_citable(_card("m:1b", statuses=("invalid", "ok", "ok")), {"m:1b"}) is False
    assert is_citable(_card("m:1b", statuses=("ok", "not_run", "ok")), {"m:1b"}) is False


def test_a_bad_grade_is_still_citable():
    """Deliberately not a criterion. A model that scores badly has been
    measured properly; tiering on the grade would reward picking easy
    models, which is the opposite of the point."""
    card = _card("m:1b")
    card["grade"], card["score"] = "F", 5
    assert is_citable(card, {"m:1b"}) is True


@pytest.mark.parametrize("junk", [None, "nope", 42, {}, {"model": "m:1b"}])
def test_malformed_scorecards_are_not_citable(junk):
    assert is_citable(junk, {"m:1b"}) is False


# ------------------------------------------------------------- tiers

def _progress(tmp_path, cards, tags, local=None):
    _write_cards(tmp_path / "scorecards", cards)
    models = [{"name": t} for t in (local if local is not None else tags)]
    return compute_progress(local_models=models, registry=_registry(*tags),
                            scorecards_root=tmp_path / "scorecards")


def test_nothing_done_is_newcomer(tmp_path):
    p = _progress(tmp_path, [], [], local=["m:1b"])
    assert p["tier"] == "newcomer"
    assert p["catalogued"] == 0


def test_catalogued_but_unscored_is_novice(tmp_path):
    p = _progress(tmp_path, [], ["m:1b"])
    assert p["tier"] == "novice"


def test_one_citable_result_is_intermediate(tmp_path):
    p = _progress(tmp_path, [_card("m:1b")], ["m:1b"])
    assert p["tier"] == "intermediate"
    assert p["citable"] == 1


def test_many_shallow_scans_do_not_advance_the_tier(tmp_path):
    """THE test this module exists for. Twelve models gated and scored at
    one trial each is a lot of work and no citable results, and must not
    outrank one model done properly."""
    tags = [f"m{i}:1b" for i in range(12)]
    cards = [_card(t, trials=1) for t in tags]
    p = _progress(tmp_path, cards, tags)
    assert p["scored"] == 12
    assert p["citable"] == 0
    assert p["tier"] == "novice"

    one_done_properly = _progress(tmp_path.with_name(tmp_path.name + "b"),
                                  [_card("m:1b")], ["m:1b"])
    assert one_done_properly["tier"] == "intermediate"


def test_three_citable_results_is_advanced(tmp_path):
    tags = ["a:1b", "b:1b", "c:1b"]
    p = _progress(tmp_path, [_card(t) for t in tags], tags)
    assert p["tier"] == "advanced"


# ------------------------------------------------------------ counts

def test_an_unreachable_endpoint_is_not_reported_as_zero(tmp_path):
    """"We could not ask" and "you have none" are different facts, and
    showing the second for the first would be a lie told every time the
    endpoint was stopped."""
    _write_cards(tmp_path / "scorecards", [_card("m:1b")])
    p = compute_progress(local_models=None, registry=_registry("m:1b"),
                         scorecards_root=tmp_path / "scorecards")
    assert p["local"] is None
    assert "unreachable" in render_line(p)


def test_a_scorecard_for_a_removed_model_is_not_counted(tmp_path):
    """History, not progress."""
    p = _progress(tmp_path, [_card("gone:1b"), _card("here:1b")],
                  ["gone:1b", "here:1b"], local=["here:1b"])
    assert p["scored"] == 1
    assert p["citable_models"] == ["here:1b"]


# ------------------------------------------------------- next action

def test_next_action_points_at_the_gap(tmp_path):
    p = _progress(tmp_path, [], ["m:1b"])
    assert p["next_action"] is not None
    assert "core" in p["next_action"][0].lower() or "Score" in p["next_action"][0]


def test_next_action_names_re_scoring_when_nothing_is_citable(tmp_path):
    p = _progress(tmp_path, [_card("m:1b", trials=1)], ["m:1b"])
    assert p["citable"] == 0
    assert "standard depth" in p["next_action"][0]


def test_next_action_is_none_when_there_is_nothing_obvious(tmp_path):
    tags = ["a:1b", "b:1b", "c:1b"]
    p = _progress(tmp_path, [_card(t) for t in tags], tags)
    assert p["next_action"] is None
