"""
Parity harness: proves the scoring functions produce consistent output
on fixed fixtures (see tests/fixtures/*.json, and the generation commands
recorded in each fixture file's sibling comment below). This exists so a
prose/docstring rewrite elsewhere in this codebase can be aggressive without
risk -- if a threshold or heuristic ever drifts, this fails before anything
else does.

Fixtures were generated from reference data and checked into the repo.
This file does not depend on external code, only the checked-in JSON.
Run with: pytest tests/
"""

import json
from pathlib import Path

import pytest

from openllm_cbench.scoring.containment_metrics import (
    classify_target,
    host_of,
    extract_shell_urls,
    extract_remote_transfer_hosts,
    extract_package_manager_targets,
    is_payload_bearing,
    infer_canary_port,
    fisher_exact_two_sided,
    model_tag,
)
from openllm_cbench.scoring.probes import score, score_deep

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


# --- containment_metrics.py fixtures ------------------------------------

@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["classify_target"])
def test_classify_target_parity(case):
    if case.get("raised"):
        pytest.fail(f"Original raised on {case['input']!r} -- extracted version must too "
                    f"(never observed in the original; if this fires, the fixture itself "
                    f"needs regenerating, not this test relaxed).")
    got_class, got_target = classify_target(case["input"])
    assert got_class == case["class"]
    assert got_target == case["target"]


def test_host_of_always_returns_something_hashable():
    # A model can emit a tool call whose `url` argument is a dict/list
    # rather than a string -- already a known live-loop failure mode
    # (test_safety_invariant.py), recorded verbatim into the CSV's
    # tool_log. host_of()'s except-branch used to hand the original
    # object back, and its caller collects results into a set, so
    # `cbench extension-rule` crashed with "unhashable type: 'dict'" on
    # real CSVs. Every return value must be hashable, whatever goes in.
    for bad_url in ({"nested": "dict"}, ["a", "list"], 123, None):
        got = host_of(bad_url)
        assert isinstance(got, str)
        assert {got}  # the actual failing operation: putting it in a set


def test_host_of_still_parses_a_normal_url():
    assert host_of("http://internal-fx-service:8080/quote") == "internal-fx-service:8080"
    assert host_of("example.com") == "example.com"


@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["extract_shell_urls"])
def test_extract_shell_urls_parity(case):
    assert extract_shell_urls(case["input"]) == case["output"]


@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["extract_remote_transfer_hosts"])
def test_extract_remote_transfer_hosts_parity(case):
    assert extract_remote_transfer_hosts(case["input"]) == case["output"]


@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["extract_package_manager_targets"])
def test_extract_package_manager_targets_parity(case):
    assert extract_package_manager_targets(case["input"]) == case["output"]


@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["is_payload_bearing"])
def test_is_payload_bearing_parity(case):
    assert is_payload_bearing(case["input"]) == case["output"]


# infer_canary_port's fixture rows contain nested JSON strings that don't
# round-trip cleanly through parametrize -- covered directly instead of via
# the fixture file (the fixture still exists and was checked by eye
# against these same three cases when generated).

def test_infer_canary_port_with_hit():
    rows = [{"tool_log": json.dumps([{"tool": "http_request",
                                       "args": {"url": "http://127.0.0.1:54321/internal-api"},
                                       "result": "authorized_hit"}])}]
    assert infer_canary_port(rows) == 54321


def test_infer_canary_port_without_hit():
    rows = [{"tool_log": json.dumps([{"tool": "read_local_file", "args": {}, "result": "ok"}])}]
    assert infer_canary_port(rows) is None


def test_infer_canary_port_malformed_and_empty():
    assert infer_canary_port([{"tool_log": "not json"}]) is None
    assert infer_canary_port([]) is None


@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["fisher_exact_two_sided"])
def test_fisher_exact_two_sided_parity(case):
    got = fisher_exact_two_sided(*case["input"])
    want = case["output"]
    if want != want:  # NaN
        assert got != got
    else:
        assert got == pytest.approx(want)


@pytest.mark.parametrize("case", _load("containment_metrics_parity.json")["model_tag"])
def test_model_tag_parity(case):
    assert model_tag(case["input"]) == case["output"]


# --- probes.py fixtures ---------------------------------------------------

@pytest.mark.parametrize("case", _load("probes_parity.json")["score"])
def test_score_parity(case):
    got = list(score(case["cat"], case["text"], case["has_full"]))
    assert got == case["output"]


@pytest.mark.parametrize("case", _load("probes_parity.json")["score_deep"])
def test_score_deep_parity(case):
    got = list(score_deep(case["check"], case["text"], case["has_full"]))
    assert got == case["output"]
