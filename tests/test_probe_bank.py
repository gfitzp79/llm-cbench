"""Invariants for the packaged probe bank.

These are here because writing them caught two real defects on first run:
a canary token that is a substring of another token, and six probes whose
prompt does not contain the string they are scored on. Both are silent --
they do not crash, they produce wrong verdicts on real models.
"""

import json
from collections import Counter

import pytest

from openllm_cbench.core.paths import data_file
from openllm_cbench.scoring.probes import DEEP_CATS, PROBE_CATS

BANK = json.loads(data_file("probes", "eval_prompts.json").read_text(encoding="utf-8"))
SCOREABLE = [p for p in BANK
             if p.get("category") in PROBE_CATS or p.get("category") in DEEP_CATS]
CANARIES = [p for p in BANK if str(p.get("check", "")).startswith("canary:")]


def token_of(p):
    return p["check"].split(":", 1)[1]


def test_every_probe_has_the_fields_the_suite_reads():
    for p in BANK:
        assert p.get("id"), p
        assert p.get("category"), p["id"]
        assert p.get("prompt"), p["id"]


def test_probe_ids_are_unique():
    """A duplicate id silently merges two probes into one cluster, which
    inflates ICC and shrinks n_eff without anything saying so."""
    dupes = [i for i, n in Counter(p["id"] for p in BANK).items() if n > 1]
    assert not dupes, dupes


def test_canary_tokens_are_unique():
    dupes = [t for t, n in Counter(token_of(p) for p in CANARIES).items() if n > 1]
    assert not dupes, dupes


def test_no_canary_token_is_a_substring_of_another():
    """A row is scored against its OWN probe's token, so a substring
    relation means a model emitting the WRONG token satisfies a different
    probe's check and manufactures a failure.

    ZW-3 inside ZW-3C is grandfathered: it ships in the packaged bank and
    predates this test. Renaming either would silently mis-score the
    existing corpus, because re-scoring reads the check from the bank by
    prompt_id -- rows shown ZW-3C would be tested for a token they were
    never given and would all come back a clean PASS. Corrupting real
    data to satisfy a guard is the wrong trade."""
    grandfathered = {("ZW-3", "ZW-3C")}
    tokens = [token_of(p) for p in CANARIES]
    bad = [(a, b) for a in tokens for b in tokens
           if a != b and a in b and (a, b) not in grandfathered]
    assert not bad, bad


def test_a_canary_probe_shows_the_model_the_token_it_is_scored_on():
    """Unless it declares `token_encoded`, in which case obfuscating the
    token IS the technique -- base64, ROT13, reversal, letter-spacing."""
    missing = [p["id"] for p in CANARIES
               if not p.get("token_encoded") and token_of(p) not in p["prompt"]]
    assert not missing, missing


def test_encoded_probes_deliberately_do_not_contain_their_token():
    """The flag must mean something. A probe that declares its token
    encoded and then shows it in plain text is mislabelled, and the
    encoding it claims to test is not being tested."""
    wrong = [p["id"] for p in CANARIES
             if p.get("token_encoded") and token_of(p) in p["prompt"]]
    assert not wrong, wrong


def test_leak_probes_use_the_leak_check():
    for p in BANK:
        if p.get("category") == "leakage_deep":
            assert p.get("check") == "leak", p["id"]


def test_every_deep_probe_declares_a_check():
    """A deep probe with no check falls through score_deep to "unknown
    deep check" -> REVIEW, for every row, forever."""
    for p in BANK:
        if p.get("category") in DEEP_CATS:
            assert p.get("check"), p["id"]


def test_the_bank_is_large_enough_for_the_power_it_claims():
    """n_eff scales with the number of DISTINCT probes, not trials -- at
    20 probes and the measured ICC the ceiling was n_eff 45, which gave a
    real comparison 24% power. The bank was sized from that calculation,
    so a change that shrinks it below the target should fail loudly."""
    assert len(SCOREABLE) >= 100, f"only {len(SCOREABLE)} scoreable probes"


def test_techniques_are_distinct():
    """Diversity is the point, not count. Near-duplicate probes inflate
    ICC and buy no independent observations, so a hundred rewordings of
    one attack would cost the runtime and return almost nothing."""
    techniques = [p["technique"] for p in BANK if p.get("technique")]
    dupes = [t for t, n in Counter(techniques).items() if n > 1]
    assert not dupes, dupes


@pytest.mark.parametrize("category", sorted(set(DEEP_CATS)))
def test_each_deep_category_has_enough_probes_to_cluster(category):
    n = sum(1 for p in BANK if p.get("category") == category)
    assert n >= 10, f"{category} has only {n}"
