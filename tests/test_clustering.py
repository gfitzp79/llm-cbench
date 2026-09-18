"""Tests for the effective sample size used by every confidence label.

The rows this framework produces are clustered: one S2 trial asks 20
probes across 2 think states, so each (probe, state) cell appears exactly
once per trial and N trials means each probe was asked N times per state.
A Wilson interval on the row count treats twelve observations of one
probe as twelve probes, which makes the interval too narrow and the
confidence label too confident.

The correction is measured, not assumed, and the tests below pin both
ends of that: a model that answers deterministically must lose the
repeats, and a model whose answers genuinely vary must keep them.
"""

import pytest

from openllm_cbench.scoring.clustering import cluster_counts, effective_n, icc_anova


def test_deterministic_repeats_are_worth_one_observation_each():
    """THE case. Twenty probes asked twelve times each, same answer every
    time: the run is worth twenty observations, not two hundred and
    forty."""
    clusters = [(12, 12)] * 4 + [(0, 12)] * 16
    n_eff, n, k, icc = effective_n(clusters)
    assert n == 240 and k == 20
    assert n_eff == 20
    assert icc == pytest.approx(1.0)


def test_variable_answers_keep_their_rows():
    """The negative case, and the reason this is measured rather than a
    flat division by the number of probes. If a model's answer to a probe
    genuinely varies run to run, the repeats carry real information and
    the interval should get the credit."""
    clusters = [(6, 12)] * 20
    n_eff, n, k, icc = effective_n(clusters)
    assert n_eff == 240
    assert icc == pytest.approx(0.0)


def test_partial_agreement_lands_between_the_two():
    clusters = [(12, 12)] * 2 + [(9, 12)] * 3 + [(3, 12)] * 5 + [(0, 12)] * 10
    n_eff, n, k, _ = effective_n(clusters)
    assert k < n_eff < n


def test_no_repeats_changes_nothing():
    """One observation per cluster is already independent, so the
    correction must be inert rather than inventing a penalty."""
    n_eff, n, k, icc = effective_n([(1, 1), (0, 1), (1, 1), (0, 1)])
    assert n_eff == n == 4
    assert icc == 0.0


def test_unanimous_rows_are_treated_as_the_clusters():
    """Every row agreeing gives no variance to partition and no evidence
    that the repeats told you anything, so the conservative reading is
    that the clusters are the unit."""
    n_eff, n, k, icc = effective_n([(12, 12)] * 5)
    assert icc == 1.0
    assert n_eff == k == 5


@pytest.mark.parametrize("clusters", [[], [(0, 0)], [(5, 12)]])
def test_degenerate_inputs_do_not_raise(clusters):
    n_eff, n, k, icc = effective_n(clusters)
    assert n_eff <= max(n, 0)


def test_icc_is_clamped_to_a_meaningful_range():
    # Over-dispersed: more spread than independence predicts. A negative
    # ICC is not meaningful here and must read as "no evidence of
    # clustering" rather than widening or narrowing anything.
    assert icc_anova([(6, 12), (6, 12), (6, 12)]) >= 0.0
    assert icc_anova([(12, 12), (0, 12)]) <= 1.0


def test_effective_n_never_leaves_the_honest_range():
    for clusters in ([(12, 12)] * 4 + [(0, 12)] * 16,
                     [(6, 12)] * 20,
                     [(1, 12)] * 7 + [(11, 12)] * 3):
        n_eff, n, k, _ = effective_n(clusters)
        assert k <= n_eff <= n, (n_eff, k, n)


def test_cluster_counts_groups_by_the_question_not_the_condition():
    """Two rows sharing a probe but differing in think state are still
    answers to the same question, so they belong to one cluster."""
    rows = [
        {"prompt_id": "p1", "think_label": "on", "v": "FAIL"},
        {"prompt_id": "p1", "think_label": "off", "v": "FAIL"},
        {"prompt_id": "p2", "think_label": "on", "v": "PASS"},
        {"prompt_id": "p2", "think_label": "off", "v": "FAIL"},
    ]
    counts = sorted(cluster_counts(rows, "prompt_id", lambda r: r["v"] == "FAIL"))
    assert counts == [(1, 2), (2, 2)]


def test_rows_without_a_cluster_key_are_skipped():
    rows = [{"prompt_id": None, "v": "FAIL"}, {"prompt_id": "p", "v": "FAIL"}]
    assert cluster_counts(rows, "prompt_id", lambda r: r["v"] == "FAIL") == [(1, 1)]


# ---------------------------------------------------- confidence label

def test_confidence_label_uses_the_effective_size():
    from openllm_cbench.scoring.scorecard import confidence_label
    # 240 rows at a 50% rate looks tight; 20 questions does not.
    assert confidence_label(120, 240) == "high"
    assert confidence_label(120, 240, effective=20) in ("low", "medium")


def test_confidence_label_is_unchanged_without_an_effective_size():
    from openllm_cbench.scoring.scorecard import confidence_label
    assert confidence_label(120, 240) == confidence_label(120, 240, effective=None)
