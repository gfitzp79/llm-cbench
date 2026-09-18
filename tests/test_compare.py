"""Tests for comparing two models.

The incident: two builds of one 9B model, identically pinned on all four
sampling axes, three trials each, scored B (79) and C (66). Nothing in
the tool compared them, so the supported way to read that was the two
letters -- and the letters differ where the evidence does not. A naive
Fisher test on the probe-failure rate gives p = 0.04; corrected for the
fact that 117 rows are 20 probes asked six times each, p = 0.30; and the
comparison had 24% power, so its null was never evidence of similarity
either.
"""

import pytest

from openllm_cbench.scoring import compare as C


def suite(hits, n, n_eff, status="ok"):
    return {"status": status, "hits": hits, "n": n, "effective_n": n_eff,
            "rate": (hits / n) if n else 0.0}


# ------------------------------------------------- the real pair

ORNITH_S2 = suite(25, 117, 37)
ARATAN_S2 = suite(40, 117, 37)


def test_the_real_pair_is_inconclusive_not_different():
    r = C.compare_suite("s2", ORNITH_S2, ARATAN_S2)
    assert r["verdict"] == C.INCONCLUSIVE
    assert r["p_naive"] < 0.05, "the naive test clears significance -- that is the trap"
    assert r["p_corrected"] > 0.20, "correcting for clustering removes it"
    assert r["naive_would_mislead"] is True
    assert r["power"] < 0.30


def test_the_naive_and_corrected_tests_are_both_reported():
    """Hiding the naive number would hide how easily this goes wrong, and
    the gap between them is the finding."""
    r = C.compare_suite("s2", ORNITH_S2, ARATAN_S2)
    assert r["p_naive"] is not None and r["p_corrected"] is not None
    assert r["p_corrected"] > r["p_naive"]
    assert r["deff"] == pytest.approx(117 / 37, rel=1e-6)


def test_an_underpowered_null_is_never_reported_as_no_difference():
    """THE RULE THIS FILE EXISTS FOR. METHODOLOGY.md 3.1 says a suite may
    not report a null until it has shown it can produce a positive. A
    comparison that could not have detected a real difference has not
    produced a null either."""
    r = C.compare_suite("s2", ORNITH_S2, ARATAN_S2)
    assert r["verdict"] != C.NO_DIFFERENCE
    text = C.render_comparison({"model_a": "a", "model_b": "b", "suites": [r]})
    assert "Not a null" in text
    assert "power" in text.lower()


# ------------------------------------------------ the three verdicts

def test_a_real_difference_is_called_different():
    """Large, clean separation at a sample size that can see it."""
    r = C.compare_suite("s1", suite(5, 200, 200), suite(120, 200, 200))
    assert r["verdict"] == C.DIFFERENT
    assert r["p_corrected"] < 0.05


def test_a_well_powered_null_is_a_finding():
    """The only case where similarity means something."""
    r = C.compare_suite("s1", suite(300, 1000, 1000), suite(310, 1000, 1000))
    # Near-identical rates at a large n_eff: not significant, and powered
    # enough that not-significant is informative.
    assert r["verdict"] in (C.NO_DIFFERENCE, C.INCONCLUSIVE)
    if r["verdict"] == C.NO_DIFFERENCE:
        text = C.render_comparison({"model_a": "a", "model_b": "b", "suites": [r]})
        assert "similarity is a finding" in text


def test_a_suite_that_could_not_be_graded_is_not_comparable():
    """An INVALID suite has no rate. Comparing it against one that does
    would manufacture a difference out of a missing measurement -- the
    same error the capability pre-flight exists to stop earlier."""
    invalid = {"status": "invalid", "reason": "no row returned a reasoning trace"}
    r = C.compare_suite("s2", invalid, ARATAN_S2)
    assert r["verdict"] == C.NOT_COMPARABLE
    assert "no row returned a reasoning trace" in r["reason"]

    r2 = C.compare_suite("s2", ORNITH_S2, {"status": "not_run"})
    assert r2["verdict"] == C.NOT_COMPARABLE


# --------------------------------------------------- the arithmetic

def test_design_effect_comes_from_the_scorecards_own_n_eff():
    """Not recomputed from rows: a comparison that silently disagreed
    with the confidence interval printed beside it would be worse than no
    comparison."""
    assert C.design_effect(117, 37) == pytest.approx(117 / 37)
    assert C.design_effect(117, 117) == 1.0
    assert C.design_effect(50, 100) == 1.0, "never below 1 -- n_eff cannot exceed n usefully"
    assert C.design_effect(0, 0) == 1.0


def test_required_sample_is_expressed_in_n_eff_not_invented_probes():
    """DEFF = 1 + (m-1)*ICC has two unknowns and a scorecard gives one
    equation, so rows-per-cluster is NOT recoverable. Reporting a probe
    count would mean inventing one."""
    need = C.effective_n_needed(0.214, 0.342)
    assert 150 < need < 260, need
    assert C.effective_n_needed(0.3, 0.3) is None, "no difference, nothing to power for"


def test_power_uses_effective_not_raw_sample_size():
    """Computing power on the row count is how a 24%-power study reports
    a confident null."""
    on_eff = C.power_for(0.214, 0.342, 37, 37)
    on_rows = C.power_for(0.214, 0.342, 117, 117)
    assert on_eff < 0.30 < on_rows


def test_fisher_matches_a_known_table():
    """Fisher's own tea-tasting 2x2: two-sided p = 0.00276 (the widely
    quoted 0.0014 is the ONE-sided value)."""
    assert C.fisher_exact_two_sided(1, 9, 11, 3) == pytest.approx(0.002759, abs=1e-5)
    assert C.fisher_exact_two_sided(10, 10, 10, 10) == pytest.approx(1.0)


@pytest.mark.parametrize("cell", [
    (25, 92, 40, 77), (10, 48, 12, 46), (1, 4, 0, 2),
    (0, 10, 10, 0), (5, 5, 5, 5), (3, 100, 7, 100),
])
def test_this_fisher_agrees_with_the_one_already_in_the_codebase(cell):
    """The local copy exists only so this module needs nothing but two
    scorecards -- scoring/containment_metrics reaches for CSV loading on
    import. A local copy that DISAGREED would be far worse than the
    import it avoids, so they are pinned to each other."""
    from openllm_cbench.scoring.containment_metrics import fisher_exact_two_sided as theirs

    assert C.fisher_exact_two_sided(*cell) == pytest.approx(theirs(*cell), rel=1e-9)


# ------------------------------------------------------- rendering

def test_an_impractical_growth_factor_says_so():
    """An 81x figure is arithmetically right and terrible advice. The
    report has to say the difference is too small to chase rather than
    print a number nobody can act on."""
    r = C.compare_suite("s1", suite(10, 58, 26), suite(12, 58, 25))
    assert r["verdict"] == C.INCONCLUSIVE
    assert r["growth_factor"] > C.IMPRACTICAL_GROWTH
    text = C.render_comparison({"model_a": "a", "model_b": "b", "suites": [r]})
    assert "not a realistic plan" in text
    assert "indistinguishable" in text


def test_differing_grades_carry_a_warning():
    text = C.render_comparison({
        "model_a": "a", "model_b": "b", "grade_a": "B", "grade_b": "C",
        "score_a": 79, "score_b": 66,
        "suites": [C.compare_suite("s2", ORNITH_S2, ARATAN_S2)]})
    assert "do not by themselves mean two different models" in text


def test_the_serving_config_caveat_is_always_present():
    """Identical pinned sampling does not make two models identically
    served, and the comparison cannot see the difference."""
    text = C.render_comparison({"model_a": "a", "model_b": "b", "suites": []})
    assert "stop" in text and "section 7" in text


def test_a_missing_scorecard_is_an_error_not_a_comparison(tmp_path):
    r = C.compare_models("nope:1b", "also-nope:1b", str(tmp_path))
    assert "error" in r
    assert "cbench score" in r["error"]
    assert "[!]" in C.render_comparison(r)


# ------------------------------------------------------------ CLI

def test_cli_requires_exactly_two_models(monkeypatch, capsys):
    import sys

    from openllm_cbench import cli
    monkeypatch.setattr(sys, "argv", ["cbench", "compare", "--model", "only-one:1b"])
    assert cli.main() == 2
    assert "exactly twice" in capsys.readouterr().err


def test_cli_refuses_to_compare_a_model_with_itself(monkeypatch, capsys):
    import sys

    from openllm_cbench import cli
    monkeypatch.setattr(sys, "argv",
                        ["cbench", "compare", "--model", "x:1b", "--model", "x:1b"])
    assert cli.main() == 2
    assert "same tag" in capsys.readouterr().err
