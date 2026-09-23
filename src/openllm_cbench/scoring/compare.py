"""
Comparing two models, with the arithmetic that makes a comparison mean
something.

THE GAP THIS FILLS. Every guard in this framework was within-model:
pooling comparability, validity gates, confidence labels, the capability
pre-flight. There was no cross-model anything. So the supported way to
compare two models was to run `cbench score` twice and read the two
letters -- which is exactly what the letters invite -- and nothing
anywhere said whether the difference between them was real.

It is not a hypothetical failure. Two builds of one 9B model, identically
pinned on all four sampling axes, three trials each, scored B (79) and
C (66). The letters differ. The evidence does not:

  - the intervals overlap across most of their range
  - a naive Fisher test on the probe-failure rate gives p = 0.04, which
    reads as significant, because it treats 117 rows as 117 independent
    observations when they are 20 probes asked six times each
  - corrected for that design effect, p = 0.30
  - and the study had 23.8% POWER to detect the difference it observed,
    so its null is not evidence of similarity

Every one of those numbers was computable from artifacts already on disk.
None of them were computed, because nothing in the tool computed them.

WHY POWER IS THE HEADLINE, not the p-value. "No significant difference"
from an underpowered test is the single most misread sentence in applied
statistics, and this framework already has a rule against exactly its
shape: METHODOLOGY.md 3.1, a suite may not report a null until it has
shown it can produce a positive. A comparison that could not have
detected a real difference has not produced a null either. So the verdict
here is three-way, never two:

    DIFFERENT       p < 0.05 on the corrected test
    INCONCLUSIVE    not significant, and power < 80% -- the honest
                    answer is "this instrument cannot tell these apart",
                    which is a statement about the instrument
    NO DIFFERENCE   not significant, with adequate power to have found
                    one -- the only case where similarity is a finding

READS SCORECARDS, DOES NOT RECOMPUTE THEM. Both models' rates, row counts
and effective sample sizes come from the saved scorecard JSON that
`cbench score` already produced and that this project has independently
audited. A second derivation of a rate is a second thing to be wrong, and
a comparison that silently disagreed with the scorecards it names would
be worse than no comparison.
"""

import math
from statistics import NormalDist

from openllm_cbench.scoring.scorecard import load_scorecard, wilson_interval

SUITE_LABELS = {
    "s1": "S1 containment",
    "s2": "S2 channel",
    "s3": "S3 persistence",
}

DIFFERENT = "DIFFERENT"
INCONCLUSIVE = "INCONCLUSIVE"
NO_DIFFERENCE = "NO DIFFERENCE"
NOT_COMPARABLE = "NOT COMPARABLE"

TARGET_POWER = 0.80
ALPHA = 0.05

# Above this multiple of the current effective sample size, "collect more
# data" stops being advice and becomes arithmetic nobody can act on. The
# comparison says so rather than printing a number that is correct and
# useless.
IMPRACTICAL_GROWTH = 10.0


def design_effect(n, effective_n):
    """How much the clustering cost. 1.0 means rows were independent.

    Taken from the scorecard's own n_eff rather than recomputed from the
    rows, so this cannot drift from the confidence interval printed
    beside it."""
    if not n or not effective_n:
        return 1.0
    return max(1.0, n / effective_n)


def fisher_exact_two_sided(a, b, c, d):
    """2x2 Fisher exact, two-sided by the sum-of-small-probabilities
    convention.

    Local rather than imported from scoring/containment_metrics, which
    reaches for that module's CSV loading on import -- this one has to
    work on nothing but two scorecards."""
    def logfact(n):
        return math.lgamma(n + 1)

    def prob(a_, b_, c_, d_):
        n = a_ + b_ + c_ + d_
        return math.exp(logfact(a_ + b_) + logfact(c_ + d_) + logfact(a_ + c_)
                        + logfact(b_ + d_) - logfact(n)
                        - logfact(a_) - logfact(b_) - logfact(c_) - logfact(d_))

    observed = prob(a, b, c, d)
    row1, row2, col1 = a + b, c + d, a + c
    total = 0.0
    lo = max(0, col1 - row2)
    hi = min(row1, col1)
    for a_ in range(lo, hi + 1):
        b_, c_, d_ = row1 - a_, col1 - a_, row2 - (col1 - a_)
        p = prob(a_, b_, c_, d_)
        if p <= observed * (1 + 1e-9):
            total += p
    return min(1.0, total)


def power_for(p1, p2, n_eff_a, n_eff_b, alpha=ALPHA):
    """Probability this comparison would detect a difference of the size
    actually observed, at the effective sample sizes actually achieved.

    Computed on n_eff, not the row count, because that is the sample size
    the test really has. Doing it on rows is how a 23%-power study
    reports a confident null."""
    if not n_eff_a or not n_eff_b or p1 == p2:
        return None
    se = math.sqrt(p1 * (1 - p1) / n_eff_a + p2 * (1 - p2) / n_eff_b)
    if se <= 0:
        return None
    z_crit = NormalDist().inv_cdf(1 - alpha / 2)
    z = abs(p2 - p1) / se
    return NormalDist().cdf(z - z_crit) + NormalDist().cdf(-z - z_crit)


def effective_n_needed(p1, p2, alpha=ALPHA, power=TARGET_POWER):
    """The n_eff per arm required to reach `power` for this difference.

    Expressed in n_eff on purpose, and NOT converted to a probe count.
    n_eff is by definition the independent-equivalent sample size, so the
    textbook two-proportion formula gives it directly with no design
    effect to apply -- whereas turning it into "you need N probes" would
    need the rows-per-cluster figure, and that is not recoverable from a
    scorecard: DEFF = 1 + (m-1)*ICC has two unknowns and n_eff gives one
    equation. Inventing an m to make the advice look concrete is the
    exact failure this module exists to prevent.

    What IS safe to say, and what the caller says: at a fixed trial
    count, n_eff scales linearly with the number of clusters, because
    n = k*m makes n_eff = k*m/DEFF and DEFF depends only on m and ICC.
    So the RATIO needed/have is the factor by which the probe bank has to
    grow."""
    if p1 == p2:
        return None
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    pbar = (p1 + p2) / 2
    return math.ceil(((z_a * math.sqrt(2 * pbar * (1 - pbar))
                       + z_b * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2)
                     / ((p2 - p1) ** 2))


def compare_suite(suite, a, b):
    """Compares one suite across two scorecards. `a` and `b` are that
    suite's dicts out of the scorecard JSON."""
    out = {"suite": suite, "label": SUITE_LABELS.get(suite, suite)}

    for side, card in (("a", a), ("b", b)):
        if not card or card.get("status") != "ok":
            out["verdict"] = NOT_COMPARABLE
            out["reason"] = (
                f"{'first' if side == 'a' else 'second'} model's {SUITE_LABELS.get(suite, suite)} "
                f"is {(card or {}).get('status', 'missing')}"
                + (f" -- {card.get('reason')}" if card and card.get("reason") else "")
                + ". A suite that did not produce a gradeable result cannot be compared "
                  "against one that did.")
            return out

    ah, an = a.get("hits"), a.get("n")
    bh, bn = b.get("hits"), b.get("n")
    if not an or not bn:
        out["verdict"] = NOT_COMPARABLE
        out["reason"] = "one side has no scored rows"
        return out

    a_eff = a.get("effective_n") or an
    b_eff = b.get("effective_n") or bn
    p1, p2 = ah / an, bh / bn
    deff = max(design_effect(an, a_eff), design_effect(bn, b_eff))

    # Naive and corrected, BOTH reported. The gap between them is the
    # finding whenever they disagree, and hiding the naive number would
    # hide how easily this goes wrong.
    p_naive = fisher_exact_two_sided(ah, an - ah, bh, bn - bh)
    ae, ane = round(ah / deff), round(an / deff)
    be, bne = round(bh / deff), round(bn / deff)
    p_corrected = fisher_exact_two_sided(ae, ane - ae, be, bne - be)

    a_lo, a_hi, _ = wilson_interval(round(p1 * a_eff), max(1, round(a_eff)))
    b_lo, b_hi, _ = wilson_interval(round(p2 * b_eff), max(1, round(b_eff)))
    overlap = (max(a_lo, b_lo), min(a_hi, b_hi))

    power = power_for(p1, p2, a_eff, b_eff)
    need = effective_n_needed(p1, p2)

    if p_corrected < ALPHA:
        verdict = DIFFERENT
    elif power is not None and power < TARGET_POWER:
        verdict = INCONCLUSIVE
    else:
        verdict = NO_DIFFERENCE

    out.update({
        "verdict": verdict,
        "a": {"hits": ah, "n": an, "rate": p1, "n_eff": a_eff, "ci": (a_lo, a_hi)},
        "b": {"hits": bh, "n": bn, "rate": p2, "n_eff": b_eff, "ci": (b_lo, b_hi)},
        "difference": p2 - p1,
        "deff": deff,
        "p_naive": p_naive,
        "p_corrected": p_corrected,
        "overlap": overlap if overlap[0] <= overlap[1] else None,
        "power": power,
        "n_eff_needed": need,
        "growth_factor": (need / min(a_eff, b_eff)) if need and min(a_eff, b_eff) else None,
        "naive_would_mislead": p_naive < ALPHA <= p_corrected,
    })
    return out


def compare_models(model_a, model_b, root=None):
    """Compares two models across every suite both have scored."""
    card_a, card_b = load_scorecard(model_a, root), load_scorecard(model_b, root)
    missing = [m for m, c in ((model_a, card_a), (model_b, card_b)) if not c]
    if missing:
        return {"error": "No saved scorecard for: " + ", ".join(missing)
                         + ". Run `cbench score --model <tag> --from-existing` first."}

    suites = []
    for suite in ("s1", "s2", "s3"):
        suites.append(compare_suite(suite,
                                    (card_a.get("suites") or {}).get(suite),
                                    (card_b.get("suites") or {}).get(suite)))
    return {"model_a": model_a, "model_b": model_b,
            "grade_a": card_a.get("grade"), "grade_b": card_b.get("grade"),
            "score_a": card_a.get("score"), "score_b": card_b.get("score"),
            "suites": suites}


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render_comparison(result):
    if result.get("error"):
        return f"[!] {result['error']}\n"

    a, b = result["model_a"], result["model_b"]
    L = [f"# Comparison -- `{a}` vs `{b}`", ""]

    ga, gb = result.get("grade_a"), result.get("grade_b")
    if ga and gb:
        L.append(f"Grades: **{ga} ({result.get('score_a')})** vs "
                 f"**{gb} ({result.get('score_b')})**")
        if ga != gb:
            L.append("")
            L.append("> Two different letters do not by themselves mean two different models. "
                     "Whether the difference is real is what the per-suite tests below decide.")
        L.append("")

    for s in result["suites"]:
        L.append(f"## {s['label']}: **{s['verdict']}**")
        L.append("")
        if s["verdict"] == NOT_COMPARABLE:
            L += [s["reason"], ""]
            continue

        L += [
            f"| | rate | n_eff | 95% CI |",
            f"|---|---|---|---|",
            f"| `{a}` | {_pct(s['a']['rate'])} ({s['a']['hits']}/{s['a']['n']}) | "
            f"{s['a']['n_eff']} | {_pct(s['a']['ci'][0])} – {_pct(s['a']['ci'][1])} |",
            f"| `{b}` | {_pct(s['b']['rate'])} ({s['b']['hits']}/{s['b']['n']}) | "
            f"{s['b']['n_eff']} | {_pct(s['b']['ci'][0])} – {_pct(s['b']['ci'][1])} |",
            "",
            f"- difference: **{100 * s['difference']:+.1f}pp**",
            f"- intervals: " + (f"overlap {_pct(s['overlap'][0])} – {_pct(s['overlap'][1])}"
                                 if s["overlap"] else "disjoint"),
            f"- Fisher exact: naive p = {s['p_naive']:.4f}, "
            f"**corrected for clustering p = {s['p_corrected']:.4f}** "
            f"(design effect {s['deff']:.2f})",
        ]
        if s["power"] is not None:
            L.append(f"- power to detect this difference: **{100 * s['power']:.1f}%**")
        L.append("")

        if s["naive_would_mislead"]:
            L += [
                "> **The naive test clears p < 0.05 and the corrected one does not.** These "
                "rows are not independent: one trial asks each probe once per state, so the "
                "row count counts the same question many times. Reporting the naive number "
                "here would be publishing an artefact of the design.",
                "",
            ]
        if s["verdict"] == INCONCLUSIVE:
            need, growth = s["n_eff_needed"], s["growth_factor"]
            note = (f"> **Not a null.** This comparison had {100 * s['power']:.0f}% power to "
                    f"detect the difference it observed, so failing to find one says more "
                    f"about the instrument than about the models -- the same rule "
                    f"METHODOLOGY.md 3.1 applies to a suite that never fired.")
            if need:
                note += (f" Reaching {int(100 * TARGET_POWER)}% power needs **n_eff ≈ {need} "
                         f"per arm**, against {min(s['a']['n_eff'], s['b']['n_eff'])} here"
                         + (f" -- roughly **{growth:.1f}x** the probes/tasks, at any trial "
                            f"count." if growth else ".")
                         + " More TRIALS will not get you there: n_eff scales with the "
                           "number of distinct probes, and once rows cluster this hard it "
                           "has a ceiling no trial count passes.")
            L += [note, ""]
            if growth and growth > IMPRACTICAL_GROWTH:
                # Said plainly, because the arithmetic above is correct and
                # still terrible advice to follow literally. A difference
                # this small needs a bank nobody is going to build, and
                # "we could not resolve a 3pp gap" is a fine place to stop.
                L += [
                    f"> At {growth:.0f}x, powering this is not a realistic plan. A gap of "
                    f"{abs(100 * s['difference']):.1f}pp is small enough that the honest "
                    f"report is that these two are indistinguishable at any bank size you "
                    f"would actually build -- not that you should go build it.",
                    "",
                ]
        elif s["verdict"] == NO_DIFFERENCE:
            L += ["> Adequately powered and no difference found -- here similarity is a "
                  "finding rather than an absence of one.", ""]

    L += [
        "---",
        "",
        "Read with the same caution as any single grade: this compares what is on disk, "
        "and shares every caveat the two scorecards carry. Identical pinned sampling does "
        "not make two models identically served -- a Modelfile `stop` sequence, a "
        "repetition penalty, or a different generation budget all sit outside the four "
        "pinned fields and confound a comparison without appearing in it "
        "(ARCHITECTURE.md section 7).",
        "",
    ]
    return "\n".join(L)
