"""
How much independent information a set of rows actually carries.

THE PROBLEM. Every confidence figure this framework prints is a Wilson
interval, and a Wilson interval assumes n independent observations. The
rows are not independent. One S2 trial is 20 probes across 2 think
states, so each (probe, state) cell appears exactly once per trial and
six trials means each probe was asked six times per state. Twelve rows
for one probe are twelve observations of the same question, not twelve
questions.

Treating them as independent makes the interval too narrow, which makes
the confidence label too confident, which is the one direction this
framework has spent its whole existence trying not to be wrong in. A
model that deterministically fails the same four probes every time was
reporting "high" confidence off 240 rows when it had asked 20 questions.

S1 and S3 have the same shape: 12 tasks across 2 boundary arms, 2
scenarios, repeated per trial.

WHAT THIS DOES, AND WHY IT IS MEASURED RATHER THAN ASSUMED. The correction
is the classic survey-statistics design effect:

    DEFF  = 1 + (m - 1) * ICC
    n_eff = n / DEFF

where m is the observations per cluster and ICC is how strongly rows
within a cluster agree with each other. ICC is estimated FROM THE DATA by
the usual ANOVA estimator, not picked. That matters here, because the two
extremes are both real and the answer should track which one you are in:

  - A model that answers a probe the same way every time has ICC near 1,
    so DEFF is near m and n_eff falls to roughly the NUMBER OF PROBES.
    Asking the same question twelve times told you what one question
    tells you.

  - A model whose answer to a probe varies run to run has ICC near 0, so
    DEFF is near 1 and n_eff stays near n. The repeats are carrying real
    information and the interval should get the credit.

n_eff therefore lands between "number of clusters" and "number of rows",
which is the honest range, and a run that genuinely earned a tight
interval still gets one.

THE ROW IS STILL THE UNIT for the rate itself; see METHODOLOGY_TECHNICAL
section 1.1. This changes only how much confidence that rate is entitled
to, not what the rate is.
"""


def icc_anova(clusters):
    """Intra-cluster correlation for a binary outcome, ANOVA estimator.

    `clusters` is an iterable of (hits, n) per cluster. Returns a float
    clamped to [0, 1] -- a negative estimate means the data are more
    dispersed than independence would predict, which for this purpose is
    indistinguishable from independent, and a value above 1 is not
    meaningful.

    Returns 0.0 when there is nothing to estimate from (fewer than two
    clusters, or every cluster of size 1), which makes the design effect
    1 and leaves the interval exactly as it was. Degrading to "no
    correction" rather than to an arbitrary one keeps this from quietly
    changing numbers it has no evidence about."""
    cl = [(h, n) for h, n in clusters if n > 0]
    k = len(cl)
    total_n = sum(n for _, n in cl)
    if k < 2 or total_n <= k:
        return 0.0

    p = sum(h for h, _ in cl) / total_n
    if p in (0.0, 1.0):
        # Every row agrees. There is no variance to partition, and no
        # evidence that repeats are informative, so treat the clusters as
        # the unit -- the conservative reading.
        return 1.0

    msb = sum(n * ((h / n) - p) ** 2 for h, n in cl) / (k - 1)
    msw = sum(n * (h / n) * (1 - (h / n)) for h, n in cl) / (total_n - k)

    # Mean cluster size, Fleiss-style correction for unequal sizes.
    m0 = (total_n - sum(n * n for _, n in cl) / total_n) / (k - 1)
    if m0 <= 1:
        return 0.0

    denom = msb + (m0 - 1) * msw
    if denom <= 0:
        return 0.0
    return max(0.0, min(1.0, (msb - msw) / denom))


def effective_n(clusters):
    """Returns (n_eff, n, k, icc).

    n_eff is what the rows are worth as independent observations, between
    k (every repeat told you nothing new) and n (every repeat was a fresh
    observation). Rounded to a whole number because it is used as a
    sample size, and never below k or above n."""
    cl = [(h, n) for h, n in clusters if n > 0]
    k = len(cl)
    n = sum(x for _, x in cl)
    if k == 0:
        return 0, 0, 0, 0.0
    if k == n:
        return n, n, k, 0.0

    icc = icc_anova(cl)
    m_bar = n / k
    deff = 1 + (m_bar - 1) * icc
    if deff <= 1:
        return n, n, k, icc
    n_eff = int(round(n / deff))
    return max(k, min(n, n_eff)), n, k, icc


def cluster_counts(rows, cluster_key, hit_pred):
    """Groups rows into (hits, n) per cluster.

    `cluster_key` names the repeated unit -- the probe, the task, the
    scenario. NOT the (unit, arm) pair: two rows sharing a probe but
    differing in think state are still answers to the same question, so
    the conservative grouping is the question."""
    buckets = {}
    for row in rows:
        key = row.get(cluster_key)
        if key is None:
            continue
        hits, n = buckets.get(key, (0, 0))
        buckets[key] = (hits + (1 if hit_pred(row) else 0), n + 1)
    return list(buckets.values())
