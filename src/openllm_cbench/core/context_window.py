"""
Did the context window bind during this run?

WHY THIS EXISTS
===============
`num_predict` and `num_ctx` are next to each other in the options payload
and fail in completely different ways.

`num_predict` FAILS LOUDLY. Hit the cap and the response carries
`done_reason == "length"`, the row is labelled TRUNCATED, it leaves the
denominator, and the aggregate prints a warning telling the operator to
raise it. Every step of that is visible.

`num_ctx` FAILS SILENTLY. Overflow the window and the server does not
error -- it drops tokens off the front of the conversation and answers
anyway. Nothing in the response says it happened. In a multi-turn suite
the thing at the front is the boundary instruction (S1) or the model's
own earlier log entries (S3), which is to say: the exact content the
suite is scoring the model against.

So a too-small window does not produce an error row. It produces a model
that genuinely cannot see the rule it is being judged by, and a scorer
that faithfully records the failure. An instrument failure wearing a
finding's clothes, which is the worst class of defect this project has.

This module is the detector for that, and nothing more. It does not
change any budget, and deliberately does not size one dynamically:
sizing the window to the prompt makes the treatment correlate with the
behaviour under test, since a model that talks more would get a bigger
window than one that does not.

THE COUNT IS CLAMPED, WHICH IS THE WHOLE DIFFICULTY
===================================================
The obvious detector -- "warn when the token count approaches `num_ctx`"
-- does not work, and the reason is worth stating precisely, because the
first version of this module shipped it and it was wrong.

`prompt_eval_count` reports the tokens the server ACTUALLY EVALUATED,
which is the count AFTER any truncation. Measured directly, one 128-token
prompt against shrinking windows (qwen3:0.6b, `num_predict` 32):

    num_ctx   prompt_eval   eval   truncated?
      4096        128         32      no
       512        128         32      no
       256        128         32      no
       128         66         32     YES
        96         50         32     YES
        64         34         32     YES

Truncation does not push the count UP against the window. It pulls the
count DOWN, to roughly half the window. An evicted run therefore looks
*comfortable*: at `num_ctx` 64 the naive check saw "34 of 64, 53% used"
and reported healthy headroom on a prompt that had lost three quarters of
its content.

WHAT IS SOUND, GIVEN THAT
=========================
One observation cannot prove a prompt was NOT truncated, but it can prove
it was, and it can prove the opposite in one direction:

  Below half the window  => provably untruncated. Truncation would have
                            pushed the count UP to about half, so a count
                            well under half cannot be a truncated one.

  At or above half       => cannot be ruled out. A prompt that genuinely
                            fills half the window and one truncated down
                            to half report the same number. The framework
                            does not know which it saw, and says so.

That is the `AT_RISK` verdict. It is deliberately conservative in the one
direction that matters: it never reports "fine" about a run that was
actually truncated. It will sometimes flag a tight-but-honest fit, which
costs an operator a larger `--num-ctx` on the next run and nothing else.

`EVICTED` is the separate, stronger finding: prompt plus generated tokens
reached the window, so the window bound DURING generation regardless of
what the prompt did. In the table above that is the `num_ctx` 64 row,
where 34 + 32 = 66 exceeded the 64-token window.

EXPECT THIS TO BE QUIET ON REAL RUNS
====================================
Across every S1 CSV in this repo, 673 rows with a usable count: the
largest prompt ever sent was 3607 tokens against an 8192 window, median
938, and not one row reached even 4096. On the models measured (0.6b-9b)
the window is not close to binding.

That is why `summary()` reports the peak on EVERY run rather than only on
a warning. This project has already audited detectors that turned out to
have fired zero times in their lifetime and had gone unnoticed, because
silence and absence look identical. A detector that prints "peak 977 of
8192" when all is well is one whose health an operator can see.
"""

# Half the window. Not a comfort margin -- the specific value at which a
# truncated prompt becomes indistinguishable from one that fit, per the
# measurement in this module's docstring. Moving it up trades away
# soundness; moving it down only adds false alarms.
AT_RISK_FRACTION = 0.5

OK = "OK"
AT_RISK = "AT_RISK"
EVICTED = "EVICTED"
UNKNOWN = "UNKNOWN"

# Both columns are load-bearing and neither substitutes for the other:
# `max_prompt_tokens` drives the truncation test (which is about the
# PROMPT against the window), and `peak_context_tokens` drives the
# eviction test (which is about prompt PLUS generation). A suite that
# records only the second cannot answer the first.
CONTEXT_FIELDS = ("max_prompt_tokens", "peak_context_tokens")


def _as_int(value):
    """CSV cells come back as strings, in-memory rows as ints, and a
    missing count as None or "". One coercion so no call site invents its
    own and gets the empty case subtly wrong."""
    if value is None or value == "":
        return None
    try:
        n = int(value)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def prompt_tokens(data):
    """Tokens the server evaluated in the prompt of one /api/chat call.

    Post-truncation, always -- see this module's docstring. That is what
    makes it usable for the one-sided test and useless for the naive one."""
    return data.get("prompt_eval_count")


def call_occupancy(data):
    """Tokens resident at the end of one call: prompt + generated.

    None when the server reported neither count, so an endpoint that omits
    them yields UNKNOWN rather than a confident zero."""
    prompt = data.get("prompt_eval_count")
    generated = data.get("eval_count")
    if prompt is None and generated is None:
        return None
    return (prompt or 0) + (generated or 0)


def peak(values):
    """High-water mark across a scenario's calls, ignoring calls that
    reported nothing."""
    seen = [v for v in values if isinstance(v, int) and v > 0]
    return max(seen) if seen else None


def series_is_cumulative(counts):
    """True if a per-turn `prompt_eval_count` series reports the whole
    prompt each turn rather than only the newly evaluated tokens.

    Load-bearing for multi-turn suites: if the server ever reported only
    the cache miss, the per-turn counts would go small exactly where the
    conversation is longest and the peak would mean nothing. Measured
    across this repo's S1 corpus -- 577 of 577 multi-turn rows are
    monotonically non-decreasing, e.g. [477, 573, 671, 765, 875, 977] --
    and asserted in the tests against that data rather than trusted here."""
    seen = [c for c in counts if isinstance(c, int)]
    return all(b >= a for a, b in zip(seen, seen[1:]))


def headroom_verdict(max_prompt_tokens, peak_context_tokens, num_ctx):
    """EVICTED / AT_RISK / OK / UNKNOWN for one row.

    Checked strongest first: eviction during generation is a fact about
    what happened, while AT_RISK is a statement that the framework cannot
    tell. Reporting the weaker one over the stronger would understate it.

    UNKNOWN never means safe. A row with no counts cannot be cleared
    retroactively, the same way a blank budget column means unknown
    rather than equal."""
    ctx = _as_int(num_ctx)
    prompt = _as_int(max_prompt_tokens)
    occupancy = _as_int(peak_context_tokens)
    if ctx is None or (prompt is None and occupancy is None):
        return UNKNOWN
    if occupancy is not None and occupancy >= ctx:
        return EVICTED
    if prompt is None:
        # Occupancy alone cannot answer the truncation question: it is
        # prompt plus generation, so a small prompt with a long answer and
        # a truncated prompt with a long answer look alike.
        return UNKNOWN
    if prompt >= AT_RISK_FRACTION * ctx:
        return AT_RISK
    return OK


def row_verdict(row):
    return headroom_verdict(row.get("max_prompt_tokens"),
                            row.get("peak_context_tokens"),
                            row.get("num_ctx"))


class HeadroomTally:
    """Streaming accumulator over rows.

    Exists because the suites hold their rows in memory and the aggregate
    layer does not -- it folds thousands of rows from many CSVs into
    counters and never keeps them. Rather than give those two callers two
    checks that could drift apart, both go through this; the list-taking
    helpers below are thin wrappers that build one."""

    def __init__(self):
        self.evicted = 0
        self.at_risk = 0
        self.unknown = 0
        self.ok = 0
        self.peak_prompt = None
        self.peak_context = None
        self._windows = set()

    def add(self, row):
        verdict = row_verdict(row)
        if verdict == EVICTED:
            self.evicted += 1
        elif verdict == AT_RISK:
            self.at_risk += 1
        elif verdict == UNKNOWN:
            self.unknown += 1
        else:
            self.ok += 1

        prompt = _as_int(row.get("max_prompt_tokens"))
        if prompt is not None and (self.peak_prompt is None or prompt > self.peak_prompt):
            self.peak_prompt = prompt
        occupancy = _as_int(row.get("peak_context_tokens"))
        if occupancy is not None and (self.peak_context is None or occupancy > self.peak_context):
            self.peak_context = occupancy
        ctx = _as_int(row.get("num_ctx"))
        if ctx is not None:
            self._windows.add(ctx)
        return verdict

    @property
    def window(self):
        """The window, only when every row agrees on it.

        Rows disagree only when something pooled two runs with different
        windows, which the comparability guard already refuses. Rather
        than pick one and print a misleading percentage, report None and
        let the caller state the counts alone."""
        return next(iter(self._windows)) if len(self._windows) == 1 else None

    @property
    def observed_peak(self):
        """The largest count of either kind, for the headline number."""
        seen = [v for v in (self.peak_prompt, self.peak_context) if v is not None]
        return max(seen) if seen else None

    @property
    def fired(self):
        return bool(self.evicted or self.at_risk)

    def summary(self):
        peak_seen = self.observed_peak
        if peak_seen is None:
            return ("Context window: not measured -- no row recorded a token count "
                    "(rows written before this check, or an endpoint that reports none).")
        ctx = self.window
        if ctx is not None:
            return (f"Context window: peak {peak_seen} of {ctx} tokens "
                    f"({peak_seen / ctx:.0%} of the window).")
        # Two ways to have no single window, and they are not the same
        # problem. Saying "not recorded consistently" about rows that
        # simply predate the column invents a drift that did not happen.
        if not self._windows:
            return (f"Context window: peak {peak_seen} tokens, but no row records the "
                    f"window it ran under, so the headroom cannot be checked. Rows "
                    f"written from this version on carry `num_ctx`.")
        return (f"Context window: peak {peak_seen} tokens across "
                f"{len(self._windows)} different windows "
                f"({', '.join(str(w) for w in sorted(self._windows))}) -- reported as a "
                f"count rather than a percentage, because no single window is the "
                f"denominator for all of these rows.")

    def caveat(self):
        if not self.fired:
            return ""
        ctx = self.window
        window_phrase = f" against a {ctx}-token window" if ctx else ""
        if self.evicted:
            return (
                f"> **[!] The context window bound on {self.evicted} row(s)**"
                f"{window_phrase} -- prompt plus generated tokens reached `num_ctx`. "
                f"Past that the server does not error: it drops tokens off the front of "
                f"the conversation and answers anyway. The front is where the boundary "
                f"instruction and the model's own earlier turns live, so an affected row "
                f"may record the model failing a rule it could no longer see. **Treat "
                f"those verdicts as unusable and re-run with a larger `--num-ctx`.**"
            )
        return (
            f"> **[!] {self.at_risk} row(s) may have been truncated**{window_phrase} -- "
            f"the prompt filled at least half the window. The server reports the token "
            f"count it evaluated AFTER truncating, so a prompt that fit and one cut down "
            f"to fit report the same number and this framework cannot tell them apart. "
            f"Nothing here proves truncation happened; it proves it cannot be ruled out. "
            f"**Re-run with a larger `--num-ctx` before citing these rows** -- if the "
            f"counts stay the same, nothing was truncated and the verdicts stand."
        )


def tally(rows):
    t = HeadroomTally()
    for row in rows:
        t.add(row)
    return t


def headroom_summary(rows):
    """One line stating what the window did, printed whether or not it
    fired. See this module's docstring on why silence is not evidence."""
    return tally(rows).summary()


def headroom_caveat(rows):
    """Markdown warning block for a report, or "" when there is nothing to
    say. Mirrors how the TRUNCATED caveat behaves: speak only when it
    fires, so a warning in a report means something happened."""
    return tally(rows).caveat()
