"""
Which suites this model can produce a gradeable result for, decided
BEFORE the run rather than after it.

THE PROBLEM THIS FIXES. The gate already measured everything needed to
answer this and then threw the answer away. Three real models, gate-
checked on the machine this was written for:

  - `exaone-deep:7.8b` reports one capability, `completion`. No tools, no
    thinking. The gate caveated the missing tools and said nothing about
    the missing thinking. Three suites were run anyway; all three came
    back INVALID, and the scorecard's own advice was "check tool calling
    works for this model (`cbench gate`)" -- pointing back at the check
    that had already been run and had already known.

  - `llama3.1:8b` reports `completion, tools`. The gate printed "Channel
    check at think=on: skipped (model does not report a thinking
    capability)" and then, four lines later, "**Clean.** No caveats
    found." A skipped check is not a passed one. S2 against this model
    cannot return a CHANNEL_LEAK whatever the model does, because there
    is no separate channel to leak from.

  - a 4B model reports `tools, thinking, completion` and returns an empty
    thinking field on essentially every row. This is the worst of the
    three, because the advertised capability makes it look supported. The
    gate saw `thinking=0 chars` and reported clean.

So the framework's own organising rule -- a suite may not report a null
until it has shown it can produce a positive (METHODOLOGY.md 3.1) -- was
enforced at scoring time, hours after the cost was sunk, using facts that
were available in seconds before it.

For the second and third models the answer is now different from the one
this module first gave. S2 grades two rates, and probe failure needs no
reasoning trace, so both are scored on probe failure with the channel
leak marked not applicable; only S1 and S3 can be predicted INVALID here
(see _channel_verdict). The first model still loses S1 and S3.

WHAT THIS MODULE IS. The one mapping from gate findings to per-suite
verdicts. Deliberately not inline in `gate.py`, because `assess` needs
the identical judgement before it spends an hour, and a second copy of a
rule is how `gate.py`'s hard-coded `<think>` test drifted from the
channel suite's four-delimiter family -- see `check_channel_at`'s comment
for what that cost.

THREE VERDICTS, and the distinction between the last two is the whole
point:

  ready       the suite's own validity guard should pass on this model.
  INVALID     it structurally cannot fire. Running it burns the time and
              produces a row of dashes.
  unverified  a check needed to decide this did not complete. NOT a
              failure, and never a reason to stop -- "the tool call timed
              out" says something about this machine, not about the
              model, and a verdict derived from a stopwatch is worse than
              no verdict.

WHAT THIS DOES NOT DO. It does not decide whether to run anything. It
reports; callers choose. `cbench gate` prints it and still exits on its
own caveats; `cbench assess` stops on an INVALID and takes an override
flag, the same shape as the run-lock guard next to it.
"""

SUITE_LABELS = {
    "s1": "S1 containment",
    "s2": "S2 channel",
    "s3": "S3 persistence",
}

READY = "ready"
INVALID = "invalid"
UNVERIFIED = "unverified"


def _tool_calling_verdict(result):
    """S1 and S3 both need one thing: a real tool call that round-trips.

    S1's positive control fires by the model calling a tool. S3's
    scenario requires the model to write a step to the action log, which
    it does with a tool call -- a row with no log has no step for the
    challenge turn to be about, which is `s3_precondition_met`'s whole
    subject. One requirement, so one function, so they cannot drift."""
    if result.get("has_tools_capability") is None:
        return UNVERIFIED, ("the endpoint's model-info route could not be read, so whether "
                            "this model supports tool calling is unknown")
    if not result.get("has_tools_capability"):
        return INVALID, ("the endpoint does not report a `tools` capability for this model, "
                         "so it cannot make the tool calls this suite scores")
    if result.get("tool_call_timed_out"):
        # Deliberately not INVALID. The endpoint reported the capability
        # and a timeout cannot overrule that -- same doctrine as the
        # caveat wording in gate.run_gate().
        return UNVERIFIED, ("the tool call check did not finish in time on this machine, which "
                            "is a performance finding rather than a capability one; the "
                            "endpoint does report `tools`")
    if result.get("tool_call_no_response"):
        # Same doctrine: no response at all is about the connection.
        return UNVERIFIED, ("the tool call check got no response from the endpoint, which is "
                            "a connection finding rather than a capability one; the "
                            "endpoint does report `tools`")
    if not result.get("tool_call_ok"):
        return INVALID, ("a real tool call did not round-trip: "
                         f"{result.get('tool_call_detail', 'no detail')}")
    return READY, "a real tool call round-tripped correctly"


def _channel_verdict(result):
    """S2 grades two rates, and only one of them needs a reasoning trace.

    Probe failure (the visible answer doing what an attack prompt asked)
    needs only an answer, so S2 can grade any model that answers. The
    channel leak needs a separate reasoning trace to have leaked FROM. So
    S2 is never INVALID here: the reason says which half applies, and
    UNVERIFIED means only that whether a trace comes back is unknown.

    It was INVALID for a model with no trace, matching a scorecard that
    discarded the whole suite in that case. Both moved together (owner
    decision, 2026-09-23): 14 of 31 catalogued models had no reasoning
    channel and were graded without S2 at all. `tests/test_preflight.py`
    asserts this and the scorecard still agree.

    A model that advertises `thinking` and returns an empty trace is the
    case worth the extra code: the capability says the leak half can be
    measured and it cannot, so the reason says so rather than letting a 0%
    leak rate describe the instrument."""
    advertised = result.get("has_thinking_capability")
    if advertised is None:
        return UNVERIFIED, ("the endpoint's model-info route could not be read, so whether "
                            "this model returns a separate reasoning trace is unknown; probe "
                            "failure is measured either way")
    if not advertised:
        return READY, ("no reasoning channel, so S2 grades probe failure only; the channel "
                       "leak does not apply")

    think_on = result.get("channel_think_on")
    checks = [think_on, result.get("channel_think_off")]
    completed = [c for c in checks if c and c.get("ok")]
    if not completed:
        return UNVERIFIED, ("neither channel check completed, so whether a reasoning trace "
                            "actually comes back could not be established on this machine")
    if any((c.get("thinking_len") or 0) > 0 for c in completed):
        return READY, ("a separate reasoning trace came back, so both probe failure and the "
                       "channel leak can be measured")
    # Only the think=on check can show that a trace is NOT delivered: no
    # trace at think=off is what thinking off means. When think=on did not
    # complete (a timeout), an empty think=off result proved nothing, and
    # reading it as "advertised and not delivered" refused S2 over a
    # stopwatch -- the one thing this module says it never does.
    if not (think_on and think_on.get("ok")):
        return UNVERIFIED, ("the think=on channel check did not complete, and no trace at "
                            "think=off is expected, so whether a reasoning trace comes back "
                            "could not be established on this machine")
    return READY, ("the endpoint reports a `thinking` capability but returned an EMPTY "
                   "reasoning trace at every state checked, so the channel leak cannot be "
                   "measured (advertised and not delivered) and S2 grades probe failure only")


# The suites a pre-flight can find unable to run: both need a tool call
# that round-trips. S2 is never INVALID here (see _channel_verdict), so a
# stored S2 verdict of INVALID in an older catalogue entry is out of date.
TOOL_SUITES = ("s1", "s3")


def suite_readiness(result):
    """Maps one `gate.run_gate()` result to {suite: (verdict, reason)}.

    Never raises: a gate result missing a key it expected yields
    UNVERIFIED for whatever depended on it, because "I could not tell" is
    a true statement and a guessed verdict is not."""
    tool_verdict, tool_reason = _tool_calling_verdict(result)
    channel_verdict, channel_reason = _channel_verdict(result)
    return {
        "s1": (tool_verdict, tool_reason),
        "s2": (channel_verdict, channel_reason),
        "s3": (tool_verdict, tool_reason),
    }


def unrunnable(result, suites=("s1", "s2", "s3")):
    """The requested suites that will produce INVALID, in order.

    UNVERIFIED is deliberately absent from this list. A caller acting on
    it -- refusing to start, say -- would turn a slow machine into a
    capability verdict, which is the exact mistake `warm_up()` exists to
    prevent."""
    readiness = suite_readiness(result)
    return [s for s in suites if readiness.get(s, (UNVERIFIED, ""))[0] == INVALID]


def render_readiness(result, suites=("s1", "s2", "s3")):
    """The report block. Returns a list of lines, no trailing newline, so
    it can be spliced into either the gate report or a CLI warning."""
    readiness = suite_readiness(result)
    mark = {READY: "CAN RUN", INVALID: "WILL BE INVALID", UNVERIFIED: "UNVERIFIED"}
    lines = ["## What this model can be scored on", ""]
    for suite in suites:
        verdict, reason = readiness.get(suite, (UNVERIFIED, "not checked"))
        lines.append(f"- **{SUITE_LABELS[suite]}: {mark[verdict]}** ({reason})")
    bad = [s for s in suites if readiness.get(s, (UNVERIFIED, ""))[0] == INVALID]
    if bad:
        lines += ["", _advice(bad, suites)]
    return lines


def _advice(bad, suites):
    """What to do about it, as one sentence a person can act on.

    Says what `cbench score` and `cbench assess` will do about it, since
    they act on this same verdict: skip a suite that cannot measure the
    model and run the rest, refusing only when nothing is left."""
    names = ", ".join(SUITE_LABELS[s] for s in bad)
    good = [s for s in suites if s not in bad]
    if not good:
        return (f"Nothing here would produce a gradeable result: {names} would all come back "
                f"INVALID. `cbench score` and `cbench assess` will refuse this model unless you "
                f"pass `--force-uncheckable`.")
    return (f"{names} would come back INVALID, so `cbench score` and `cbench assess` will skip "
            f"{'it' if len(bad) == 1 else 'them'} and run "
            f"{', '.join(SUITE_LABELS[s] for s in good)}. Read the grade as covering only "
            f"{'that suite' if len(good) == 1 else 'those suites'}.")
