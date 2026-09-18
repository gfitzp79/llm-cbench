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
                            "is a performance finding rather than a capability one -- the "
                            "endpoint does report `tools`")
    if not result.get("tool_call_ok"):
        return INVALID, ("a real tool call did not round-trip: "
                         f"{result.get('tool_call_detail', 'no detail')}")
    return READY, "a real tool call round-tripped correctly"


def _channel_verdict(result):
    """S2 needs a separate reasoning trace to have leaked FROM.

    Two ways to fail, and the second is the one worth the extra code. A
    model that never advertises `thinking` is at least honest about it.
    A model that advertises `thinking` and returns an empty one looks
    supported right up until the scorecard reports a 0% leak rate that is
    a property of the instrument.

    NOTE, and it matters if S2's scoring changes. S2 grades two rates,
    and only the leak rate needs a trace -- `scoring/capability.py`'s
    `s2_could_detect_a_leak` governs that one alone, while the probe
    failure rate stays measurable on a model that returns no trace at
    all. The scorecard currently discards the whole suite when the leak
    rate is ungradeable, so that is what this reports, because this
    module's job is to predict what the run will actually produce. If
    that scoring decision is revisited, this verdict must move with it --
    `tests/test_preflight.py` asserts the two agree."""
    advertised = result.get("has_thinking_capability")
    if advertised is None:
        return UNVERIFIED, ("the endpoint's model-info route could not be read, so whether "
                            "this model returns a separate reasoning trace is unknown")
    if not advertised:
        return INVALID, ("the endpoint does not report a `thinking` capability, so there is no "
                         "separate reasoning channel for a CHANNEL_LEAK to be found in -- a 0% "
                         "leak rate here would describe the instrument, not the model")

    checks = [result.get("channel_think_on"), result.get("channel_think_off")]
    completed = [c for c in checks if c and c.get("ok")]
    if not completed:
        return UNVERIFIED, ("neither channel check completed, so whether a reasoning trace "
                            "actually comes back could not be established on this machine")
    if any((c.get("thinking_len") or 0) > 0 for c in completed):
        return READY, "a separate reasoning trace came back, so a leak could be detected"
    return INVALID, ("the endpoint reports a `thinking` capability but returned an EMPTY "
                     "reasoning trace at every state checked -- the capability is advertised "
                     "and not delivered, which reads as support right up until the leak rate "
                     "comes back 0% for want of anything to measure")


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
        lines.append(f"- **{SUITE_LABELS[suite]}: {mark[verdict]}** -- {reason}")
    bad = [s for s in suites if readiness.get(s, (UNVERIFIED, ""))[0] == INVALID]
    if bad:
        lines += ["", _advice(bad, suites)]
    return lines


def _advice(bad, suites):
    """What to do about it, as one sentence a person can act on.

    Names the narrowed command when something is still worth running,
    because "two of your three suites are dead" is much less useful than
    the flags that skip them."""
    names = ", ".join(SUITE_LABELS[s] for s in bad)
    good = [s for s in suites if s not in bad]
    if not good:
        return (f"Nothing here would produce a gradeable result: {names} would all come back "
                f"INVALID. `cbench assess` will refuse this model unless you pass "
                f"`--force-uncheckable`.")
    return (f"{names} would come back INVALID and the time spent on it is wasted. Run the rest "
            f"with `--suites {','.join(good)}`, and read any resulting grade as covering only "
            f"those suites.")
