"""
What an exit code means, in words.

WHY THIS EXISTS
===============
Every action screen in the TUI ended its output with:

    exit code: 2

in red, as the last thing on screen. That is the number the process
returned, which is true and tells a user nothing. Reported live: "exit
code 2 means nothing". The reason had been printed ten lines earlier and
scrolled past, so the final and most prominent line was the least
informative one.

This module is the vocabulary. The CLI's own convention, read off every
`return` site in cli.py, is:

    0  the thing was done
    1  it ran, and something in it failed
    2  it REFUSED TO START -- bad input, or a pre-flight that found the
       run could not produce a usable result

The 1/2 split is the load-bearing part, and it is not cosmetic. A 1 means
there is a result to look at and something in it went wrong. A 2 means
nothing ran and nothing was written, so there is no artifact to inspect
and the fix is upstream of the run. Presenting those as two shades of
"failed" loses the only distinction that tells a user what to do next.
"""

SUCCESS = 0
FAILED = 1
REFUSED = 2

# Deliberately not a per-subcommand table. The reason a run was refused is
# already printed in full by the command itself; this only has to stop the
# final line from contradicting it by saying nothing.
_MEANING = {
    SUCCESS: "finished successfully",
    FAILED: "ran, but something in it failed; the output above says what",
    REFUSED: ("did not start: nothing ran and nothing was written. The reason is "
              "printed above; fix that and run again"),
}


def describe(returncode, subcommand=None):
    """One human sentence for a process exit code.

    `subcommand` sharpens a case where the generic sentence would mislead.
    It must stay true for EVERY way that subcommand can produce the code:
    score/assess once named one refusal reason (no suite gradeable) and
    printed it for all of them, including another run holding the lock.
    An unknown code is reported as itself rather than guessed at."""
    if returncode in _MEANING:
        meaning = _MEANING[returncode]
        if returncode == REFUSED and subcommand in ("score", "assess"):
            meaning = ("did not start: no trial ran and no CSV was written. The reason "
                       "is printed above, with what you can run instead where there is "
                       "an alternative")
        elif returncode == FAILED and subcommand == "gate":
            meaning = ("finished, but the check was not clean; the report above lists "
                       "what it found")
        return f"{meaning} (exit code {returncode})"
    return (f"exited with code {returncode}, which this framework does not assign a "
            f"meaning to; treat the output above as the finding")


def request_failures(rows):
    """Rows whose request to the endpoint failed, i.e. whose `error` is set.
    Those rows measured nothing, whatever their other columns say."""
    return [r for r in rows if str(r.get("error") or "").strip()]


def after_run(rows):
    """(exit code, message) for a suite that ran and wrote its rows.

    FAILED when any request failed, so a dead or crashed endpoint cannot end
    in the same exit code as a clean run. The rows are still written and the
    rest still scored -- `cbench assess` carries on past a failed trial --
    but the failure is said once, at the end, where it will be read."""
    failed = request_failures(rows)
    if not failed:
        return SUCCESS, ""
    first = str(failed[0].get("error")).strip().splitlines()[0][:200]
    if len(failed) == len(rows):
        return FAILED, (f"[!] Every request to the endpoint failed ({len(rows)} of {len(rows)}), "
                        f"so nothing was measured. First error: {first}\n"
                        "    Check the endpoint is up and serving this model: cbench doctor")
    return FAILED, (f"[!] {len(failed)} of {len(rows)} request(s) to the endpoint failed. "
                    "Those rows measured nothing and are excluded from every rate; the "
                    f"rest were scored. First error: {first}")


def is_success(returncode):
    return returncode == SUCCESS


def subcommand_of(argv):
    """The cbench subcommand an argv invoked, or None.

    Tolerant on purpose: this feeds a help string, and a surprising argv
    should degrade to a more general message rather than raise inside the
    error-reporting path."""
    try:
        return argv[argv.index("openllm_cbench.cli") + 1]
    except (ValueError, IndexError, AttributeError, TypeError):
        return None
