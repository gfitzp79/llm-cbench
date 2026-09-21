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
    FAILED: "ran, but something in it failed -- the output above says what",
    REFUSED: ("did not start -- nothing ran and nothing was written. The reason is "
              "printed above; fix that and run again"),
}


def describe(returncode, subcommand=None):
    """One human sentence for a process exit code.

    `subcommand` sharpens the REFUSED case where the framework has exactly
    one reason to refuse, so the user does not have to scroll back for it.
    An unknown code is reported as itself rather than guessed at."""
    if returncode in _MEANING:
        meaning = _MEANING[returncode]
        if returncode == REFUSED and subcommand in ("score", "assess"):
            meaning = ("did not start -- the pre-flight found no selected suite could "
                        "produce a gradeable result on this model, so no time was spent "
                        "and no CSV was written. See the per-suite reasons above")
        return f"{meaning} (exit code {returncode})"
    return (f"exited with code {returncode}, which this framework does not assign a "
            f"meaning to -- treat the output above as the finding")


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
