"""
The one sentence every entry point in this framework repeats.

Kept as a single string so it can never drift between `cbench doctor`,
the top-level `cbench --help`, and each suite's own `--help` epilog --
four surfaces that have to say the same thing, not four independently
maintained copies of it.
"""

SAFETY_INVARIANT = (
    "SAFETY INVARIANT: this suite measures ATTEMPT, never success. Its "
    "http_request tool can only ever reach a loopback canary (127.0.0.1) "
    "that asserts its own bind address before starting; any other host is "
    "intercepted before a socket is opened and answered with a synthetic "
    "failure. run_shell is always a logged no-op -- nothing it is given is "
    "ever executed. See ARCHITECTURE.md and tests/test_safety_invariant.py "
    "(runs with no model and no network)."
)


def epilog():
    return SAFETY_INVARIANT
