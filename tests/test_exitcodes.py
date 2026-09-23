"""Exit codes report what happened, not what number was returned.

Reported live: "exit code 2 means nothing". It was the last and most
prominent line of every TUI run, and it was the least informative one.
"""
from openllm_cbench.core import exitcodes


def test_zero_reads_as_success():
    assert "successfully" in exitcodes.describe(0)
    assert exitcodes.is_success(0)


def test_one_says_something_ran_and_failed():
    # The distinction that matters: there IS a result to look at.
    said = exitcodes.describe(1)
    assert "ran" in said
    assert not exitcodes.is_success(1)


def test_two_says_nothing_ran():
    # ...and here there is NOT. Presenting 1 and 2 as two shades of
    # "failed" loses the only thing that tells a user what to do next.
    said = exitcodes.describe(2)
    assert "did not start" in said
    assert "nothing was written" in said


def test_two_on_score_is_true_for_every_way_score_refuses():
    # It used to name one reason -- no suite gradeable -- and printed it
    # for every refusal, including another run holding the lock. It must
    # say only what is true of all of them, and point at the real reason.
    said = exitcodes.describe(2, "score")
    assert "no trial ran" in said
    assert "no CSV was written" in said
    assert "gradeable" not in said
    assert exitcodes.describe(2, "assess") == said


def test_one_on_gate_reads_as_a_finding_not_a_crash():
    said = exitcodes.describe(1, "gate")
    assert "not clean" in said
    assert "exit code 1" in said


def test_every_message_still_carries_the_number():
    # The number is what a script or a bug report needs; it just stops
    # being the whole message.
    for code in (0, 1, 2):
        assert f"exit code {code}" in exitcodes.describe(code)


def test_unknown_code_is_reported_not_guessed():
    said = exitcodes.describe(137)
    assert "137" in said
    assert "does not assign a meaning" in said


def test_subcommand_of_reads_the_argv():
    argv = ["python", "-u", "-m", "openllm_cbench.cli", "score", "--model", "x:1b"]
    assert exitcodes.subcommand_of(argv) == "score"


def test_subcommand_of_degrades_rather_than_raising():
    # This feeds the error-reporting path, so a surprising argv must not
    # raise inside the code that reports a failure.
    assert exitcodes.subcommand_of([]) is None
    assert exitcodes.subcommand_of(["python", "-m", "something_else"]) is None
    assert exitcodes.subcommand_of(None) is None
    assert exitcodes.subcommand_of(["python", "-m", "openllm_cbench.cli"]) is None
