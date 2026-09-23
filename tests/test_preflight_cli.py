"""The pre-flight where it actually costs something: the two commands
that spend hours.

`cbench assess` had exactly one launch-time guard -- the run lock -- and
it guarded against contention, not against running a model that cannot
produce a result. `cbench score` did not even have that, and `cbench
score` is what the TUI's Score screen runs, so the only interactive path
into a multi-hour run was the unguarded one. That is how a model
reporting nothing but `completion` got three suites and a scorecard whose
every row was a dash.
"""

import sys

import pytest

from openllm_cbench import cli
from openllm_cbench.core import gate


NO_CAPABILITIES = {
    "capabilities": ["completion"],
    "details": {"family": "exaone", "parameter_size": "7.8B",
                "quantization_level": "Q4_K_M"},
    "modelfile": "",
}
TOOLS_ONLY = {
    "capabilities": ["completion", "tools"],
    "details": {"family": "llama", "parameter_size": "8.0B",
                "quantization_level": "Q4_K_M"},
    "modelfile": "",
}


@pytest.fixture
def endpoint(monkeypatch):
    """A gate-able endpoint with no network. Returns a setter for which
    capabilities it reports."""
    state = {"show": TOOLS_ONLY, "tool_ok": True}

    monkeypatch.setattr(gate, "fetch_show_info", lambda *a, **k: state["show"])
    monkeypatch.setattr(gate, "warm_up", lambda *a, **k: (True, 1.0, 50.0))
    monkeypatch.setattr(gate, "check_tool_call",
                        lambda *a, **k: (state["tool_ok"], "round-tripped" if state["tool_ok"]
                                         else "no tool_calls in response"))
    monkeypatch.setattr(gate, "check_channel_at", lambda *a, **k: {
        "ok": True, "content_len": 500, "thinking_len": 300, "done_reason": "stop",
        "merged_channel_suspected": False, "merge_evidence": [], "truncated": False})

    def _set(show=None, tool_ok=None):
        if show is not None:
            state["show"] = show
        if tool_ok is not None:
            state["tool_ok"] = tool_ok
    return _set


def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cbench"] + argv)
    return cli.main()


def _never_runs(monkeypatch):
    """Fails loudly if a suite is launched. The point of the guard is
    that nothing downstream of it executes."""
    def boom(*a, **k):
        raise AssertionError("a suite was launched past the pre-flight refusal")
    monkeypatch.setattr(cli, "_assess_body", boom)


# ----------------------------------------------------------- refusing

def test_assess_refuses_a_model_with_no_usable_capability(endpoint, monkeypatch, capsys):
    endpoint(show=NO_CAPABILITIES, tool_ok=False)
    _never_runs(monkeypatch)
    assert _run(["assess", "--model", "exaone-deep:7.8b", "--trials", "3"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "NOT STARTING" in err
    assert "No suite here would produce anything" in err


def test_assess_refuses_s2_and_names_the_narrowed_command(endpoint, monkeypatch, capsys):
    """llama3.1:8b: S1 and S3 are fine, S2 cannot fire. A user who wanted
    three suites and can have two is better served by the command than by
    the diagnosis."""
    endpoint(show=TOOLS_ONLY)
    _never_runs(monkeypatch)
    assert _run(["assess", "--model", "llama3.1:8b", "--trials", "3"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "--suites s1,s3" in err
    assert "--trials 3" in err


def test_score_is_guarded_too_and_suggests_a_score_command(endpoint, monkeypatch, capsys):
    """THE PATH THE TUI RUNS. Telling someone who ran `cbench score` to
    run `cbench assess` would hand them a command that produces no
    grade."""
    endpoint(show=TOOLS_ONLY)
    _never_runs(monkeypatch)
    assert _run(["score", "--model", "llama3.1:8b", "--depth", "quick"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "cbench score --model llama3.1:8b --suites s1,s3 --depth quick" in err


def test_the_requested_suites_are_what_gets_judged(endpoint, monkeypatch, capsys):
    """A user who already excluded S2 must not be refused because of it."""
    endpoint(show=TOOLS_ONLY)
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    assert _run(["assess", "--model", "llama3.1:8b", "--suites", "s1,s3"], monkeypatch) == 0
    assert "NOT STARTING" not in capsys.readouterr().err


# ---------------------------------------------------------- overriding

def test_force_uncheckable_runs_anyway_without_claiming_not_to(endpoint, monkeypatch, capsys):
    """The override must not print NOT STARTING and then start -- that
    contradiction was in the first draft of this guard."""
    endpoint(show=NO_CAPABILITIES, tool_ok=False)
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    rc = _run(["assess", "--model", "exaone-deep:7.8b", "--force-uncheckable"], monkeypatch)
    err = capsys.readouterr().err
    assert rc == 0
    assert "NOT STARTING" not in err
    assert "Running anyway" in err


def test_skip_preflight_makes_no_gate_call_at_all(endpoint, monkeypatch, capsys):
    def boom(*a, **k):
        raise AssertionError("--skip-preflight still ran the gate")
    monkeypatch.setattr(cli, "_assess_preflight", boom)
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    assert _run(["assess", "--model", "whatever:1b", "--skip-preflight"], monkeypatch) == 0


def test_dry_run_skips_the_preflight(endpoint, monkeypatch):
    """Same reasoning as the run lock: a dry run makes no model calls, so
    a pre-flight that did would defeat the point of it."""
    def boom(*a, **k):
        raise AssertionError("--dry-run ran the pre-flight")
    monkeypatch.setattr(cli, "_assess_preflight", boom)
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    assert _run(["assess", "--model", "exaone-deep:7.8b", "--dry-run"], monkeypatch) == 0


def test_from_existing_is_not_preflighted(monkeypatch, capsys, tmp_path):
    """Scoring CSVs already on disk calls no model, so there is nothing to
    pre-flight -- and refusing would make an existing corpus unreadable
    because of what the endpoint says about a tag today."""
    def boom(*a, **k):
        raise AssertionError("--from-existing ran the pre-flight")
    monkeypatch.setattr(cli, "_assess_preflight", boom)
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path))
    _run(["score", "--model", "anything:1b", "--from-existing"], monkeypatch)


# --------------------------------------------- what must not block a run

def test_an_unreachable_endpoint_refuses_before_spending_the_run(monkeypatch, capsys):
    """REVERSED 2026-09-23. This test used to assert the opposite: an
    unreachable endpoint "is not evidence about the model", so the run
    went ahead. It still is not evidence about the model, and the refusal
    does not claim it is -- it says nothing would be measured. But going
    ahead was measured live: every trial made the same failing call,
    spent its time, and left a file of error rows that counted toward the
    trial total. Only BOTH routes failing refuses (see the next test for
    one route answering); a pre-flight that crashes still never blocks
    (the test after that)."""
    monkeypatch.setattr(gate, "fetch_show_info",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("connection refused")))
    monkeypatch.setattr(gate, "warm_up", lambda *a, **k: (False, 0.0, None))
    monkeypatch.setattr(gate, "check_tool_call", lambda *a, **k: (False, "request failed"))
    ran = []
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: ran.append(1) or 0)
    assert _run(["assess", "--model", "unreachable:8b"], monkeypatch) == 2
    err = capsys.readouterr().err
    assert "NOT STARTING" in err
    assert "nothing would be measured" in err
    assert not ran


def test_a_model_that_answers_chat_is_not_refused_for_a_missing_info_route(monkeypatch, capsys):
    """An endpoint without Ollama's /api/show still serves chat. One
    route answering is enough -- that is what keeps this refusal from
    becoming a pre-flight that blocks a working setup."""
    monkeypatch.setattr(gate, "fetch_show_info",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("404 not found")))
    monkeypatch.setattr(gate, "warm_up", lambda *a, **k: (True, 1.0, 50.0))
    monkeypatch.setattr(gate, "check_tool_call", lambda *a, **k: (False, "request failed"))
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    _run(["assess", "--model", "no-show-route:8b"], monkeypatch)
    assert "could not reach" not in capsys.readouterr().err


def test_the_preflight_cannot_block_a_run_by_failing_itself(monkeypatch, capsys):
    """The defensive catch, exercised for real by making run_gate() raise.

    `run_gate()` documents that it never raises past itself, which makes
    this handler unreachable today -- deliberately kept anyway, because
    it is a contract rather than a guarantee, and the cost of that
    contract quietly changing is a guard that blocks legitimate runs. A
    guard that can stop your work by malfunctioning is worse than no
    guard, so the fallback is tested rather than assumed."""
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    monkeypatch.setattr(
        "openllm_cbench.core.gate.run_gate",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("preflight itself is broken")))

    assert _run(["assess", "--model", "whatever:8b"], monkeypatch) == 0
    err = capsys.readouterr().err
    assert "Pre-flight could not complete" in err
    assert "continuing anyway" in err
    assert "NOT STARTING" not in err


def test_a_timed_out_tool_call_does_not_block(endpoint, monkeypatch, capsys):
    """UNVERIFIED is not INVALID. Turning a slow machine into a capability
    verdict is the mistake gate.warm_up() exists to prevent."""
    endpoint(show={"capabilities": ["completion", "tools", "thinking"],
                   "details": {}, "modelfile": ""})
    monkeypatch.setattr(gate, "check_tool_call",
                        lambda *a, **k: (False, "TIMEOUT after 300s -- this says nothing about"))
    monkeypatch.setattr(cli, "_assess_body", lambda *a, **k: 0)
    assert _run(["assess", "--model", "huge:70b"], monkeypatch) == 0
    assert "NOT STARTING" not in capsys.readouterr().err
