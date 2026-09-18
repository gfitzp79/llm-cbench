"""Tests for the pre-run capability check.

Every fixture in here is a real gate result shape, taken from a model
that actually behaved this way on the machine this was written for. The
three that matter:

  exaone-deep:7.8b   reports `completion` and nothing else. Three suites
                     were run against it; all three scored INVALID.
  llama3.1:8b        reports `completion, tools`. The gate printed
                     "Clean. No caveats found." while S2 against it could
                     not have returned a verdict.
  a 4B thinking model that reports `tools, thinking, completion` and
                     returns an empty thinking field -- advertised and
                     not delivered, which is the case that looks
                     supported.
"""

from openllm_cbench.core import preflight as pf
from openllm_cbench.core.preflight import INVALID, READY, UNVERIFIED


def gate_result(**over):
    """A gate result for a model where everything works."""
    base = {
        "model": "fine:8b",
        "show_info_ok": True,
        "has_tools_capability": True,
        "has_thinking_capability": True,
        "tool_call_ok": True,
        "tool_call_detail": "well-formed tool call round-tripped correctly",
        "tool_call_timed_out": False,
        "channel_think_on": {"ok": True, "content_len": 500, "thinking_len": 300},
        "channel_think_off": {"ok": True, "content_len": 500, "thinking_len": 0},
    }
    base.update(over)
    return base


# ------------------------------------------------------- the happy path

def test_a_capable_model_is_ready_for_everything():
    readiness = pf.suite_readiness(gate_result())
    assert [v for v, _ in readiness.values()] == [READY, READY, READY]
    assert pf.unrunnable(gate_result()) == []


def test_a_trace_at_either_think_state_is_enough():
    """Matches scoring/capability.py's s2_could_detect_a_leak, which is a
    whole-run check: a model may legitimately return no trace at
    think=off, and requiring both would fail a valid model."""
    only_off = gate_result(
        channel_think_on={"ok": True, "content_len": 500, "thinking_len": 0},
        channel_think_off={"ok": True, "content_len": 500, "thinking_len": 120})
    assert pf.suite_readiness(only_off)["s2"][0] == READY


# ---------------------------------------------- the three real failures

def test_exaone_reports_no_capability_at_all():
    """`completion` only. Every suite is dead, and the previous gate
    mentioned only the missing tools."""
    r = gate_result(has_tools_capability=False, has_thinking_capability=False,
                    tool_call_ok=False, tool_call_detail="request failed: 400 Client Error",
                    channel_think_on=None, channel_think_off=None)
    readiness = pf.suite_readiness(r)
    assert [v for v, _ in readiness.values()] == [INVALID, INVALID, INVALID]
    assert pf.unrunnable(r) == ["s1", "s2", "s3"]


def test_a_tools_only_model_is_not_clean_for_s2():
    """THE REGRESSION THIS FILE EXISTS FOR. llama3.1:8b gate-checked
    "Clean. No caveats found." while its S2 could not fire. The gate's
    own report said "skipped (model does not report a thinking
    capability)" four lines above the word Clean. A skipped check is not
    a passed one."""
    r = gate_result(has_thinking_capability=False,
                    channel_think_on=None, channel_think_off=None)
    readiness = pf.suite_readiness(r)
    assert readiness["s1"][0] == READY
    assert readiness["s3"][0] == READY
    assert readiness["s2"][0] == INVALID
    assert pf.unrunnable(r) == ["s2"]


def test_advertised_thinking_that_comes_back_empty_is_invalid():
    """The worst of the three, because the capability makes it look
    supported. A 0% leak rate from this model describes the instrument."""
    r = gate_result(
        channel_think_on={"ok": True, "content_len": 900, "thinking_len": 0},
        channel_think_off={"ok": True, "content_len": 900, "thinking_len": 0})
    verdict, reason = pf.suite_readiness(r)["s2"]
    assert verdict == INVALID
    assert "advertised and not delivered" in reason


# ------------------------------------------- what must NOT be a verdict

def test_a_timed_out_tool_call_is_unverified_not_invalid():
    """The endpoint reported the capability and a stopwatch cannot
    overrule it. Same doctrine as gate.warm_up()."""
    r = gate_result(tool_call_ok=False, tool_call_timed_out=True,
                    tool_call_detail="TIMEOUT after 300s -- ...")
    assert pf.suite_readiness(r)["s1"][0] == UNVERIFIED
    assert pf.unrunnable(r) == [], "UNVERIFIED must never stop a run"


def test_an_unreadable_endpoint_is_unverified_everywhere():
    r = gate_result(has_tools_capability=None, has_thinking_capability=None,
                    channel_think_on=None, channel_think_off=None)
    assert [v for v, _ in pf.suite_readiness(r).values()] == [UNVERIFIED] * 3
    assert pf.unrunnable(r) == []


def test_a_channel_check_that_errored_is_unverified():
    """Distinct from 'returned an empty trace': one is a measurement, the
    other is a missing one."""
    r = gate_result(channel_think_on={"ok": False, "error": "read timeout"},
                    channel_think_off={"ok": False, "error": "read timeout"})
    assert pf.suite_readiness(r)["s2"][0] == UNVERIFIED


def test_a_gate_result_missing_keys_does_not_raise():
    """A verdict guessed from absent data is worse than 'I could not
    tell'."""
    assert [v for v, _ in pf.suite_readiness({}).values()] == [UNVERIFIED] * 3


# ---------------------------------------------------------- the report

def test_the_report_names_the_narrowed_command_when_something_survives():
    r = gate_result(has_thinking_capability=False,
                    channel_think_on=None, channel_think_off=None)
    text = "\n".join(pf.render_readiness(r))
    assert "WILL BE INVALID" in text
    assert "--suites s1,s3" in text


def test_the_report_says_so_when_nothing_survives():
    r = gate_result(has_tools_capability=False, has_thinking_capability=False,
                    tool_call_ok=False, channel_think_on=None, channel_think_off=None)
    text = "\n".join(pf.render_readiness(r))
    assert "Nothing here would produce a gradeable result" in text
    assert "--suites" not in text, "there is no narrowed command to offer"


def test_the_report_honours_a_suite_subset():
    r = gate_result(has_thinking_capability=False,
                    channel_think_on=None, channel_think_off=None)
    text = "\n".join(pf.render_readiness(r, suites=("s1", "s3")))
    assert "S2" not in text
    assert "WILL BE INVALID" not in text


# ------------------------------------------------- agreement with scoring

def test_preflight_agrees_with_the_scorer_about_s2():
    """These two live in different modules and must not drift. The scorer
    decides after the run from rows; the pre-flight decides before it
    from the gate. They have to reach the same answer about the same
    model or the pre-flight is lying about what the run will produce."""
    from openllm_cbench.scoring.capability import s2_could_detect_a_leak

    empty = gate_result(
        channel_think_on={"ok": True, "content_len": 900, "thinking_len": 0},
        channel_think_off={"ok": True, "content_len": 900, "thinking_len": 0})
    assert pf.suite_readiness(empty)["s2"][0] == INVALID
    assert s2_could_detect_a_leak(0) is False

    trace = gate_result()
    assert pf.suite_readiness(trace)["s2"][0] == READY
    assert s2_could_detect_a_leak(7) is True


# -------------------------------------------------------- gate wiring

def test_the_gate_caveats_an_invalid_suite(monkeypatch):
    """The gate's own caveat list must carry these, because `clean` is
    derived from it and the TUI reads the exit code that follows.

    monkeypatch, not direct assignment: an earlier test in this suite
    replaced gate.check_channel_at on the module with no restore and
    leaked the fake into every test that ran after it."""
    import openllm_cbench.core.gate as gate

    monkeypatch.setattr(gate, "fetch_show_info", lambda *a, **k: {
        "capabilities": ["completion", "tools"],
        "details": {"family": "llama", "parameter_size": "8.0B",
                    "quantization_level": "Q4_K_M"},
        "modelfile": ""})
    monkeypatch.setattr(gate, "warm_up", lambda *a, **k: (True, 1.0, 50.0))
    monkeypatch.setattr(gate, "check_tool_call",
                        lambda *a, **k: (True, "well-formed tool call round-tripped correctly"))

    result = gate.run_gate("llama3.1:8b", "http://localhost:11434")

    assert result["clean"] is False, "a tools-only model is not clean -- S2 cannot fire"
    assert any("S2 channel will come back INVALID" in c for c in result["caveats"])
    assert result["suite_readiness"]["s1"][0] == READY
    assert result["suite_readiness"]["s2"][0] == INVALID
    assert "What this model can be scored on" in gate.render_gate_report(result)
