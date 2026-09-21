"""
Coverage for core/gate.py's pure functions -- no model, no network.
"""

from openllm_cbench.core import gate
from openllm_cbench.core.gate import (
    _numeric_params_b, render_gate_report, scaled_timeout, summarize_gate_output,
    to_registry_entry,
)


# --- Timeout scaling -----------------------------------------------------

def test_scaled_timeout_never_shortens_a_check():
    # A fast model must behave exactly as it did before scaling existed.
    assert scaled_timeout(60, 0.2, tokens_per_sec=80, num_predict=512) == 60
    assert scaled_timeout(90, None) == 90


def test_scaled_timeout_buys_a_slow_model_more_room():
    # 2048 tokens at 4 tok/s is ~512s of generation -- 90s was never going
    # to be enough, which is how a large model's channel check "failed".
    assert scaled_timeout(90, 30.0, tokens_per_sec=4, num_predict=2048) > 90


def test_scaled_timeout_is_capped():
    # Past some point the honest answer is "could not be verified", not a
    # twenty-minute block. The cap is why an unverified check must caveat.
    assert scaled_timeout(90, 600.0, tokens_per_sec=0.1, num_predict=2048, ceiling=300) == 300


def test_numeric_params_b_strips_b_suffix():
    assert _numeric_params_b("14.8B") == 14.8
    assert _numeric_params_b("8.0B") == 8
    assert _numeric_params_b("20.9B") == 20.9


def test_numeric_params_b_integral_values_become_int_not_float():
    # Matches the hand-written seed entries' style (12, not 12.0).
    assert _numeric_params_b("12.0B") == 12
    assert isinstance(_numeric_params_b("12.0B"), int)


def test_numeric_params_b_falls_back_on_unparseable_input():
    assert _numeric_params_b("unknown") == "unknown"
    assert _numeric_params_b(None) is None
    assert _numeric_params_b(7) == 7  # already numeric, passed through


def test_numeric_params_b_converts_millions_suffix_to_billions():
    # Found live: smollm2:135m reports parameter_size as "134.52M", not
    # "0.13B" -- an 'M' value must be divided by 1000, not just have its
    # suffix stripped like a 'B' value, or it reads as 1000x too large.
    assert _numeric_params_b("134.52M") == 0.13452
    assert _numeric_params_b("500M") == 0.5


def test_to_registry_entry_produces_numeric_params_b():
    result = {
        "architecture": "qwen3", "param_size": "14.8B", "quant": "Q4_K_M",
        "has_tools_capability": True, "has_thinking_capability": False,
        "channel_think_on": None, "channel_think_off": None,
        "caveats": [],
    }
    entry = to_registry_entry(result)
    assert entry["params_b"] == 14.8
    assert isinstance(entry["params_b"], (int, float))


# summarize_gate_output() is deliberately tested through render_gate_report()'s
# own real output rather than hand-typed report text -- that way a future
# change to the report's wording breaks these tests instead of silently
# leaving the TUI's parser reading a format that no longer exists.

def _report_lines(result):
    return render_gate_report(result).splitlines()


def test_summarize_gate_output_detects_hard_failure_when_endpoint_unreachable():
    result = {"model": "x:1b", "show_info_ok": False, "show_info_error": "Connection refused"}
    summary = summarize_gate_output(_report_lines(result))
    assert summary["hard_failure"] is True
    assert "connection refused" in summary["reason"].lower()
    assert summary["clean"] is None
    assert summary["caveats"] == []


def test_summarize_gate_output_extracts_every_caveat_when_not_clean():
    result = {
        "model": "granite4.1-guardian:8b", "show_info_ok": True,
        "architecture": "granite", "param_size": "8B", "quant": "Q4_K_M",
        "capabilities": [], "has_tools_capability": False, "has_thinking_capability": False,
        "tool_call_ok": False, "tool_call_detail": "no tool_calls in response",
        "channel_think_on": None, "channel_think_off": None, "sampling_params": {},
        "caveats": [
            "Endpoint does not report a tools capability -- the containment and "
            "persistence suites need real tool calling and will not produce valid data.",
        ],
        "clean": False,
    }
    summary = summarize_gate_output(_report_lines(result))
    assert summary["hard_failure"] is False
    assert summary["clean"] is False
    assert len(summary["caveats"]) == 1
    assert "tools capability" in summary["caveats"][0]


def test_summarize_gate_output_reports_clean_with_no_caveats():
    result = {
        "model": "smollm2:135m", "show_info_ok": True,
        "architecture": "llama", "param_size": "135M", "quant": "Q4_K_M",
        "capabilities": ["tools"], "has_tools_capability": True, "has_thinking_capability": False,
        "tool_call_ok": True, "tool_call_detail": "well-formed tool call round-tripped correctly",
        "channel_think_on": None, "channel_think_off": None, "sampling_params": {},
        "caveats": [], "clean": True,
    }
    summary = summarize_gate_output(_report_lines(result))
    assert summary["hard_failure"] is False
    assert summary["clean"] is True
    assert summary["caveats"] == []


def test_an_unverified_channel_check_is_never_reported_as_clean(monkeypatch):
    # Found live on a 27.9B model: the think=on channel check timed out,
    # contributed nothing to `clean`, and the report said "Clean. No
    # caveats found." -- a clean bill of health for a check that never
    # ran, on the exact property the check exists to establish.
    from openllm_cbench.core.gate import run_gate

    import openllm_cbench.core.gate as gate

    def fake_show(model, endpoint=None, timeout=30):
        return {"capabilities": ["completion", "tools", "thinking"],
                "details": {"family": "x", "parameter_size": "27.9B",
                             "quantization_level": "Q4_K_M"},
                "modelfile": ""}

    # monkeypatch, not direct assignment: these used to be assigned onto
    # the module with no restore, so they leaked into every test that ran
    # after this one in the same session.
    monkeypatch.setattr(gate, "fetch_show_info", fake_show)
    monkeypatch.setattr(gate, "warm_up", lambda *a, **k: (True, 1.0, 50.0))
    monkeypatch.setattr(gate, "check_tool_call", lambda *a, **k: (True, "fine"))
    # think=on never completes; think=off is clean.
    monkeypatch.setattr(gate, "check_channel_at", lambda model, think, *a, **k: (
        {"ok": False, "error": "Read timed out. (read timeout=300)"} if think
        else {"ok": True, "content_len": 800, "thinking_len": 0,
              "done_reason": "stop", "merged_channel_suspected": False, "truncated": False}))

    result = run_gate("big:30b", "http://localhost:11434")

    assert result["clean"] is False, "a check that never completed is not a pass"
    assert any("could NOT be verified" in c for c in result["caveats"])
    assert any("think=on" in c for c in result["caveats"])
    assert "**Clean.**" not in render_gate_report(result)


def test_summarize_gate_output_degrades_gracefully_on_unrecognized_text():
    # e.g. the process crashed before printing any report at all.
    summary = summarize_gate_output(["Traceback (most recent call last):", "RuntimeError: boom"])
    assert summary["hard_failure"] is False
    assert summary["clean"] is None
    assert summary["caveats"] == []


def test_tool_check_retries_with_thinking_off(monkeypatch):
    """A model that emits a reasoning trace and then stops must not be
    reported as unable to call tools.

    Measured on `SuhasDevmane55/geollm-qwen3-4b-v8`: no tool call with
    thinking on, one well-formed call with `think=false`. Its base model
    qwen3:4b calls the tool in both states, so the fine-tune changed it and
    nothing in the metadata says so. With one probe the pre-flight refused
    S1 and S3 on a model that can run them -- a missing measurement
    reported as a capability finding."""
    seen = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_post(url, json=None, timeout=None):
        seen.append(json.get("think", "<unset>"))
        if json.get("think") is False:
            return _Resp({"message": {"tool_calls": [
                {"function": {"name": "echo", "arguments": {"text": "gate-check-ok"}}}]}})
        return _Resp({"message": {"content": "", "thinking": "hmm"}})

    monkeypatch.setattr(gate.requests, "post", fake_post)
    ok, detail = gate.check_tool_call("fine-tune:4b")

    assert ok is True
    # Default state, then the SAME state again (one failure is not evidence
    # about the state -- see test_a_single_flaky_failure_...), then off.
    assert seen == ["<unset>", "<unset>", False]
    assert "ONLY with the reasoning channel off" in detail
    assert "think=false" in detail


def test_tool_check_reports_failure_in_both_states(monkeypatch):
    """A model that really cannot tool-call still fails -- and the detail
    says both states were tried, so the finding is not mistaken for the
    one-probe version it replaced."""
    class _Resp:
        def raise_for_status(self):
            pass

        def json(self):
            return {"message": {"content": "I cannot do that."}}

    monkeypatch.setattr(gate.requests, "post", lambda *a, **k: _Resp())
    ok, detail = gate.check_tool_call("no-tools:1b")

    assert ok is False
    assert "with thinking on" in detail
    assert "think=false" in detail


def test_tool_check_does_not_retry_a_timeout(monkeypatch):
    """A timeout says nothing about capability, and a second call would
    only double the wait before saying so."""
    calls = []

    def fake_post(*a, **k):
        calls.append(1)
        raise gate.requests.exceptions.Timeout()

    monkeypatch.setattr(gate.requests, "post", fake_post)
    ok, detail = gate.check_tool_call("slow:70b", timeout=5)

    assert ok is False
    assert len(calls) == 1, "a timeout must not be retried"
    assert "TIMEOUT" in detail


def test_show_info_failure_heading_is_matched_by_the_summarizer():
    """The heading and the matcher must stay one string.

    They were the same literal written twice. Editing the wording in the
    report alone would have silently stopped summarize_gate_output()
    recognising a hard failure -- which fails OPEN: the gate reports no
    problem about a model it could not read at all."""
    report = render_gate_report({
        "model": "broken:27b",
        "show_info_ok": False,
        "show_info_error": "HTTP 500: unsupported tensor size overflows",
    })
    assert gate.SHOW_INFO_FAILED_HEADING in report
    assert summarize_gate_output(report)["hard_failure"] is True


def test_show_info_failure_does_not_claim_the_endpoint_was_unreachable():
    """Measured on a real catalogue entry: Ollama answered /api/show
    promptly with HTTP 500 and `read GGUF metadata ...: unsupported tensor
    "output.weight" size overflows` -- a corrupt model file. Calling that
    "could not reach" sends someone to check their network when the fix is
    to re-pull or drop the tag."""
    report = render_gate_report({
        "model": "broken:27b",
        "show_info_ok": False,
        "show_info_error": "the endpoint REJECTED the request with HTTP 500: bad GGUF",
    })
    assert "Could not reach" not in report
    assert "re-pull the tag" in report


def test_summarize_accepts_the_whole_report_not_only_lines():
    """Handing it render_gate_report()'s return value must work.

    It took lines only. Passing the whole string iterated it CHARACTER by
    character, matched nothing, and reported no hard failure for a model
    the gate could not read -- a fail-open on the field a caller uses to
    decide whether to start a run."""
    result = {"model": "x:1b", "show_info_ok": False, "show_info_error": "Connection refused"}
    report = render_gate_report(result)
    assert summarize_gate_output(report) == summarize_gate_output(report.splitlines())
    assert summarize_gate_output(report)["hard_failure"] is True


def test_a_single_flaky_failure_does_not_pin_a_model_to_thinking_off(monkeypatch):
    """One failed call is not evidence about the reasoning state.

    Measured: granite4.1:8b failed this probe once during a catalogue
    re-gate, then called the tool 5/5 in BOTH states when checked. The
    single failure had written it a permanent `think: false` override,
    silently changing the conditions S1/S3 measure it under."""
    states = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    calls = {"n": 0}

    def fake_post(url, json=None, timeout=None):
        calls["n"] += 1
        states.append(json.get("think", "<unset>"))
        if calls["n"] == 1:          # flaky miss
            return _Resp({"message": {"content": "Sure, let me help."}})
        return _Resp({"message": {"tool_calls": [
            {"function": {"name": "echo", "arguments": {"text": "gate-check-ok"}}}]}})

    monkeypatch.setattr(gate.requests, "post", fake_post)
    ok, detail = gate.check_tool_call("flaky:8b")

    assert ok is True
    assert states == ["<unset>", "<unset>"], "must repeat the DEFAULT state before blaming it"
    assert gate.TOOLS_NEED_THINKING_OFF not in detail
    assert "run-to-run variance" in detail
    # The whole point: no override gets written for a flaky model.
    assert to_registry_entry({"tools_need_thinking_off": False})["config_overrides"] == {}


def test_a_consistent_failure_still_reaches_the_thinking_off_retry(monkeypatch):
    """The repeat must not mask a real state-dependent gap.

    geollm-qwen3-4b-v8 is 0/8 with thinking on and 8/8 with it off; that
    shape must still be found after the extra same-state attempt."""
    states = []

    class _Resp:
        def __init__(self, payload):
            self._payload = payload

        def raise_for_status(self):
            pass

        def json(self):
            return self._payload

    def fake_post(url, json=None, timeout=None):
        states.append(json.get("think", "<unset>"))
        if json.get("think") is False:
            return _Resp({"message": {"tool_calls": [
                {"function": {"name": "echo", "arguments": {"text": "gate-check-ok"}}}]}})
        return _Resp({"message": {"content": "", "thinking": "hmm"}})

    monkeypatch.setattr(gate.requests, "post", fake_post)
    ok, detail = gate.check_tool_call("statebound:4b")

    assert ok is True
    assert states == ["<unset>", "<unset>", False]
    assert gate.TOOLS_NEED_THINKING_OFF in detail
