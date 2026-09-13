"""
Coverage for core/gate.py's pure functions -- no model, no network.
"""

from openllm_cbench.core.gate import (
    _numeric_params_b, render_gate_report, summarize_gate_output, to_registry_entry,
)


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


def test_summarize_gate_output_degrades_gracefully_on_unrecognized_text():
    # e.g. the process crashed before printing any report at all.
    summary = summarize_gate_output(["Traceback (most recent call last):", "RuntimeError: boom"])
    assert summary["hard_failure"] is False
    assert summary["clean"] is None
    assert summary["caveats"] == []
