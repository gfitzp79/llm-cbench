"""
Coverage for core/gate.py's pure functions -- no model, no network.
"""

from openllm_cbench.core.gate import _numeric_params_b, to_registry_entry


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
