"""
Coverage for core/hardware.py's pure arithmetic -- no subprocess, no GPU
needed. detect_gpu_vram_mb()/detect_system_ram_mb()/probe() themselves are
platform-dependent and best-effort by design (see the module docstring);
what's testable and load-bearing without real hardware is the arithmetic
check_model_fit() runs on whatever numbers it's given.
"""

from openllm_cbench.core.hardware import check_model_fit


def test_check_model_fit_unknown_when_any_input_missing():
    assert check_model_fit(None, "Q4_K_M", 8192)["known"] is False
    assert check_model_fit(12.2, None, 8192)["known"] is False
    assert check_model_fit(12.2, "Q4_K_M", None)["known"] is False
    assert check_model_fit(12.2, "Q4_K_M", 0)["known"] is False


def test_check_model_fit_unknown_when_params_b_is_the_literal_unknown_string():
    # A catalogue entry that couldn't be measured stores params_b as the
    # string "unknown" (core/gate.py:_numeric_params_b's own fallback),
    # not a number or None -- this must not reach the arithmetic below
    # and raise a TypeError multiplying a string by a float.
    fit = check_model_fit("unknown", "unknown", 8192)
    assert fit["known"] is False
    assert fit["fits"] is None


def test_check_model_fit_true_when_comfortably_within_budget():
    # ~1B params at Q4_K_M needs well under 1GB; a 24GB GPU fits it easily.
    fit = check_model_fit(1, "Q4_K_M", 24576)
    assert fit["known"] is True
    assert fit["fits"] is True
    assert fit["needed_mb"] < fit["usable_mb"]


def test_check_model_fit_false_when_model_exceeds_vram():
    # gemma3:12b (12.2B, Q4_K_M) needs ~6.8GB -- doesn't fit a 2GB card.
    fit = check_model_fit(12.2, "Q4_K_M", 2048)
    assert fit["known"] is True
    assert fit["fits"] is False
    assert fit["needed_mb"] > fit["usable_mb"]


def test_check_model_fit_subtracts_context_overhead_from_usable_vram():
    fit = check_model_fit(1, "Q4_K_M", 1024, ctx_overhead_mb=1024)
    assert fit["usable_mb"] == 0
    assert fit["fits"] is False  # any nonzero need can't fit zero usable VRAM


def test_check_model_fit_falls_back_to_q4_bytes_per_param_for_unknown_quant():
    known_quant = check_model_fit(7, "Q4_K_M", 8192)
    unknown_quant = check_model_fit(7, "some-future-quant", 8192)
    assert known_quant["needed_mb"] == unknown_quant["needed_mb"]
