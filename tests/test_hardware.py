"""
Coverage for core/hardware.py's pure arithmetic -- no subprocess, no GPU
needed. detect_gpu_vram_mb()/detect_system_ram_mb()/probe() themselves are
platform-dependent and best-effort by design (see the module docstring);
what's testable and load-bearing without real hardware is the arithmetic
check_model_fit() runs on whatever numbers it's given.
"""

from openllm_cbench.core import hardware as H
from openllm_cbench.core.hardware import (
    check_model_fit, fit_assessment, is_moe, parse_active_params_b,
)


# --- Apple Silicon GPU budget ---------------------------------------------

# Ollama 0.35.0's startup line on a 32 GB Apple M5, verbatim.
METAL_LINE = (
    'time=2026-10-02T18:06:30.519+01:00 level=INFO source=types.go:32 '
    'msg="inference compute" id=0 filter_id=0 library=Metal compute=0.0 name=MTL0 '
    'description="Apple M5" libdirs="" driver=0.0 pci_id="" type=iGPU '
    'total="25.0 GiB" available="25.0 GiB"'
)


def _sysctl(values):
    """A stand-in for _run() that answers `sysctl -n <key>` from a dict."""
    def run(cmd, timeout=10):
        if cmd[:2] == ["sysctl", "-n"] and cmd[2] in values:
            return values[cmd[2]]
        return None
    return run


def test_metal_total_is_read_from_the_last_inference_compute_line():
    older = METAL_LINE.replace("25.0 GiB", "21.3 GiB")
    assert H.parse_metal_total_mb(older + "\n" + METAL_LINE) == 25600
    assert H.parse_metal_total_mb(METAL_LINE.replace('total="25.0 GiB"', 'total="512.0 MiB"')) == 512
    assert H.parse_metal_total_mb('msg="inference compute" library=CUDA total="24.0 GiB"') is None
    assert H.parse_metal_total_mb("") is None


def test_apple_budget_is_not_total_ram(monkeypatch):
    # Measured: doctor reported 32768 MB and a ~56B band on a Mac whose
    # GPU could wire 25 GiB. The cap is what decides whether a model spills.
    monkeypatch.setattr(H, "_run", _sysctl({"hw.optional.arm64": "1\n",
                                            "iogpu.wired_limit_mb": "0\n",
                                            "hw.memsize": str(32 * 1024 ** 3)}))
    monkeypatch.setattr(H, "_metal_total_from_ollama_log", lambda: 25600)
    assert H.detect_apple_gpu_budget_mb() == (25600, "Metal limit reported by Ollama")


def test_an_operator_wired_limit_wins_over_the_logged_figure(monkeypatch):
    monkeypatch.setattr(H, "_run", _sysctl({"hw.optional.arm64": "1\n",
                                            "iogpu.wired_limit_mb": "28000\n"}))
    monkeypatch.setattr(H, "_metal_total_from_ollama_log", lambda: 25600)
    assert H.detect_apple_gpu_budget_mb() == (28000, "iogpu.wired_limit_mb")


def test_without_a_log_the_budget_is_an_estimate_and_says_so(monkeypatch):
    monkeypatch.setattr(H, "_run", _sysctl({"hw.optional.arm64": "1\n",
                                            "hw.memsize": str(32 * 1024 ** 3)}))
    monkeypatch.setattr(H, "_metal_total_from_ollama_log", lambda: None)
    mb, source = H.detect_apple_gpu_budget_mb()
    assert mb == 24576 and source.startswith("estimated")


def test_an_intel_mac_reports_no_gpu_rather_than_ram_as_vram(monkeypatch):
    monkeypatch.setattr(H, "_run", _sysctl({"hw.optional.arm64": "0\n",
                                            "hw.memsize": str(32 * 1024 ** 3)}))
    monkeypatch.setattr(H.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(H.shutil, "which", lambda name: None)
    mb, source = H.detect_apple_gpu_budget_mb()
    assert mb is None and "Intel" in source
    assert H.detect_gpu_vram_mb() == (None, None)


def test_the_report_says_where_an_apple_figure_came_from():
    report = H.format_report({"gpu_vendor": "apple-unified", "gpu_vram_mb": 25600,
                              "gpu_vram_source": "Metal limit reported by Ollama",
                              "system_ram_mb": 32768, "advisory_max_params_b_q4": 43.6})
    assert "25600 MB of unified memory usable by the GPU (Metal limit reported by Ollama)" in report


# --- MoE detection -------------------------------------------------------

def test_parse_active_params_from_the_community_tag_convention():
    assert parse_active_params_b("qwen3:30b-a3b") == 3.0
    assert parse_active_params_b("nemotron-3-nano:30b-a3b-q4_K_M") == 3.0
    assert parse_active_params_b("some-model:80b-a13b") == 13.0


def test_parse_active_params_returns_none_for_a_dense_tag():
    assert parse_active_params_b("muse-glimmer:30b") is None
    assert parse_active_params_b("qwen3:14b") is None
    assert parse_active_params_b("") is None


def test_is_moe_from_either_architecture_or_tag():
    assert is_moe(architecture="qwen3moe") is True
    assert is_moe(architecture="nemotron_h_moe") is True
    assert is_moe(architecture="llama", tag="x:30b-a3b") is True
    assert is_moe(architecture="muse-glimmer", tag="muse-glimmer:30b") is False


# --- Fit assessment ------------------------------------------------------

def test_fit_assessment_separates_a_spilling_moe_from_a_spilling_dense_model():
    # Both exceed VRAM by a similar margin, but an MoE only reads a
    # fraction of its weights per token. Ranking them identically would
    # talk someone out of a model that is actually usable.
    dense = fit_assessment(12282, tag="muse-glimmer:30b", architecture="muse-glimmer",
                            params_b=27.9, quant="Q4_K_M", size_mb=17316)
    moe = fit_assessment(12282, tag="qwen3:30b-a3b", architecture="qwen3moe",
                          params_b=30.5, quant="Q4_K_M", size_mb=17697)

    assert dense["tier"] == "spills" and moe["tier"] == "spills"
    assert dense["moe"] is False and moe["moe"] is True
    assert dense["headline"] != moe["headline"]
    assert "5-20x slower" in dense["note"]
    assert "mixture-of-experts" in moe["note"]
    assert "not unusable" in moe["note"]


def test_fit_assessment_tiers():
    # Comfortably under, just under, and over the usable budget.
    assert fit_assessment(12282, size_mb=2000)["tier"] == "fits"
    assert fit_assessment(12282, size_mb=11000)["tier"] == "tight"
    assert fit_assessment(12282, size_mb=17316)["tier"] == "spills"


def test_fit_assessment_tight_warns_about_context_headroom():
    # "Fits" with no headroom still spills once context grows -- the user
    # needs to know that before a long run, not during one.
    a = fit_assessment(12282, size_mb=11000)
    assert "headroom" in a["note"]


def test_fit_assessment_unknown_when_nothing_to_compare():
    a = fit_assessment(None, tag="x:1b")
    assert a["tier"] == "unknown"
    assert a["note"] == ""


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


def test_check_model_fit_prefers_a_real_on_disk_size_over_the_estimate():
    # The endpoint reports each model's actual size. That's a measurement,
    # not a calculation from a rough bytes-per-param table, so it wins
    # whenever it's available.
    estimated = check_model_fit(27.9, "Q4_K_M", 12282)
    measured = check_model_fit(27.9, "Q4_K_M", 12282, size_mb=17316)
    assert estimated["source"] == "estimate"
    assert measured["source"] == "size"
    assert measured["needed_mb"] == 17316
    assert measured["needed_mb"] != estimated["needed_mb"]
    assert measured["fits"] is False


def test_check_model_fit_uses_real_size_even_with_no_params_or_quant():
    # An uncatalogued model may have no params_b/quant anywhere, but the
    # endpoint still knows how big the file is -- enough to warn on.
    fit = check_model_fit(None, None, 12282, size_mb=17316)
    assert fit["known"] is True
    assert fit["fits"] is False


def test_check_model_fit_falls_back_to_q4_bytes_per_param_for_unknown_quant():
    known_quant = check_model_fit(7, "Q4_K_M", 8192)
    unknown_quant = check_model_fit(7, "some-future-quant", 8192)
    assert known_quant["needed_mb"] == unknown_quant["needed_mb"]
