"""
Tests for the verified-model registry and the advisory hardware probe.
No network calls, no model calls.
"""

from openllm_cbench.core.hardware import recommend_band, BYTES_PER_PARAM
from openllm_cbench.core.registry import load_registry, lookup, banner


def test_recommend_band_arithmetic():
    # 12288 MB VRAM, default 1024 MB overhead reserved -> 11264 MB usable.
    # At Q4_K_M (0.55 GB/B == 0.55*1024 MB/B), that's 11264/(0.55*1024) B.
    got = recommend_band(12288)
    want = round((12288 - 1024) / (BYTES_PER_PARAM["Q4_K_M"] * 1024), 1)
    assert got == want
    assert got > 0


def test_recommend_band_none_when_no_vram():
    assert recommend_band(None) is None
    assert recommend_band(0) is None


def test_registry_loads_and_is_a_dict():
    reg = load_registry()
    assert isinstance(reg, dict)
    assert "models" in reg
    assert isinstance(reg["models"], dict)


def test_registry_has_seed_entries():
    reg = load_registry()
    # At least the worked examples shipped with the package should be present.
    assert len(reg["models"]) >= 1


def test_lookup_unlisted_model_returns_none():
    reg = load_registry()
    assert lookup("definitely-not-a-real-model-tag:999b", reg) is None


def test_lookup_exact_match_only_no_substring_match():
    """A fork tag must never match its base's registry entry (or vice
    versa) via substring/prefix logic -- exact key match only."""
    reg = {"models": {"base-model:8b": {"tools": True}}}
    assert lookup("base-model:8b-fork", reg) is None
    assert lookup("base-model:8b", reg) is not None


def test_banner_for_unlisted_model_says_ungated():
    text = banner("not-a-real-model:1b", {"models": {}})
    assert "UNGATED" in text
    assert "not-a-real-model:1b" in text


def test_banner_for_listed_clean_model():
    reg = {"models": {"clean-model:1b": {"caveats": [], "config_overrides": {}}}}
    text = banner("clean-model:1b", reg)
    assert "clean" in text.lower()
    assert "UNGATED" not in text


def test_banner_for_listed_model_with_caveats():
    reg = {"models": {"caveat-model:1b": {"caveats": ["watch out for X"], "config_overrides": {"num_predict": 4096}}}}
    text = banner("caveat-model:1b", reg)
    assert "watch out for X" in text
    assert "num_predict" in text
