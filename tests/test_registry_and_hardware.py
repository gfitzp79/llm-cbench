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


def test_registry_heals_a_params_b_saved_in_millions(tmp_path, monkeypatch):
    # `cbench gate --save` used to store Ollama's "751.63M" verbatim, so
    # every small model catalogued before that fix has a non-numeric
    # params_b on disk. Anything doing arithmetic on it (the Score
    # screen's VRAM fit warning) silently got nothing -- for exactly the
    # models most likely to be run on a small machine. Normalising on
    # read heals every existing entry without rewriting anyone's file.
    import json

    from openllm_cbench.core.registry import load_registry, lookup

    overlay = tmp_path / "models.json"
    overlay.write_text(json.dumps({"models": {
        "tiny:1b": {"architecture": "x", "params_b": "751.63M", "quant": "Q4_K_M",
                    "tools": True, "thinking": False, "channel_separation": None,
                    "config_overrides": {}, "caveats": []},
    }}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))

    entry = lookup("tiny:1b", load_registry())
    assert entry["params_b"] == 0.75163
    assert isinstance(entry["params_b"], float)


def test_registry_leaves_an_unmeasurable_params_b_alone(tmp_path, monkeypatch):
    # "unknown" is the documented sentinel for a genuinely unmeasured
    # entry -- healing must not invent a number for it.
    import json

    from openllm_cbench.core.registry import load_registry, lookup

    overlay = tmp_path / "models.json"
    overlay.write_text(json.dumps({"models": {
        "mystery:1b": {"params_b": "unknown", "quant": "unknown"},
    }}), encoding="utf-8")
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(overlay))

    assert lookup("mystery:1b", load_registry())["params_b"] == "unknown"
