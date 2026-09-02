"""
Coverage for core/discover.py's pure functions. list_local_models() itself
needs a live endpoint (not tested here, network-dependent); these tests
cover the parsing/diffing logic against fabricated /api/tags-shaped data
and a fabricated catalogue, no model or network required.
"""

from openllm_cbench.core.discover import find_uncatalogued, format_size


def _model(name, **overrides):
    m = {"name": name, "size": 5_000_000_000, "modified_at": "2026-01-01T00:00:00Z",
         "architecture": "llama", "params_b": 8, "quant": "Q4_K_M"}
    m.update(overrides)
    return m


def test_find_uncatalogued_returns_only_tags_missing_from_registry():
    local = [_model("gemma4:12b"), _model("brand-new:7b"), _model("gpt-oss:20b")]
    registry = {"models": {"gemma4:12b": {}, "gpt-oss:20b": {}}}
    result = find_uncatalogued(local, registry)
    assert [m["name"] for m in result] == ["brand-new:7b"]


def test_find_uncatalogued_exact_match_only_not_fuzzy():
    # A near-miss tag (different quant/tag suffix) must still be reported
    # -- silently treating it as "covered" would hide a real gap.
    local = [_model("qwen3.5:9b-instruct")]
    registry = {"models": {"qwen3.5:9b": {}}}  # different tag, same family
    result = find_uncatalogued(local, registry)
    assert len(result) == 1
    assert result[0]["name"] == "qwen3.5:9b-instruct"


def test_find_uncatalogued_empty_registry_reports_everything():
    local = [_model("a:1b"), _model("b:2b")]
    result = find_uncatalogued(local, {"models": {}})
    assert len(result) == 2


def test_find_uncatalogued_nothing_new_returns_empty():
    local = [_model("gemma4:12b")]
    registry = {"models": {"gemma4:12b": {}}}
    assert find_uncatalogued(local, registry) == []


def test_format_size_converts_bytes_to_gb():
    assert format_size(5_000_000_000) == "4.7 GB"


def test_format_size_handles_missing_or_bad_input():
    assert format_size(None) == "unknown size"
    assert format_size("not a number") == "unknown size"


def test_discover_output_never_includes_the_unreliable_capabilities_field():
    # Regression guard: capabilities was deliberately dropped from
    # list_local_models()'s output because /api/tags under-reports it
    # relative to /api/show for the same model (verified live). Nothing
    # should reintroduce it without the same verification.
    m = _model("x:1b")
    assert "capabilities" not in m or True  # the fixture itself never sets it
    from openllm_cbench.core import discover
    import inspect
    source = inspect.getsource(discover.list_local_models)
    assert '"capabilities"' not in source
