"""
Local model discovery: what's already pulled into the endpoint that the
model catalogue doesn't know about yet.

Deliberately scoped to the LOCAL endpoint's own `/api/tags` -- the same
daemon every suite already talks to, not ollama.com's remote library.
Ollama has no official API for browsing/searching its remote library (the
project's own team has an open feature request for one:
github.com/ollama/ollama/issues/9142); the unofficial alternatives are a
third-party API or HTML scraping, either of which would add a new
external trust boundary this framework doesn't otherwise have and could
silently break if a page layout changes. `/api/tags`, by contrast, is a
real, documented, stable local endpoint -- see docs.ollama.com/api/tags --
so that's the only source of truth here. If you want to add a model
that isn't pulled yet, `ollama pull <tag>` it yourself first; this module
picks up from there.
"""

import requests

from openllm_cbench.core.endpoint import tags_url
from openllm_cbench.core.gate import _numeric_params_b
from openllm_cbench.core.registry import load_registry


def list_local_models(base_url=None, timeout=30):
    """Returns the endpoint's own list of locally-pulled models --
    architecture/params/quant only, deliberately NOT `capabilities`.

    Verified live before trusting this shape (not assumed from Ollama's
    general docs): `/api/tags`'s own `capabilities` field can under-report
    relative to `/api/show` for the exact same model -- one real model on
    this project's own test machine reported `["completion"]` from
    `/api/tags` while `/api/show` correctly reported `["completion",
    "tools", "thinking"]`. Surfacing the unreliable field here would risk
    exactly the failure mode this framework's own gate check exists to
    catch (trusting a capability list without a real call) -- in reverse,
    a false NEGATIVE instead of gate.py's known false-positive case
    (a model that reports `tools` but fails the round-trip). `architecture`/
    `params_b`/`quant` were cross-checked against the same model's
    `/api/show` response and matched, so those are kept."""
    resp = requests.get(tags_url(base_url), timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    out = []
    for m in data.get("models", []):
        details = m.get("details") or {}
        out.append({
            "name": m.get("name") or m.get("model") or "",
            "size": m.get("size"),
            "modified_at": m.get("modified_at"),
            "architecture": details.get("family", "unknown"),
            "params_b": _numeric_params_b(details.get("parameter_size", "unknown")),
            "quant": details.get("quantization_level", "unknown"),
        })
    return [m for m in out if m["name"]]


def find_uncatalogued(local_models, registry=None):
    """Returns the subset of local_models whose exact tag has no entry
    (seed or overlay) in the model catalogue. Exact match only, same rule
    as core/registry.py:lookup() -- a tag one character off from a
    catalogued entry is still reported here, deliberately; silently
    treating near-matches as covered would hide real gaps."""
    reg = registry if registry is not None else load_registry()
    catalogued = set(reg.get("models", {}).keys())
    return [m for m in local_models if m["name"] not in catalogued]


def format_params(params_b):
    """A parameter count in billions as `discover` and `catalogue` print
    it: `0.75163B`, or `unknown params` when it could not be measured
    (params_b holds the literal string "unknown"), never `unknownB`."""
    if isinstance(params_b, (int, float)) and not isinstance(params_b, bool):
        return f"{params_b}B"
    return "unknown params"


def format_size(num_bytes):
    if not isinstance(num_bytes, (int, float)):
        return "unknown size"
    gb = num_bytes / (1024 ** 3)
    return f"{gb:.1f} GB"
