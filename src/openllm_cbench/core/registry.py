"""
Model catalogue: what approach a given model tag actually needs to
produce valid data from this framework.

The lab-testing observation this exists to operationalize: different
models need different config to work at all -- a raised num_predict here,
an effort-level sweep instead of a boolean think toggle there, a known
channel-merge state at think=off somewhere else. The fix is not custom
per-model code (an if/elif ladder in every suite would rot the moment a
new model needs a new special case) -- it's a declarative catalogue every
suite consults generically: "for this model tag, apply these overrides."
The suite code never branches on a model name; only the data does.

Two layers, merged:
  - the PACKAGED SEED (data/models/verified.json) -- a small set of worked
    examples shipped with the package, read-only.
  - a LOCAL OVERLAY (default ./models.json in the current directory,
    override with --registry-file or OPENLLM_CBENCH_MODELS_FILE) -- grows
    as you gate-check your own models. `cbench gate --model X --save`
    writes to this file; it is never the packaged seed, so your local
    catalogue and the package's worked examples never collide on disk.

An overlay entry for a tag that also exists in the seed REPLACES that
seed entry entirely (not a deep merge) -- if your own gate check found
something different from the shipped example, your result should win
outright, not blend confusingly with a stale one.

No suite refuses to run against an unlisted model tag -- the framework
talks to any HTTP chat endpoint and is intentionally not bounded to a
fixed roster. An unlisted model just runs with the suite's own hardcoded
defaults and an "ungated" banner, not a hard stop.
"""

import json
import os
from pathlib import Path

from openllm_cbench.core.paths import data_file

DEFAULT_OVERLAY_FILENAME = "models.json"


def default_overlay_path(override=None):
    if override:
        return Path(override)
    env = os.environ.get("OPENLLM_CBENCH_MODELS_FILE")
    if env:
        return Path(env)
    return Path.cwd() / DEFAULT_OVERLAY_FILENAME


def _load_seed():
    try:
        text = data_file("models", "verified.json").read_text(encoding="utf-8")
        data = json.loads(text)
        return data.get("models", {})
    except Exception:
        return {}


def _load_overlay(path):
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("models", {})
    except Exception:
        return {}


def load_registry(overlay_path=None):
    """Returns {"models": {...}}, seed entries with any overlay entries
    replacing same-tag seed entries outright. Never raises -- a
    missing/malformed file at either layer degrades to skipping that
    layer, since an unlisted or unreadable-catalogue model must still be
    runnable."""
    models = dict(_load_seed())
    models.update(_load_overlay(default_overlay_path(overlay_path)))
    return {"models": {tag: _normalize_entry(e) for tag, e in models.items()}}


def _normalize_entry(entry):
    """Heals a catalogue entry read from disk into the schema's own
    types, without rewriting the file.

    Exists for `params_b`, which the schema documents as a number of
    billions. Ollama reports sub-billion models in millions ("751.63M"),
    and `cbench gate --save` used to store that string verbatim -- so
    every small model catalogued before that was fixed still has a
    non-numeric params_b on disk. Anything doing arithmetic on it (the
    Score screen's VRAM fit warning) silently got nothing, for exactly
    the models most likely to be run on a small machine. Normalising on
    read fixes every existing entry at once, where a migration would fix
    only the people who ran it.

    Returns the entry unchanged if there's nothing to heal, so this stays
    cheap on the common path."""
    if not isinstance(entry, dict):
        return entry
    raw = entry.get("params_b")
    if not isinstance(raw, str):
        return entry
    from openllm_cbench.core.gate import _numeric_params_b
    healed = _numeric_params_b(raw)
    if healed == raw:
        return entry
    entry = dict(entry)
    entry["params_b"] = healed
    return entry


def lookup(model, registry=None):
    """Returns the registry entry for a model tag, or None if unlisted.
    Exact match only -- a fork and its base are different tags with
    potentially different config needs, and should never be conflated by
    a substring or prefix match."""
    registry = registry if registry is not None else load_registry()
    return registry.get("models", {}).get(model)


def config_overrides_for(model, keys=None, registry=None):
    """Returns the subset of this model's config_overrides matching
    `keys` (or all of them if keys is None). Empty dict for an unlisted
    model or one with no overrides -- always safe to call unconditionally
    before falling back to a suite's own hardcoded defaults."""
    entry = lookup(model, registry)
    overrides = (entry or {}).get("config_overrides", {})
    if keys is None:
        return dict(overrides)
    return {k: overrides[k] for k in keys if k in overrides}


def thinking_mode_for(model, registry=None):
    """Returns the catalogued 'thinking_mode' for a model, or None if
    unlisted/unspecified. Known values: 'effort' (graded reasoning depth,
    channel suite should default to an --effort sweep instead of
    --think), 'ignores_think' (the think param has no effect -- channel
    suite should not bother sweeping it), or None (ordinary boolean
    think toggle, the default assumption)."""
    entry = lookup(model, registry)
    return (entry or {}).get("thinking_mode")


def save_entry(model, entry, overlay_path=None):
    """Writes/replaces one model's entry in the local overlay file,
    creating the file if it doesn't exist yet. Does not touch the
    packaged seed. Returns the path written to."""
    path = default_overlay_path(overlay_path)
    data = {"models": _load_overlay(path)}
    data["models"][model] = entry
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def banner(model, registry=None):
    """Returns a short status line for CLI output. Never blocks a run --
    this is purely informational."""
    entry = lookup(model, registry)
    if entry is None:
        return (
            f"[!] '{model}' is not in the model catalogue -- this run is UNGATED. "
            f"Results may be affected by an unverified tool-calling gap, a merged "
            f"reasoning channel, or a generation-budget truncation this framework has "
            f"no way to warn you about automatically. Run `cbench gate --model {model} "
            f"--save` first, or proceed and read the transcripts carefully."
        )
    status = "gated: " + ("clean" if not entry.get("caveats") else "gated with caveats -- see below")
    lines = [f"'{model}' found in the model catalogue ({status})."]
    for c in entry.get("caveats", []):
        lines.append(f"  - {c}")
    overrides = entry.get("config_overrides", {})
    if overrides:
        lines.append(f"  Applying catalogued config overrides: {overrides}")
    mode = entry.get("thinking_mode")
    if mode:
        lines.append(f"  Catalogued thinking mode: {mode}")
    return "\n".join(lines)
