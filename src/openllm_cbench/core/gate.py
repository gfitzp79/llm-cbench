"""
Model gate-check: verifies a model tag actually supports what this
framework needs before a real suite run is spent discovering a gap the
hard way. Ported into shipped code from a checklist that used to live
only as tribal knowledge -- run this on any new model before trusting its
results, and definitely before comparing it against another model.

Checks, in order:
  1. capabilities   -- does the endpoint report tools/thinking support
  2. tool call      -- does a real tool call round-trip and parse cleanly
  3. channel split, BOTH think states -- a model can split content/thinking
     cleanly at one state and merge them at the other; checking only one
     state is a known way to miss a real bug
  4. sampling params -- read from the endpoint's reported model info, so a
     base/fork pair can be diffed for an unmatched parameter before either
     one is ever run through a real suite

Uses the chat endpoint's model-info route and one or two chat calls -- no
suite-specific state, safe to run standalone.
"""

import json
import re

import requests

from openllm_cbench.core.endpoint import chat_url, show_url

_PARAM_RE = re.compile(r"^\s*PARAMETER\s+(\S+)\s+(.+?)\s*$", re.MULTILINE)


def fetch_show_info(model, endpoint=None, timeout=30):
    """One call to the endpoint's model-info route. Returns the parsed
    JSON dict. Raises on failure -- unlike the runtime checks inside a
    suite, a gate check should fail loudly and immediately if the
    endpoint can't even describe the model."""
    resp = requests.post(endpoint or show_url(), json={"model": model}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def parse_sampling_params(modelfile_text):
    """Extracts PARAMETER <name> <value> lines from a Modelfile-shaped
    text blob. Returns {} on anything unparseable rather than raising --
    this is diagnostic output, not something that should abort a gate
    check over a formatting surprise."""
    if not modelfile_text:
        return {}
    try:
        return {name: value for name, value in _PARAM_RE.findall(modelfile_text)}
    except Exception:
        return {}


def check_tool_call(model, endpoint=None, timeout=60):
    """Sends one trivial tool-schema prompt and checks whether a
    well-formed tool_calls block comes back. Returns (ok, detail)."""
    tools = [{
        "type": "function",
        "function": {
            "name": "echo",
            "description": "Echo back the given text.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"],
            },
        },
    }]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Call the echo tool with text='gate-check-ok'."}],
        "tools": tools,
        "stream": False,
        "options": {"num_ctx": 4096, "num_predict": 512},
    }
    try:
        resp = requests.post(endpoint or chat_url(), json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return False, f"request failed: {e}"

    tool_calls = (data.get("message") or {}).get("tool_calls") or []
    if not tool_calls:
        return False, "no tool_calls in response -- model may not support tool calling, or needs a different prompt"
    fn = tool_calls[0].get("function", {})
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except Exception:
            return False, f"tool_calls present but arguments field is not valid JSON: {args!r}"
    if fn.get("name") != "echo" or "text" not in args:
        return False, f"tool call malformed or wrong tool: {fn}"
    return True, "well-formed tool call round-tripped correctly"


def check_channel_at(model, think_value, endpoint=None, num_predict=2048, timeout=90):
    """One chat call at a specific think state. Returns a dict with
    content, thinking, done_reason, and a merged_channel_suspected flag --
    same signature the channel suite's own runtime guard uses, so a gate
    check and a live run apply the identical test."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Explain briefly why the sky appears blue."}],
        "stream": False,
        "think": think_value,
        "options": {"num_ctx": 4096, "num_predict": num_predict, "presence_penalty": 0},
    }
    try:
        resp = requests.post(endpoint or chat_url(), json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

    msg = data.get("message", {})
    content = msg.get("content", "") or ""
    thinking = msg.get("thinking", "") or ""
    done_reason = data.get("done_reason", "") or ""
    merged_suspected = (not thinking.strip()) and ("<think>" in content.lower() or "</think>" in content.lower())
    return {
        "ok": True,
        "content_len": len(content),
        "thinking_len": len(thinking),
        "done_reason": done_reason,
        "merged_channel_suspected": merged_suspected,
        "truncated": done_reason == "length" and not content.strip(),
    }


def run_gate(model, base_url=None):
    """Runs the full gate check and returns a structured result dict.
    Never raises past this point -- every sub-check catches its own
    failure and reports it in the result instead.

    `base_url` is the endpoint's base (e.g. http://localhost:11434, same
    value core.endpoint.resolve_base_url() would produce) -- NOT a
    pre-built /api/chat or /api/show URL. This function derives both the
    chat and show endpoints from it itself, so a caller can't accidentally
    hand the chat URL to the show-info check (or vice versa) the way a
    single shared "endpoint" parameter invites."""
    from openllm_cbench.core.endpoint import chat_url as _chat_url, show_url as _show_url
    chat_endpoint = _chat_url(base_url)
    show_endpoint = _show_url(base_url)
    result = {"model": model, "chat_endpoint": chat_endpoint, "show_endpoint": show_endpoint}

    try:
        show_info = fetch_show_info(model, show_endpoint)
        caps = show_info.get("capabilities") or []
        result["capabilities"] = caps
        result["has_tools_capability"] = "tools" in caps
        result["has_thinking_capability"] = "thinking" in caps
        result["architecture"] = (show_info.get("details") or {}).get("family", "unknown")
        result["param_size"] = (show_info.get("details") or {}).get("parameter_size", "unknown")
        result["quant"] = (show_info.get("details") or {}).get("quantization_level", "unknown")
        result["sampling_params"] = parse_sampling_params(show_info.get("modelfile", ""))
        result["show_info_ok"] = True
    except Exception as e:
        result["show_info_ok"] = False
        result["show_info_error"] = str(e)
        result["capabilities"] = []
        result["has_tools_capability"] = None
        result["has_thinking_capability"] = None
        result["sampling_params"] = {}

    tool_ok, tool_detail = check_tool_call(model, chat_endpoint)
    result["tool_call_ok"] = tool_ok
    result["tool_call_detail"] = tool_detail

    if result.get("has_thinking_capability"):
        result["channel_think_on"] = check_channel_at(model, True, chat_endpoint)
        result["channel_think_off"] = check_channel_at(model, False, chat_endpoint)
    else:
        result["channel_think_on"] = None
        result["channel_think_off"] = None

    caveats = []
    if not result.get("has_tools_capability"):
        caveats.append(
            "Endpoint does not report a tools capability -- the containment and "
            "persistence suites need real tool calling and will not produce valid data."
        )
    elif not tool_ok:
        caveats.append(f"Tool call check failed: {tool_detail}")
    for label, key in (("think=on", "channel_think_on"), ("think=off", "channel_think_off")):
        ch = result.get(key)
        if ch and ch.get("merged_channel_suspected"):
            caveats.append(
                f"Channel merge suspected at {label} -- reasoning text is leaking into "
                f"the visible answer instead of a separate field. Every channel-suite "
                f"verdict at {label} would be unreliable."
            )
    result["caveats"] = caveats
    result["clean"] = not caveats and bool(result.get("show_info_ok")) and tool_ok

    return result


def _numeric_params_b(param_size):
    """Ollama's /api/show reports parameter_size as a string like '14.8B'
    (or '20.9B', 'unknown', etc) for most models -- but a small enough
    model (found live: smollm2:135m) gets reported in millions instead,
    e.g. '134.52M', not '0.13B'. verified.json's own schema documents
    `params_b` as a plain number of *billions* regardless, so an 'M'
    value is divided by 1000 on the way in, not just suffix-stripped the
    way a 'B' value is -- treating '134.52M' as if it meant 134.52B would
    be a three-order-of-magnitude error in anything that reads this field
    (e.g. core/hardware.py's VRAM-fit check).

    Falls back to the original string unchanged if it doesn't match
    either expected shape, rather than guessing -- this is also how a
    genuinely unmeasured entry's literal 'unknown' string round-trips
    unchanged, and callers that need a real number must check
    isinstance(..., (int, float)) rather than truthiness, since a
    fallen-back string is still truthy."""
    if not isinstance(param_size, str):
        return param_size
    s = param_size.strip()
    upper = s.upper()
    divisor = 1.0
    if upper.endswith("B"):
        s = s[:-1]
    elif upper.endswith("M"):
        s = s[:-1]
        divisor = 1000.0
    try:
        n = float(s) / divisor
        return int(n) if n == int(n) else n
    except ValueError:
        return param_size


def to_registry_entry(result):
    """Converts a run_gate() result into the shape core/registry.py's
    catalogue expects (same schema as data/models/verified.json). Only
    fills in what a gate check can actually discover automatically --
    `config_overrides` is left empty (a generic capability/channel check
    doesn't know the right num_predict for a given model's task-shaped
    behaviour; that needs a real run to discover, same as it did for
    every worked example already in the packaged seed). Hand-edit the
    saved entry to add config_overrides once you've found what a real run
    needed."""
    channel_separation = None
    if result.get("channel_think_on") is not None or result.get("channel_think_off") is not None:
        def _label(ch):
            if ch is None:
                return "n/a"
            if not ch.get("ok"):
                return f"error: {ch.get('error')}"
            if ch.get("merged_channel_suspected"):
                return "UNRELIABLE (merge suspected)"
            return "clean"
        channel_separation = {
            "think_on": _label(result.get("channel_think_on")),
            "think_off": _label(result.get("channel_think_off")),
        }
    return {
        "architecture": result.get("architecture", "unknown"),
        "params_b": _numeric_params_b(result.get("param_size", "unknown")),
        "quant": result.get("quant", "unknown"),
        "tools": bool(result.get("has_tools_capability")),
        "thinking": bool(result.get("has_thinking_capability")),
        "channel_separation": channel_separation,
        "config_overrides": {},
        "caveats": list(result.get("caveats", [])),
    }


def render_gate_report(result):
    L = [f"# Gate check -- `{result['model']}`", ""]
    if not result.get("show_info_ok"):
        L.append(f"**Could not reach the endpoint's model-info route**: {result.get('show_info_error')}")
        L.append("Nothing further was checked. Confirm the endpoint is reachable and the model is pulled.")
        return "\n".join(L) + "\n"

    L += [
        f"- Architecture: `{result.get('architecture')}`, params: `{result.get('param_size')}`, "
        f"quant: `{result.get('quant')}`",
        f"- Capabilities reported: `{', '.join(result.get('capabilities', [])) or '(none)'}`",
        f"- Tool call check: {'OK' if result['tool_call_ok'] else 'FAILED'} -- {result['tool_call_detail']}",
    ]
    for label, key in (("think=on", "channel_think_on"), ("think=off", "channel_think_off")):
        ch = result.get(key)
        if ch is None:
            L.append(f"- Channel check at {label}: skipped (model does not report a thinking capability)")
        elif not ch.get("ok"):
            L.append(f"- Channel check at {label}: request failed -- {ch.get('error')}")
        else:
            flag = " [MERGE SUSPECTED]" if ch.get("merged_channel_suspected") else ""
            flag += " [TRUNCATED]" if ch.get("truncated") else ""
            L.append(
                f"- Channel check at {label}: content={ch['content_len']} chars, "
                f"thinking={ch['thinking_len']} chars, done_reason={ch['done_reason']!r}{flag}"
            )
    if result.get("sampling_params"):
        L.append(f"- Modelfile sampling parameters: `{result['sampling_params']}`")
        L.append(
            "  (If gate-checking a base/fork pair, diff this against the other arm "
            "explicitly -- an unmatched parameter here silently confounds any comparison.)"
        )

    L += ["", "## Result", ""]
    if result["clean"]:
        L.append("**Clean.** No caveats found. Safe to add to your own verified-model registry.")
    else:
        L.append("**Caveats found -- read before trusting a real run against this model:**")
        for c in result["caveats"]:
            L.append(f"- {c}")
    return "\n".join(L) + "\n"


def summarize_gate_output(lines):
    """Extracts a compact verdict from `cbench gate`'s own printed report
    (render_gate_report()'s exact text, captured verbatim as subprocess
    output by a caller like the TUI) without re-running the check or
    re-implementing it a second time.

    Exists because a caller watching gate as a subprocess only has an
    exit code (0/1, from run_gate()['clean']) and a wall of text to go
    on -- and a nonzero exit code alone conflates two very different
    situations a human reading it needs to tell apart: the endpoint
    genuinely couldn't be reached at all (a hard failure -- a real run
    against this model right after would hit the identical wall), versus
    the check ran fine and found real caveats (e.g. no tool-calling
    support) -- a softer, more specific finding this framework's own
    policy is to warn about, never block on.

    Returns {"hard_failure": bool, "reason": str|None, "clean": bool|None,
    "caveats": list[str]}. `clean` is None if neither a "Clean." nor a
    "Caveats found" line was seen at all (e.g. the process crashed before
    printing a report). Never raises -- an unrecognized or future-changed
    report shape just yields empty/None fields rather than guessing."""
    hard_failure = False
    reason = None
    clean = None
    caveats = []
    collecting = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("**Could not reach the endpoint's model-info route**"):
            hard_failure = True
            reason = stripped.split(":", 1)[1].strip() if ":" in stripped else stripped
            collecting = False
        elif stripped.startswith("**Clean.**"):
            clean = True
            collecting = False
        elif stripped.startswith("**Caveats found"):
            clean = False
            collecting = True
        elif collecting:
            if stripped.startswith("- "):
                caveats.append(stripped[2:])
            elif stripped:
                collecting = False
    return {"hard_failure": hard_failure, "reason": reason, "clean": clean, "caveats": caveats}
