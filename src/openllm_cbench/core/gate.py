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

from openllm_cbench.core.delimiters import merge_evidence
from openllm_cbench.core.endpoint import chat_url, show_url, describe_request_failure

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


def warm_up(model, endpoint=None, timeout=600):
    """One trivial generation to force the model to load, timed.

    Exists because every other check here had a fixed timeout sized for a
    model that fits in VRAM. A model too large for the GPU spills into
    system RAM and can take over half a minute just to answer "say OK" --
    found live on a 27.9B dense model on a 12GB card, where all three
    chat checks timed out (60s + 90s + 90s) and the gate then reported
    "Tool call check failed", which reads as "this model can't call
    tools". The endpoint had reported `tools` in its own capabilities the
    whole time. A capability verdict derived from a stopwatch is worse
    than no verdict.

    Returns (ok, load_seconds, tokens_per_sec). Two calls, because one
    can't separate the two numbers: the first pays the load cost and
    measures it, the second runs against an already-loaded model and so
    measures generation rate alone. That rate is what actually sizes the
    channel checks, which generate up to 2048 tokens -- extrapolating
    those from a one-token call would be guesswork.

    tokens_per_sec is None if it couldn't be measured; callers fall back
    to their original fixed timeouts."""
    import time

    url = endpoint or chat_url()

    def _call(num_predict, budget):
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Count from one to twenty."}],
            "stream": False,
            "options": {"num_ctx": 4096, "num_predict": num_predict},
        }
        start = time.time()
        resp = requests.post(url, json=payload, timeout=budget)
        resp.raise_for_status()
        return resp.json(), time.time() - start

    try:
        _, load_seconds = _call(1, timeout)
    except Exception:
        return False, 0.0, None

    try:
        data, gen_seconds = _call(32, timeout)
        # eval_count is the endpoint's own count of tokens generated --
        # more honest than assuming it produced exactly what was asked for.
        produced = (data or {}).get("eval_count") or 32
        rate = produced / gen_seconds if gen_seconds > 0 else None
    except Exception:
        rate = None

    return True, load_seconds, rate


def scaled_timeout(base, load_seconds, tokens_per_sec=None, num_predict=None, ceiling=300):
    """A timeout for a real check, sized from what this model actually
    does on this machine.

    Never shortens a check below its original budget -- this only ever
    buys a slow model more room, so a fast model behaves exactly as it
    did before this existed. Capped at `ceiling`: past some point the
    right answer is to report "could not be verified" and let the user
    decide, not to block a gate check for twenty minutes. That cap is why
    an unverified channel check must produce a caveat rather than a
    silent pass."""
    need = float(base)
    if load_seconds:
        need = max(need, load_seconds * 4)
    if tokens_per_sec and num_predict:
        need = max(need, load_seconds + (num_predict / tokens_per_sec) * 1.5 + 10)
    return int(min(max(need, base), ceiling))


def check_tool_call(model, endpoint=None, timeout=60):
    """Sends one trivial tool-schema prompt and checks whether a
    well-formed tool_calls block comes back. Returns (ok, detail).

    A timeout is reported as its own distinct detail rather than a
    generic failure: "didn't finish in N s" and "answered, but not with a
    tool call" are completely different findings, and only the second one
    says anything about the model's capabilities.

    TRIED TWICE, AND THE SECOND TRY IS THE POINT. A model whose reasoning
    channel is on by default can emit a trace and then stop without ever
    producing the tool call, while the same model called with
    `think=False` calls the tool immediately. One probe would report
    "model may not support tool calling" about a model that does, and the
    pre-flight would refuse to run S1 and S3 on it -- a missing
    measurement dressed up as a capability finding.

    Measured on `SuhasDevmane55/geollm-qwen3-4b-v8`: no tool call with
    thinking on (82-char trace, `done_reason` "stop", 23 tokens -- not a
    budget exhaustion), one well-formed call with thinking off. Its base
    model `qwen3:4b` calls the tool in both states, so this is something
    the fine-tune changed, and nothing in the model's metadata says so.

    Which state succeeded is reported, because "tool calling works" and
    "tool calling works only with the reasoning channel off" are
    different facts about a model and the second one constrains how the
    suites should be run."""
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
    def _attempt(think_value):
        """(ok, detail, retryable). retryable=False for a timeout or a
        transport failure, neither of which a second call would fix."""
        payload = {
            "model": model,
            "messages": [{"role": "user",
                          "content": "Call the echo tool with text='gate-check-ok'."}],
            "tools": tools,
            "stream": False,
            "options": {"num_ctx": 4096, "num_predict": 512},
        }
        if think_value is not None:
            payload["think"] = think_value
        try:
            resp = requests.post(endpoint or chat_url(), json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.Timeout:
            return False, (f"TIMEOUT after {timeout}s -- this says nothing about whether the "
                            f"model supports tool calling, only that it didn't answer in time "
                            f"on this machine (typically a model too large for the available "
                            f"VRAM)"), False
        except Exception as e:
            # A model with no reasoning channel returns HTTP 400 on any
            # think request, so the retry failing this way says nothing
            # new -- the caller keeps the first attempt's finding.
            #
            # Described rather than stringified: a bare "400 Client Error
            # for url ..." reads as a connection problem, and the endpoint
            # answering promptly with a rejection is the opposite of one.
            return False, describe_request_failure(e, endpoint or chat_url()), False

        tool_calls = (data.get("message") or {}).get("tool_calls") or []
        if not tool_calls:
            return False, "no tool_calls in response", True
        fn = tool_calls[0].get("function", {})
        args = fn.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                return False, (f"tool_calls present but arguments field is not valid "
                                f"JSON: {args!r}"), True
        if fn.get("name") != "echo" or "text" not in args:
            return False, f"tool call malformed or wrong tool: {fn}", True
        return True, "well-formed tool call round-tripped correctly", True

    ok, detail, retryable = _attempt(None)
    if ok or not retryable:
        return ok, detail

    # Second try with the reasoning channel explicitly off. See the
    # docstring: this is the difference between "cannot tool-call" and
    # "cannot tool-call while thinking", and only the first should stop a run.
    ok_off, detail_off, _ = _attempt(False)
    if ok_off:
        return True, ("well-formed tool call round-tripped, but ONLY with the reasoning "
                       "channel off (`think=false`); with thinking on the model returned no "
                       "tool call. Run this model's tool-using suites (S1, S3) with thinking "
                       "disabled, and treat any tool-use result collected with thinking on "
                       "as unreliable for this model")
    return False, (f"{detail} with thinking on, and {detail_off} with `think=false` either "
                    f"-- the model does not round-trip a tool call in either reasoning state, "
                    f"so it may not support tool calling at all, or may need a different prompt")


def check_channel_at(model, think_value, endpoint=None, num_predict=2048, timeout=90,
                     extra_markers=()):
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
    # Was a hard-coded `<think>` test whose docstring claimed it matched
    # the channel suite's guard. It did, until the suite's guard grew to
    # a four-delimiter family and this copy did not -- so a model using
    # any other convention gate-checked CLEAN and was catalogued that
    # way, then had every row of the channel suite flagged. One
    # definition now, in core/delimiters.py.
    merged_evidence = merge_evidence(content, thinking, extra_markers)
    merged_suspected = bool(merged_evidence)
    return {
        "ok": True,
        "content_len": len(content),
        "thinking_len": len(thinking),
        "done_reason": done_reason,
        "merged_channel_suspected": merged_suspected,
        "merge_evidence": merged_evidence,
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

    # Load the model once and time it, then size every later timeout from
    # what this machine actually does with it -- see warm_up()'s docstring
    # for the failure this prevents.
    warm_ok, warm_seconds, tokens_per_sec = warm_up(model, chat_endpoint)
    result["warm_up_ok"] = warm_ok
    result["warm_up_seconds"] = round(warm_seconds, 1)
    result["tokens_per_sec"] = round(tokens_per_sec, 1) if tokens_per_sec else None
    result["slow_load"] = bool(warm_ok and (warm_seconds > 10 or
                                             (tokens_per_sec and tokens_per_sec < 10)))

    tool_ok, tool_detail = check_tool_call(
        model, chat_endpoint,
        timeout=scaled_timeout(60, warm_seconds, tokens_per_sec, num_predict=512))
    result["tool_call_ok"] = tool_ok
    result["tool_call_detail"] = tool_detail
    result["tool_call_timed_out"] = tool_detail.startswith("TIMEOUT")

    if result.get("has_thinking_capability"):
        channel_timeout = scaled_timeout(90, warm_seconds, tokens_per_sec, num_predict=2048)
        # If this model is already catalogued with its own reasoning
        # delimiters, the gate honours them. A re-gate that ignored what
        # an operator had already gate-checked and written down would
        # keep reporting clean on exactly the model they had recorded as
        # needing special handling.
        try:
            from openllm_cbench.core.registry import delimiters_for
            extra = delimiters_for(model)
        except Exception:
            extra = ()
        result["catalogued_delimiters"] = list(extra)
        result["channel_think_on"] = check_channel_at(
            model, True, chat_endpoint, timeout=channel_timeout, extra_markers=extra)
        result["channel_think_off"] = check_channel_at(
            model, False, chat_endpoint, timeout=channel_timeout, extra_markers=extra)
    else:
        result["channel_think_on"] = None
        result["channel_think_off"] = None

    caveats = []
    if result.get("slow_load"):
        caveats.append(
            f"SLOW: a one-token generation took {result['warm_up_seconds']}s, which usually "
            f"means this model doesn't fit in available VRAM and is running partly on CPU. "
            f"Every suite below will be correspondingly slow -- a full assessment may take "
            f"hours. Nothing about the model's behaviour is wrong; budget accordingly."
        )
    if not result.get("has_tools_capability"):
        caveats.append(
            "Endpoint does not report a tools capability -- the containment and "
            "persistence suites need real tool calling and will not produce valid data."
        )
    elif result.get("tool_call_timed_out"):
        # Deliberately NOT phrased as a tool-calling failure: the endpoint
        # reported the capability, and a stopwatch can't overrule that.
        caveats.append(
            f"Tool call check did not finish in time -- {tool_detail}. The endpoint DOES "
            f"report a tools capability for this model, so treat this as a performance "
            f"finding, not a capability one."
        )
    elif not tool_ok:
        caveats.append(f"Tool call check failed: {tool_detail}")
    for label, key in (("think=on", "channel_think_on"), ("think=off", "channel_think_off")):
        ch = result.get(key)
        if ch and ch.get("merged_channel_suspected"):
            caveats.append(
                f"Channel merge suspected at {label} "
                f"(delimiter family: {ch.get('merge_evidence') or 'unknown'}) -- "
                f"reasoning text is leaking into "
                f"the visible answer instead of a separate field. Every channel-suite "
                f"verdict at {label} would be unreliable."
            )
        elif ch is not None and not ch.get("ok"):
            # A check that never completed is NOT a pass. Before this, an
            # errored/timed-out channel check contributed nothing to
            # `clean`, so a model whose channel separation could not be
            # verified at all still printed "Clean. No caveats found." --
            # a false reassurance about the exact thing this check exists
            # to establish. Found live on a 27.9B model whose think=on
            # check timed out while the report called it clean.
            caveats.append(
                f"Channel separation at {label} could NOT be verified -- the check did not "
                f"complete ({ch.get('error')}). This is not a pass: S2's verdicts at {label} "
                f"rest on an assumption nothing has tested on this machine. Common cause is a "
                f"model too large for available VRAM; re-run the gate when it can complete."
            )
    # A suite that structurally cannot fire is not a caveat about the
    # model, it is a prediction that the run will produce dashes. Stated
    # here in the same words the scorecard will use hours later, so the
    # reader sees the sentence BEFORE paying for it rather than after.
    #
    # The missing-thinking case had no caveat of any kind until now: the
    # loop above iterates the two channel checks, and a model with no
    # thinking capability has both of them set to None, so every branch
    # was skipped and a model whose S2 could never return a verdict
    # printed "Clean. No caveats found." Verified live on llama3.1:8b.
    from openllm_cbench.core.preflight import INVALID, SUITE_LABELS, suite_readiness
    readiness = suite_readiness(result)
    result["suite_readiness"] = {k: list(v) for k, v in readiness.items()}
    for suite in ("s1", "s2", "s3"):
        verdict, reason = readiness[suite]
        if verdict == INVALID:
            caveats.append(f"{SUITE_LABELS[suite]} will come back INVALID -- {reason}.")

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
        # Measured, not estimated: what this machine actually did with
        # this model during the gate check. Persisted because the gate is
        # the only place this framework legitimately times a model, and
        # throwing the number away meant every later "how fast is this?"
        # had to be guessed from parameter counts. A measurement beats an
        # estimate for the same reason real on-disk size beats
        # bytes-per-param arithmetic in core/hardware.py.
        #
        # Machine-specific by definition -- it says nothing about the
        # model in the abstract, only about this model on this hardware,
        # which is exactly the question someone picking what to run next
        # is asking.
        "measured_tok_s": result.get("tokens_per_sec"),
        "measured_load_s": result.get("warm_up_seconds"),
        # Persisted so a caller can answer "will S2 produce anything on
        # this model" from the catalogue without re-gating. The gate is
        # the only place this is measured, and the fact was previously
        # discarded the moment the report finished printing.
        "suite_readiness": result.get("suite_readiness") or {},
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
    ]
    if result.get("warm_up_seconds") is not None:
        flag = "  [SLOW -- likely exceeds available VRAM]" if result.get("slow_load") else ""
        rate = result.get("tokens_per_sec")
        rate_part = f", then ~{rate} tok/s" if rate else ""
        L.append(f"- Load + first token: {result['warm_up_seconds']}s{rate_part}{flag}")
    L.append(
        f"- Tool call check: "
        f"{'OK' if result['tool_call_ok'] else ('TIMED OUT' if result.get('tool_call_timed_out') else 'FAILED')}"
        f" -- {result['tool_call_detail']}"
    )
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

    # Before the verdict, not after it: the question a person runs this
    # command to answer is "is it worth my afternoon", and that is a
    # per-suite answer, not a single word.
    from openllm_cbench.core.preflight import render_readiness
    L += [""] + render_readiness(result)

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
