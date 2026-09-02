"""
Pulls a model into the local endpoint via its own official `/api/pull`
route -- NDJSON streaming progress, verified against a real response
before relying on this shape.

Not a new external trust boundary: this is the identical mechanism
`ollama pull <tag>` already uses from a terminal, invoked over HTTP
against the same local daemon every other command in this framework
talks to, instead of shelling out to the `ollama` binary. It needs no
new permission model beyond what pulling a model already requires today
-- and it downloads real data, potentially several GB, which is why it's
a deliberate, separate command (`cbench pull`) rather than something any
suite does implicitly.
"""

import json

import requests

from openllm_cbench.core.endpoint import resolve_base_url


def pull_model(tag, base_url=None, on_progress=None, timeout=None):
    """Streams a real pull from the endpoint's /api/pull route.
    on_progress(dict) is called once per NDJSON line as it arrives (keys
    vary: status/digest/total/completed, or error on failure) -- the
    caller decides how to surface that (a terminal print, a TUI log
    widget), this function makes no assumption about presentation.
    Returns (ok, final_status_or_error). Raises only if the endpoint
    itself can't be reached at all -- same discipline as
    core/gate.py:fetch_show_info(); a genuine connection failure should
    surface loudly and immediately, not be swallowed."""
    url = resolve_base_url(base_url).rstrip("/") + "/api/pull"
    resp = requests.post(url, json={"model": tag, "stream": True}, stream=True, timeout=timeout)
    resp.raise_for_status()
    last_status = ""
    error = None
    for raw_line in resp.iter_lines():
        if not raw_line:
            continue
        try:
            evt = json.loads(raw_line)
        except json.JSONDecodeError:
            continue
        if on_progress:
            on_progress(evt)
        if "error" in evt:
            error = evt["error"]
        if "status" in evt:
            last_status = evt["status"]
    if error:
        return False, error
    return last_status == "success", last_status


def throttled_progress_printer(print_fn):
    """Returns an on_progress callback for pull_model() that prints one
    line per status change and, for a layer with a byte total, one line
    per 10 percentage points -- not one line per NDJSON event, which for
    a multi-GB layer is dozens of near-identical lines. Verified live
    against a real pull: Ollama sends many progress events per layer with
    the same digest and a growing `completed` count."""
    state = {"digest": None, "last_pct": -1}

    def on_progress(evt):
        if "error" in evt:
            print_fn(f"[!] {evt['error']}")
            return
        status = evt.get("status", "")
        digest = evt.get("digest")
        total = evt.get("total")
        completed = evt.get("completed")
        if digest and total:
            if digest != state["digest"]:
                state["digest"] = digest
                state["last_pct"] = -1
            pct = int((completed or 0) / total * 100)
            if pct >= state["last_pct"] + 10 or completed == total:
                state["last_pct"] = pct
                print_fn(f"  {status} {digest[:19]}...  {pct}%")
        else:
            print_fn(f"  {status}")

    return on_progress
