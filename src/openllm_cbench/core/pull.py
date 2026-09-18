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

# Prefixes a pasted command starts with. Ollama's own CLI verbs only --
# not a general "does this look like a shell command" heuristic, which
# would eventually reject a legitimate tag for looking wrong.
_PASTED_COMMAND_PREFIXES = (
    "ollama run", "ollama pull", "ollama create", "ollama cp", "ollama show",
    "cbench pull", "cbench search",
)


def check_tag(raw):
    """Whether this can be a model tag at all, checked before anything is
    sent. Returns (ok, complaint_or_empty).

    THE CASE THIS EXISTS FOR. `ollama run qwen3.8` pasted into a model
    field went to /api/pull verbatim as the model name and came back a
    400, reported as "Could not reach http://localhost:11434". Three
    separate things were wrong with that: the endpoint was reached, the
    tag was recoverable from what was typed, and nothing said so.

    DELIBERATELY REFUSES RATHER THAN CORRECTING. Stripping the `ollama
    run` and pulling `qwen3.8` would be friendlier right up until it
    downloads several GB of something the user did not literally ask
    for. The complaint names the exact command to re-run instead, which
    costs one keystroke and cannot pull the wrong model. Same principle
    as core/config.py: a tool that acts on what it guessed you meant is
    one you cannot predict.

    Not a validity check on the tag's characters -- Ollama's namespace
    includes `hf.co/...` paths, digests and registry hosts, and a
    whitelist here would reject a legitimate tag sooner or later. It
    tests only for things that cannot be any tag: emptiness, internal
    whitespace, and a pasted command."""
    if raw is None:
        return False, "No model tag given."
    tag = raw.strip()
    if not tag:
        return False, "No model tag given -- the field is empty."

    lowered = tag.lower()
    for prefix in _PASTED_COMMAND_PREFIXES:
        if lowered.startswith(prefix + " "):
            meant = tag[len(prefix):].strip()
            return False, (
                f"That looks like a pasted command, not a model tag: {tag!r}.\n"
                f"    The whole string would be sent as the model name, which no "
                f"registry has.\n"
                f"    You probably want the tag on its own:  {meant}")

    if any(c.isspace() for c in tag):
        return False, (
            f"A model tag cannot contain spaces, and this one does: {tag!r}.\n"
            f"    If you pasted a command line, use just the tag from it.")
    return True, ""


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


def check_model_availability(tag, base_url=None, timeout=15):
    """Checks whether a tag exists in Ollama's registry WITHOUT
    downloading it -- a real, live search against Ollama's actual
    registry, not a guess or a cached list.

    Deliberately reuses `/api/pull` rather than adding a new external
    destination: browsing/searching Ollama's remote library has no
    official API (see core/discover.py's docstring for the full
    reasoning -- unofficial alternatives are scraping or a third-party
    wrapper, both rejected as a new trust boundary this framework
    doesn't otherwise have). `/api/pull` itself, however, fetches the
    real manifest from that same registry as its very first step, before
    any layer data flows -- verified live: a real tag's first two events
    are `{"status": "pulling manifest"}` then a `{"status": "pulling
    <digest>", "total": <bytes>}` with the real download size already
    known; a nonexistent tag's manifest fetch fails immediately instead
    (`{"error": "pull model manifest: file does not exist"}`). Reading
    only that far and then closing the connection (`resp.close()`, before
    `iter_lines()` pulls any actual layer bytes off the wire) gets a real
    existence check and a real size for the cost of an aborted request,
    not a multi-GB download just to ask "does this exist."

    Returns {"exists": bool, "size_bytes": int | None, "error": str | None}.
    Raises only on a genuine connection failure to the endpoint itself."""
    url = resolve_base_url(base_url).rstrip("/") + "/api/pull"
    resp = requests.post(url, json={"model": tag, "stream": True}, stream=True, timeout=timeout)
    resp.raise_for_status()
    result = {"exists": False, "size_bytes": None, "error": None}
    try:
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            try:
                evt = json.loads(raw_line)
            except json.JSONDecodeError:
                continue
            if "error" in evt:
                result["error"] = evt["error"]
                break
            if evt.get("total"):
                result["exists"] = True
                result["size_bytes"] = evt["total"]
                break
            # "pulling manifest" alone (no total yet) -- keep reading a
            # couple more lines for the real size before giving up.
    finally:
        resp.close()  # abort the download -- nothing past the manifest is ever read
    return result


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
