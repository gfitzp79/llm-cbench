"""
Chat endpoint resolution, shared by every suite.

The harness is model-scale-agnostic -- it communicates with an HTTP chat
endpoint, and the canary/scoring/metrics logic compute identically
regardless of model size or hosting. The only hardcoded assumption is
which endpoint to call, which is why it is centralized and made
overridable in one place rather than duplicated per suite.

Precedence: explicit `--endpoint` CLI flag > `OPENLLM_CBENCH_ENDPOINT`
environment variable > default (a local Ollama install).

Default value and URL shape (`/api/chat`, `/api/show`) are unchanged from
the original harness, which targeted Ollama's native API specifically
(not its OpenAI-compatible `/v1` surface -- see ARCHITECTURE.md for why
that distinction matters for the channel-divergence suite)."""

import os

DEFAULT_BASE_URL = "http://localhost:11434"


def resolve_base_url(cli_value=None):
    return cli_value or os.environ.get("OPENLLM_CBENCH_ENDPOINT") or DEFAULT_BASE_URL


def chat_url(cli_value=None):
    return resolve_base_url(cli_value).rstrip("/") + "/api/chat"


def show_url(cli_value=None):
    return resolve_base_url(cli_value).rstrip("/") + "/api/show"


def tags_url(cli_value=None):
    return resolve_base_url(cli_value).rstrip("/") + "/api/tags"


def describe_request_failure(exc, base_url):
    """Says what actually went wrong, instead of calling everything a
    connection failure.

    THE BUG THIS FIXES. Several commands wrapped their whole request in
    `except Exception` and printed "Could not reach {base_url}: {e}". A
    user who pasted `ollama run qwen3.8` into a model field got:

        [!] Could not reach http://localhost:11434: 400 Client Error

    The endpoint was reached. It answered, promptly, and rejected the
    request -- the string in the model field was a shell command, not a
    tag. Reporting that as unreachable sends someone to check whether
    their daemon is running, which is the one thing that was fine. It is
    the same mistake as reporting a timed-out tool call as a missing
    capability (see core/gate.py:check_tool_call), and it costs the same
    thing: time spent debugging the wrong layer.

    Returns a sentence, no prefix and no trailing newline, so the caller
    decides how to mark it."""
    import requests

    if isinstance(exc, requests.exceptions.ConnectionError):
        return (f"Could not reach {base_url} -- nothing is listening there, or the host "
                f"is wrong. Check the endpoint is running ({exc}).")
    if isinstance(exc, requests.exceptions.Timeout):
        return (f"{base_url} accepted the connection but did not answer in time. The "
                f"endpoint is up; it is busy or stuck ({exc}).")
    response = getattr(exc, "response", None)
    if response is not None:
        # The endpoint answered. Its body carries the real reason far more
        # often than the status line does, and throwing it away is how a
        # precise complaint ("invalid model name") became a guess.
        detail = ""
        try:
            body = response.json()
            detail = body.get("error") or ""
        except Exception:
            text = " ".join((response.text or "").split())
            if text[:200].lstrip().lower().startswith(("<!doctype", "<html")):
                # Don't paste 300 characters of error-page boilerplate --
                # the FACT that it is HTML is the finding. Ollama answers
                # with JSON, so an HTML body means whatever is on this
                # address is not the endpoint you think it is, which is a
                # far more useful thing to be told than the page's text.
                detail = ("the response was an HTML page, not JSON -- whatever is "
                          "listening here does not look like an Ollama endpoint")
            else:
                detail = text[:300]
        detail = " ".join(detail.split())[:300]
        return (f"{base_url} REJECTED the request with HTTP {response.status_code}"
                f"{': ' + detail if detail else ''}. The endpoint is reachable -- it is "
                f"the request it did not accept.")
    return f"Request to {base_url} failed: {exc}"
