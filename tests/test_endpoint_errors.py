"""Tests for saying which layer actually failed.

The incident: `ollama run qwen3.8` was pasted into a model field. The
whole string went to /api/pull as the model name, the endpoint answered
with a 400, and the tool printed:

    [!] Could not reach http://localhost:11434: 400 Client Error

The endpoint was reached. It answered promptly and rejected the request.
That message sends someone to check whether their daemon is running,
which is the one thing that was fine -- the same class of mistake as
reporting a timed-out tool call as a missing capability.
"""

import sys

import pytest
import requests

from openllm_cbench import cli
from openllm_cbench.core.endpoint import describe_request_failure
from openllm_cbench.core.pull import check_tag

BASE = "http://localhost:11434"


class _Response:
    def __init__(self, status_code, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


# ------------------------------------------------ naming the right layer

def test_an_http_rejection_is_not_reported_as_unreachable():
    """THE REGRESSION. A 400 means the endpoint answered."""
    exc = requests.exceptions.HTTPError("400 Client Error: Bad Request")
    exc.response = _Response(400, {"error": "invalid model name"})
    msg = describe_request_failure(exc, BASE)
    assert "REJECTED" in msg
    assert "invalid model name" in msg, "the body carries the real reason"
    assert "reachable" in msg
    assert "Could not reach" not in msg


def test_a_connection_failure_still_says_unreachable():
    exc = requests.exceptions.ConnectionError("refused")
    msg = describe_request_failure(exc, BASE)
    assert "Could not reach" in msg
    assert "REJECTED" not in msg


def test_a_timeout_is_its_own_thing():
    """Reached, and did not answer. Neither of the other two, and it has
    a different fix from both."""
    msg = describe_request_failure(requests.exceptions.Timeout("read timeout"), BASE)
    assert "did not answer in time" in msg
    assert "endpoint is up" in msg


def test_an_html_error_body_still_produces_a_sentence():
    """A proxy or a wrong host answers with HTML, not JSON. The message
    must survive that rather than raising inside the error handler, and
    must stay on one line -- this goes on a single stderr row."""
    exc = requests.exceptions.HTTPError("502")
    exc.response = _Response(502, None, text="<html>\n  <body>Bad Gateway</body>\n</html>")
    msg = describe_request_failure(exc, BASE)
    assert "502" in msg
    assert "not look like an Ollama endpoint" in msg
    assert "\n" not in msg


def test_a_non_html_text_body_is_still_shown():
    """Naming HTML as HTML must not swallow a plain-text reason, which is
    a real thing a reverse proxy returns."""
    exc = requests.exceptions.HTTPError("413")
    exc.response = _Response(413, None, text="request entity too large")
    assert "request entity too large" in describe_request_failure(exc, BASE)


def test_an_unrecognised_failure_is_not_dressed_up():
    msg = describe_request_failure(ValueError("something else entirely"), BASE)
    assert "something else entirely" in msg
    assert "Could not reach" not in msg


# ---------------------------------------------------- catching the paste

@pytest.mark.parametrize("pasted,meant", [
    ("ollama run qwen3.8", "qwen3.8"),
    ("ollama pull gemma3:12b", "gemma3:12b"),
    ("OLLAMA RUN qwen3:4b", "qwen3:4b"),
    ("cbench pull --model x", "--model x"),
])
def test_a_pasted_command_is_refused_and_the_tag_named(pasted, meant):
    ok, complaint = check_tag(pasted)
    assert ok is False
    assert "pasted command" in complaint
    assert meant in complaint


def test_a_pasted_command_is_not_silently_corrected():
    """Stripping the prefix and pulling anyway would be friendlier right
    up until it downloads several GB of something nobody asked for."""
    ok, _ = check_tag("ollama run qwen3.8")
    assert ok is False, "refuse and name the tag; never act on the guess"


def test_any_internal_whitespace_is_refused():
    ok, complaint = check_tag("qwen3 4b")
    assert ok is False and "cannot contain spaces" in complaint


@pytest.mark.parametrize("tag", [
    "qwen3.8",
    "gemma3:12b",
    "hf.co/empero-ai/Qwen3.8-9B-GGUF:Q4_K_M",
    "SuhasDevmane55/geollm-qwen3-4b-v8:latest",
    "registry.example.com:5000/team/model:v2",
])
def test_real_tags_are_not_rejected(tag):
    """Ollama's namespace includes registry hosts, HF paths and ports. A
    character whitelist here would reject a legitimate tag sooner or
    later, so this checks only for what cannot be any tag."""
    ok, complaint = check_tag(tag)
    assert ok is True, complaint


@pytest.mark.parametrize("empty", ["", "   ", None])
def test_nothing_is_a_tag(empty):
    assert check_tag(empty)[0] is False


def test_surrounding_whitespace_is_tolerated():
    assert check_tag("  qwen3.8  ") == (True, "")


# --------------------------------------------------------------- the CLI

def _run(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["cbench"] + argv)
    return cli.main()


@pytest.mark.parametrize("command", ["pull", "search"])
def test_the_cli_refuses_a_pasted_command_before_any_request(command, monkeypatch, capsys):
    """Before any request: the point is that no multi-GB download is
    started for a string that cannot be a tag."""
    import openllm_cbench.core.pull as pull_mod

    def boom(*a, **k):
        raise AssertionError("a request was sent for a string that cannot be a tag")
    monkeypatch.setattr(pull_mod, "pull_model", boom)
    monkeypatch.setattr(pull_mod, "check_model_availability", boom)

    assert _run([command, "--model", "ollama run qwen3.8"], monkeypatch) == 2
    assert "pasted command" in capsys.readouterr().err


@pytest.mark.parametrize("command", ["pull", "search"])
def test_the_cli_reports_a_rejection_honestly(command, monkeypatch, capsys):
    import openllm_cbench.core.pull as pull_mod

    exc = requests.exceptions.HTTPError("400 Client Error")
    exc.response = _Response(400, {"error": "invalid model name"})

    def raise_http(*a, **k):
        raise exc
    monkeypatch.setattr(pull_mod, "pull_model", raise_http)
    monkeypatch.setattr(pull_mod, "check_model_availability", raise_http)

    assert _run([command, "--model", "whatever:1b"], monkeypatch) == 1
    err = capsys.readouterr().err
    assert "REJECTED" in err and "invalid model name" in err
    assert "Could not reach" not in err


def test_an_html_body_is_named_as_html_not_pasted():
    """A 404 page is 300 characters of boilerplate that buries the
    finding. The FACT that it is HTML is the finding: Ollama answers with
    JSON, so an HTML body means this address is not the endpoint you
    think it is. Provoked live against a plain static file server."""
    exc = requests.exceptions.HTTPError("404")
    exc.response = _Response(404, None, text=(
        "<!DOCTYPE HTML><html lang='en'><head><title>Error response</title></head>"
        "<body><h1>Error response</h1><p>Error code: 404</p></body></html>"))
    msg = describe_request_failure(exc, BASE)
    assert "does not look like an Ollama endpoint" in msg
    assert "DOCTYPE" not in msg and "<h1>" not in msg
    assert "404" in msg


@pytest.mark.parametrize("command", ["discover", "catalogue"])
def test_the_tags_commands_report_a_rejection_honestly(command, monkeypatch, capsys):
    """These two were missed by the first pass of this fix: both wrapped
    list_local_models() in a blanket except and called an HTTP status a
    connection failure. Caught by provoking them against a real server
    that answers 404 to everything."""
    import openllm_cbench.core.discover as discover_mod

    exc = requests.exceptions.HTTPError("404 Client Error")
    exc.response = _Response(404, {"error": "no such route"})

    def raise_http(*a, **k):
        raise exc
    monkeypatch.setattr(discover_mod, "list_local_models", raise_http)

    assert _run([command], monkeypatch) == 1
    err = capsys.readouterr().err
    assert "REJECTED" in err and "no such route" in err
    assert "Could not reach" not in err


@pytest.mark.parametrize("command", ["discover", "catalogue"])
def test_the_tags_commands_still_say_unreachable_when_it_is(command, monkeypatch, capsys):
    import openllm_cbench.core.discover as discover_mod

    def refuse(*a, **k):
        raise requests.exceptions.ConnectionError("refused")
    monkeypatch.setattr(discover_mod, "list_local_models", refuse)

    assert _run([command], monkeypatch) == 1
    assert "Could not reach" in capsys.readouterr().err
