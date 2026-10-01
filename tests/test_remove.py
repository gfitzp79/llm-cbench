"""Tests for the one destructive operation in this framework.

`cbench remove` deletes model weights from the local endpoint. Everything
else here reads, measures, or writes into results/. The tests below are
weighted towards the cases where nothing should happen, because the cost
of a false negative (a delete that should have been refused and was not)
is gigabytes and a re-download, while the cost of a false positive is an
error message.
"""

import pytest

from openllm_cbench.core import remove as rm


class _Resp:
    def __init__(self, status=200, text="ok"):
        self.status_code = status
        self.text = text
        self.ok = 200 <= status < 300


def test_an_empty_tag_is_never_sent(monkeypatch):
    """An endpoint asked to delete "" may do something surprising, and
    finding out what is not worth it."""
    called = {}

    def boom(*a, **k):
        called["sent"] = True
        return _Resp()

    monkeypatch.setattr(rm.requests, "delete", boom)
    for tag in ("", "   ", None):
        ok, detail = rm.remove_model(tag)
        assert ok is False
        assert "refusing" in detail
    assert "sent" not in called, "nothing should reach the endpoint"


def test_a_successful_delete_reports_the_tag(monkeypatch):
    seen = {}

    def fake_delete(url, json=None, timeout=None):
        seen["url"], seen["json"] = url, json
        return _Resp()

    monkeypatch.setattr(rm.requests, "delete", fake_delete)
    ok, detail = rm.remove_model("m:1b")
    assert ok is True
    assert "m:1b" in detail
    assert seen["json"] == {"model": "m:1b"}
    assert seen["url"].endswith("/api/delete")


def test_the_exact_tag_is_sent_never_a_prefix(monkeypatch):
    """The whole reason this takes an exact tag: a prefix or substring
    match is how someone deletes qwen3:14b while meaning qwen3:1.7b."""
    seen = {}
    monkeypatch.setattr(rm.requests, "delete",
                        lambda url, json=None, timeout=None: seen.update(json=json) or _Resp())
    rm.remove_model("qwen3:1.7b")
    assert seen["json"]["model"] == "qwen3:1.7b"


def test_a_missing_model_is_reported_not_raised(monkeypatch):
    monkeypatch.setattr(rm.requests, "delete",
                        lambda *a, **k: _Resp(404, "not found"))
    ok, detail = rm.remove_model("ghost:1b")
    assert ok is False
    assert "404" in detail and "Nothing was deleted" in detail


def test_a_transport_failure_is_reported_not_raised(monkeypatch):
    def boom(*a, **k):
        raise OSError("connection refused")

    monkeypatch.setattr(rm.requests, "delete", boom)
    ok, detail = rm.remove_model("m:1b")
    assert ok is False
    assert "request failed" in detail


def test_a_server_error_is_reported(monkeypatch):
    monkeypatch.setattr(rm.requests, "delete", lambda *a, **k: _Resp(500, "boom"))
    ok, detail = rm.remove_model("m:1b")
    assert ok is False
    assert "500" in detail


# ------------------------------------------------------------- the CLI

def test_cli_refuses_without_yes(monkeypatch, capsys):
    """The confirmation is required in the same invocation. There is no
    remembered consent for this."""
    called = {}
    monkeypatch.setattr("openllm_cbench.core.remove.remove_model",
                        lambda *a, **k: called.setdefault("ran", True) or (True, "deleted"))
    from openllm_cbench import cli
    monkeypatch.setattr("sys.argv", ["cbench", "remove", "--model", "m:1b"])
    rc = cli.main()
    assert rc == 2
    assert "ran" not in called, "nothing should be deleted without --yes"
    assert "NOT DELETING" in capsys.readouterr().err


def test_cli_deletes_with_yes(monkeypatch, capsys):
    monkeypatch.setattr("openllm_cbench.core.remove.remove_model",
                        lambda tag, base=None, **k: (True, f"deleted '{tag}'"))
    from openllm_cbench import cli
    monkeypatch.setattr("sys.argv", ["cbench", "remove", "--model", "m:1b", "--yes"])
    rc = cli.main()
    assert rc == 0
    out = capsys.readouterr().out
    assert "deleted 'm:1b'" in out
    # The measurement outlives the weights, and the tool says so.
    assert "results/" in out


def test_cli_refuses_a_blank_tag_with_exit_2(monkeypatch, capsys):
    """A blank tag is bad input that runs nothing, which every command
    reports with exit 2. This command once printed "Deleting ''" and
    returned 1, the code for a command that ran and failed, because only
    core/remove.py caught the blank tag. With or without --yes, nothing is
    deleted and nothing is reported as attempted."""
    called = {}
    monkeypatch.setattr("openllm_cbench.core.remove.remove_model",
                        lambda *a, **k: called.setdefault("ran", True) or (False, "x"))
    from openllm_cbench import cli
    for argv in (["--model", "", "--yes"], ["--model", "   ", "--yes"], ["--model", ""]):
        monkeypatch.setattr("sys.argv", ["cbench", "remove"] + argv)
        assert cli.main() == 2, argv
        captured = capsys.readouterr()
        assert "NOT STARTING" in captured.err, argv
        assert "Deleting" not in captured.out, argv
    assert "ran" not in called, "a blank tag must never reach the delete"


def test_cli_sends_the_tag_without_surrounding_spaces(monkeypatch):
    seen = {}
    monkeypatch.setattr("openllm_cbench.core.remove.remove_model",
                        lambda tag, base=None, **k: seen.setdefault("tag", tag) and (True, "deleted"))
    from openllm_cbench import cli
    monkeypatch.setattr("sys.argv", ["cbench", "remove", "--model", "  m:1b ", "--yes"])
    assert cli.main() == 0
    assert seen["tag"] == "m:1b"


def test_cli_reports_failure_as_nonzero(monkeypatch):
    monkeypatch.setattr("openllm_cbench.core.remove.remove_model",
                        lambda *a, **k: (False, "404"))
    from openllm_cbench import cli
    monkeypatch.setattr("sys.argv", ["cbench", "remove", "--model", "m:1b", "--yes"])
    assert cli.main() == 1
