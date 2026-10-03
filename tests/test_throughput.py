"""Generation speed, the time limit, and why a model call failed, per row.

Recorded, not scored. These tests pin the arithmetic, the classification of
a failed call, the per-row meter (which must send exactly the request a suite
sent before and never break a run), the report lines, the path from a CSV
column to the scorecard, and that runs written before the columns existed
still pool and read as not measured.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import requests
from urllib3.exceptions import ReadTimeoutError

from openllm_cbench.core import throughput as tp
from openllm_cbench.core.paths import data_file
from openllm_cbench.scoring import aggregate as agg
from openllm_cbench.scoring import scorecard as sc
from openllm_cbench.scoring.containment_metrics import incomplete_cause, is_incomplete_row
from openllm_cbench.suites import channel, containment, persistence

SRC = Path(__file__).parent.parent / "src"


def _reply(count, nanos, **extra):
    return {"message": {"role": "assistant", "content": "done"}, "done_reason": "stop",
            "eval_count": count, "eval_duration": nanos, **extra}


# --- the arithmetic ----------------------------------------------------------

def test_the_rate_is_tokens_over_generation_seconds():
    assert tp.generation_rate([_reply(100, 2_000_000_000)]) == pytest.approx(50.0)


def test_the_rate_is_token_weighted_not_a_mean_of_call_rates():
    # 100 tokens in 1 s (100 tok/s) and 300 in 6 s (50 tok/s): the row made
    # 400 tokens in 7 s, 57.1 tok/s, not the 75 a mean of the two would give.
    rate = tp.generation_rate([_reply(100, 1_000_000_000), _reply(300, 6_000_000_000)])
    assert rate == pytest.approx(400 / 7)


@pytest.mark.parametrize("reply", [
    {}, None, {"eval_count": 10}, {"eval_duration": 10},
    _reply(10, 0), _reply(True, 1_000), _reply(10, True), _reply("10", 1_000),
    _reply(10, "1000"), _reply(-1, 1_000), _reply(10, -5), _reply(float("nan"), 1_000),
])
def test_unusable_counts_measure_nothing(reply):
    assert tp.eval_pair(reply) is None
    assert tp.generation_rate([reply]) is None


def test_a_reply_without_counts_does_not_dilute_one_with_them():
    assert tp.generation_rate([{}, _reply(50, 1_000_000_000)]) == pytest.approx(50.0)


# --- classifying a failed call ------------------------------------------------

def _http_error(status=500):
    resp = requests.Response()
    resp.status_code = status
    return requests.exceptions.HTTPError(f"{status} Server Error", response=resp)


@pytest.mark.parametrize("exc,kind", [
    (requests.exceptions.ReadTimeout("Read timed out. (read timeout=120)"), "timeout"),
    (requests.exceptions.ConnectTimeout("connect timed out"), "connection"),
    (requests.exceptions.ConnectionError("Connection refused"), "connection"),
    # requests reports a read timeout during the body as a ConnectionError.
    (requests.exceptions.ConnectionError(ReadTimeoutError(None, "/api/chat", "Read timed out.")),
     "timeout"),
    (_http_error(500), "http"),
    (requests.exceptions.HTTPError("no response attached"), "other"),
    (ValueError("Expecting value: line 1 column 1"), "other"),
    ("not an exception at all", "other"),
])
def test_a_failure_is_classified_by_what_went_wrong(exc, kind):
    assert tp.classify_error(exc) == kind


# --- the per-row meter --------------------------------------------------------

class _Clock:
    def __init__(self, *ticks):
        self.ticks = list(ticks)

    def __call__(self):
        return self.ticks.pop(0)


class _Resp:
    def __init__(self, data, status=200):
        self.data, self.status = data, status

    def raise_for_status(self):
        if self.status >= 400:
            raise _http_error(self.status)

    def json(self):
        return self.data


def test_the_meter_sends_exactly_the_request_it_is_given(monkeypatch):
    seen = {}

    def fake_post(url, json=None, timeout=None):
        seen.update(url=url, json=json, timeout=timeout)
        return _Resp(_reply(10, 1_000_000_000))

    monkeypatch.setattr(tp.requests, "post", fake_post)
    payload = {"model": "m:1", "messages": [{"role": "user", "content": "hi"}]}
    data = tp.CallMeter(clock=_Clock(0.0, 2.5)).post("http://h:9/api/chat", payload, 120)
    assert seen == {"url": "http://h:9/api/chat", "json": payload, "timeout": 120}
    assert data["eval_count"] == 10


def test_the_meter_keeps_the_slowest_call_and_the_token_weighted_rate(monkeypatch):
    replies = iter([_Resp(_reply(100, 1_000_000_000)), _Resp(_reply(300, 6_000_000_000))])
    monkeypatch.setattr(tp.requests, "post", lambda *a, **k: next(replies))
    m = tp.CallMeter(clock=_Clock(0.0, 1.5, 10.0, 17.0))
    m.post("u", {}, 120)
    m.post("u", {}, 120)
    assert m.row_fields() == {"gen_tokens_per_s": "57.1", "slowest_call_s": "7.0",
                              "error_kind": ""}


def test_a_failed_call_is_timed_classified_and_re_raised_unchanged(monkeypatch):
    boom = requests.exceptions.ReadTimeout("Read timed out. (read timeout=120)")

    def fake_post(*a, **k):
        raise boom

    monkeypatch.setattr(tp.requests, "post", fake_post)
    m = tp.CallMeter(clock=_Clock(0.0, 120.4))
    with pytest.raises(requests.exceptions.ReadTimeout) as caught:
        m.post("u", {}, 120)
    assert caught.value is boom
    assert m.row_fields() == {"gen_tokens_per_s": "", "slowest_call_s": "120.4",
                              "error_kind": "timeout"}


def test_an_error_status_is_a_failure(monkeypatch):
    monkeypatch.setattr(tp.requests, "post", lambda *a, **k: _Resp({}, status=500))
    m = tp.CallMeter(clock=_Clock(0.0, 0.2))
    with pytest.raises(requests.exceptions.HTTPError):
        m.post("u", {}, 120)
    assert m.error_kind == "http"


def test_the_first_failure_in_a_row_is_the_one_recorded(monkeypatch):
    errors = iter([_http_error(500), requests.exceptions.ReadTimeout("t")])

    def fake_post(*a, **k):
        raise next(errors)

    monkeypatch.setattr(tp.requests, "post", fake_post)
    m = tp.CallMeter(clock=_Clock(0, 1, 2, 3))
    for _ in range(2):
        with pytest.raises(requests.exceptions.RequestException):
            m.post("u", {}, 120)
    assert m.error_kind == "http"


def test_a_broken_clock_never_breaks_a_call(monkeypatch):
    monkeypatch.setattr(tp.requests, "post", lambda *a, **k: _Resp(_reply(5, 1_000_000_000)))

    def broken():
        raise RuntimeError("no clock")

    m = tp.CallMeter(clock=broken)
    assert m.post("u", {}, 120)["eval_count"] == 5
    assert m.row_fields() == {"gen_tokens_per_s": "5.0", "slowest_call_s": "", "error_kind": ""}


# --- the CSV cells ------------------------------------------------------------

def test_the_cells():
    assert tp.tenths_cell(57.142) == "57.1"
    assert tp.tenths_cell(None) == ""
    assert tp.tenths_cell(True) == ""
    assert tp.limit_cell(120) == "120"
    assert tp.limit_cell(180.5) == "180.5"
    assert tp.limit_cell(0) == ""
    assert tp.timeout_row_fields(120) == {"request_timeout_s": "120"}


@pytest.mark.parametrize("value,expected", [("57.1", 57.1), ("120", 120.0), (3, 3.0),
                                            ("", None), (None, None), ("n/a", None),
                                            ("-1", None), (True, None), ("inf", None)])
def test_cells_read_back_from_a_csv(value, expected):
    got = tp.as_number(value)
    assert got == (pytest.approx(expected) if expected is not None else None)


def test_blank_cells_survive_a_csv_round_trip(tmp_path):
    path = tmp_path / "rows.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[*tp.THROUGHPUT_FIELDS, "error"])
        w.writeheader()
        w.writerow({"gen_tokens_per_s": "", "slowest_call_s": "", "request_timeout_s": "120",
                    "error_kind": "", "error": ""})
    with open(path, newline="", encoding="utf-8") as f:
        t = tp.tally(csv.DictReader(f))
    assert (t.rates, t.timed, t.limits, t.fired) == ([], 0, {120.0}, False)


# --- the tally ----------------------------------------------------------------

def _row(rate="50.0", slowest="10.0", limit="120", kind="", error=""):
    return {"gen_tokens_per_s": rate, "slowest_call_s": slowest, "request_timeout_s": limit,
            "error_kind": kind, "error": error}


def test_the_summary_prints_when_nothing_failed():
    t = tp.tally([_row(), _row(rate="40.0", slowest="20.0")])
    assert not t.fired and t.caveat() == ""
    s = t.summary()
    assert "median 45.0 tokens/s (range 40.0 to 50.0) over 2 measured row(s)" in s
    assert "slowest call 20.0 s of a 120 s limit" in s
    assert "0 row(s) timed out" in s


def test_one_timeout_fires_the_caveat_and_names_the_machine():
    t = tp.tally([_row(), _row(rate="", slowest="120.3", kind="timeout",
                               error="Read timed out.")])
    assert t.fired and t.timeouts == 1
    c = t.caveat()
    assert "1 row(s) measured nothing because a model call failed** (1 timed out)" in c
    assert "--timeout" in c and "mixture-of-experts" in c
    assert "cbench doctor" not in c


def test_near_the_limit_means_more_than_half_and_leaves_out_timeouts():
    rows = [_row(slowest="60.0"), _row(slowest="60.1"),
            _row(slowest="121.0", kind="timeout", error="x")]
    assert tp.tally(rows).near_limit == 1


def test_a_server_error_points_at_the_endpoint_not_the_time_limit():
    c = tp.tally([_row(kind="http", error="500 Server Error")]).caveat()
    assert "(1 server error)" in c and "cbench doctor" in c
    assert "--timeout" not in c


def test_a_failed_follow_up_is_counted_apart_from_rows_that_left_the_rate():
    t = tp.tally([_row(kind="timeout", error="")])
    assert t.failed == {} and t.scored_after_failure == {"timeout": 1}
    assert "scored on the reply the model had already given" in t.caveat()
    assert "1 row(s) were scored after their follow-up call failed (1 timed out)" in t.summary()


def test_rows_from_before_the_columns_read_as_not_measured():
    t = tp.tally([{"error": ""}, {"gen_tokens_per_s": "", "slowest_call_s": ""}])
    assert not t.measured and not t.fired
    assert "not measured" in t.summary()


def test_an_unknown_kind_is_counted_as_other():
    assert tp.tally([_row(kind="weird", error="x")]).failed == {"other": 1}


def test_the_dict_is_ready_for_a_json_scorecard():
    d = tp.tally([_row(), _row(kind="timeout", error="x", slowest="121")]).as_dict()
    json.dumps(d)
    assert d["limits_s"] == [120.0]
    assert d["failed"] == {"timeout": 1}
    assert d["slowest_call_s"] == 121.0 and d["slowest_limit_s"] == 120.0


# --- why an S1 row was left out -----------------------------------------------

@pytest.mark.parametrize("row,cause", [
    ({"error": "Read timed out.", "final_text": "x"}, "failed"),
    ({"final_text": "done", "done_reasons": '["length"]'}, None),
    ({"final_text": "", "done_reasons": '["stop", "length"]'}, "reply_limit"),
    ({"final_text": "", "last_done_reason": "length", "done_reasons": "garbage"}, "reply_limit"),
    ({"final_text": "", "done_reasons": '["stop"]', "turns_used": "6"}, "turns"),
    ({"final_text": "", "done_reasons": '["stop"]', "turns_used": "2"}, None),
    ({"final_text": "", "turns_used": "n/a"}, None),
])
def test_an_incomplete_row_names_its_cause(row, cause):
    assert incomplete_cause(row) == cause
    assert is_incomplete_row(row) is (cause is not None)


def test_the_cause_follows_the_turn_budget_it_is_given():
    row = {"final_text": "", "done_reasons": '["stop"]', "turns_used": "8"}
    assert incomplete_cause(row, max_turns=12) is None
    assert incomplete_cause(row, max_turns=8) == "turns"


# --- at the call sites ----------------------------------------------------------

def test_s1_records_the_columns_on_a_row(monkeypatch):
    monkeypatch.setattr(containment.requests, "post",
                        lambda *a, **k: _Resp(_reply(20, 1_000_000_000)))
    monkeypatch.setattr(containment, "read_fraction", lambda model, endpoint: 1.0)
    result = containment.run_task("m:1", {"id": "t", "prompt": "hi"}, False, {}, 1,
                                  8192, 2048, 30, 6, endpoint="http://h:9/api/chat")
    assert result["gen_tokens_per_s"] == "20.0"
    assert result["error_kind"] == ""
    assert result["slowest_call_s"] != ""


def test_s1_records_a_timeout_and_the_row_is_still_left_out(monkeypatch):
    def fake_post(*a, **k):
        raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=30)")

    monkeypatch.setattr(containment.requests, "post", fake_post)
    result = containment.run_task("m:1", {"id": "t", "prompt": "hi"}, False, {}, 1,
                                  8192, 2048, 30, 6, endpoint="http://h:9/api/chat")
    assert result["error"] and result["error_kind"] == "timeout"
    row = {**result, "done_reasons": json.dumps(result["done_reasons"])}
    assert incomplete_cause(row) == "failed"


def test_s3_records_a_failed_follow_up_without_failing_the_row(monkeypatch):
    """The follow-up that asks for the model's answer after a challenge reply
    made only of tool calls used to swallow its exception, leaving no trace
    that the answer was never received."""
    replies = iter([
        _Resp({"message": {"role": "assistant", "content": "done"}, "done_reason": "stop"}),
        _Resp({"message": {"role": "assistant", "content": "",
                           "tool_calls": [{"function": {"name": "list_log", "arguments": {}}}]},
               "done_reason": "stop"}),
    ])

    def fake_post(*a, **k):
        try:
            return next(replies)
        except StopIteration:
            raise requests.exceptions.ReadTimeout("Read timed out. (read timeout=30)")

    monkeypatch.setattr(persistence.requests, "post", fake_post)
    monkeypatch.setattr(persistence, "read_fraction", lambda model, endpoint: 1.0)
    result = persistence.run_scenario(
        "m:1", {"id": "s", "task_prompt": "do it", "challenge_prompt": "why?"},
        8192, 2048, 30, 4, endpoint="http://h:9/api/chat")
    assert result["error"] == "", "the row is still scored, as before"
    assert result["challenge_response"] == ""
    assert result["error_kind"] == "timeout"


def test_s3_carries_the_columns_on_an_early_failure(monkeypatch):
    def fake_post(*a, **k):
        raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(persistence.requests, "post", fake_post)
    result = persistence.run_scenario(
        "m:1", {"id": "s", "task_prompt": "do it", "challenge_prompt": "why?"},
        8192, 2048, 30, 4, endpoint="http://h:9/api/chat")
    assert result["error"] and result["error_kind"] == "connection"
    assert result["slowest_call_s"] != ""


def test_s2_fills_the_meter_it_is_passed_and_keeps_its_return_shape(monkeypatch):
    reply = _reply(30, 1_000_000_000,
                   message={"role": "assistant", "content": "no", "thinking": "hmm"})
    monkeypatch.setattr(channel.requests, "post", lambda *a, **k: _Resp(reply))
    m = tp.CallMeter()
    out = channel.call_model("m:1", "hi", True, 4096, 512, 30,
                             endpoint="http://h:9/api/chat", meter=m)
    assert len(out) == 6 and out[0] == "no" and out[1] == "hmm" and out[5] == ""
    assert m.row_fields()["gen_tokens_per_s"] == "30.0"


def test_s2_without_a_meter_returns_what_it_always_did(monkeypatch):
    def fake_post(*a, **k):
        raise requests.exceptions.ConnectionError("Connection refused")

    monkeypatch.setattr(channel.requests, "post", fake_post)
    out = channel.call_model("m:1", "hi", True, 4096, 512, 30, endpoint="http://h:9/api/chat")
    assert out[:5] == ("", "", "", None, None)
    assert "refused" in out[5]


# --- through the aggregate to the scorecard -----------------------------------

S1_FIELDS = ["model", "task_id", "task_class", "boundary", "escape_attempt",
             "escape_attempt_real", "escape_turn", "escalated", "authorized_use",
             "turns_used", "error", "tool_log", "final_text", "last_done_reason",
             "done_reasons", "prompt_token_counts", "max_prompt_tokens",
             "gpu_resident_fraction", *tp.THROUGHPUT_FIELDS]


def _s1_row(**overrides):
    row = {"model": "test:1b", "task_id": "email_report", "task_class": "pressure",
           "boundary": "stated", "escape_attempt": "False", "escape_attempt_real": "False",
           "escape_turn": "", "escalated": "False", "authorized_use": "False",
           "turns_used": "1", "error": "", "tool_log": "[]", "final_text": "done",
           "last_done_reason": "stop", "done_reasons": '["stop"]',
           "prompt_token_counts": "[]", "max_prompt_tokens": "0",
           "gpu_resident_fraction": "1.0000", "gen_tokens_per_s": "48.0",
           "slowest_call_s": "30.0", "request_timeout_s": "120", "error_kind": ""}
    row.update(overrides)
    return row


_TIMED_OUT = dict(error="Read timed out. (read timeout=120)", error_kind="timeout",
                  final_text="", gen_tokens_per_s="", slowest_call_s="120.2")
_OUT_OF_TURNS = dict(final_text="", turns_used="6", done_reasons='["stop", "stop"]')


def _write(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


@pytest.fixture
def dirs(tmp_path, monkeypatch):
    for key in ("S1_DIR", "S2_DIR", "S3_DIR"):
        folder = tmp_path / key
        folder.mkdir()
        monkeypatch.setattr(agg, key, folder)
    return tmp_path


def _three_s1_trials(dirs, rows, fields=S1_FIELDS):
    for i in range(3):
        _write(dirs / "S1_DIR" / f"containment_test-1b_20260101_00{i:02d}00.csv", rows, fields)


def _incomplete_note(card):
    [note] = [c for c in card["suites"]["s1"]["caveats"] if "INCOMPLETE" in c]
    return note


def test_the_aggregate_carries_the_columns_and_the_causes(dirs):
    _three_s1_trials(dirs, [_s1_row(), _s1_row(**_TIMED_OUT), _s1_row(**_OUT_OF_TURNS)])
    report, stats = agg.aggregate_s1("test:1b")
    assert stats["throughput"]["failed"] == {"timeout": 3}
    assert stats["incomplete_causes"] == {"failed": 3, "reply_limit": 0, "turns": 3}
    assert "3 failed request(s), 0 at the reply limit, 3 out of turns" in report
    assert "Failed requests: 3 timed out." in report


def test_the_s1_caveat_names_turns_not_the_reply_budget(dirs):
    _three_s1_trials(dirs, [_s1_row(), _s1_row(**_OUT_OF_TURNS)])
    note = _incomplete_note(sc.compute_scorecard("test:1b"))
    assert "3 that ran out of turns" in note and "retry loop" in note
    assert "--num-predict" not in note


def test_the_s1_caveat_gives_the_time_remedy_for_a_timeout(dirs):
    _three_s1_trials(dirs, [_s1_row(), _s1_row(**_TIMED_OUT)])
    note = _incomplete_note(sc.compute_scorecard("test:1b"))
    assert "3 failed request(s) (3 timed out)" in note
    assert "raise --timeout" in note


def test_the_s1_caveat_still_advises_the_reply_budget_for_a_reply_limit_row(dirs):
    _three_s1_trials(dirs, [_s1_row(), _s1_row(final_text="", done_reasons='["length"]')])
    note = _incomplete_note(sc.compute_scorecard("test:1b"))
    assert "3 that reached the reply limit" in note and "raise --num-predict" in note


def test_old_runs_pool_with_new_ones_and_read_as_not_measured(dirs):
    old = [f for f in S1_FIELDS if f not in tp.THROUGHPUT_FIELDS]
    _write(dirs / "S1_DIR" / "containment_test-1b_20260101_000000.csv", [_s1_row()], old)
    _write(dirs / "S1_DIR" / "containment_test-1b_20260101_000100.csv", [_s1_row()], S1_FIELDS)
    _, stats = agg.aggregate_s1("test:1b")
    assert not stats["pooling_incomparable"]
    assert stats["throughput"]["rows_rate_measured"] == 1


def test_a_failed_row_from_before_the_columns_adds_no_kind_and_no_new_caveat(dirs):
    old = [f for f in S1_FIELDS if f not in tp.THROUGHPUT_FIELDS]
    _three_s1_trials(dirs, [_s1_row(), _s1_row(error="Read timed out.", final_text="")], old)
    card = sc.compute_scorecard("test:1b")
    note = _incomplete_note(card)
    assert "3 failed request(s)" in note and "timed out" not in note
    assert len(card["suites"]["s1"]["caveats"]) == 1
    assert "were not recorded for these rows" in sc.render_scorecard_markdown(card)


def test_the_card_states_speed_and_the_limit(dirs):
    _three_s1_trials(dirs, [_s1_row(), _s1_row(**_TIMED_OUT)])
    md = sc.render_scorecard_markdown(sc.compute_scorecard("test:1b"))
    assert "Generation ran at a median of 48 tokens/s" in md
    assert "the slowest call took 120 s of a 120 s limit, and 3 row(s) timed out" in md
    assert "same per-request time limit" in md


def test_a_card_saved_before_the_columns_renders_without_the_line(dirs):
    _three_s1_trials(dirs, [_s1_row()])
    card = sc.compute_scorecard("test:1b")
    del card["generation"]
    md = sc.render_scorecard_markdown(card)
    assert "Generation ran" not in md and "Generation speed" not in md


def test_the_grade_is_unchanged_by_speed_and_the_limit(dirs):
    _three_s1_trials(dirs, [_s1_row()])
    fast = sc.compute_scorecard("test:1b")["score"]
    for p in (dirs / "S1_DIR").glob("*.csv"):
        p.unlink()
    _three_s1_trials(dirs, [_s1_row(gen_tokens_per_s="3.0", slowest_call_s="119.0")])
    assert sc.compute_scorecard("test:1b")["score"] == fast


S2_FIELDS = ["model", "prompt_id", "category", "think_label", "content_verdict",
             "thinking_verdict", "combined_verdict", "content_note", "thinking_note",
             "content_full", "thinking_full", "done_reason", "error",
             "merged_channel_suspected", "truncation_suspected", "scoring_version",
             "gpu_resident_fraction", *tp.THROUGHPUT_FIELDS]


def _s2_row(pid, **overrides):
    row = {"model": "test:1b", "prompt_id": pid, "category": "jailbreak_resistance",
           "think_label": "on", "content_verdict": "PASS", "thinking_verdict": "PASS",
           "combined_verdict": "CLEAN", "content_full": "an answer",
           "thinking_full": "a trace", "done_reason": "stop", "error": "",
           "merged_channel_suspected": "False", "truncation_suspected": "False",
           "scoring_version": str(sc.SCORING_VERSION), "gpu_resident_fraction": "1.0000",
           "gen_tokens_per_s": "48.0", "slowest_call_s": "20.0", "request_timeout_s": "180",
           "error_kind": ""}
    row.update(overrides)
    return row


def _three_s2_trials(dirs, rows):
    for i in range(3):
        _write(dirs / "S2_DIR" / f"channel_test-1b_20260101_00{i:02d}00.csv", rows, S2_FIELDS)


def test_s2_marks_the_grade_when_a_request_failed(dirs):
    """The `*` this adds is the intended change: S2 used to leave a failed
    row out of its rates with no caveat at all."""
    clean = [_s2_row(f"p{i}") for i in range(12)]
    _three_s2_trials(dirs, clean)
    card = sc.compute_scorecard("test:1b")
    assert not card["suites"]["s2"]["caveats"]
    assert "*" not in card["compact_summary"]

    for p in (dirs / "S2_DIR").glob("*.csv"):
        p.unlink()
    failed = _s2_row("p12", error="Read timed out. (read timeout=180)", error_kind="timeout",
                     combined_verdict="", content_verdict="", content_full="",
                     thinking_full="", slowest_call_s="180.1")
    _three_s2_trials(dirs, clean + [failed])
    card = sc.compute_scorecard("test:1b")
    [note] = card["suites"]["s2"]["caveats"]
    assert "3 row(s) left the rate because a model call failed (3 timed out)" in note
    assert "raise --timeout" in note
    assert "*" in card["compact_summary"]


def test_s2_rows_from_before_the_column_add_no_caveat(dirs):
    old = [f for f in S2_FIELDS if f not in tp.THROUGHPUT_FIELDS]
    rows = [_s2_row(f"p{i}") for i in range(12)] + [
        _s2_row("p12", error="Read timed out.", combined_verdict="", content_verdict="")]
    for i in range(3):
        _write(dirs / "S2_DIR" / f"channel_test-1b_20260101_00{i:02d}00.csv", rows, old)
    assert not sc.compute_scorecard("test:1b")["suites"]["s2"]["caveats"]


def test_the_failed_request_caveat_reads_both_kinds_of_failure():
    note = sc._failed_request_caveat({"throughput": {"failed": {"http": 1},
                                                     "scored_after_failure": {"timeout": 2}}})
    assert "1 row(s) left the rate because a model call failed (1 server error)" in note
    assert "2 row(s) were scored on the reply the model had already given" in note
    assert "raise --timeout" in note and "cbench doctor" in note
    assert sc._failed_request_caveat({"throughput": {"failed": {}}}) is None
    assert sc._failed_request_caveat({}) is None


# --- every suite writes the columns ---------------------------------------------

def _first_row(csv_dir, prefix):
    [path] = list(Path(csv_dir).rglob(f"{prefix}_*.csv"))
    with open(path, newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def _run(module, *args, results_dir):
    return subprocess.run(
        [sys.executable, "-m", f"openllm_cbench.suites.{module}", "--model", "dead:1b",
         "--endpoint", "http://127.0.0.1:9", "--results-dir", str(results_dir),
         "--no-catalogue", *args],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )


def _check_dead_endpoint_row(row, limit):
    for field in tp.THROUGHPUT_FIELDS:
        assert field in row
    assert row["error_kind"] == "connection"
    assert row["request_timeout_s"] == limit
    assert row["slowest_call_s"] != ""
    assert row["gen_tokens_per_s"] == ""


def test_s1_writes_the_columns_and_reports_them(tmp_path):
    _run("containment", "--task", "email_report", "--boundary", "stated", results_dir=tmp_path)
    _check_dead_endpoint_row(_first_row(tmp_path, "containment"), "120")
    [report] = list(tmp_path.rglob("containment_report_*.md"))
    text = report.read_text(encoding="utf-8")
    assert "Failed requests: 1 connection failure." in text
    assert "cbench doctor" in text


def test_s2_writes_the_columns(tmp_path):
    bank = json.loads(data_file("probes", "eval_prompts.json").read_text(encoding="utf-8"))
    one = [p for p in bank if p.get("category") in channel.PROBE_CATS][:1]
    probes = tmp_path / "one_probe.json"
    probes.write_text(json.dumps(one), encoding="utf-8")
    _run("channel", "--prompts-file", str(probes), "--think", "false", results_dir=tmp_path)
    _check_dead_endpoint_row(_first_row(tmp_path, "channel"), "180")


def test_s3_writes_the_columns(tmp_path):
    _run("persistence", "--scenario", "dedup_customer_records", results_dir=tmp_path)
    _check_dead_endpoint_row(_first_row(tmp_path, "persistence"), "120")
