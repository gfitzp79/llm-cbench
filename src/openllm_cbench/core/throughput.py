"""
How fast did the model generate, and how close did each call come to its time limit?

WHY THIS EXISTS
===============
Sampling is pinned and seeded, and every budget is set in turns and tokens, so
a faster machine runs the same rows sooner and no verdict depends on speed.
Speed reaches a rate through one channel only: the per-request time limit. A
call still generating when the limit closes it fails, the row leaves the rate
as a failed request, and the rate moves with no change in the model's
behaviour. A model that does not fit in GPU memory is the usual cause (see
core/residency.py); another process sharing the GPU is the other.

A failed request already records that it failed (`error`), but not why in a
form a reader can count, and nothing recorded how close the calls that did
finish came to the limit. This module records both on every row, beside the
residency reading. Recorded, never scored: no rate, band or grade reads these
columns.

WHAT IS MEASURED
================
  gen_tokens_per_s   Tokens generated per second of generation over the row's
                     successful calls: the sum of `eval_count` over the sum of
                     `eval_duration`. Token-weighted, never a mean of per-call
                     rates. It is the server's own measure of generation
                     alone, so it leaves out prompt processing, model load and
                     the HTTP round trip.
  slowest_call_s     The longest client-side wall time of any model call in
                     the row, failed calls included: a call that timed out
                     records how long it ran. It is the quantity the time
                     limit races, so it includes a cold load on the first
                     call, as the limit does.
  request_timeout_s  The per-request time limit the run used.
  error_kind         The kind of the first model call in the row that failed:
                     `timeout` (the server accepted the request and was still
                     generating when the limit closed it), `connection`
                     (nothing answered, including a connect timeout), `http`
                     (the server answered with an error status) or `other`.
                     Blank when every call succeeded.

`gen_tokens_per_s` is not `core/gate.py:warm_up()`'s `tokens_per_sec`, which
the catalogue stores as `measured_tok_s`. That one divides a short warm-up
call's token count by its wall time, to size the gate's own timeouts. They
measure different things and will not agree; keep the names apart.

One call can fail without its row leaving the rate. In S3, a challenge reply
made only of tool calls is followed by one more call that asks for the model's
answer. When that call fails, the suite still scores the row on the reply it
already has, so `error` stays blank while `error_kind` records the failure.
The summary and the caveat count those rows separately.

WHAT IT CANNOT DO
=================
Speed is not a residency signal. A mixture-of-experts model computes only its
active experts for each token, so it can generate quickly with much of itself
on the CPU; read `gpu_resident_fraction` for that. An endpoint that does not
report `eval_count` and `eval_duration` leaves `gen_tokens_per_s` blank, which
means not measured, never zero. Rows written before these columns existed read
the same way.

Nothing here can fail a run or lose a row: every function returns a value or
None and never raises.

PRINTED ON EVERY RUN
====================
The summary line prints whether or not anything failed, for the reason
core/residency.py gives: a detector that never prints looks exactly like one
that is not running.
"""

import time

import requests

# Every suite writes these columns, and every reader goes through this
# module, so the names exist once.
THROUGHPUT_FIELDS = ("gen_tokens_per_s", "slowest_call_s", "request_timeout_s", "error_kind")

ERROR_KINDS = ("timeout", "connection", "http", "other")

_KIND_WORDS = {
    "timeout": "timed out",
    "connection": "connection failure",
    "http": "server error",
    "other": "other failure",
}

# A row whose slowest call took more than this share of its time limit is
# counted as near the limit. A run sharing its GPU with another process has
# been measured at about twice its normal wall time, so a call past half the
# limit would have timed out under that contention.
NEAR_TIMEOUT_FRACTION = 0.5


def _number(value):
    """A finite int or float that is not a bool, or None."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def eval_pair(data):
    """(tokens generated, generation nanoseconds) from one parsed `/api/chat`
    reply, or None when either is missing, unusable or not positive. Pure."""
    try:
        count = _number((data or {}).get("eval_count"))
        duration = _number((data or {}).get("eval_duration"))
    except Exception:
        return None
    if count is None or duration is None or count < 0 or duration <= 0:
        return None
    return count, duration


def generation_rate(replies):
    """Token-weighted generation rate over a row's successful replies: total
    tokens over total generation time, in tokens per second. None when no
    reply reported both fields. Pure, so the arithmetic is tested without a
    server."""
    tokens = nanos = 0
    for data in replies or ():
        pair = eval_pair(data)
        if pair is not None:
            tokens += pair[0]
            nanos += pair[1]
    return tokens / (nanos / 1e9) if nanos > 0 else None


def _wraps_read_timeout(exc):
    """True when requests reports a read timeout as a ConnectionError, which
    it does when the limit closes the connection while the body is being
    read rather than while the server is still preparing it."""
    try:
        from urllib3.exceptions import ReadTimeoutError
        return any(isinstance(arg, ReadTimeoutError) for arg in (getattr(exc, "args", ()) or ()))
    except Exception:
        return False


def classify_error(exc):
    """The kind of a failed model call, from its exception. Never raises.

    ConnectTimeout subclasses both ConnectionError and Timeout. Nothing
    answered in time to connect, so it is a connection failure, the reading
    core/endpoint.py:describe_request_failure() also gives it. A ReadTimeout
    means the server accepted the request and was still generating when the
    limit closed it."""
    try:
        if isinstance(exc, requests.exceptions.ConnectTimeout):
            return "connection"
        if isinstance(exc, requests.exceptions.ConnectionError):
            return "timeout" if _wraps_read_timeout(exc) else "connection"
        if isinstance(exc, requests.exceptions.Timeout):
            return "timeout"
        if (isinstance(exc, requests.exceptions.HTTPError)
                and getattr(exc, "response", None) is not None):
            return "http"
    except Exception:
        pass
    return "other"


def tenths_cell(value):
    """The CSV cell for a rate or a duration: one decimal place, or blank."""
    v = _number(value)
    return "" if v is None or v < 0 else f"{v:.1f}"


def limit_cell(value):
    """The CSV cell for a time limit: as given (120, not 120.0), or blank."""
    v = _number(value)
    return "" if v is None or v <= 0 else f"{v:g}"


def as_number(value):
    """A cell read back from a CSV (a string) or an in-memory row (a number),
    or None for blank, unreadable or negative. One coercion, so no reader
    invents its own and gets the blank case wrong."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v != v or v in (float("inf"), float("-inf")) or v < 0:
        return None
    return v


def timeout_row_fields(timeout):
    """The per-row mapping for the time limit. Stamped at write time beside
    the sampling and budget fields, for the reason they are: a suite builds
    rows in several places, and one stamping site cannot drift from
    itself."""
    return {"request_timeout_s": limit_cell(timeout)}


class CallMeter:
    """One row's model calls, timed, classified and summed for the columns.

    One per row. `post()` stands in for a suite's `requests.post`,
    `raise_for_status` and `json` sequence and sends exactly the request the
    suite sent before; the meter only reads the clock around it and the
    counts in the reply. A failure is recorded and then re-raised unchanged,
    so the suite's own error handling is untouched."""

    def __init__(self, clock=time.monotonic):
        self._clock = clock
        self.slowest = None
        self.tokens = 0
        self.nanos = 0
        self.error_kind = ""

    def post(self, url, payload, timeout):
        start = self._now()
        try:
            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            self._took(start)
            if not self.error_kind:
                self.error_kind = classify_error(exc)
            raise
        self._took(start)
        self._count(data)
        return data

    def _now(self):
        try:
            return self._clock()
        except Exception:
            return None

    def _took(self, start):
        try:
            if start is None:
                return
            elapsed = self._clock() - start
            if elapsed >= 0 and (self.slowest is None or elapsed > self.slowest):
                self.slowest = elapsed
        except Exception:
            pass

    def _count(self, data):
        pair = eval_pair(data)
        if pair is not None:
            self.tokens += pair[0]
            self.nanos += pair[1]

    @property
    def rate(self):
        return self.tokens / (self.nanos / 1e9) if self.nanos > 0 else None

    def row_fields(self):
        """The row's columns, except the time limit, which is stamped at write
        time (`timeout_row_fields`)."""
        return {"gen_tokens_per_s": tenths_cell(self.rate),
                "slowest_call_s": tenths_cell(self.slowest),
                "error_kind": self.error_kind}


def kinds_text(counts):
    """`{"timeout": 2, "http": 1}` as "2 timed out, 1 server error", in a
    fixed order so a caveat string is stable."""
    parts = []
    for kind in ERROR_KINDS:
        n = (counts or {}).get(kind) or 0
        if n:
            parts.append(f"{n} {_KIND_WORDS[kind]}")
    return ", ".join(parts)


def _median(values):
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


class ThroughputTally:
    """Streaming accumulator over rows, for the suites (which hold their rows)
    and the aggregate layer (which folds rows from many files and keeps
    none). One class, so the two cannot disagree."""

    def __init__(self):
        self.rows = 0
        self.rates = []
        self.timed = 0
        self.slowest = None
        self.slowest_limit = None
        self.near_limit = 0
        self.limits = set()
        # error_kind -> rows that left the rate because a call failed
        self.failed = {}
        # error_kind -> rows still scored although a call failed (S3's
        # follow-up call, see this module's docstring)
        self.scored_after_failure = {}

    def add(self, row):
        self.rows += 1
        rate = as_number(row.get("gen_tokens_per_s"))
        if rate is not None:
            self.rates.append(rate)
        slowest = as_number(row.get("slowest_call_s"))
        limit = as_number(row.get("request_timeout_s"))
        if limit is not None and limit > 0:
            self.limits.add(limit)
        else:
            limit = None
        kind = str(row.get("error_kind") or "").strip()
        if kind and kind not in ERROR_KINDS:
            kind = "other"
        if slowest is not None:
            self.timed += 1
            if self.slowest is None or slowest > self.slowest:
                self.slowest, self.slowest_limit = slowest, limit
            if limit and slowest > NEAR_TIMEOUT_FRACTION * limit and kind != "timeout":
                self.near_limit += 1
        if kind:
            bucket = (self.failed if str(row.get("error") or "").strip()
                      else self.scored_after_failure)
            bucket[kind] = bucket.get(kind, 0) + 1

    @property
    def timeouts(self):
        return self.failed.get("timeout", 0) + self.scored_after_failure.get("timeout", 0)

    @property
    def measured(self):
        return bool(self.rates or self.timed or self.limits or self.failed
                    or self.scored_after_failure)

    @property
    def fired(self):
        return bool(self.failed or self.scored_after_failure)

    def as_dict(self):
        return {
            "rows": self.rows,
            "rows_rate_measured": len(self.rates),
            "rate_median": _median(self.rates) if self.rates else None,
            "rate_min": min(self.rates) if self.rates else None,
            "rate_max": max(self.rates) if self.rates else None,
            "rows_timed": self.timed,
            "slowest_call_s": self.slowest,
            "slowest_limit_s": self.slowest_limit,
            "limits_s": sorted(self.limits),
            "rows_near_limit": self.near_limit,
            "failed": dict(self.failed),
            "scored_after_failure": dict(self.scored_after_failure),
        }

    def summary(self):
        if not self.measured:
            return ("Generation speed and time limit: not measured. No row recorded them "
                    "(rows written before this check, or an endpoint that does not report "
                    "them).")
        if self.rates:
            lo, hi = min(self.rates), max(self.rates)
            speed = (f"Generation speed: median {_median(self.rates):.1f} tokens/s (range "
                     f"{lo:.1f} to {hi:.1f}) over {len(self.rates)} measured row(s).")
        else:
            speed = "Generation speed: not reported by the endpoint for these rows."
        if self.slowest is None:
            limit = "Time limit: no call was timed."
        else:
            of = (f" of a {self.slowest_limit:g} s limit" if self.slowest_limit
                  else " (no time limit recorded)")
            limit = (f"Time limit: slowest call {self.slowest:.1f} s{of}; "
                     f"{self.near_limit} row(s) used more than half the limit without "
                     f"reaching it; {self.timeouts} row(s) timed out.")
        tail = ""
        if self.failed:
            tail += f" Failed requests: {kinds_text(self.failed)}."
        if self.scored_after_failure:
            n = sum(self.scored_after_failure.values())
            tail += (f" {n} row(s) were scored after their follow-up call failed "
                     f"({kinds_text(self.scored_after_failure)}).")
        return f"{speed} {limit}{tail}"

    def caveat(self):
        if not self.fired:
            return ""
        parts = []
        if self.failed:
            n = sum(self.failed.values())
            parts.append(f"**[!] {n} row(s) measured nothing because a model call failed** "
                         f"({kinds_text(self.failed)}), so they are left out of every rate.")
        if self.scored_after_failure:
            n = sum(self.scored_after_failure.values())
            parts.append(f"**[!] {n} row(s) were scored on the reply the model had already "
                         f"given, because the follow-up call that asks for its answer failed** "
                         f"({kinds_text(self.scored_after_failure)}). A denial in those rows "
                         f"could not be seen.")
        if self.timeouts:
            limit = (f"the {self.slowest_limit:g} s" if self.slowest_limit else "the")
            parts.append(f"A timeout means the model was still generating when {limit} "
                         f"time limit closed the call: a reason in this machine, not in the "
                         f"model. The usual causes are a model partly outside GPU memory (see "
                         f"the residency line) and another process sharing the GPU. Speed is "
                         f"not a residency signal: a mixture-of-experts model can generate "
                         f"quickly with much of itself on the CPU. **Raise --timeout, or run "
                         f"where the model fits, before comparing this result with one from "
                         f"another machine.**")
        if any(k != "timeout" for k in list(self.failed) + list(self.scored_after_failure)):
            parts.append("A connection or server error is a failure of the endpoint, not of "
                         "the model: check it with `cbench doctor`.")
        return "> " + " ".join(parts)


def tally(rows):
    t = ThroughputTally()
    for row in rows:
        t.add(row)
    return t


def throughput_summary(rows):
    """One line on generation speed and the time limit, printed whether or
    not anything failed."""
    return tally(rows).summary()


def throughput_caveat(rows):
    """Markdown warning block for a report, or "" when no model call failed."""
    return tally(rows).caveat()
