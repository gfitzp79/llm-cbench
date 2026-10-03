"""
How much of the model was in GPU memory while a row ran?

WHY THIS EXISTS
===============
A model that does not fit in GPU memory still runs. The server places what
fits on the GPU, runs the rest on the CPU, and says nothing about it in the
chat response. With sampling pinned the model generates the same text, but
more slowly, and a slower request is more likely to reach its time limit
before it finishes. Those rows leave the rates as failed requests, so a rate
can move with no change in the model's behaviour, and it moves most for
exactly the models that are too large for the machine running them.
core/throughput.py records how close each row's calls came to that limit.

Speed is not a reliable sign of this. A mixture-of-experts model computes
only its active experts for each token, so it can generate quickly with a
large share of its weights outside GPU memory. The quantity that decides
whether the CPU is doing part of the work is the share of the loaded model's
BYTES in GPU memory, and that is what this module records.

WHAT IS MEASURED
================
The server's process list (`/api/ps` beside `/api/chat`) reports, for every
loaded model, its size in memory and how much of that is in GPU memory. The
ratio is recorded per row as `gpu_resident_fraction`. The size includes the
context cache the run allocated, so the figure answers for the budget the
suite actually used, not for the weights alone.

It is read after every successful model call, when the model is certainly
loaded, and a row keeps the LOWEST value across its calls: a model that is
evicted and reloaded with less of itself on the GPU part-way through a row
is caught rather than averaged away.

WHAT IT CANNOT DO
=================
An endpoint that is not shaped like Ollama's, or that does not report sizes,
yields no value. A blank cell means NOT MEASURED, never "fully resident",
in the same way a blank budget column means unknown rather than equal.

Reading the process list never fails a run. Every failure path returns None
and the row is written with a blank cell, because losing a measurement is
better than losing the row it describes.

EXPECT 1.0 ON MODELS THAT FIT, AND SAY SO
=========================================
The summary line prints on every run, not only when something is wrong, for
the reason `core/context_window.py` gives: a detector that has never fired
looks exactly like one that is not running. "The whole model was in GPU
memory on all 72 measured rows" is a statement an operator can check.
"""

import requests

# Every suite writes this column, and every reader goes through this
# module, so the name exists once.
RESIDENCY_FIELDS = ("gpu_resident_fraction",)

# The server reports whole bytes, so a model that fits reads exactly 1.0.
# The margin only keeps a rounding artefact from reading as a spill; a real
# partial offload is a whole layer or more, far below this.
FULLY_RESIDENT = 0.999

# A local GET against the model server. Short, because it runs after every
# model call, and a slow one is a problem the chat call will already have
# reported.
PS_TIMEOUT_S = 3


def ps_url_for(chat_endpoint):
    """The process-list URL beside a chat endpoint, or None when the
    endpoint is not the `/api/chat` shape and so has no known equivalent."""
    if not isinstance(chat_endpoint, str):
        return None
    base = chat_endpoint.strip().rstrip("/")
    if not base.endswith("/api/chat"):
        return None
    return base[: -len("/api/chat")] + "/api/ps"


def _names(model):
    """The spellings the server may use for this tag. A tag given without a
    version is listed with `:latest`, and the reverse."""
    tag = str(model or "").strip()
    names = {tag}
    last = tag.rsplit("/", 1)[-1]
    if ":" not in last:
        names.add(tag + ":latest")
    elif tag.endswith(":latest"):
        names.add(tag[: -len(":latest")])
    return names


def fraction_from_ps(ps_json, model):
    """The share of `model` in GPU memory from one parsed process-list
    response, or None when the model is not listed or its sizes are not
    usable. Pure, so the arithmetic is tested without a server."""
    try:
        wanted = _names(model)
        for entry in (ps_json or {}).get("models") or []:
            if not isinstance(entry, dict):
                continue
            if entry.get("name") not in wanted and entry.get("model") not in wanted:
                continue
            size, vram = entry.get("size"), entry.get("size_vram")
            if isinstance(size, bool) or isinstance(vram, bool):
                return None
            if not isinstance(size, (int, float)) or not isinstance(vram, (int, float)):
                return None
            if size <= 0 or vram < 0:
                return None
            return round(min(vram / size, 1.0), 4)
    except Exception:
        return None
    return None


def read_fraction(model, chat_endpoint, timeout=PS_TIMEOUT_S):
    """The share of `model` in GPU memory right now, or None when it cannot
    be measured. Never raises: see this module's docstring."""
    url = ps_url_for(chat_endpoint)
    if url is None:
        return None
    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        return fraction_from_ps(resp.json(), model)
    except Exception:
        return None


def lowest(values):
    """The lowest reading across a row's calls, ignoring calls that could
    not be measured. None when none could."""
    seen = [v for v in values if isinstance(v, (int, float)) and not isinstance(v, bool)]
    return min(seen) if seen else None


def cell(value):
    """The CSV cell for one row: four decimal places, or blank."""
    return "" if value is None else f"{value:.4f}"


def as_fraction(value):
    """A cell read back from a CSV (a string) or an in-memory row (a
    float), or None for blank or unreadable. One coercion, so no reader
    invents its own and gets the blank case wrong."""
    if value is None or value == "" or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if 0.0 <= f <= 1.0 else None


def _pct(value):
    return f"{value:.0%}"


class ResidencyTally:
    """Streaming accumulator over rows, for the suites (which hold their
    rows) and the aggregate layer (which folds rows from many files and
    keeps none). One class, so the two cannot disagree."""

    def __init__(self):
        self.measured = 0
        self.not_measured = 0
        self.below_full = 0
        self.lowest = None

    def add(self, row):
        value = as_fraction(row.get("gpu_resident_fraction"))
        if value is None:
            self.not_measured += 1
            return None
        self.measured += 1
        if value < FULLY_RESIDENT:
            self.below_full += 1
        if self.lowest is None or value < self.lowest:
            self.lowest = value
        return value

    @property
    def fired(self):
        return self.below_full > 0

    def as_dict(self):
        return {"rows_measured": self.measured, "rows_below_full": self.below_full,
                "rows_not_measured": self.not_measured, "lowest": self.lowest}

    def summary(self):
        if not self.measured:
            return ("GPU residency: not measured. No row recorded it (rows written before "
                    "this check, or an endpoint that does not report it).")
        tail = (f" {self.not_measured} row(s) were not measured."
                if self.not_measured else "")
        if not self.fired:
            return (f"GPU residency: the whole model was in GPU memory on all "
                    f"{self.measured} measured row(s).{tail}")
        return (f"GPU residency: lowest {_pct(self.lowest)}. Part of the model was outside "
                f"GPU memory on {self.below_full} of {self.measured} measured row(s).{tail}")

    def caveat(self):
        if not self.fired:
            return ""
        return (
            f"> **[!] Part of the model ran outside GPU memory on {self.below_full} of "
            f"{self.measured} row(s)** (lowest {_pct(self.lowest)} in GPU memory). The "
            f"server runs what does not fit on the CPU and says nothing about it in its "
            f"reply. The model generates the same text more slowly, so more requests reach "
            f"their time limit before they finish, and those rows leave the rates as failed "
            f"requests. Which rows survive then depends on this machine as well as on the "
            f"model. "
            f"**Compare this result only with runs from the same machine, and prefer one "
            f"where the model fits before citing it.** Speed is not a reliable sign of "
            f"this: a mixture-of-experts model can run quickly with much of itself on the "
            f"CPU."
        )


def tally(rows):
    t = ResidencyTally()
    for row in rows:
        t.add(row)
    return t


def residency_summary(rows):
    """One line stating where the model ran, printed whether or not it
    fired."""
    return tally(rows).summary()


def residency_caveat(rows):
    """Markdown warning block for a report, or "" when the model stayed in
    GPU memory throughout."""
    return tally(rows).caveat()
