"""
When a run actually happened, recorded in the data rather than inferred
from the filesystem.

WHY THIS EXISTS. A file's mtime is not a property of the run that
produced it. It is a property of whatever tool last touched the file.
`git checkout`, `git clone` and `git merge` all rewrite it, so a results
tree that has been through version control carries the checkout's
timestamp on every file, not the generation time. In the lab this
framework was extracted from, that cost real work: 110 of 337 result
CSVs ended up stamped with a merge commit's time, and an audit tool that
read mtime as a run's end time was unreliable for every one of them.

The general rule, and the reason this module is three functions rather
than a comment somewhere: **if you need to know when something ran,
record it when it runs.** Anything derived from the filesystem afterwards
is a guess that looks like a measurement, which is the worst kind.

Two things follow from that:

  - Every suite row carries `run_started_at`, written at the moment the
    run starts. It survives a clone, a checkout, a zip round-trip and
    being emailed to somebody.

  - Where a timestamp must still be recovered from a file that predates
    this column, prefer the one encoded in the FILENAME (these suites
    have always named their output `..._YYYYmmdd_HHMMSS.csv`) over the
    mtime, and say which one is being shown. A filename is data the tool
    wrote. An mtime is not.

This framework has no concurrency detector of the kind that made this
expensive elsewhere -- it has a run lock, which prevents the problem
rather than detecting it afterwards. If one is ever added here, it must
read `run_started_at`, never mtime.
"""

import re
from datetime import datetime

# Written into every suite CSV, alongside the sampling columns.
RUN_TIME_FIELDS = ("run_started_at",)

# The stamp all three suites have always put in their output filenames.
FILENAME_STAMP = re.compile(r"_(\d{8})_(\d{6})(?:\D|$)")


def run_started_now():
    """The value to stamp on this run's rows. ISO-8601 local time with a
    UTC offset, so a result collected in one timezone and read in another
    is still unambiguous -- a bare local timestamp is only interpretable
    by the person who produced it."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def run_time_row_fields(started_at):
    """The per-row mapping, matching `sampling_row_fields`'s shape so the
    suites stamp both the same way at the same place."""
    return {"run_started_at": started_at}


def time_from_filename(name):
    """Recovers a datetime from a `..._YYYYmmdd_HHMMSS...` filename, or
    None if there is no stamp in it.

    Returns None rather than raising or guessing: a caller that cannot
    recover a real time should say so, not substitute one that looks
    equally authoritative."""
    m = FILENAME_STAMP.search(str(name))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return None
