"""
How many rows a run will produce, said once by the code that produces them.

A progress estimate needs the size of the whole job before the job starts.
Counting trials is the wrong unit: in one measured quick score the four
trials took 2:39, 11:54, 0:33 and 0:29, so an estimate built on trials
predicted a third of the real time and then stood still through the longest
one. Rows are the unit that moves steadily, and every suite prints one
`  -> <id> ...` line per row as it finishes.

The row count is not computed here. Each suite's `--plan` flag runs its own
start-up (arguments, catalogue, think variants, the capability check) and
prints `plan_line()` instead of calling the model, so the count comes from
the same code as the loop that produces the rows. A second copy of that
logic would drift from the first.
"""

import re

_PLAN = "Planned rows: {n}"
_PLAN_RE = re.compile(r"^Planned rows: (\d+)$", re.M)
_TOTAL_RE = re.compile(r"^Plan: .*?(\d+) rows? in total\.$")
_ROW_RE = re.compile(r"^  -> \S+ \.\.\.")


def plan_line(n):
    """The line a suite's `--plan` prints: its rows for one trial."""
    return _PLAN.format(n=n)


def parse_plan(text):
    """Rows for one trial from a suite's `--plan` output, or None."""
    m = _PLAN_RE.search(text or "")
    return int(m.group(1)) if m else None


def total_line(parts, total):
    """The summary `cbench score` and `cbench assess` print before the
    first trial. parts: [(suite, rows_per_trial, trials)]."""
    each = ", ".join(f"{s.upper()} {rows} rows x {t} trial(s)" for s, rows, t in parts)
    return f"Plan: {each}; {total} rows in total."


def parse_total(line):
    """The total from total_line()'s output, or None for any other line."""
    m = _TOTAL_RE.match(line.strip())
    return int(m.group(1)) if m else None


def is_row_line(line):
    """True for the line every suite prints as one row finishes."""
    return bool(_ROW_RE.match(line))
