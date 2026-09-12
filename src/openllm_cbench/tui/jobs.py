"""
Job model for the TUI: builds and runs the exact `cbench` command line a
user could type by hand, as a real subprocess, and streams its output back
line by line.

DESIGN INVARIANT: the TUI is a control panel
over the CLI, not a second implementation of suite logic. Every action it
can take must be traceable to a `cbench` subcommand a reviewer could run
themselves and get the identical result. That's why this module shells
out to `[sys.executable, "-m", "openllm_cbench.cli", ...]` -- the same
entry point `pip install`'s `cbench` console script resolves to -- rather
than importing and calling suite `main()` functions in-process. A
Textual event loop and a suite's own blocking `requests` calls have no
business sharing a thread; a subprocess boundary also means a hung or
crashed suite run can never take the TUI process down with it.
"""

import asyncio
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

# Passthrough suites the TUI exposes a form for. Kept as a plain tuple of
# (label, subcommand) rather than importing each suite module -- the TUI
# never needs to know a suite's own flags beyond --model/--dry-run; anything
# else goes through the free-text "extra args" field, which is what keeps
# this file from having to be kept in sync with every suite's argparse setup.
RUNNABLE_SUITES = (
    ("Containment (S1)", "containment"),
    ("Channel (S2)", "channel"),
    ("Persistence (S3)", "persistence"),
)


def cbench_command(subcommand: str, args: list[str]) -> list[str]:
    """The literal argv this job runs -- the same as `cbench <subcommand>
    ...` on the command line plus `-u` (unbuffered stdout/stderr), shown
    in the UI verbatim before a run starts, so what the TUI is about to
    do is never a black box. `-u` matters specifically here: Python
    block-buffers stdout when it isn't a real terminal (i.e. always, for
    a subprocess piped through run_job()), so without it a long-running
    command's output arrives in large delayed chunks instead of live --
    most visible on `cbench pull`'s per-layer progress lines, but it
    affects every subcommand's streamed output the same way."""
    return [sys.executable, "-u", "-m", "openllm_cbench.cli", subcommand, *args]


def build_args(model: str, dry_run: bool, extra_args: str) -> list[str]:
    """Assembles a suite's argv from the TUI form fields. `extra_args` is
    passed through shlex so a user can type real flags ("--boundary both
    --sandbox extended") without the TUI having to model every suite's
    own options. Raises ValueError on unbalanced quoting -- surfaced to
    the user rather than silently dropped."""
    args: list[str] = []
    if model:
        args += ["--model", model]
    if dry_run:
        args.append("--dry-run")
    if extra_args.strip():
        args += shlex.split(extra_args)
    return args


@dataclass
class JobResult:
    argv: list[str]
    returncode: int | None
    lines: list[str] = field(default_factory=list)
    error: str = ""


async def run_job(argv: list[str], on_line: Callable[[str], None], cwd: Path | None = None) -> JobResult:
    """Runs argv as a subprocess, calling on_line(line) as each line of
    combined stdout/stderr arrives (so a long-running suite streams live
    rather than dumping everything at the end), and returns the full
    result once the process exits. Never raises past this point -- a
    failure to even start the process is reported as an error string,
    not an unhandled exception in the UI's event loop."""
    result = JobResult(argv=list(argv), returncode=None)
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            cwd=str(cwd) if cwd else None,
        )
    except Exception as e:
        result.error = f"Failed to start process: {e}"
        on_line(f"[error] {result.error}")
        return result

    assert proc.stdout is not None
    while True:
        raw = await proc.stdout.readline()
        if not raw:
            break
        line = raw.decode(errors="replace").rstrip("\n")
        result.lines.append(line)
        on_line(line)

    result.returncode = await proc.wait()
    return result


# Lines the on-screen (live) view keeps, before a run reaches "Assessment
# complete" -- see condensed_line_filter()'s own docstring for why this is
# an allowlist, not a blocklist, and why it's a prefix match against
# stripped()'d lines rather than a regex.
_CONDENSED_KEEP_PREFIXES = (
    "$ ", "=== ", "--- ", "Scoring '", "Full assessment:", "Assessment complete",
    "Aggregating ", "Report:", "CSV:", "Saved:", "[run-lock]", "[!]",
    "Model not catalogued", "Gate-checking", "Full log saved",
)


def condensed_line_filter():
    """Returns a fresh `should_show(line) -> bool` closure for one run's
    on-screen (live) RichLog -- NEVER for the saved log file, which
    always gets every line regardless via run_job()'s own unconditional
    result.lines.append(); this only controls what's echoed to the
    screen while a run is in progress.

    Found live: a real 3-suite/3-trial run against a real model produced
    800+ lines on screen, and a security practitioner watching it had no
    way to tell what mattered. Two things dominated the noise, neither of
    them the actual outcome: a verdict line for every one of dozens of
    task/probe rows per trial, and each suite's own full single-run
    report -- paragraphs of methodology prose plus a results table --
    printed fresh after every trial, repeated 3 times per suite for a
    3-trial run. All of that is still in the saved log file (open it for
    the play-by-play); it was never the right thing to force someone to
    watch scroll past live. What actually renders on screen instead:
    suite/trial headers, aggregation/report/save pointers, warnings --
    and once the run reaches "Assessment complete", every remaining line
    verbatim, because nothing comes after that but the final grade card
    itself, which is the entire point of watching."""
    seen_complete = False

    def should_show(line):
        nonlocal seen_complete
        if seen_complete:
            return True
        stripped = line.strip()
        if stripped.startswith("Assessment complete"):
            seen_complete = True
            return True
        if not stripped:
            return False
        return any(stripped.startswith(p) for p in _CONDENSED_KEEP_PREFIXES)

    return should_show


def save_job_log(result: JobResult) -> Path:
    """Writes a completed job's full stdout/stderr (result.lines) to a
    timestamped file under results/tui-logs/, and returns its path.

    Exists because a RichLog widget is in-memory only -- its content is
    gone the moment you navigate away from the screen or close the app,
    and Textual has no built-in copy-to-clipboard that works reliably
    across every terminal this might run in. This is the actual,
    inspectable artifact instead: a real file, in the same results/ tree
    every suite already writes its own output to, openable in any editor
    and attachable to a bug report without needing to screenshot a
    terminal.

    The subcommand name (for the filename) is read straight out of
    result.argv (cbench_command()'s own layout: [..., "-m",
    "openllm_cbench.cli", subcommand, *args]) rather than asked for
    separately -- one caller passing a stale or mismatched label is a
    whole class of bug this sidesteps entirely. Never raises past this
    point: if the file can't be written (e.g. a read-only results/ tree),
    the caller still has result.lines in memory and this failure is
    theirs to report, not something to crash the TUI over."""
    import datetime

    from openllm_cbench.core.paths import results_dir

    try:
        subcommand = result.argv[result.argv.index("openllm_cbench.cli") + 1]
    except (ValueError, IndexError):
        subcommand = "unknown"

    d = results_dir("tui-logs")
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = d / f"{subcommand}_{ts}.log"
    footer = (f"\n[exit code: {result.returncode}]\n" if not result.error
              else f"\n[error: {result.error}]\n")
    path.write_text(f"$ {' '.join(result.argv)}\n\n" + "\n".join(result.lines) + footer,
                     encoding="utf-8")
    return path
