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
import re
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
    log_path: "Path | None" = None


async def run_job(argv: list[str], on_line: Callable[[str], None], cwd: Path | None = None,
                  live_log: bool = False) -> JobResult:
    """Runs argv as a subprocess, calling on_line(line) as each line of
    combined stdout/stderr arrives (so a long-running suite streams live
    rather than dumping everything at the end), and returns the full
    result once the process exits. Never raises past this point -- a
    failure to even start the process is reported as an error string,
    not an unhandled exception in the UI's event loop.

    live_log=True writes each line to the job's log file as it arrives,
    rather than all at once when the job exits: a score run stopped
    part-way (a fanless laptop getting too hot, a closed lid, a quit TUI)
    otherwise left no log at all. save_job_log() then adds the exit line
    to the same file."""
    result = JobResult(argv=list(argv), returncode=None)
    fh = None
    if live_log:
        result.log_path, fh = _open_job_log(result.argv)
    try:
        return await _stream(result, argv, on_line, cwd, fh)
    finally:
        if fh is not None:
            fh.close()


async def _stream(result, argv, on_line, cwd, fh):
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
        line = raw.decode(errors="replace").rstrip("\r\n")
        result.lines.append(line)
        if fh is not None:
            try:
                fh.write(line + "\n")
                fh.flush()
            except OSError:
                # The screen and result.lines still have every line.
                fh = None
        on_line(line)

    result.returncode = await proc.wait()
    return result


# Lines the on-screen (live) view keeps, before a run reaches "Assessment
# complete" -- see condensed_line_filter()'s own docstring for why this is
# an allowlist, not a blocklist, and why it's a prefix match against
# stripped()'d lines rather than a regex.
def _preflight_prefixes():
    # The per-suite verdict lines the pre-flight prints ("S1 containment:
    # READY -- ..."), from the same constant it prints them with. Dropping
    # them left a refusal on screen with its reasons filtered out.
    from openllm_cbench.core.preflight import SUITE_LABELS
    return ("Pre-flight",) + tuple(f"{label}:" for label in SUITE_LABELS.values())


_CONDENSED_KEEP_PREFIXES = (
    "$ ", "=== ", "--- ", "Scoring '", "Full assessment:", "Assessment complete",
    "Aggregating ", "Report:", "CSV:", "Saved:", "[run-lock]", "[!]",
    "Model not catalogued", "Gate-checking", "Full log saved", "Generation budget:",
    "Plan:",
) + _preflight_prefixes()

# After this line nothing else runs, and what follows is the explanation:
# what you CAN run instead, and how to override. Shown in full.
_REFUSAL_PIVOT = "[!] NOT STARTING"


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
    and once the run reaches the end of the work, every remaining line
    verbatim, because nothing comes after that but the final grade card
    itself, which is the entire point of watching.

    TWO pivot lines, not one. "Assessment complete" marks the end of a
    run that actually ran trials -- but `cbench score --from-existing`
    runs no trials at all, so that line never appears, and pivoting on it
    alone meant the grade card was filtered out entirely on exactly the
    path whose only output IS the grade card. Found live: a --from-existing
    run showed "Scoring ..." and "Saved: ..." and nothing in between.
    "# Grade:" is render_scorecard_markdown()'s own first line, so it
    catches that path regardless of how the run got there."""
    seen_complete = False

    def should_show(line):
        nonlocal seen_complete
        if seen_complete:
            return True
        stripped = line.strip()
        if (stripped.startswith("Assessment complete") or stripped.startswith("# Grade:")
                or stripped.startswith(_REFUSAL_PIVOT)):
            seen_complete = True
            return True
        if not stripped:
            return False
        return any(stripped.startswith(p) for p in _CONDENSED_KEEP_PREFIXES)

    return should_show


_TRIAL_HEADER_RE = re.compile(r"^---\s+\S+\s+trial\s+(\d+)/(\d+)\s+---$")


def parse_trial_header(line: str) -> tuple[int, int] | None:
    """Detects one of `_assess_body`'s own `--- <suite> trial N/M ---`
    headers (cli.py, printed once per trial, for every command that runs
    suites N times -- `cbench assess` and `cbench score` alike) and
    returns (N, M) if this line is one, else None.

    Exists so a screen can drive a live trial-progress counter (see
    ScoreScreen) from the exact same line it's already streaming to the
    log, rather than teaching this module -- or the screen -- anything
    new about suite internals beyond a print format `_assess_body`
    already commits to for human readability."""
    m = _TRIAL_HEADER_RE.match(line.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def _job_log_path(argv) -> Path:
    import datetime

    from openllm_cbench.core.paths import results_dir

    try:
        subcommand = argv[argv.index("openllm_cbench.cli") + 1]
    except (ValueError, IndexError):
        subcommand = "unknown"
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return results_dir("tui-logs") / f"{subcommand}_{ts}.log"


def _open_job_log(argv):
    """(path, open file) for a job's live log, or (None, None) when the
    results folder cannot be written; the job still runs."""
    try:
        path = _job_log_path(argv)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(path, "w", encoding="utf-8")
        fh.write(f"$ {' '.join(argv)}\n\n")
        fh.flush()
        return path, fh
    except OSError:
        return None, None


def save_job_log(result: JobResult) -> "Path | None":
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
    try:
        subcommand = result.argv[result.argv.index("openllm_cbench.cli") + 1]
    except (ValueError, IndexError):
        subcommand = "unknown"

    # Says what happened, not just the number -- the saved log is read
    # later, out of context, so it needs the meaning more than the screen
    # does. See core/exitcodes.py.
    from openllm_cbench.core import exitcodes
    footer = (f"\n[{exitcodes.describe(result.returncode, subcommand)}]\n"
              if not result.error else f"\n[error: {result.error}]\n")
    # The promise above was not kept: an uncreatable results folder raised
    # here, inside a worker, and took every TUI action down with it.
    try:
        if result.log_path is not None and result.log_path.exists():
            # Written line by line as the job ran (run_job's live_log);
            # the lines are already there, so only the exit line is added.
            with open(result.log_path, "a", encoding="utf-8") as f:
                f.write(footer.lstrip("\n"))
            return result.log_path
        path = _job_log_path(result.argv)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"$ {' '.join(result.argv)}\n\n" + "\n".join(result.lines) + footer,
                        encoding="utf-8")
    except OSError:
        return None
    return path
