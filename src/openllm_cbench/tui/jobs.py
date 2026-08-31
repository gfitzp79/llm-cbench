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
    """The literal argv this job runs, and the same argv `cbench <subcommand>
    ...` resolves to on the command line -- shown in the UI verbatim before
    a run starts, so what the TUI is about to do is never a black box."""
    return [sys.executable, "-m", "openllm_cbench.cli", subcommand, *args]


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
