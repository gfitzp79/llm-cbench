#!/usr/bin/env python
"""Launch-time exclusivity guard: one battery at a time, on an idle machine.

WHY THIS EXISTS. A multi-trial assessment is long, expensive, and produces
usable data only on completion. Two of them running at once on one GPU do not
fail -- they silently halve each other's throughput and inflate every wall-time
measurement taken from them. In the research lab this framework was extracted
from, exactly that happened: two batches ran concurrently for three hours after
one was believed killed but never verified dead. It was noticed only because a
trial took 117 minutes against a gated 52, and it was very nearly written up as
model variance rather than contention.

Every other guard in this framework and that lab is ANALYSIS-time: schema
checks, trial-count checks, generation-mixing checks. They all read CSVs that
already exist, which is detection after the GPU is spent. This is the only
guard that runs BEFORE the work, and it is the only one that can prevent rather
than report.

DEPENDENCIES: standard library only, deliberately. This package depends on
`requests` and nothing else; pulling in `psutil` for a guard would be a poor
trade, and the platform-specific parts here are ~40 lines of `ctypes`.

USAGE, Python (this is what `cbench assess` does):
    from openllm_cbench.core.runlock import RunLock, RunLockBusy
    with RunLock(label="assess gemma4:12b"):
        ...                      # refuses by raising RunLockBusy

USAGE, shell (for external batch scripts driving `cbench`):
    python -m openllm_cbench.core.runlock hold --ready-file /tmp/r &
    # ...wait for the ready file, then run the batch, then kill the holder.
"""
from __future__ import annotations

import argparse
import ctypes
import errno
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Suite runners whose presence means the machine is already busy. A hand-run
# trial outside any batch collides exactly as badly as a second batch and no
# lockfile can see it, which is why the preflight exists alongside the lock
# rather than instead of it.
RUNNER_PATTERNS = (
    "openllm_cbench.suites.containment",
    "openllm_cbench.suites.channel",
    "openllm_cbench.suites.persistence",
    # Bare module names too, for `python -m` and direct-path invocations.
    "suites/containment", "suites/channel", "suites/persistence",
    "suites\containment", "suites\channel", "suites\persistence",
)

DEFAULT_LOCK = os.environ.get("CBENCH_LOCK_DIR", ".cbench/run.lock")
UNKNOWN = -1  # the preflight could not run; NOT the same as "zero runners"


class RunLockBusy(RuntimeError):
    """Another batch holds the lock, or a suite runner is already live."""


# --------------------------------------------------------------------------
# process liveness
# --------------------------------------------------------------------------
def parent_pid() -> int:
    """PID of the process that invoked us, at the OS level.

    NOT USED TO OWN THE LOCK ON WINDOWS, and the reason is measured rather
    than assumed. The obvious CLI design -- `acquire` records its parent, the
    batch shell -- fails under Git Bash: it forks a transient subshell per
    command, so two consecutive calls from the SAME script returned parents
    103784 and 105120, both already gone. Owning the lock with a pid that dies
    immediately means every later acquire reads the lock as stale, reclaims it,
    and the guard silently protects nothing while printing "acquired". That is
    strictly worse than no guard, because it manufactures assurance.

    The CLI therefore uses the `hold` action, whose owner is a long-lived
    process by construction. This function is retained: it is correct on POSIX
    (`os.getppid()`) and useful for diagnostics.

    Windows note: `os.getppid()` in Git Bash reports an MSYS pid, which is not
    the Windows pid `pid_alive()` checks via OpenProcess -- different number
    spaces -- so the toolhelp snapshot is used. Pure ctypes, no subprocess.
    """
    if os.name != "nt":
        return os.getppid()
    TH32CS_SNAPPROCESS = 0x00000002

    class PROCESSENTRY32(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong),
                    ("cntUsage", ctypes.c_ulong),
                    ("th32ProcessID", ctypes.c_ulong),
                    ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
                    ("th32ModuleID", ctypes.c_ulong),
                    ("cntThreads", ctypes.c_ulong),
                    ("th32ParentProcessID", ctypes.c_ulong),
                    ("pcPriClassBase", ctypes.c_long),
                    ("dwFlags", ctypes.c_ulong),
                    ("szExeFile", ctypes.c_char * 260)]

    k32 = ctypes.windll.kernel32
    me = os.getpid()
    snap = k32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snap == -1:
        return 0
    try:
        e = PROCESSENTRY32()
        e.dwSize = ctypes.sizeof(PROCESSENTRY32)
        if not k32.Process32First(snap, ctypes.byref(e)):
            return 0
        while True:
            if e.th32ProcessID == me:
                return int(e.th32ParentProcessID)
            if not k32.Process32Next(snap, ctypes.byref(e)):
                return 0
    finally:
        k32.CloseHandle(snap)


def pid_alive(pid: int) -> bool:
    """Is this PID live? Cross-platform, stdlib only.

    NOTE for Windows: `os.kill(pid, 0)` must NOT be used. On Windows CPython
    routes any signal other than CTRL_C_EVENT/CTRL_BREAK_EVENT to
    TerminateProcess -- so the POSIX idiom for "is it alive" would KILL the
    process it is asking about. Use OpenProcess/GetExitCodeProcess instead.

    PID reuse is not defended against, and that is the safe direction: a reused
    PID reads as ALIVE, which refuses a launch. A false "alive" costs one
    manual lock removal; a false "dead" reclaims the lock out from under a
    running batch and recreates the incident this module exists to prevent.
    """
    if pid <= 0:
        return False
    if os.name == "nt":
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not h:
            return False
        try:
            code = ctypes.c_ulong()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return True  # cannot tell -> assume alive (safe direction)
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        # EPERM means it exists but belongs to someone else -> alive.
        return e.errno == errno.EPERM


# --------------------------------------------------------------------------
# runner preflight
# --------------------------------------------------------------------------
def live_runners(patterns=RUNNER_PATTERNS, exclude_pids=()):
    """Return [(pid, cmdline)] of live suite runners, or UNKNOWN on failure.

    Matching is restricted to PYTHON processes before the command line is
    examined. That ordering is load-bearing: a command-line-only match also
    matches the shell or PowerShell process running the query itself, so the
    count never reaches zero and the operator learns to ignore the guard. That
    exact self-matching artifact was observed while investigating a live
    duplicate-batch incident (see this module's own docstring above).

    Returning UNKNOWN rather than [] when the probe fails is the other half of
    the same principle: a check that cannot run must not report success.
    """
    exclude = {os.getpid()} | set(exclude_pids)
    try:
        if os.name == "nt":
            ps = (
                "Get-CimInstance Win32_Process | "
                "Where-Object { $_.Name -like 'python*' } | "
                "ForEach-Object { $_.ProcessId.ToString() + ' ' + $_.CommandLine }"
            )
            out = subprocess.run(
                ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", ps],
                capture_output=True, text=True, timeout=30,
            )
        else:
            out = subprocess.run(
                ["ps", "-eo", "pid=,args="], capture_output=True, text=True, timeout=30
            )
        if out.returncode != 0:
            return UNKNOWN
    except Exception:
        return UNKNOWN

    hits = []
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        head, _, rest = line.partition(" ")
        try:
            pid = int(head)
        except ValueError:
            continue
        if pid in exclude:
            continue
        if os.name != "nt" and "python" not in rest.lower():
            continue
        if any(p in rest for p in patterns):
            hits.append((pid, rest[:160]))
    return hits


# --------------------------------------------------------------------------
# the lock
# --------------------------------------------------------------------------
class RunLock:
    """Exclusive batch lock. `os.mkdir` is the atomic step.

    Do NOT "simplify" this to `if not exists: create`. That check-then-act is
    not atomic and races precisely when two launches are seconds apart, which
    is exactly the two-batches-on-one-GPU scenario this module's own
    docstring is modelled on.
    """

    def __init__(self, lock_dir=DEFAULT_LOCK, label="", patterns=RUNNER_PATTERNS,
                 allow_unverified=False, force=False, log=None, owner_pid=None):
        self.dir = Path(lock_dir)
        self.label = label or "unlabelled"
        # Whose liveness makes this lock valid. Defaults to this process, which
        # is right for an in-process orchestrator (cbench assess). The CLI
        # passes the PARENT -- see parent_pid() for why that is not optional.
        self.owner_pid = int(owner_pid) if owner_pid else os.getpid()
        self.patterns = tuple(patterns)
        self.allow_unverified = allow_unverified
        self.force = force
        self._log = log or (lambda m: print("[run-lock] %s" % m, file=sys.stderr))
        self.held = False

    # -- internals ---------------------------------------------------------
    def _owner(self):
        try:
            parts = (self.dir / "owner").read_text(encoding="utf-8").splitlines()
        except Exception:
            return None
        while len(parts) < 3:
            parts.append("")
        try:
            pid = int(parts[0])
        except ValueError:
            pid = 0
        return {"pid": pid, "label": parts[1], "started": parts[2]}

    def _preflight(self):
        runners = live_runners(self.patterns)
        if runners == UNKNOWN:
            if self.allow_unverified:
                self._log("WARNING: runner preflight could not run. Proceeding "
                          "under allow_unverified. The machine was NOT verified "
                          "idle; any timing measured here is suspect.")
                return
            raise RunLockBusy(
                "the runner preflight could not run, so the machine cannot be "
                "confirmed idle. A check that cannot run must not report "
                "success. Pass --allow-unverified only after confirming by hand."
            )
        if runners:
            detail = "\n".join("    PID %d  %s" % (p, c) for p, c in runners)
            msg = ("%d suite runner process(es) already live:\n%s\n"
                   "  Stop them and VERIFY THE PROCESS TABLE before relaunching.\n"
                   "  A kill is not done until the process table says it is done."
                   % (len(runners), detail))
            if not self.force:
                raise RunLockBusy(msg)
            self._log(msg)
            self._log("force=True -- proceeding into a contended machine.")

    # -- api ---------------------------------------------------------------
    def acquire(self):
        self._preflight()
        self.dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.mkdir(self.dir)
        except FileExistsError:
            own = self._owner()
            if own and pid_alive(own["pid"]):
                msg = ("batch lock held by a LIVE process.\n"
                       "  owner pid : %s\n  label     : %s\n  started   : %s\n"
                       "  This is the exact failure mode this guard exists to prevent "
                       "(see runlock.py's own module docstring). Do not launch a second batch."
                       % (own["pid"], own["label"] or "?", own["started"] or "?"))
                if not self.force:
                    raise RunLockBusy(msg)
                self._log(msg)
                self._log("force=True -- proceeding anyway.")
            else:
                # Stale. Reclaim, but say so: a batch that died without
                # releasing is itself a reason to distrust its last artifact.
                self._log("STALE lock reclaimed. Previous owner did NOT exit "
                          "cleanly: pid %s / %s / started %s. Check that "
                          "batch's last artifact before trusting it."
                          % ((own or {}).get("pid", "?"),
                             (own or {}).get("label", "?"),
                             (own or {}).get("started", "?")))
            try:
                (self.dir / "owner").unlink()
            except OSError:
                pass
            try:
                self.dir.rmdir()
                os.mkdir(self.dir)
            except OSError as e:
                raise RunLockBusy("could not reclaim lock: %s" % e)
        (self.dir / "owner").write_text(
            "%d\n%s\n%s\n" % (self.owner_pid, self.label,
                                time.strftime("%Y-%m-%d %H:%M:%S")),
            encoding="utf-8")
        self.held = True
        self._log("acquired (%s, owner pid %d). Machine verified idle."
                  % (self.label, self.owner_pid))
        return self

    def release(self):
        if not self.dir.exists():
            return
        try:
            (self.dir / "owner").unlink()
        except OSError:
            pass
        try:
            self.dir.rmdir()
        except OSError:
            return
        self.held = False
        self._log("released %s" % self.dir)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
        return False


# --------------------------------------------------------------------------
# CLI -- so shell batch scripts share this one implementation
# --------------------------------------------------------------------------
def hold(lock, ready_file=None):
    """Acquire, signal readiness, then stay alive holding the lock.

    This is how a SHELL batch script owns a lock safely. The holder process is
    the owner, so `pid_alive(owner)` is true for exactly as long as the batch
    should be considered running -- no dependence on shell process semantics,
    which differ between Git Bash, WSL and POSIX shells (see parent_pid()).

    On a clean signal the lock is released here. On an abrupt kill (MSYS `kill`
    against a native Windows process is TerminateProcess, so handlers do not
    run) the lock dir survives with a DEAD owner, and the next acquire correctly
    reclaims it as stale and says so. Both paths are safe; only one is tidy.
    """
    def _bye(*_a):
        lock.release()
        sys.exit(0)

    for sig in ("SIGTERM", "SIGINT", "SIGBREAK"):
        s = getattr(signal, sig, None)
        if s is not None:
            try:
                signal.signal(s, _bye)
            except (ValueError, OSError):
                pass
    lock.acquire()
    if ready_file:
        Path(ready_file).write_text(str(os.getpid()), encoding="utf-8")
    try:
        while True:
            time.sleep(3600)
    finally:
        lock.release()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=["acquire", "release", "status", "hold"])
    ap.add_argument("--ready-file", default="",
                    help="hold: file to create once the lock is actually held, "
                         "so a shell caller can distinguish success from refusal "
                         "without parsing output.")
    ap.add_argument("--lock-dir", default=DEFAULT_LOCK)
    ap.add_argument("--label", default="")
    ap.add_argument("--allow-unverified", action="store_true",
                    default=os.environ.get("CBENCH_LOCK_ALLOW_UNVERIFIED") == "1")
    ap.add_argument("--force", action="store_true",
                    default=os.environ.get("CBENCH_LOCK_FORCE") == "1")
    ap.add_argument("--owner-pid", type=int, default=0,
                    help="PID whose liveness owns the lock. Defaults to the "
                         "parent process, because this CLI exits immediately "
                         "after acquiring and must not record itself.")
    a = ap.parse_args(argv)

    # `hold` owns with its own (long-lived) pid, which is the point of it.
    owner = a.owner_pid or (parent_pid() if a.action == "acquire" else 0)
    lock = RunLock(a.lock_dir, label=a.label,
                   owner_pid=(None if a.action == "hold" else (owner or None)),
                   allow_unverified=a.allow_unverified, force=a.force)
    if a.action == "hold":
        try:
            hold(lock, a.ready_file or None)
        except RunLockBusy as e:
            print("[run-lock] REFUSING: %s" % e, file=sys.stderr)
            return 1
        return 0
    if a.action == "release":
        lock.release()
        return 0
    if a.action == "status":
        own = lock._owner()
        runners = live_runners()
        print("lock     : %s" % ("held by %s (pid %s, %s)"
                                 % (own["label"], own["pid"], own["started"])
                                 if own else "free"))
        print("owner    : %s" % ("ALIVE" if own and pid_alive(own["pid"])
                                 else "dead/stale" if own else "-"))
        print("runners  : %s" % ("UNVERIFIABLE" if runners == UNKNOWN
                                 else "%d live" % len(runners)))
        if runners not in (UNKNOWN,):
            for p, c in runners:
                print("    PID %d  %s" % (p, c))
        return 0
    try:
        lock.acquire()
    except RunLockBusy as e:
        print("[run-lock] REFUSING: %s" % e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
