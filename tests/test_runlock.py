"""Tests for core/runlock.py -- the launch-time exclusivity guard.

No model, no network, no GPU. Cross-platform: these run unchanged on Windows
and POSIX, which matters because the liveness and process-listing code paths
are platform-specific by necessity.
"""
import json
import os
import subprocess
import sys
import time

import pytest

from openllm_cbench.core.runlock import (
    UNKNOWN, RunLock, RunLockBusy, live_runners, pid_alive,
)

NONE = ("__no_such_pattern__",)


@pytest.fixture()
def lockdir(tmp_path):
    return tmp_path / "run.lock"


# --- liveness --------------------------------------------------------------
def test_own_pid_is_alive():
    assert pid_alive(os.getpid())


def test_bogus_pids_are_dead():
    assert not pid_alive(0)
    assert not pid_alive(-1)
    assert not pid_alive(4294967)


def test_probing_self_does_not_kill_self():
    """Regression guard for the Windows trap this module exists to avoid.

    On Windows, CPython's os.kill(pid, 0) routes to TerminateProcess -- the
    POSIX "is it alive?" idiom would KILL the process it asks about. If someone
    ever 'simplifies' pid_alive() back to os.kill, this test dies with the
    interpreter.
    """
    for _ in range(3):
        assert pid_alive(os.getpid())


# --- lock lifecycle --------------------------------------------------------
def test_acquire_creates_lock_and_records_owner(lockdir):
    with RunLock(lockdir, label="t", patterns=NONE, log=lambda m: None):
        assert lockdir.is_dir()
        owner = (lockdir / "owner").read_text(encoding="utf-8").splitlines()
        assert owner[0] == str(os.getpid())
        assert owner[1] == "t"
    assert not lockdir.exists()


def test_second_acquire_is_refused_while_held(lockdir):
    with RunLock(lockdir, label="a", patterns=NONE, log=lambda m: None):
        with pytest.raises(RunLockBusy, match="LIVE process"):
            RunLock(lockdir, label="b", patterns=NONE, log=lambda m: None).acquire()


def test_lock_is_released_even_when_the_body_raises(lockdir):
    with pytest.raises(ValueError):
        with RunLock(lockdir, label="a", patterns=NONE, log=lambda m: None):
            raise ValueError("boom")
    assert not lockdir.exists()


def test_stale_lock_is_reclaimed_and_reported(lockdir):
    """A dead owner must not block forever -- but the reclaim must be loud.

    A batch that died without releasing is itself a reason to distrust whatever
    artifact it last wrote, so a silent reclaim would discard real signal.
    """
    lockdir.mkdir(parents=True)
    (lockdir / "owner").write_text("4294967\ndead\n2026-01-01 00:00:00\n",
                                   encoding="utf-8")
    seen = []
    lk = RunLock(lockdir, label="new", patterns=NONE, log=seen.append).acquire()
    assert lk.held
    assert any("STALE" in m for m in seen), seen
    lk.release()


# --- runner preflight ------------------------------------------------------
@pytest.mark.real_process_table
def test_preflight_detects_a_real_live_suite_process(lockdir, monkeypatch):
    """Uses a REAL process, not a mock: the detector's only job is reading the
    real process table, and a mock would test the mock.

    COLUMNS=80 forces the adverse case everywhere: ps honours it, and on a
    CI runner's long interpreter path the module name fell past column 80,
    so this test failed on Linux only, and only where COLUMNS was set."""
    monkeypatch.setenv("COLUMNS", "80")
    child = subprocess.Popen(
        [sys.executable, "-c",
         "import time  # openllm_cbench.suites.containment marker\ntime.sleep(20)"])
    try:
        deadline = time.time() + 15
        hits = UNKNOWN
        while time.time() < deadline:
            hits = live_runners(patterns=("openllm_cbench.suites.containment",))
            if hits != UNKNOWN and any(p == child.pid for p, _ in hits):
                break
            time.sleep(0.5)
        assert hits != UNKNOWN, "process listing unavailable on this platform"
        assert any(p == child.pid for p, _ in hits), hits
        assert all(p != os.getpid() for p, _ in hits), "must exclude own pid"

        with pytest.raises(RunLockBusy, match="already live"):
            RunLock(lockdir, label="x",
                    patterns=("openllm_cbench.suites.containment",),
                    log=lambda m: None).acquire()

        forced = RunLock(lockdir, label="x",
                         patterns=("openllm_cbench.suites.containment",),
                         force=True, log=lambda m: None).acquire()
        assert forced.held
        forced.release()
    finally:
        child.terminate()
        child.wait(timeout=10)


def test_idle_machine_reads_as_idle():
    """Guards the self-matching artifact: a command-line-only match would also
    match the shell running the query, so the count would never reach zero and
    the operator would learn to ignore the guard."""
    hits = live_runners(patterns=("__definitely_not_running__",))
    assert hits != UNKNOWN
    assert hits == []


# The child asks the preflight; the marker reaches it as an argument, so its
# own command line matches and must be excluded as self.
_ASK = (
    "import json, sys\n"
    "from openllm_cbench.core.runlock import UNKNOWN, live_runners\n"
    "hits = live_runners(patterns=(sys.argv[1],))\n"
    "print(json.dumps('UNKNOWN' if hits == UNKNOWN else [p for p, _ in hits]))\n"
)
# The parent stands in for a launcher: its command line carries the marker
# too, and it waits for the child, exactly as a Windows venv's
# Scripts\python.exe waits for the real interpreter it started. ONE line,
# with the marker ahead of the child's code: the process listing yields one
# command line per output line, so a marker after a multi-line argument
# would never be matched, and the test would pass without the fix. (It did,
# first time; a real launcher's command line is a single line.)
_LAUNCH = ("import subprocess, sys; "
           "out = subprocess.run([sys.executable, '-c', sys.argv[2], sys.argv[1]], "
           "capture_output=True, text=True); "
           "sys.stdout.write(out.stdout); sys.stderr.write(out.stderr)")


@pytest.mark.real_process_table
def test_a_run_ignores_its_own_launcher_but_still_sees_a_second_run():
    """Found live: every `cbench score` from a Windows virtual environment
    refused to start, naming its own launcher (`.venv\\Scripts\\python.exe`,
    whose command line is the run's own) as the live run. Real processes,
    no mock: a launcher-shaped parent must not count, and an unrelated
    second run started beside it still must."""
    marker = "openllm_cbench.suites.persistence"
    second = subprocess.Popen(
        [sys.executable, "-c", f"import time  # {marker}\ntime.sleep(30)"])
    try:
        seen, launcher_pid = [], None
        deadline = time.time() + 20
        while time.time() < deadline:
            launcher = subprocess.Popen([sys.executable, "-c", _LAUNCH, marker, _ASK],
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                        text=True)
            out, err = launcher.communicate(timeout=60)
            launcher_pid = launcher.pid
            seen = json.loads(out.strip().splitlines()[-1])
            assert seen != "UNKNOWN", err
            if second.pid in seen:
                break
            time.sleep(0.5)
        assert second.pid in seen, (seen, second.pid)
        assert launcher_pid not in seen, "a run must not detect the launcher above it"
    finally:
        second.terminate()
        second.wait(timeout=10)


def test_unverifiable_preflight_refuses_by_default(lockdir, monkeypatch):
    """A check that cannot run must not report success."""
    import openllm_cbench.core.runlock as rl
    monkeypatch.setattr(rl, "live_runners", lambda *a, **k: UNKNOWN)
    with pytest.raises(RunLockBusy, match="cannot be confirmed idle"):
        RunLock(lockdir, label="x", patterns=("y",), log=lambda m: None).acquire()
    lk = RunLock(lockdir, label="x", patterns=("y",), allow_unverified=True,
                 log=lambda m: None).acquire()
    assert lk.held
    lk.release()


def test_force_concurrent_gets_past_an_unreadable_process_table(lockdir, monkeypatch):
    """`cbench score`/`assess` expose only --force-concurrent. The refusal
    once named --allow-unverified, which they do not accept, so where the
    process table could not be read (no `ps`) no run could ever start."""
    import openllm_cbench.core.runlock as rl
    monkeypatch.setattr(rl, "live_runners", lambda *a, **k: UNKNOWN)
    with pytest.raises(RunLockBusy, match="--force-concurrent"):
        RunLock(lockdir, label="x", patterns=("y",), log=lambda m: None).acquire()
    lk = RunLock(lockdir, label="x", patterns=("y",), force=True,
                 log=lambda m: None).acquire()
    assert lk.held
    lk.release()


def test_the_idle_check_sees_single_suite_runs_but_not_the_offline_rescorer():
    from openllm_cbench.core.runlock import RUNNER_PATTERNS
    live = [
        "python -u -m openllm_cbench.cli containment --model x:1b",   # the TUI's Run screen
        "/venv/bin/python /venv/bin/cbench persistence --model x:1b",
        '"C:\\Py\\python.exe" "C:\\venv\\Scripts\\cbench.exe" channel --model x:1b',
    ]
    for line in live:
        assert any(p in line for p in RUNNER_PATTERNS), line
    offline = "/venv/bin/python /venv/bin/cbench score-containment --pair a b"
    assert not any(p in offline for p in RUNNER_PATTERNS)


def test_the_lock_path_resolves_when_asked_not_at_import(tmp_path, monkeypatch):
    """REGRESSION GUARD. `DEFAULT_LOCK` is bound the first time this
    module is imported, and it used to be the default ARGUMENT of
    RunLock.__init__ as well -- so $CBENCH_LOCK_DIR was unreachable to
    anything setting it after import.

    That was not theoretical. Running `pytest` while a genuine assessment
    held the lock failed five tests that have nothing to do with locking:
    they constructed RunLock with no lock_dir, got the real one, and were
    correctly refused. Resolve-at-call-time is how every other path in
    this package works."""
    from openllm_cbench.core.runlock import RunLock, default_lock_dir

    chosen = tmp_path / "elsewhere" / "run.lock"
    monkeypatch.setenv("CBENCH_LOCK_DIR", str(chosen))
    assert default_lock_dir() == str(chosen)
    assert RunLock(label="t").dir == chosen, "constructed with no lock_dir"

    explicit = tmp_path / "explicit.lock"
    assert RunLock(lock_dir=str(explicit), label="t").dir == explicit, \
        "an explicit path still wins, so a contention test can still be written"
