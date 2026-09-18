"""Shared test isolation.

The persistent config (core/config.py) sits between the environment
variable and the directory-relative default when resolving where results
go. That means a developer who has run `cbench config --set-results-dir`
would, without this file, have their real results directory silently
substituted into every test that relies on the CWD default -- tests
passing or failing based on a setting outside the repository, and in the
worst case a test run writing into a real corpus.

Pointing $OPENLLM_CBENCH_CONFIG at a per-test temp path makes the config
layer real but empty: the resolution order is still exercised, and there
is nothing in it to leak.
"""

import pytest


@pytest.fixture(autouse=True)
def isolate_user_config(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_CBENCH_CONFIG",
                       str(tmp_path / "cbench-test-config" / "config.json"))


@pytest.fixture(autouse=True)
def isolate_run_lock(tmp_path, monkeypatch):
    """Keeps the suite off the real launch-time run lock.

    Without this, any test that exercises the path past the capability
    pre-flight acquires the SAME lock a genuine assessment holds -- so
    running `pytest` while a real run was in progress failed five tests
    that have nothing to do with locking. The lock was behaving exactly
    as designed; the tests were reaching for the live one.

    A test that wants to exercise contention still can, by constructing
    RunLock with an explicit lock_dir."""
    monkeypatch.setenv("CBENCH_LOCK_DIR", str(tmp_path / "run.lock"))
