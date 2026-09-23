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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "real_process_table: let this test's run lock read the machine's real process table",
    )


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


@pytest.fixture(autouse=True)
def isolate_process_table(request, monkeypatch):
    """Keeps the run lock's live-runner check off the machine's processes.

    Besides the lock file, the lock refuses to start while any `cbench
    score`, `assess` or suite process is live, and it reads that from the
    real process table, which isolate_run_lock does not cover. Running the
    tests during a real run failed five pre-flight tests with "NOT
    STARTING: 1 suite runner process(es) already live". A test about that
    check itself is marked real_process_table."""
    if request.node.get_closest_marker("real_process_table"):
        return
    import openllm_cbench.core.runlock as runlock
    monkeypatch.setattr(runlock, "live_runners", lambda *a, **k: [])


@pytest.fixture(autouse=True)
def isolate_results_and_catalogue(tmp_path, monkeypatch):
    """Keeps every test off the working directory's results/ and models.json.

    With neither set, both resolve relative to wherever pytest runs. A
    scorecard test wrote `results/scorecards/test-1b.*` into the checkout,
    and a test reading the catalogue read the developer's own local overlay
    -- passing or failing on a file that CI never has. A test about the
    resolution order sets or deletes these itself."""
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(tmp_path / "models.json"))


@pytest.fixture(autouse=True)
def isolate_endpoint_runtime(monkeypatch):
    """Keeps community packaging off the live endpoint.

    Packaging records the endpoint's runtime version, read from its
    /api/version, and leaves the field empty when nothing answers -- which
    validation then refuses, as designed. Six tests passed only on a machine
    with Ollama running and failed on every CI runner, which has none.

    A test about the undetectable case still overrides this with None."""
    import openllm_cbench.core.community as community
    monkeypatch.setattr(community, "detect_endpoint_runtime",
                        lambda *a, **k: "ollama 0.0.0-test")
