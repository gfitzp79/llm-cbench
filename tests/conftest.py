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
