"""
Path resolution for openllm-cbench.

Two different kinds of path, resolved two different ways:

DATA (read-only, shipped with the package -- task sets, probe sets,
scenario files, the verified-model registry): resolved via
importlib.resources against the installed package, so it works
identically whether the package is running from an editable checkout
(`pip install -e .`) or an installed wheel. `--tasks-file` / `--prompts-file`
/ `--scenarios-file` CLI overrides on every suite still take an explicit
path and bypass this entirely.

RESULTS (write-only, generated per run -- CSVs and markdown reports):
never packaged. This is deliberately the opposite resolution strategy
from DATA -- output should never be written inside an installed package.

Resolved most-specific-first:

  1. `--results-dir` on the command       this invocation only
  2. `$OPENLLM_CBENCH_RESULTS_DIR`        this shell only
  3. the persistent config file           this user, everywhere
  4. `./results` relative to the CWD      whatever directory you are in

Layer 3 exists because layer 4 alone made "where are my results?" depend
on where you happened to launch from. Running the TUI from a home
directory and the CLI from a project produced two unrelated trees with
the same name, each invisible to the other, and a scorecard computed over
whichever subset shared a directory with it. See core/config.py.
"""

import os
from importlib import resources
from pathlib import Path


def data_file(*parts):
    """Returns a Traversable for a packaged data file under
    src/openllm_cbench/data/<parts>. Supports .read_text()/.read_bytes()
    directly -- no filesystem extraction needed for JSON reads."""
    return resources.files("openllm_cbench").joinpath("data", *parts)


def read_data_text(*parts, encoding="utf-8"):
    return data_file(*parts).read_text(encoding=encoding)


def results_root(override: str | None = None) -> Path:
    """The results root, by the precedence in this module's docstring.

    Read fresh on every call rather than cached at import: a test that
    points the environment variable at a temp directory, and a TUI that
    changes the configured location while running, both have to take
    effect without a restart."""
    base = override or os.environ.get("OPENLLM_CBENCH_RESULTS_DIR")
    if base:
        return Path(base).expanduser()
    # Imported here rather than at module scope: core.config imports
    # nothing from this module today, and keeping it that way means a
    # future edit cannot make the two circular.
    from openllm_cbench.core.config import configured_results_dir
    configured = configured_results_dir()
    if configured:
        return configured
    return Path.cwd() / "results"


def results_dir(suite: str, override: str | None = None) -> Path:
    """suite is one of 's1_containment', 's2_channel', 's3_persistence',
    's5_inspect'."""
    return results_root(override) / suite
