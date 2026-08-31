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
never packaged. Defaults to `./results/<suite>` under the current working
directory, overridable with `--results-dir` or the `OPENLLM_CBENCH_RESULTS_DIR`
environment variable. This is deliberately the opposite resolution strategy
from DATA -- output should never be written inside an installed package.
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


def results_dir(suite: str, override: str | None = None) -> Path:
    """suite is one of 's1_containment', 's2_channel', 's3_persistence',
    's5_inspect'. Precedence: explicit --results-dir override > env var >
    default './results/<suite>' relative to CWD."""
    base = override or os.environ.get("OPENLLM_CBENCH_RESULTS_DIR")
    root = Path(base) if base else Path.cwd() / "results"
    d = root / suite
    return d
