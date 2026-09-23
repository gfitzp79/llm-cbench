"""Tests for the persistent results location.

The bug this fixes: results resolved to `./results` next to the current
working directory, so launching the TUI from a home directory and the CLI
from a project produced two unrelated results trees with the same name.
That is not merely untidy -- `cbench score` globs every CSV for a model
tag in ONE tree, so a run that landed in the other is silently absent
from the aggregate and the scorecard is computed over whatever subset
shared a directory with it. Four such trees existed on the machine this
was written for.
"""

import json
import os
from pathlib import Path

import pytest

from openllm_cbench.core import config as cfg
from openllm_cbench.core.paths import results_dir, results_root


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_CBENCH_CONFIG", str(tmp_path / "cfg" / "config.json"))
    monkeypatch.delenv("OPENLLM_CBENCH_RESULTS_DIR", raising=False)
    # Both, not just the results one: these tests are about what applies
    # when neither variable is set, and a developer who had exported the
    # catalogue one failed four of them for a reason outside the repo.
    monkeypatch.delenv("OPENLLM_CBENCH_MODELS_FILE", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ------------------------------------------------------- precedence

def test_nothing_configured_falls_back_to_the_working_directory(home):
    """Unchanged behaviour for anyone who never runs `cbench config`."""
    path, source, pinned = cfg.resolution()
    assert path == home / "results"
    assert pinned is False
    assert "working directory" in source


def test_the_config_beats_the_working_directory(home):
    cfg.set_results_dir(home / "pinned")
    path, source, pinned = cfg.resolution()
    assert path == (home / "pinned").resolve()
    assert pinned is True
    assert "config file" in source
    assert results_dir("s1_containment") == (home / "pinned").resolve() / "s1_containment"


def test_the_environment_beats_the_config(home, monkeypatch):
    """A shell that pins a path must keep winning, so a script or a CI job
    cannot be redirected by a setting someone made months ago."""
    cfg.set_results_dir(home / "pinned")
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(home / "from_env"))
    path, source, pinned = cfg.resolution()
    assert path == home / "from_env"
    assert "$OPENLLM_CBENCH_RESULTS_DIR" in source


def test_an_explicit_flag_beats_everything(home, monkeypatch):
    cfg.set_results_dir(home / "pinned")
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(home / "from_env"))
    path, source, pinned = cfg.resolution(override=str(home / "from_flag"))
    assert path == home / "from_flag"
    assert "--results-dir" in source
    assert results_root(str(home / "from_flag")) == home / "from_flag"


# ---------------------------------------------------------- storing

def test_the_stored_path_is_absolute(home):
    """A relative path in a config read from several directories would
    reintroduce the exact bug this exists to fix."""
    os.makedirs(home / "sub", exist_ok=True)
    cfg.set_results_dir("sub")
    stored = json.loads(cfg.config_path().read_text(encoding="utf-8"))["results_dir"]
    assert Path(stored).is_absolute()


def test_unset_returns_to_the_working_directory(home):
    cfg.set_results_dir(home / "pinned")
    assert cfg.unset_results_dir() is True
    _path, _source, pinned = cfg.resolution()
    assert pinned is False
    assert cfg.unset_results_dir() is False, "removing nothing reports nothing removed"


def test_a_corrupt_config_does_not_stop_a_run(home):
    """The worst a broken config may cost is the directory-relative
    default, which is where the run would have gone anyway."""
    cfg.config_path().parent.mkdir(parents=True, exist_ok=True)
    cfg.config_path().write_text("{not json", encoding="utf-8")
    assert cfg.load_config() == {}
    path, _source, pinned = cfg.resolution()
    assert path == home / "results" and pinned is False


def test_nothing_is_written_until_asked(home):
    """A first run must not silently create a config or move files. A
    tool that relocates output on its own is one you cannot predict."""
    cfg.resolution()
    cfg.configured_results_dir()
    results_dir("s1_containment")
    assert not cfg.config_path().exists()


# ----------------------------------------------------- the warning

def test_an_unpinned_location_warns(home):
    warning = cfg.unconfigured_warning()
    assert "no persistent location is set" in warning
    assert "cbench config --set-results-dir" in warning


def test_a_pinned_location_is_silent(home):
    """The caller prints this unconditionally, so it has to be empty when
    there is nothing to say."""
    cfg.set_results_dir(home / "pinned")
    assert cfg.unconfigured_warning() == ""


# ------------------------------------------------------------- CLI

def test_cli_config_reports_the_resolution(home, capsys):
    from openllm_cbench import cli
    import sys
    sys.argv = ["cbench", "config"]
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "Results directory" in out
    assert "Precedence" in out


def test_cli_config_sets_and_unsets(home, capsys):
    from openllm_cbench import cli
    import sys
    target = home / "chosen"
    sys.argv = ["cbench", "config", "--set-results-dir", str(target)]
    assert cli.main() == 0
    assert cfg.configured_results_dir() == target.resolve()
    assert target.is_dir(), "the directory should be created, not just recorded"

    sys.argv = ["cbench", "config", "--unset-results-dir"]
    assert cli.main() == 0
    assert cfg.configured_results_dir() is None


def test_cli_config_refuses_contradictory_flags(home):
    from openllm_cbench import cli
    import sys
    sys.argv = ["cbench", "config", "--set-results-dir", str(home), "--unset-results-dir"]
    assert cli.main() == 2


# -------------------------------------------- the model catalogue

def test_the_catalogue_has_its_own_setting(home):
    """Deliberately not derived from the results directory. One catalogue
    can serve several results corpora, and tying them together would mean
    moving your results silently re-gated every model."""
    from openllm_cbench.core.registry import default_overlay_path

    cfg.set_results_dir(home / "res")
    assert default_overlay_path() == home / "models.json", "results must not move it"

    cfg.set_models_file(home / "cat" / "models.json")
    assert default_overlay_path() == (home / "cat" / "models.json").resolve()


def test_catalogue_precedence_matches_results(home, monkeypatch):
    from openllm_cbench.core.registry import default_overlay_path

    cfg.set_models_file(home / "pinned.json")
    _p, source, pinned = cfg.models_resolution()
    assert pinned and "config file" in source

    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(home / "env.json"))
    assert default_overlay_path() == home / "env.json"
    assert default_overlay_path("explicit.json") == Path("explicit.json")


def test_unset_catalogue_returns_to_the_working_directory(home):
    cfg.set_models_file(home / "pinned.json")
    assert cfg.unset_models_file() is True
    _p, _s, pinned = cfg.models_resolution()
    assert pinned is False
    assert cfg.unset_models_file() is False


def test_an_unpinned_catalogue_is_not_silently_the_same_one(home, monkeypatch):
    """The failure mode worth its own test: an unpinned catalogue does not
    lose data, it presents a DIFFERENT file, so a model already
    gate-checked reads as uncatalogued and the next run goes out
    ungated."""
    from openllm_cbench.core.registry import default_overlay_path

    here = default_overlay_path()
    other = home / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    assert default_overlay_path() != here, "the CWD default follows the directory"

    cfg.set_models_file(home / "one.json")
    monkeypatch.chdir(home)
    assert default_overlay_path() == (home / "one.json").resolve()
    monkeypatch.chdir(other)
    assert default_overlay_path() == (home / "one.json").resolve(), \
        "a pinned catalogue is the same file from anywhere"


def test_cli_config_sets_and_unsets_the_catalogue(home, capsys):
    from openllm_cbench import cli
    import sys
    target = home / "cat" / "models.json"
    sys.argv = ["cbench", "config", "--set-models-file", str(target)]
    assert cli.main() == 0
    assert cfg.configured_models_file() == target.resolve()

    sys.argv = ["cbench", "config", "--unset-models-file"]
    assert cli.main() == 0
    assert cfg.configured_models_file() is None


def test_cli_config_refuses_contradictory_catalogue_flags(home):
    from openllm_cbench import cli
    import sys
    sys.argv = ["cbench", "config", "--set-models-file", str(home),
                "--unset-models-file"]
    assert cli.main() == 2


def test_config_output_reports_both_settings(home, capsys):
    from openllm_cbench import cli
    import sys
    sys.argv = ["cbench", "config"]
    assert cli.main() == 0
    out = capsys.readouterr().out
    assert "Results directory" in out
    assert "Model catalogue" in out
