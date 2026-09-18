"""
Persistent user settings, which currently means one thing: where results
go.

THE PROBLEM THIS FIXES. Results resolved to `./results` relative to the
current working directory, so the answer to "where are my results?"
depended on which directory you happened to be in when you launched. Run
the TUI from your home directory and the CLI from a project, and you get
two unrelated results trees with the same name, each invisible to the
other. On the machine this was written for that had already produced
three: one under the home directory, one under the repository, one under
a scratch folder, with the newest scorecard in a different tree from the
reports browser that was looking for it.

It is worse than untidy. `cbench score` globs every CSV on disk for a
model tag, so a run that lands in a second tree is not merely hard to
find -- it is silently absent from the aggregate, and the scorecard is
computed over whatever subset happened to share a directory with it.

THE RESOLUTION ORDER, most specific first:

  1. `--results-dir` on the command             this invocation only
  2. `$OPENLLM_CBENCH_RESULTS_DIR`              this shell only
  3. this config file                           this user, everywhere
  4. `./results`                                whatever directory you are in

Layer 3 is the new one. The three that surrounded it are unchanged, so a
script pinning a path with the flag or the environment variable behaves
exactly as it did, and someone who has never run `cbench config` sees the
old behaviour with a louder warning about it.

NOTHING IS WRITTEN UNTIL ASKED. A first run does not silently create a
config or move anybody's files: a tool that relocates output on its own
is a tool you cannot predict. `cbench config --set-results-dir` writes;
everything else here reads.
"""

import json
import os
from pathlib import Path

CONFIG_FILENAME = "config.json"
APP_DIR = "openllm-cbench"


def config_path():
    """Where the persistent config lives.

    `$OPENLLM_CBENCH_CONFIG` wins, which is what the test suite uses to
    keep a developer's real settings out of a test run. Otherwise the
    platform's own convention: %APPDATA% on Windows, $XDG_CONFIG_HOME or
    ~/.config elsewhere."""
    override = os.environ.get("OPENLLM_CBENCH_CONFIG")
    if override:
        return Path(override)
    appdata = os.environ.get("APPDATA")
    if appdata and os.name == "nt":
        return Path(appdata) / APP_DIR / CONFIG_FILENAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / APP_DIR / CONFIG_FILENAME


def load_config():
    """The stored settings, or {} for no file / an unreadable one.

    Never raises. A corrupt config must not stop a suite running: the
    worst it should cost is falling back to the directory-relative
    default, which is where the run would have gone anyway."""
    path = config_path()
    try:
        if not path.is_file():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_config(data):
    """Writes the settings, creating the directory. Returns the path."""
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return path


def configured_results_dir():
    """The persisted results root, or None if none is set."""
    value = load_config().get("results_dir")
    if not value or not isinstance(value, str):
        return None
    return Path(value).expanduser()


def set_results_dir(path):
    """Persists the results root. Stored ABSOLUTE and expanded, because a
    relative path in a config read from several directories would
    reintroduce exactly the bug this exists to fix."""
    resolved = Path(path).expanduser().resolve()
    data = load_config()
    data["results_dir"] = str(resolved)
    save_config(data)
    return resolved


def unset_results_dir():
    """Removes the persisted root, returning to directory-relative
    behaviour. Returns True if something was actually removed."""
    data = load_config()
    if "results_dir" not in data:
        return False
    del data["results_dir"]
    save_config(data)
    return True


def resolution(override=None):
    """Explains where results will go and which layer decided it.

    Returns (path, source, is_pinned). `source` is a short phrase for a
    human. `is_pinned` is False only for the directory-relative fallback,
    which is the case worth warning about, because it is the one where
    the answer changes depending on where the command was run."""
    if override:
        return Path(override).expanduser(), "--results-dir on this command", True
    env = os.environ.get("OPENLLM_CBENCH_RESULTS_DIR")
    if env:
        return Path(env).expanduser(), "$OPENLLM_CBENCH_RESULTS_DIR in this shell", True
    configured = configured_results_dir()
    if configured:
        return configured, f"config file ({config_path()})", True
    return (Path.cwd() / "results",
            "the current working directory, because nothing is configured", False)


def unconfigured_warning(override=None):
    """The line to print when results are landing somewhere that depends
    on the caller's directory. Empty string when a location is pinned, so
    callers can print it unconditionally."""
    path, _source, pinned = resolution(override)
    if pinned:
        return ""
    return (f"[!] Results will go to {path}, because no persistent location is set.\n"
            f"    Launch from a different directory and they land somewhere else, "
            f"invisible to this one.\n"
            f"    Fix it once with:  cbench config --set-results-dir <path>")
