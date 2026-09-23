"""
Where things are: the local Ollama install, its logs and model storage,
and this tool's own files -- for `cbench doctor`.

Every location here is FOUND and then CHECKED, never assumed from a
platform default. Measured on the machine this was written on:

- The default model folder (~/.ollama/models) existed and was EMPTY while
  the server listed 39 models. They were on another drive, chosen in the
  Ollama app's own settings -- not an environment variable this process
  could see. Reporting the default would have been confidently wrong. So
  the model folder is read from the server's own startup log first, and a
  folder only counts once the manifests in it match the models the server
  lists.
- A directory listing showed the live server.log as 0 bytes while it
  held 3 MB -- a stale size for a file the server holds open. Sizes here
  come from os.stat, which asks the file.

Nothing here writes anything. The only requests go to the configured
endpoint (its version and its model list); everything local is read-only.
"""

import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

LOOPBACK_HOSTS = ("localhost", "127.0.0.1", "::1")
REGISTRY_HOST = "registry.ollama.ai"
SERVER_CONFIG_MARK = 'msg="server config"'


# --- the endpoint ------------------------------------------------------------

def _is_loopback(host):
    host = (host or "").lower().strip("[]")
    return host in LOOPBACK_HOSTS or host.startswith("127.")


def endpoint_is_local(base_url):
    """Whether the endpoint is on this machine, so its files can be too."""
    return _is_loopback(urlparse(base_url).hostname)


def listens_beyond_loopback(ollama_host):
    """True when a server configured with this OLLAMA_HOST accepts
    connections from other machines. Empty means Ollama's own default,
    127.0.0.1. Ollama has no authentication, so a server bound to 0.0.0.0
    or a LAN address lets anyone on that network use its models and GPU."""
    if not ollama_host:
        return False
    url = ollama_host if "://" in ollama_host else f"http://{ollama_host}"
    return not _is_loopback(urlparse(url).hostname)


def _get_json(base_url, route, timeout=5):
    try:
        import requests
        resp = requests.get(f"{base_url.rstrip('/')}{route}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def server_version(base_url):
    data = _get_json(base_url, "/api/version")
    return data.get("version") if isinstance(data, dict) else None


def server_tags(base_url):
    """The model tags the server itself lists, or None if it did not answer."""
    data = _get_json(base_url, "/api/tags")
    if not isinstance(data, dict):
        return None
    return [m.get("name") for m in data.get("models") or [] if m.get("name")]


# --- the install and its logs ----------------------------------------------

def find_ollama_binary():
    """The ollama executable: PATH first, then where each platform's
    installer puts it. None when neither has it -- which is also what a
    server running in a container looks like from here."""
    found = shutil.which("ollama")
    if found:
        # resolve(): the real file, in its real case -- `which` on Windows
        # reports the extension as PATHEXT spells it (ollama.EXE).
        return Path(found).resolve()
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        candidates = [Path(local) / "Programs" / "Ollama" / "ollama.exe"] if local else []
    elif sys.platform == "darwin":
        candidates = [Path("/Applications/Ollama.app/Contents/Resources/ollama")]
    else:
        candidates = [Path("/usr/local/bin/ollama"), Path("/usr/bin/ollama")]
    return next((c for c in candidates if c.is_file()), None)


def ollama_log_dir():
    """(folder, note). Windows and macOS write log files; on Linux the
    service logs to the systemd journal and a hand-run `ollama serve` to its
    own terminal, so there is usually no file to point at."""
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA")
        folder = Path(local) / "Ollama" if local else None
    elif sys.platform == "darwin":
        folder = Path.home() / ".ollama" / "logs"
    else:
        return None, "no log file on Linux -- a service logs to `journalctl -u ollama`"
    if folder and folder.is_dir():
        return folder, ""
    return None, f"not found (expected {folder})" if folder else "not found"


def log_files(folder):
    """[(path, size_bytes, mtime)] for the server and app logs that exist."""
    out = []
    for name in ("server.log", "app.log"):
        path = folder / name
        try:
            st = os.stat(path)
        except OSError:
            continue
        out.append((path, st.st_size, st.st_mtime))
    return out


_KEY = re.compile(r"(?:(?<=\[)|(?<= ))([A-Z][A-Z0-9_]*):")


def parse_server_config(line):
    """OLLAMA_* settings from Ollama's startup line:
    `... msg="server config" env="map[KEY:value KEY:value ...]"`.

    A value runs to the next ` KEY:`, so a Windows path's drive-letter
    colon stays inside it, and the doubled backslashes the log writes are
    undone. Keys are upper-case, which also keeps a URL inside a value
    (`[http://localhost ...]`) from being read as a key."""
    m = re.search(r'env="map\[(.*)\]"', line)
    if not m:
        return {}
    body = m.group(1)
    keys = list(_KEY.finditer(body))
    out = {}
    for i, key in enumerate(keys):
        end = keys[i + 1].start() if i + 1 < len(keys) else len(body)
        if key.group(1).startswith("OLLAMA_"):
            out[key.group(1)] = body[key.end():end].strip().replace("\\\\", "\\")
    return out


def read_server_config(folder, max_bytes=8_000_000):
    """The RUNNING server's settings: the startup line of server.log, which
    holds the current session (older sessions rotate to server-1.log and
    on). {} when there is no such line to read."""
    if folder is None:
        return {}
    try:
        with open(folder / "server.log", encoding="utf-8", errors="replace") as f:
            seen = 0
            for line in f:
                if SERVER_CONFIG_MARK in line:
                    return parse_server_config(line)
                seen += len(line)
                if seen > max_bytes:
                    break
    except OSError:
        pass
    return {}


# --- model storage -----------------------------------------------------------

def manifest_relpath(tag):
    """Where Ollama keeps a tag's manifest, relative to models/manifests:
    `qwen3:4b` -> registry.ollama.ai/library/qwen3/4b, `ns/name:t` ->
    registry.ollama.ai/ns/name/t, `hf.co/org/repo:q` -> hf.co/org/repo/q.

    By POSITION, the way Ollama parses a name -- model, namespace/model,
    host/namespace/model -- not by looking for a dot. The first version
    took any dotted first part for a host, and `qwen3.8`, `granite4.2` and
    `llama3.1` all went looking under hosts of those names: 10 of 39 real
    models reported missing from a folder that held all 39."""
    name, _, version = tag.partition(":")
    parts = name.split("/")
    model = parts[-1]
    namespace = parts[-2] if len(parts) >= 2 else "library"
    host = "/".join(parts[:-2]) if len(parts) >= 3 else REGISTRY_HOST
    return "/".join([host, namespace, model, version or "latest"])


def verify_models_dir(folder, tags):
    """How many of the server's tags have a manifest in this folder.
    Compared case-insensitively: Windows and macOS file systems are, and a
    tag's case is not always the file's."""
    root = Path(folder) / "manifests"
    if not root.is_dir():
        return 0
    on_disk = {p.relative_to(root).as_posix().lower() for p in root.rglob("*") if p.is_file()}
    return sum(1 for t in tags if manifest_relpath(t).lower() in on_disk)


def blobs_size(folder):
    total = 0
    blobs = Path(folder) / "blobs"
    if blobs.is_dir():
        for entry in os.scandir(blobs):
            try:
                total += entry.stat().st_size
            except OSError:
                continue
    return total


def default_models_dirs():
    dirs = [Path.home() / ".ollama" / "models"]
    if sys.platform.startswith("linux"):
        dirs.append(Path("/usr/share/ollama/.ollama/models"))  # the install script's service user
    return dirs


def find_models_dir(tags, server_config):
    """(folder, source, matched) for the first candidate whose manifests
    match every tag the server lists, else the best partial match, else
    None. Candidates in order of authority: the running server's own
    startup log, $OLLAMA_MODELS here (the server's may differ), then the
    platform defaults."""
    candidates = []
    if server_config.get("OLLAMA_MODELS"):
        candidates.append((Path(server_config["OLLAMA_MODELS"]), "the server's own startup log"))
    if os.environ.get("OLLAMA_MODELS"):
        candidates.append((Path(os.environ["OLLAMA_MODELS"]), "$OLLAMA_MODELS in this shell"))
    candidates += [(d, "the platform default") for d in default_models_dirs()]

    best, seen = None, set()
    for folder, source in candidates:
        key = str(folder).lower()
        if key in seen or not folder.is_dir():
            continue
        seen.add(key)
        matched = verify_models_dir(folder, tags)
        if tags and matched == len(tags):
            return folder, source, matched
        if matched and (best is None or matched > best[2]):
            best = (folder, source, matched)
        if not tags and best is None:
            best = (folder, source, 0)
    return best


# --- the report --------------------------------------------------------------

def _size(n):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024


def ollama_report(base_url):
    """Lines describing the Ollama behind `base_url`, each one checked."""
    import datetime

    if not endpoint_is_local(base_url):
        return [f"  The endpoint is not on this machine, so its install, logs and model "
                f"storage are on {urlparse(base_url).hostname}, not here."]
    lines = []
    binary = find_ollama_binary()
    lines.append(f"  installed:     {binary}" if binary else
                 "  installed:     [!] no ollama executable found on PATH or in the default "
                 "install location (a server in a container looks like this too)")
    version = server_version(base_url)
    lines.append(f"  version:       {version} (reported by the endpoint)" if version else
                 "  version:       unknown -- the endpoint did not answer /api/version")

    folder, note = ollama_log_dir()
    if folder:
        files = ", ".join(
            f"{p.name} {_size(size)}, updated "
            f"{datetime.datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M')}"
            for p, size, mtime in log_files(folder)) or "no server.log or app.log in it"
        lines.append(f"  logs:          {folder}  ({files})")
    else:
        lines.append(f"  logs:          {note}")

    config = read_server_config(folder)
    tags = server_tags(base_url)
    if tags is None:
        lines.append("  model storage: unknown -- the endpoint did not list its models, so no "
                     "folder can be checked against them")
    else:
        found = find_models_dir(tags, config)
        if found is None:
            lines.append(f"  model storage: [!] not found -- none of the usual places holds the "
                         f"{len(tags)} model(s) the server lists. It may be set in the Ollama "
                         f"app's settings or the server's own environment.")
        else:
            path, source, matched = found
            check = (f"verified: {matched} of {len(tags)} model(s) the server lists have a "
                     f"manifest there" if tags else "the server lists no models to check it against")
            if tags and matched < len(tags):
                check = f"[!] only {matched} of {len(tags)} model(s) the server lists are there"
            lines.append(f"  model storage: {path}  (from {source}; {check}; "
                         f"{_size(blobs_size(path))} of model data)")

    host_setting = config.get("OLLAMA_HOST", "")
    if listens_beyond_loopback(host_setting):
        lines.append(f"  network:       [!] the server listens on {host_setting} -- other machines "
                     f"on your network can use it, and Ollama has no authentication. Unless you "
                     f"meant that, turn off \"Expose Ollama to the network\" in the Ollama app, "
                     f"or set OLLAMA_HOST=127.0.0.1.")
    elif config:
        lines.append("  network:       loopback only (from the server's startup log)")
    return lines


def cbench_report():
    """Lines describing where this tool reads and writes, and which setting
    decided each one."""
    import openllm_cbench
    from openllm_cbench.core.config import config_path, models_resolution, resolution
    from openllm_cbench.core.runlock import default_lock_dir

    results, results_source, _ = resolution()
    catalogue, catalogue_source, _ = models_resolution()
    cfg = config_path()
    return [
        f"  installed:     {openllm_cbench.__version__} at {Path(openllm_cbench.__file__).parent}",
        f"  python:        {sys.executable}",
        f"  config file:   {cfg}" + ("" if cfg.exists() else "  (not created yet)"),
        f"  results:       {results}  (from {results_source})",
        f"  catalogue:     {catalogue}  (from {catalogue_source})"
        + ("" if catalogue.exists() else "  (not created yet)"),
        f"  TUI logs:      {results / 'tui-logs'}",
        f"  run lock:      {default_lock_dir()}  (held only while a run is going)",
    ]
