"""`cbench doctor`'s "where things are" -- each location found, then checked.

Written against a real machine where the obvious answers were wrong: the
default model folder existed and was empty while the server's 39 models
sat on another drive, chosen in the Ollama app's own settings; a
directory listing showed the live server.log as 0 bytes while it held
3 MB; and a first version of the manifest check read `qwen3.8` as a
registry host and reported 10 of 39 models missing.
"""

import os
import subprocess
import sys
from pathlib import Path

from openllm_cbench.core import locations as L

SRC = Path(__file__).parent.parent / "src"

# The shape of Ollama's startup line, with the awkward parts: a drive
# letter's colon and doubled backslashes inside a value, a path with a
# space, empty values, and URLs inside a bracketed value.
CONFIG_LINE = (
    'time=2026-09-23T09:52:07.160+01:00 level=INFO source=routes.go:1940 '
    'msg="server config" env="map[CUDA_VISIBLE_DEVICES: HTTPS_PROXY: '
    'OLLAMA_CONTEXT_LENGTH:0 OLLAMA_HOST:http://0.0.0.0:11434 OLLAMA_IGPU_ENABLE: '
    'OLLAMA_KEEP_ALIVE:5m0s OLLAMA_MODELS:D:\\\\My Models\\\\ollama OLLAMA_NOHISTORY:false '
    'OLLAMA_ORIGINS:[http://localhost https://localhost app://*] OLLAMA_SCHED_SPREAD:false '
    'ROCR_VISIBLE_DEVICES: http_proxy: https_proxy: no_proxy:]"'
)


def test_the_startup_line_yields_the_server_s_own_settings():
    config = L.parse_server_config(CONFIG_LINE)
    assert config["OLLAMA_MODELS"] == "D:\\My Models\\ollama"
    assert config["OLLAMA_HOST"] == "http://0.0.0.0:11434"
    assert config["OLLAMA_IGPU_ENABLE"] == ""
    assert config["OLLAMA_ORIGINS"].startswith("[http://localhost")
    assert all(k.startswith("OLLAMA_") for k in config)


def test_the_startup_line_is_found_wherever_it_sits(tmp_path):
    (tmp_path / "server.log").write_text("some earlier line\n" + CONFIG_LINE + "\nlater\n",
                                         encoding="utf-8")
    assert L.read_server_config(tmp_path)["OLLAMA_HOST"] == "http://0.0.0.0:11434"
    assert L.read_server_config(tmp_path / "missing") == {}
    assert L.read_server_config(None) == {}


def test_manifest_paths_follow_position_not_dots():
    assert L.manifest_relpath("qwen3:4b") == "registry.ollama.ai/library/qwen3/4b"
    # The regression: a dotted model name is not a registry host.
    assert L.manifest_relpath("qwen3.8:latest") == "registry.ollama.ai/library/qwen3.8/latest"
    assert L.manifest_relpath("llama3.1") == "registry.ollama.ai/library/llama3.1/latest"
    assert (L.manifest_relpath("aratan/Ornith-1.5-9B-GGUF:Q4_K_M")
            == "registry.ollama.ai/aratan/Ornith-1.5-9B-GGUF/Q4_K_M")
    assert (L.manifest_relpath("hf.co/empero-ai/Qwen3.8-9B-GGUF:Q4_K_M")
            == "hf.co/empero-ai/Qwen3.8-9B-GGUF/Q4_K_M")


def _store(folder, tags):
    for tag in tags:
        path = folder / "manifests" / L.manifest_relpath(tag).lower()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
    (folder / "blobs").mkdir(parents=True, exist_ok=True)
    (folder / "blobs" / "sha256-x").write_bytes(b"0" * 2048)


TAGS = ["qwen3.8:latest", "qwen3:4b", "aratan/Ornith-1.5-9B-GGUF:Q4_K_M",
        "hf.co/empero-ai/Qwen3.8-9B-GGUF:Q4_K_M"]


def test_a_folder_counts_only_when_its_manifests_match_the_server(tmp_path):
    _store(tmp_path, TAGS)
    assert L.verify_models_dir(tmp_path, TAGS) == len(TAGS)
    assert L.verify_models_dir(tmp_path, TAGS + ["never-pulled:1b"]) == len(TAGS)
    assert L.verify_models_dir(tmp_path / "nowhere", TAGS) == 0
    assert L.blobs_size(tmp_path) == 2048


def test_the_server_s_own_folder_wins_over_an_empty_default(tmp_path, monkeypatch):
    """The trap measured live: the default folder exists and is empty."""
    home = tmp_path / "home"
    empty_default = home / ".ollama" / "models"
    (empty_default / "manifests").mkdir(parents=True)
    real = tmp_path / "elsewhere"
    _store(real, TAGS)
    monkeypatch.setattr(L, "default_models_dirs", lambda: [empty_default])
    monkeypatch.delenv("OLLAMA_MODELS", raising=False)

    folder, source, matched = L.find_models_dir(TAGS, {"OLLAMA_MODELS": str(real)})
    assert folder == real and matched == len(TAGS)
    assert "startup log" in source
    # With no config to read, the empty default is NOT reported as the answer.
    assert L.find_models_dir(TAGS, {}) is None


def test_network_exposure_is_read_from_the_bind_address():
    assert L.listens_beyond_loopback("http://0.0.0.0:11434")
    assert L.listens_beyond_loopback("0.0.0.0")
    assert L.listens_beyond_loopback("192.168.1.20:11434")
    assert not L.listens_beyond_loopback("")               # Ollama's default, loopback
    assert not L.listens_beyond_loopback("127.0.0.1:11434")
    assert not L.listens_beyond_loopback("http://localhost:11434")


def test_only_a_loopback_endpoint_has_local_files():
    assert L.endpoint_is_local("http://localhost:11434")
    assert L.endpoint_is_local("http://127.0.0.1:11434")
    assert not L.endpoint_is_local("http://gpu-box.lan:11434")
    lines = L.ollama_report("http://gpu-box.lan:11434")
    assert len(lines) == 1 and "gpu-box.lan" in lines[0]


def test_log_sizes_come_from_the_file_itself(tmp_path):
    (tmp_path / "server.log").write_bytes(b"x" * 5000)
    [(path, size, _)] = L.log_files(tmp_path)
    assert path.name == "server.log" and size == 5000


def test_the_cbench_half_names_the_setting_behind_each_path(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENLLM_CBENCH_RESULTS_DIR", str(tmp_path / "res"))
    monkeypatch.setenv("OPENLLM_CBENCH_MODELS_FILE", str(tmp_path / "cat.json"))
    text = "\n".join(L.cbench_report())
    assert str(tmp_path / "res") in text and "$OPENLLM_CBENCH_RESULTS_DIR" in text
    assert str(tmp_path / "cat.json") in text and "(not created yet)" in text
    assert "run lock:" in text and "TUI logs:" in text


def test_doctor_reports_both_halves_without_an_endpoint(tmp_path):
    """CI has no Ollama. Every line must degrade to a sentence, not a traceback."""
    result = subprocess.run(
        [sys.executable, "-m", "openllm_cbench.cli", "doctor", "--endpoint", "http://127.0.0.1:9"],
        capture_output=True, text=True, timeout=120, cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(SRC)},
    )
    assert result.returncode == 0, result.stderr
    assert "Traceback" not in result.stdout + result.stderr
    assert "\nOllama:\n" in result.stdout and "\ncbench:\n" in result.stdout
    assert "model storage:" in result.stdout
