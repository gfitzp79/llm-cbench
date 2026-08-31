"""
Advisory hardware probe -- estimates a reasonable max model size for this
machine. Never blocks a run; every suite still runs against any model tag
regardless of what this reports. The harness talks to an HTTP chat
endpoint and is model-scale-agnostic -- this module exists purely to help
a user pick a sensible starting point, not to gatekeep.

Detection is best-effort and platform-dependent, wrapped so a missing
tool or an unrecognized platform degrades to "unknown" rather than
raising. Nothing here is authoritative -- always treat the reported
figures as a rough guide, not a spec.
"""

import platform
import re
import shutil
import subprocess

# Rough bytes-per-parameter at common quantizations, used only for the
# advisory band below. Real values vary by quant implementation and model
# architecture -- do not treat these as precise.
BYTES_PER_PARAM = {
    "Q4_K_M": 0.55,
    "Q5_K_M": 0.65,
    "Q8_0": 1.0,
    "F16": 2.0,
}


def _run(cmd, timeout=10):
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if out.returncode != 0:
            return None
        return out.stdout
    except Exception:
        return None


def detect_gpu_vram_mb():
    """Returns (vendor, total_mb) for the first detected GPU, or (None, None)
    if nothing could be detected. Tries nvidia-smi, then rocm-smi, then
    (on macOS) sysctl for unified memory. Never raises."""
    if shutil.which("nvidia-smi"):
        out = _run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
        if out:
            try:
                mb = int(out.strip().splitlines()[0].strip())
                return "nvidia", mb
            except (ValueError, IndexError):
                pass

    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showmeminfo", "vram", "--csv"])
        if out:
            m = re.search(r"(\d+)\s*$", out.strip().splitlines()[-1]) if out.strip() else None
            if m:
                try:
                    return "amd", int(m.group(1)) // (1024 * 1024)
                except ValueError:
                    pass

    if platform.system() == "Darwin":
        out = _run(["sysctl", "-n", "hw.memsize"])
        if out:
            try:
                # Apple Silicon: unified memory, treat total system RAM as
                # the GPU-accessible ceiling (a rough overestimate in
                # practice, since the OS and other processes also share it).
                return "apple-unified", int(out.strip()) // (1024 * 1024)
            except ValueError:
                pass

    return None, None


def detect_system_ram_mb():
    """stdlib-only per platform. Never raises; returns None if it can't
    determine a value."""
    system = platform.system()
    try:
        if system == "Linux":
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return kb // 1024
        elif system == "Windows":
            out = _run(["wmic", "computersystem", "get", "TotalPhysicalMemory"])
            if out:
                for line in out.splitlines():
                    line = line.strip()
                    if line.isdigit():
                        return int(line) // (1024 * 1024)
        elif system == "Darwin":
            out = _run(["sysctl", "-n", "hw.memsize"])
            if out:
                return int(out.strip()) // (1024 * 1024)
    except Exception:
        pass
    return None


def recommend_band(vram_mb, quant="Q4_K_M", ctx_overhead_mb=1024):
    """Returns an advisory max-parameter-count band (in billions) for the
    given VRAM budget at the given quantization. Purely arithmetic, shown
    with its reasoning so it's easy to sanity-check or override by hand:
    usable_mb = vram_mb - ctx_overhead_mb (headroom for KV cache + runtime
    overhead); max_params_b = usable_mb / (bytes_per_param * 1024)."""
    if not vram_mb:
        return None
    bytes_per_param = BYTES_PER_PARAM.get(quant, BYTES_PER_PARAM["Q4_K_M"])
    usable_mb = max(0, vram_mb - ctx_overhead_mb)
    max_params_b = usable_mb / (bytes_per_param * 1024)
    return round(max_params_b, 1)


def probe():
    """Returns a dict summarizing this machine's advisory model-size band.
    Never raises -- every field degrades to None/'unknown' independently."""
    vendor, vram_mb = detect_gpu_vram_mb()
    ram_mb = detect_system_ram_mb()
    band_b = recommend_band(vram_mb) if vram_mb else None
    return {
        "gpu_vendor": vendor,
        "gpu_vram_mb": vram_mb,
        "system_ram_mb": ram_mb,
        "advisory_max_params_b_q4": band_b,
    }


def format_report(info=None):
    info = info or probe()
    lines = ["Hardware (advisory only -- every suite runs regardless):"]
    if info["gpu_vendor"] and info["gpu_vram_mb"]:
        lines.append(f"  GPU: {info['gpu_vendor']}, {info['gpu_vram_mb']} MB VRAM")
    else:
        lines.append("  GPU: not detected (nvidia-smi/rocm-smi not found, or non-GPU/unsupported platform)")
    if info["system_ram_mb"]:
        lines.append(f"  System RAM: {info['system_ram_mb']} MB")
    else:
        lines.append("  System RAM: could not determine")
    if info["advisory_max_params_b_q4"]:
        lines.append(
            f"  Advisory band: up to ~{info['advisory_max_params_b_q4']}B params at Q4_K_M "
            f"(rough headroom estimate, not a hard limit -- context length and runtime "
            f"overhead both eat into this)"
        )
    else:
        lines.append("  Advisory band: unavailable (no VRAM figure detected)")
    return "\n".join(lines)
