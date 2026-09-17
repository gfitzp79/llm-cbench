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
            # ctypes first, not wmic: Windows 11 ships without wmic (it was
            # deprecated and then removed), so the subprocess route silently
            # returned nothing on exactly the machines most likely to be
            # running a local model. GlobalMemoryStatusEx is in kernel32 on
            # every supported Windows and needs no external binary at all.
            import ctypes

            class _MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]

            stat = _MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys) // (1024 * 1024)
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


def check_model_fit(params_b, quant, vram_mb, ctx_overhead_mb=1024, size_mb=None):
    """Advisory only -- estimates whether a model is likely to fit in the
    given VRAM budget, using the same arithmetic as recommend_band() run
    in reverse (needed_mb from params_b/quant, rather than max-params_b
    from vram_mb). Never blocks anything; a caller decides what, if
    anything, to do with a `fits: False` result.

    `size_mb` is the model's REAL on-disk size when the caller knows it
    (the endpoint reports it per model). It is preferred over the
    params/quant estimate whenever available, because it's a measurement
    rather than a calculation: it already accounts for whatever the
    quantisation actually did, embedded tokenizer/vision weights, and
    everything else this module's rough BYTES_PER_PARAM table can only
    approximate.

    Returns {"known": bool, "fits": bool|None, "needed_mb": float|None,
    "usable_mb": int|None, "source": "size"|"estimate"|None}. `known` is
    False (and every other field None) when there's nothing to compare --
    this never guesses a verdict from a partial input, since "probably
    fits" and "unknown" are different things to show a user."""
    if size_mb and vram_mb:
        usable_mb = max(0, vram_mb - ctx_overhead_mb)
        return {"known": True, "fits": size_mb <= usable_mb, "needed_mb": size_mb,
                "usable_mb": usable_mb, "source": "size"}
    # params_b comes from the model catalogue, where an entry that
    # couldn't be measured stores the literal string "unknown" (see
    # core/gate.py:_numeric_params_b's own fallback) rather than a
    # number -- isinstance, not truthiness, so that string doesn't slip
    # through and crash the arithmetic below.
    if not isinstance(params_b, (int, float)) or not quant or not vram_mb:
        return {"known": False, "fits": None, "needed_mb": None, "usable_mb": None,
                "source": None}
    bytes_per_param = BYTES_PER_PARAM.get(quant, BYTES_PER_PARAM["Q4_K_M"])
    needed_mb = params_b * bytes_per_param * 1024
    usable_mb = max(0, vram_mb - ctx_overhead_mb)
    return {"known": True, "fits": needed_mb <= usable_mb, "needed_mb": needed_mb,
            "usable_mb": usable_mb, "source": "estimate"}


# --- Fit assessment ------------------------------------------------------
#
# Turns "how big is this model vs this GPU" into a recommendation a user
# can act on BEFORE spending an hour finding out. Advisory, like
# everything else in this module -- it never blocks a run.
#
# The mixture-of-experts distinction is the part that makes this worth
# having rather than a bare size comparison. An MoE's weights must all be
# resident (or paged), so it needs the SAME memory as a dense model of
# the same total size -- but only a fraction of them are touched per
# token, so when it does spill, it degrades far less badly than a dense
# model of that size. Ranking a 30B-A3B MoE as "just as bad as" a dense
# 30B is wrong in the direction that matters: it talks someone out of a
# model that would actually have been usable.

_ACTIVE_PARAMS_RE = re.compile(r"[-_:]a(\d+(?:\.\d+)?)b\b", re.IGNORECASE)


def parse_active_params_b(tag):
    """Active (per-token) parameter count in billions from a tag that
    advertises one -- the `a3b` in `qwen3:30b-a3b` is the community
    convention for "30B total, 3B active". Returns None when the tag
    says nothing, which is the common case for a dense model."""
    if not tag:
        return None
    m = _ACTIVE_PARAMS_RE.search(tag)
    return float(m.group(1)) if m else None


def is_moe(architecture=None, tag=None):
    """Whether this looks like a mixture-of-experts model. Two
    independent signals, either sufficient: the endpoint's own
    architecture string (`qwen3moe`, `nemotron_h_moe`), and an active
    param count in the tag. Neither is authoritative -- a new MoE family
    that names itself nothing like "moe" and ships no `aNb` tag reads as
    dense here, which is the safe direction to be wrong in (it predicts a
    worse outcome than reality)."""
    if architecture and "moe" in str(architecture).lower():
        return True
    return parse_active_params_b(tag) is not None


def fit_assessment(vram_mb, tag="", architecture=None, params_b=None, quant=None,
                    size_mb=None, ctx_overhead_mb=1024):
    """Recommendation for running `tag` on a GPU with `vram_mb`.

    Returns a dict with `tier` ("fits" | "tight" | "spills" | "unknown"),
    a short `headline` for a table cell, and a full-sentence `note` for a
    warning line, plus the raw numbers behind them. Prefers a real
    on-disk `size_mb` over the params/quant estimate -- see
    check_model_fit()."""
    fit = check_model_fit(params_b, quant, vram_mb, ctx_overhead_mb, size_mb=size_mb)
    moe = is_moe(architecture, tag)
    active = parse_active_params_b(tag)

    if not fit["known"]:
        return {"tier": "unknown", "moe": moe, "active_params_b": active,
                "needed_mb": None, "usable_mb": None, "source": None,
                "headline": "unknown",
                "note": ""}

    needed, usable = fit["needed_mb"], fit["usable_mb"]
    base = {"moe": moe, "active_params_b": active, "needed_mb": needed,
            "usable_mb": usable, "source": fit["source"]}
    measured = "on disk" if fit["source"] == "size" else "estimated"

    if needed <= usable * 0.85:
        return {**base, "tier": "fits", "headline": "fits",
                "note": (f"Hardware: ~{needed:,.0f} MB {measured}, ~{usable:,} MB usable "
                         f"VRAM -- fits, should run at full GPU speed.")}

    if needed <= usable:
        return {**base, "tier": "tight", "headline": "tight",
                "note": (f"Hardware: ~{needed:,.0f} MB {measured} against ~{usable:,} MB usable "
                         f"VRAM -- fits, but with almost no headroom. A long context or a "
                         f"second process on this GPU will push it into system RAM mid-run.")}

    over = needed - usable
    if moe:
        active_note = (f"only ~{active:g}B of ~{params_b:g}B parameters are active per token"
                        if active and params_b else
                        "only a fraction of its parameters are active per token")
        speed = (f"It's a mixture-of-experts model, so {active_note} -- it degrades far less "
                 f"than a dense model this size would. Expect slower, not unusable.")
    else:
        speed = ("It's a dense model, so every parameter is read for every token -- expect "
                 "roughly 5-20x slower than a model that fits, which turns a minutes-long "
                 "run into an hours-long one.")

    return {**base, "tier": "spills", "headline": "spills (MoE)" if moe else "spills",
            "note": (f"Hardware: ~{needed:,.0f} MB {measured} but only ~{usable:,} MB of VRAM "
                     f"is usable -- about {over:,.0f} MB will spill into system RAM. {speed}")}


# --- Expected speed ------------------------------------------------------
#
# "Will it fit" and "will it be quick" are different questions with
# different answers, and a size column answers neither on its own. A
# 30B-A3B MoE is the same size on disk as a dense 30B and roughly ten
# times faster to generate with, because only the active experts are read
# per token. A 0.6B model that fits is fast for the opposite reason.
# Showing size alone invites the reader to rank those three by the one
# number that does not predict what they care about.
#
# EFFECTIVE parameters -- active for an MoE, total for a dense model --
# are what drive generation speed. Memory pressure then modifies it: a
# model that spills reads weights over the PCIe bus or from system RAM,
# and a dense model pays that per token where an MoE pays it only for the
# experts it touches.
#
# All of this is a fallback. If the model has actually been gate-checked
# on this machine, we have a MEASURED tokens/sec and use that instead --
# same principle as preferring real on-disk size over bytes-per-param
# arithmetic in check_model_fit().

SPEED_BANDS = ((2, "fast"), (9, "good"), (25, "moderate"), (float("inf"), "slow"))
MEASURED_BANDS = ((40, "fast"), (15, "good"), (6, "moderate"), (0, "slow"))


def effective_params_b(params_b, active_params_b=None):
    """Parameters actually read per token: the active count for a
    mixture-of-experts model, the full count for a dense one. None when
    neither is known."""
    if active_params_b:
        return active_params_b
    return params_b if isinstance(params_b, (int, float)) else None


def performance_estimate(fit_tier=None, params_b=None, active_params_b=None,
                          measured_tok_s=None, moe=False):
    """Expected generation speed for a model on this machine.

    Returns {"label", "source", "tok_s", "detail"}. `source` is
    "measured" when a gate check on this machine timed the model, and
    "estimate" when the label is inferred from effective parameter count
    and memory pressure. A caller should show the distinction: one is a
    fact about this hardware, the other is arithmetic."""
    if measured_tok_s:
        # A measured rate is banded on the rate itself, not on parameter
        # count -- the whole point of having measured it is that the
        # proxy is no longer needed.
        label = next(lbl for floor, lbl in MEASURED_BANDS if measured_tok_s >= floor)
        return {"label": label, "source": "measured", "tok_s": measured_tok_s,
                "detail": f"{measured_tok_s:.0f} tok/s measured on this machine"}

    eff = effective_params_b(params_b, active_params_b)
    if eff is None:
        return {"label": "unknown", "source": None, "tok_s": None, "detail": ""}

    label = next(lbl for ceiling, lbl in SPEED_BANDS if eff <= ceiling)

    # Memory pressure degrades the estimate. A dense model that spills
    # pays for every parameter on every token; an MoE pays only for the
    # experts it actually reads, so it drops one band rather than two.
    order = ["fast", "good", "moderate", "slow"]
    if fit_tier == "spills":
        label = order[min(len(order) - 1, order.index(label) + (1 if moe else 2))]
    elif fit_tier == "tight":
        label = order[min(len(order) - 1, order.index(label) + 1)]

    if active_params_b and params_b:
        detail = (f"~{active_params_b:g}B active of {params_b:g}B "
                  f"(mixture-of-experts), estimated")
    else:
        detail = f"~{eff:g}B parameters read per token, estimated"
    return {"label": label, "source": "estimate", "tok_s": None, "detail": detail}


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
