"""
`cbench` -- single entry point for every suite and utility in this
package.

Subcommands for the suites and scoring tools are thin passthroughs: this
module does not redeclare their arguments, it just hands the remaining
argv to that module's own `main()`. Run `cbench <subcommand> --help` for
that subcommand's real flags (declared once, in the module itself).

SAFETY INVARIANT: every suite in this framework measures ATTEMPT, never
success. The containment suite's canary listener binds 127.0.0.1 only and
refuses to start otherwise (see core/canary.py); no suite ever makes a
real outbound request to anywhere but the configured chat endpoint and
that loopback canary. See ARCHITECTURE.md.

Usage:
    cbench doctor
    cbench gate --model <model-tag>
    cbench containment --model <model-tag> --boundary both
    cbench channel --model <model-tag> --think both
    cbench persistence --model <model-tag>
    cbench aggregate --suite s1 --model <model-tag>
    cbench guardrail --csv <path>
    cbench tui                                   # requires: pip install "openllm-cbench[tui]"
"""

import sys

# --- Native subcommands (not passthrough) --------------------------------


def _cmd_doctor(argv):
    import argparse

    from openllm_cbench.core import hardware
    from openllm_cbench.core.endpoint import resolve_base_url, chat_url
    from openllm_cbench.core.canary import start_canary
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.registry import load_registry

    p = argparse.ArgumentParser(prog="cbench doctor", description="Environment sanity check.",
                                 epilog=safety_epilog())
    p.add_argument("--endpoint", default=None, help="Chat endpoint base URL to check.")
    args = p.parse_args(argv)

    print("openllm-cbench doctor")
    print("=" * 40)

    base_url = resolve_base_url(args.endpoint)
    print(f"\nEndpoint: {base_url}")
    try:
        import requests
        resp = requests.get(base_url, timeout=5)
        print(f"  reachable (HTTP {resp.status_code})")
    except Exception as e:
        print(f"  [!] NOT reachable: {e}")
        print("  Every suite needs a running chat endpoint at this address (or pass --endpoint / "
              "set OPENLLM_CBENCH_ENDPOINT).")

    print("\nSafety invariant:")
    try:
        server, port = start_canary()
        print(f"  canary binds loopback-only -- confirmed live (127.0.0.1:{port})")
        server.shutdown()
    except Exception as e:
        print(f"  [!] canary self-check FAILED: {e}")
        print("  Do not trust any suite's output until this is understood.")

    print("\n" + hardware.format_report())

    registry = load_registry()
    n = len(registry.get("models", {}))
    print(f"\nVerified-model registry: {n} model(s) gate-checked and on file.")
    print("  Run `cbench gate --model <tag>` on any model before a real run, "
          "or check `data/models/verified.json` for existing entries.")

    print("\nDone. This command does not call a model and makes no outbound request "
          "beyond the endpoint reachability check above.")
    return 0


def _cmd_gate(argv):
    import argparse

    from openllm_cbench.core.gate import run_gate, render_gate_report, to_registry_entry
    from openllm_cbench.core.endpoint import resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.registry import save_entry

    p = argparse.ArgumentParser(prog="cbench gate", description="Gate-check a model tag before trusting it.",
                                 epilog=safety_epilog())
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default=None, help="Endpoint base URL, e.g. http://localhost:11434 "
                                                      "(not a /api/chat or /api/show path).")
    p.add_argument("--save", action="store_true",
                    help="Write this result into the local model catalogue overlay (default ./models.json) "
                         "so every suite picks it up automatically on future runs against this tag. "
                         "config_overrides is left empty -- add any num_predict/num_ctx/etc a real run "
                         "turns out to need by hand-editing the saved entry.")
    p.add_argument("--registry-file", default=None,
                    help="Overlay file to write to with --save (default: $OPENLLM_CBENCH_MODELS_FILE "
                         "or ./models.json).")
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Gate-checking '{args.model}' against {base_url} ...")
    result = run_gate(args.model, base_url)
    print()
    print(render_gate_report(result))

    if args.save:
        entry = to_registry_entry(result)
        path = save_entry(args.model, entry, args.registry_file)
        print(f"\nSaved to {path}. Every suite will pick this up automatically for '{args.model}' "
              f"on future runs (an explicit CLI flag still always wins). Edit the file directly to "
              f"add config_overrides once a real run tells you what this model needs.")

    return 0 if result.get("clean") else 1


def _cmd_tui(argv):
    """Launches the Textual control panel. Every action it takes is a real
    `cbench` subcommand run as a subprocess -- see tui/jobs.py's module
    docstring. `textual` is an optional dependency (`pip install
    "openllm-cbench[tui]"`); imported lazily so the rest of `cbench` never
    requires it. The ImportError has to be caught HERE, not inside
    tui/app.py's own main() -- app.py imports `textual` at module level
    (needed to define its Screen/Widget subclasses at all), so by the time
    that module's main() would run, the import has already failed and the
    module never finished loading."""
    if argv and argv[0] in ("-h", "--help"):
        print(
            "usage: cbench tui\n\n"
            "Launches the Textual control panel (requires: pip install "
            "\"openllm-cbench[tui]\"). Takes no arguments -- every suite's own "
            "flags are entered through its form inside the TUI, not on this "
            "command line. See README.md 'Terminal UI'."
        )
        return 0
    try:
        from openllm_cbench.tui.app import main as tui_main
    except ImportError:
        print(
            "The TUI needs the 'textual' package, which is an optional dependency.\n"
            "Install it with:\n\n    pip install \"openllm-cbench[tui]\"\n",
            file=sys.stderr,
        )
        return 1
    return tui_main()


# --- Passthrough subcommands ----------------------------------------------

_PASSTHROUGH = {
    "containment": "openllm_cbench.suites.containment",
    "channel": "openllm_cbench.suites.channel",
    "persistence": "openllm_cbench.suites.persistence",
    "aggregate": "openllm_cbench.scoring.aggregate",
    "guardrail": "openllm_cbench.scoring.guardrail",
    "extension-rule": "openllm_cbench.scoring.extension_rule",
    "score-probes": "openllm_cbench.scoring.probes",
    "score-containment": "openllm_cbench.scoring.containment_metrics",
}

_NATIVE = {
    "doctor": _cmd_doctor,
    "gate": _cmd_gate,
    "tui": _cmd_tui,
}


def _dispatch_passthrough(module_path, argv):
    import importlib

    mod = importlib.import_module(module_path)
    old_argv = sys.argv
    try:
        sys.argv = [module_path] + argv
        return mod.main() or 0
    finally:
        sys.argv = old_argv


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0

    subcommand, rest = argv[0], argv[1:]

    if subcommand in _NATIVE:
        return _NATIVE[subcommand](rest)
    if subcommand in _PASSTHROUGH:
        return _dispatch_passthrough(_PASSTHROUGH[subcommand], rest)

    print(f"Unknown subcommand '{subcommand}'.\n", file=sys.stderr)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
