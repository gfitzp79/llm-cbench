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
    cbench discover                               # what's pulled locally but not catalogued yet
    cbench discover --gate-all                     # ...and gate-check + save all of them
    cbench search --model <model-tag>             # check it exists in Ollama's registry first
    cbench pull --model <model-tag>               # download a model into the local endpoint
    cbench assess --model <model-tag> --trials 3   # full S1+S2+S3 assessment, auto-aggregated
    cbench containment --model <model-tag> --boundary both
    cbench channel --model <model-tag> --think both
    cbench persistence --model <model-tag>
    cbench aggregate --suite s1 --model <model-tag>
    cbench guardrail --csv <path>
    cbench tui                                   # requires: pip install "openllm-cbench[tui]"
"""

import sys
from openllm_cbench.core.console import ensure_utf8_stdio
from openllm_cbench.core.runlock import RunLock, RunLockBusy

# --- Native subcommands (not passthrough) --------------------------------


def _cmd_doctor(argv):
    import argparse

    from openllm_cbench.core import hardware
    from openllm_cbench.core.endpoint import resolve_base_url
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


def _cmd_assess(argv):
    """Runs a full assessment of one model: N trials each of the selected
    suites, then auto-aggregates each suite's trials into a trial-summary
    report. This is S1 (containment) + S2 (channel) + S3 (persistence)
    ONLY -- the three suites this framework actually ships. Deliberately
    excludes two things a reader familiar with the private research lab
    this framework was extracted from might expect: "S4" (that lab's
    external inspect_evals benchmarks) was never extracted into this
    framework and doesn't exist here to run; "S5" (that lab's Inspect
    cross-validation of S1/S3) DOES exist here, as `integrations/`, but
    stays a separate, deliberate action (`inspect eval ...`) rather than
    being folded into every assessment -- it's a validity check on S1/S3's
    own scoring, not a fourth independent measurement, and doubling S1/S3
    runtime by default wasn't judged worth it for every run.

    Each suite invocation is the identical `_dispatch_passthrough()`
    machinery every other passthrough subcommand already uses -- N
    repeated real calls to that suite's own main(), not a fourth
    implementation of any of them. This automates the documented manual
    workflow (scoring/aggregate.py's own module docstring: "run those N
    times per model first ... then point this at the resulting CSVs")
    rather than replacing it -- the exact same CSVs land in the exact
    same place either way, so `cbench aggregate` still works standalone
    on whatever this command produces."""
    import argparse

    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.scoring.aggregate import aggregate_s1, aggregate_s2, aggregate_s3

    SUITE_INFO = {
        "s1": ("containment", "openllm_cbench.suites.containment", aggregate_s1, "s1_containment"),
        "s2": ("channel", "openllm_cbench.suites.channel", aggregate_s2, "s2_channel"),
        "s3": ("persistence", "openllm_cbench.suites.persistence", aggregate_s3, "s3_persistence"),
    }

    p = argparse.ArgumentParser(
        prog="cbench assess",
        description="Full assessment of one model: N trials each of the selected suites "
                     "(s1 containment, s2 channel, s3 persistence -- this framework's own "
                     "three, no more), then auto-aggregate each into a trial-summary report.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--suites", default="s1,s2,s3",
                    help="Comma-separated subset of s1,s2,s3 (default: all three).")
    p.add_argument("--trials", type=int, default=3,
                    help="Trials per suite (default 3 -- this framework's own pre-registered "
                         "minimum for a rate worth citing; see ARCHITECTURE.md).")
    p.add_argument("--dry-run", action="store_true",
                    help="Pass --dry-run through to every suite invocation -- prints each "
                         "payload, calls no model, and skips aggregation since there would "
                         "be no real CSVs to aggregate.")
    p.add_argument("--force-concurrent", action="store_true",
                    help="Run even though another assessment holds the run lock, or a "
                         "suite process is already live. There is no routine use for "
                         "this: two assessments on one GPU halve each other's throughput "
                         "and make every wall-time number they produce misleading. Use "
                         "only when the other work is provably on different hardware.")
    args = p.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in SUITE_INFO]
    if unknown:
        print(f"[!] Unknown suite(s): {', '.join(unknown)} -- choose from s1,s2,s3", file=sys.stderr)
        return 2
    if not suites:
        print("[!] --suites resolved to nothing to run.", file=sys.stderr)
        return 2
    if args.trials < 1:
        print("[!] --trials must be at least 1.", file=sys.stderr)
        return 2

    print(f"Full assessment: '{args.model}', suites={','.join(suites)}, trials={args.trials}"
          f"{' (dry-run)' if args.dry_run else ''}")
    print("S1+S2+S3 only -- see this command's own --help / module docstring for why S4/S5 "
          "aren't included.\n")

    # LAUNCH-TIME EXCLUSIVITY GUARD.
    #
    # An assessment is long, expensive, and yields usable data only on
    # completion. Two running at once on one GPU do not fail -- they halve
    # each other's throughput and corrupt every wall-time number taken from
    # them, which then gets misread as model behaviour. See
    # core/runlock.py's docstring for the incident this is modelled on.
    #
    # This is the only guard in the framework that runs BEFORE the work.
    # Every other check reads artifacts that already exist, which is
    # detection after the cost is sunk.
    #
    # --dry-run skips it deliberately: a dry run makes no model calls, so it
    # contends for nothing and must stay usable while a real assessment runs.
    lock = None
    if not args.dry_run:
        try:
            lock = RunLock(label="assess %s" % args.model,
                           force=args.force_concurrent).acquire()
        except RunLockBusy as e:
            print("\n[!] NOT STARTING -- %s" % e, file=sys.stderr)
            print("\n    An assessment already running, or a suite process still "
                  "live, means\n    this one would contend for the same GPU. "
                  "Stop it and VERIFY the process\n    table before retrying: a "
                  "kill is not done until the table says it is done.\n"
                  "    Override with --force-concurrent only if you know the "
                  "other work is on\n    different hardware.", file=sys.stderr)
            return 2

    try:
        return _assess_body(args, suites, SUITE_INFO)
    finally:
        if lock is not None:
            lock.release()


def _assess_body(args, suites, SUITE_INFO):
    from openllm_cbench.core.paths import results_dir as _results_dir
    from openllm_cbench.scoring.aggregate import model_tag

    report_paths = []
    failures = []
    for suite in suites:
        label, module_path, aggregate_fn, results_subdir = SUITE_INFO[suite]
        print(f"=== {suite.upper()} ({label}) -- {args.trials} trial(s) ===")
        for trial in range(1, args.trials + 1):
            print(f"\n--- {suite} trial {trial}/{args.trials} ---")
            trial_args = ["--model", args.model]
            if args.dry_run:
                trial_args.append("--dry-run")
            rc = _dispatch_passthrough(module_path, trial_args)
            if rc != 0:
                print(f"[!] {suite} trial {trial} exited {rc} -- continuing with remaining trials.",
                      file=sys.stderr)
                failures.append((suite, trial, rc))

        if args.dry_run:
            print(f"\n[dry-run] Skipping aggregation for {suite} -- no real CSVs were produced.\n")
            continue

        print(f"\nAggregating {suite}...")
        try:
            md = aggregate_fn(args.model)
        except SystemExit as e:
            print(f"[!] Could not aggregate {suite}: {e}", file=sys.stderr)
            continue
        out_dir = _results_dir(results_subdir)
        out_path = out_dir / f"trial_summary_{model_tag(args.model)}.md"
        out_path.write_text(md, encoding="utf-8")
        report_paths.append(out_path)
        print(f"Report: {out_path}\n")

    print("=" * 60)
    print(f"Assessment complete: {len(suites)} suite(s), {args.trials} trial(s) each"
          f"{' (dry-run, no reports)' if args.dry_run else ''}.")
    if failures:
        print(f"\n[!] {len(failures)} trial(s) exited non-zero -- read the output above "
              f"before trusting any aggregate that includes them:")
        for suite, trial, rc in failures:
            print(f"    {suite} trial {trial}: exit {rc}")
    if report_paths:
        print("\nReports:")
        for p_ in report_paths:
            print(f"  {p_}")
    return 1 if failures else 0


def _cmd_search(argv):
    """Checks whether a model tag exists in Ollama's registry -- a real
    live lookup, not a browsable catalogue. See
    core/pull.py:check_model_availability()'s docstring for why this
    reuses /api/pull's own manifest-fetch step rather than adding a new
    external destination (Ollama has no official remote-library search
    API; this framework deliberately doesn't scrape or wrap an unofficial
    one -- see core/discover.py). Downloads nothing regardless of the
    result."""
    import argparse

    from openllm_cbench.core.endpoint import resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.pull import check_model_availability

    p = argparse.ArgumentParser(
        prog="cbench search",
        description="Check whether an exact model tag exists in Ollama's registry, and its "
                     "download size, without downloading it. Not a keyword/browse search -- "
                     "you need the exact tag (as it would appear to `ollama pull`); this "
                     "confirms it exists before you commit to pulling it.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default=None)
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Checking '{args.model}' against {base_url}'s registry ...")
    try:
        result = check_model_availability(args.model, base_url)
    except Exception as e:
        print(f"[!] Could not reach {base_url}: {e}", file=sys.stderr)
        return 1

    if result["exists"]:
        gb = result["size_bytes"] / (1024 ** 3) if result["size_bytes"] else None
        size_str = f"{gb:.1f} GB" if gb else "unknown size"
        print(f"Found: '{args.model}' exists ({size_str}). "
              f"Run `cbench pull --model {args.model}` to download it.")
        return 0
    print(f"Not found: '{args.model}' -- {result['error'] or 'no matching manifest'}.",
          file=sys.stderr)
    return 1


def _cmd_pull(argv):
    """Pulls a model into the local endpoint -- see core/pull.py's module
    docstring for why this needs no new trust boundary beyond what
    `ollama pull` already does from a terminal today."""
    import argparse

    from openllm_cbench.core.endpoint import resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.pull import pull_model, throttled_progress_printer

    p = argparse.ArgumentParser(
        prog="cbench pull",
        description="Pull a model into the local endpoint. Downloads real data, "
                     "potentially several GB -- streams progress as it happens.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default=None)
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Pulling '{args.model}' into {base_url} ...")

    def _print(line):
        print(line, flush=True)

    try:
        ok, final = pull_model(args.model, base_url, on_progress=throttled_progress_printer(_print))
    except Exception as e:
        print(f"[!] Could not reach {base_url}: {e}", file=sys.stderr)
        return 1

    if ok:
        print(f"\nDone: '{args.model}' pulled successfully. Run `cbench gate --model "
              f"{args.model} --save` next to add it to your catalogue.")
        return 0
    print(f"\n[!] Pull failed: {final}", file=sys.stderr)
    return 1


def _cmd_discover(argv):
    import argparse

    from openllm_cbench.core.discover import list_local_models, find_uncatalogued, format_size
    from openllm_cbench.core.endpoint import resolve_base_url
    from openllm_cbench.core.gate import run_gate, render_gate_report, to_registry_entry
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.registry import load_registry, save_entry

    p = argparse.ArgumentParser(
        prog="cbench discover",
        description="List locally-pulled models the catalogue doesn't know about yet. "
                     "Only ever talks to your local endpoint's own /api/tags -- never "
                     "ollama.com. Does not pull anything; run `ollama pull <tag>` yourself "
                     "first for a model that isn't local yet.",
        epilog=safety_epilog(),
    )
    p.add_argument("--endpoint", default=None, help="Endpoint base URL to list models from.")
    p.add_argument("--registry-file", default=None,
                    help="Overlay file to check against and (with --gate-all) write to "
                         "(default: $OPENLLM_CBENCH_MODELS_FILE or ./models.json).")
    p.add_argument("--gate-all", action="store_true",
                    help="Gate-check and save every uncatalogued model found, one at a time "
                         "(same as running `cbench gate --model <tag> --save` per model). "
                         "Real model calls -- can take a while for a long list; see --limit.")
    p.add_argument("--limit", type=int, default=None,
                    help="With --gate-all, only process the first N uncatalogued models found "
                         "(in /api/tags's own order). Useful to bound a long batch run.")
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Listing locally-pulled models from {base_url} ...")
    try:
        local = list_local_models(base_url)
    except Exception as e:
        print(f"[!] Could not reach {base_url}/api/tags: {e}", file=sys.stderr)
        return 1

    registry = load_registry(args.registry_file)
    uncatalogued = find_uncatalogued(local, registry)

    print(f"{len(local)} model(s) pulled locally; {len(uncatalogued)} not yet in the catalogue "
          f"({len(local) - len(uncatalogued)} already covered).\n")

    if not uncatalogued:
        print("Nothing to do -- every locally-pulled model is already catalogued.")
        return 0

    for m in uncatalogued:
        print(f"  {m['name']}  ({m['architecture']}, {m['params_b']}, {m['quant']}, "
              f"{format_size(m['size'])})")

    if not args.gate_all:
        print(f"\nRun with --gate-all to gate-check and save all {len(uncatalogued)} of these, "
              f"or `cbench gate --model <tag> --save` one at a time.")
        return 0

    to_gate = uncatalogued[:args.limit] if args.limit is not None else uncatalogued
    if args.limit is not None and args.limit < len(uncatalogued):
        print(f"\n--limit {args.limit}: processing the first {len(to_gate)} of "
              f"{len(uncatalogued)} uncatalogued models.")
    print(f"\nGate-checking {len(to_gate)} model(s) -- this makes real calls to each "
          f"and can take a while:\n")
    results = []
    for m in to_gate:
        tag = m["name"]
        print(f"--- {tag} ---")
        try:
            result = run_gate(tag, base_url)
        except Exception as e:
            print(f"  [!] gate check failed to run: {e}\n")
            results.append((tag, "error", str(e)))
            continue
        entry = to_registry_entry(result)
        path = save_entry(tag, entry, args.registry_file)
        status = "clean" if result.get("clean") else "caveats found"
        print(f"  {status} -- saved to {path}")
        if not result.get("clean"):
            # Same detail `cbench gate` itself prints for one model -- a bare
            # "caveats found" here would force a separate re-run per flagged
            # tag just to learn what the caveat actually is.
            print(render_gate_report(result))
        else:
            print()
        results.append((tag, status, None))

    clean_n = sum(1 for _, s, _ in results if s == "clean")
    print(f"Done: {clean_n}/{len(results)} clean, "
          f"{sum(1 for _, s, _ in results if s == 'caveats found')} with caveats, "
          f"{sum(1 for _, s, _ in results if s == 'error')} failed to run.")
    return 0


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
    "discover": _cmd_discover,
    "search": _cmd_search,
    "pull": _cmd_pull,
    "assess": _cmd_assess,
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
    ensure_utf8_stdio()
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
