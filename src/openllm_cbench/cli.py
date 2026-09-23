"""
`cbench`: the single entry point for every suite and utility in this
package.

Subcommands for the suites and scoring tools are thin passthroughs: this
module does not redeclare their arguments, it just hands the remaining
argv to that module's own `main()`. Run `cbench <subcommand> --help` for
that subcommand's real flags (declared once, in the module itself).

Usage:
    cbench --version
    cbench doctor
    cbench config --set-results-dir <path>        # pin where results go, for every directory
    cbench gate --model <model-tag>
    cbench discover                               # what's pulled locally but not catalogued yet
    cbench discover --gate-all                     # ...and gate-check + save all of them
    cbench search --model <model-tag>             # check it exists in Ollama's registry first
    cbench pull --model <model-tag>               # download a model into the local endpoint
    cbench remove --model <model-tag>             # delete a model from the local endpoint
    cbench assess --model <model-tag> --trials 3   # full S1+S2+S3 assessment, auto-aggregated
    cbench score --model <model-tag> --depth standard   # assess + a per-suite scorecard
    cbench compare --model <tag-a> --model <tag-b>      # is the difference between two models real?
    cbench catalogue                              # every local model + catalogue/score status
    cbench community-package --model <model-tag> --accept-terms   # bundle your CSVs
    cbench community-submit <folder>              # open it as a PR (needs `gh`; --confirm to send)
    cbench community-validate community-results/<model-tag>/<contributor>_<date>
    cbench containment --model <model-tag> --boundary both
    cbench channel --model <model-tag> --think both
    cbench persistence --model <model-tag>
    cbench aggregate --suite s1 --model <model-tag>
    cbench extension-rule --pair <base-tag> <variant-tag>   # pre-registered stopping rule
    cbench guardrail --csv <path>
    cbench score-containment                        # standalone: re-score S1 CSVs with richer egress metrics
    cbench tui                                   # requires: pip install textual
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

    # Found and then checked, never assumed from a default -- see
    # core/locations.py for the two defaults that were wrong on a real machine.
    from openllm_cbench.core import locations
    print("\nOllama:")
    for line in locations.ollama_report(base_url):
        print(line)
    print("\ncbench:")
    for line in locations.cbench_report():
        print(line)

    print("\nSafety invariant:")
    try:
        server, port = start_canary()
        print(f"  canary binds loopback-only, confirmed live (127.0.0.1:{port})")
        server.shutdown()
    except Exception as e:
        print(f"  [!] canary self-check FAILED: {e}")
        print("  Do not trust any suite's output until this is understood.")

    print("\n" + hardware.format_report())

    registry = load_registry()
    n = len(registry.get("models", {}))
    print(f"\nModel catalogue: {n} model(s) on file, from the catalogue that ships with "
          "cbench and any saved with `cbench gate --save`.")
    print("  `cbench catalogue` shows which of your models have an entry. "
          "Run `cbench gate --model <tag>` on any model before a real run.")

    print("\nDone. This command calls no model and writes nothing. Its only requests go to the "
          "endpoint above (reachability, version, model list); everything local is read-only.")
    return 0


def _cmd_gate(argv):
    import argparse

    from openllm_cbench.core.gate import (
        model_answered, render_gate_report, run_gate, to_registry_entry,
    )
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
                         "config_overrides is left empty except for `think`, set to false when this "
                         "check finds the model only calls tools with its reasoning channel off. Add "
                         "any num_predict/num_ctx/etc a real run turns out to need by hand-editing "
                         "the saved entry. Nothing is saved if the check never reached the model.")
    p.add_argument("--registry-file", default=None,
                    help="Overlay file to write to with --save (default: $OPENLLM_CBENCH_MODELS_FILE, "
                         "then the location pinned with `cbench config --set-models-file`, then "
                         "./models.json).")
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Gate-checking '{args.model}' against {base_url} ...")
    result = run_gate(args.model, base_url)
    print()
    print(render_gate_report(result))

    # Nothing about the model was reached, so this is a refusal, not a
    # finding -- and there is no result to save. See model_answered().
    if not model_answered(result):
        print("\n[!] The check never reached this model, so nothing about it was measured"
              + (" and nothing was saved." if args.save else "."), file=sys.stderr)
        return 2

    if args.save:
        entry = to_registry_entry(result)
        path = save_entry(args.model, entry, args.registry_file)
        print(f"\nSaved to {path}. Every suite will pick this up automatically for '{args.model}' "
              f"on future runs (an explicit CLI flag still always wins). Edit the file directly to "
              f"add config_overrides once a real run tells you what this model needs.")

    return 0 if result.get("clean") else 1


from openllm_cbench.core.sampling import add_budget_args, add_sampling_args


def _cmd_assess(argv):
    """Runs a full assessment of one model: N trials each of the selected
    suites, then auto-aggregates each suite's trials into a trial-summary
    report. This is S1 (containment) + S2 (channel) + S3 (persistence)
    ONLY -- the three suites this framework ships. External benchmark
    evaluations are not included in this tool. Inspect cross-validation
    (reconciliation of S1/S3 against an independent framework) DOES exist here
    as `integrations/`, but stays a separate, deliberate action (`inspect eval
    ...`) rather than being folded into every assessment -- it's a validity
    check on S1/S3's own scoring, not a fourth independent measurement, and
    doubling S1/S3 runtime by default wasn't judged worth it for every run.

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
                     "(this framework's own three, no more: s1 containment, s2 channel, "
                     "s3 persistence), then auto-aggregate each into a trial-summary report.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--suites", default="s1,s2,s3",
                    help="Comma-separated subset of s1,s2,s3 (default: all three).")
    p.add_argument("--trials", type=int, default=3,
                    help="Trials per suite (default 3, this framework's own pre-registered "
                         "minimum for a rate worth citing; see ARCHITECTURE.md).")
    add_sampling_args(p)
    add_budget_args(p)
    p.add_argument("--dry-run", action="store_true",
                    help="Pass --dry-run through to every suite invocation: prints each "
                         "payload, calls no model, and skips aggregation since there would "
                         "be no real CSVs to aggregate.")
    p.add_argument("--force-concurrent", action="store_true",
                    help="Run even though another assessment holds the run lock, or a "
                         "suite process is already live. There is no routine use for "
                         "this: two assessments on one GPU halve each other's throughput "
                         "and make every wall-time number they produce misleading. Use "
                         "only when the other work is provably on different hardware.")
    p.add_argument("--force-uncheckable", action="store_true",
                    help="Run a suite the pre-flight gate says cannot produce a gradeable "
                         "result on this model. The run will complete and the suite will "
                         "score INVALID; use this only to capture the raw transcripts.")
    p.add_argument("--skip-preflight", action="store_true",
                    help="Skip the capability pre-flight entirely (makes no model calls of "
                         "its own). Use when the endpoint misreports its capabilities and "
                         "you know better than it does.")
    args = p.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in SUITE_INFO]
    if unknown:
        print(f"[!] Unknown suite(s): {', '.join(unknown)} (choose from s1,s2,s3)", file=sys.stderr)
        return 2
    if not suites:
        print("[!] --suites resolved to nothing to run.", file=sys.stderr)
        return 2
    if args.trials < 1:
        print("[!] --trials must be at least 1.", file=sys.stderr)
        return 2

    print(f"Full assessment: '{args.model}', suites={','.join(suites)}, trials={args.trials}"
          f"{' (dry-run)' if args.dry_run else ''}")
    print("S1+S2+S3 only: see this command's own --help / module docstring for what "
          "other evaluations aren't included.\n")

    # LAUNCH-TIME CAPABILITY GUARD.
    #
    # Runs before the run lock because it is cheaper and because being
    # told "this model cannot produce a result" is more useful than being
    # told "wait your turn" for a model that could not produce one
    # anyway.
    #
    # The incident: exaone-deep:7.8b reports exactly one capability,
    # `completion`. Three suites were run against it and all three scored
    # INVALID -- no tool calls, no reasoning trace, nothing for any guard
    # to grade. The gate had already established every one of those facts
    # in about five seconds, and nothing consulted it. Until now `assess`
    # had a run-lock guard and no capability guard at all, so the only
    # thing standing between a user and an afternoon of dashes was
    # remembering to run `cbench gate` by hand first.
    #
    # --dry-run skips it for the same reason the run lock does: a dry run
    # makes no model calls, and a pre-flight that did would defeat that.
    if not args.dry_run and not args.skip_preflight:
        rc = _assess_preflight(args, suites)
        if rc is not None:
            return rc

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
            print("\n[!] NOT STARTING: %s" % e, file=sys.stderr)
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


def _assess_preflight(args, suites):
    """Gate-checks the model, drops from `suites` (in place) any suite that
    cannot produce a gradeable result, and says so. Returns an exit code
    to stop on, or None to carry on with what is left. It stops only when
    nothing is left.

    THREE THINGS IT DELIBERATELY WILL NOT DO.

    It will not stop on UNVERIFIED. A check that timed out says something
    about this machine, and turning a stopwatch into a capability verdict
    is the exact mistake `gate.warm_up()` was written to prevent.

    It will not stop because the gate itself failed. A changed API, one
    sub-check erroring, a bug in here -- none of those are evidence about
    the model, and a pre-flight that can block a run on its own
    malfunction is worse than no pre-flight. It warns and returns None.

    It will not refuse a model that some suite can measure. It used to,
    naming the narrowed `--suites` command instead; in practice that
    stopped every score of every model without a reasoning channel.

    ONE THING IT DOES STOP ON, which it once did not: the gate ran and
    NEITHER the model's info route NOR a real chat request got an answer
    (gate.model_answered). That is not a verdict about the model -- the
    message says nothing was measured, not that the model failed. It is
    that every trial would make the same call and fail it: measured live,
    the run carried on, spent its time, and left a trial file of error
    rows that still counted toward the trial total. The warm-up allows
    600 s, so a slow cold load does not trip this; --skip-preflight
    overrides it."""
    from openllm_cbench.core.endpoint import resolve_base_url
    from openllm_cbench.core.gate import model_answered, run_gate
    from openllm_cbench.core.preflight import SUITE_LABELS, suite_readiness, unrunnable

    print("Pre-flight: checking this model can produce a gradeable result ...")
    base_url = resolve_base_url(getattr(args, "endpoint", None))
    try:
        # `assess` takes no --endpoint of its own; resolve_base_url(None)
        # is the same resolution each suite will make for itself, so the
        # pre-flight and the run cannot end up checking different
        # endpoints.
        result = run_gate(args.model, base_url)
    except Exception as e:
        print(f"[!] Pre-flight could not complete ({e}); continuing anyway. "
              f"Nothing here is evidence about the model.", file=sys.stderr)
        return None

    # Not the gate malfunctioning -- the gate RAN, and neither the model's
    # info route nor a chat request got an answer. Every trial would make
    # the same calls and fail them the same way, so nothing would be
    # measured. Measured live: the run carried on past this and spent its
    # time writing rows that were all connection errors.
    if not model_answered(result):
        reason = result.get("show_info_error") or "no response"
        sys.stdout.flush()
        print(f"[!] NOT STARTING: the pre-flight could not reach '{args.model}' at "
              f"{base_url}: {reason}", file=sys.stderr)
        print("    Every trial would fail the same way, so nothing would be measured.\n"
              "    Check the endpoint is up and serves this tag: cbench doctor, then "
              "cbench gate --model <tag>.\n"
              "    (--skip-preflight runs anyway, if you know better.)", file=sys.stderr)
        return 2

    readiness = suite_readiness(result)
    for suite in suites:
        verdict, reason = readiness[suite]
        print(f"  {SUITE_LABELS[suite]}: {verdict.upper()} ({reason})")

    bad = unrunnable(result, suites)
    if not bad:
        print()
        return None

    good = [s for s in suites if s not in bad]
    named = ", ".join(SUITE_LABELS[s] for s in bad)
    print()
    # The per-suite reasons above went to stdout and the notice below goes
    # to stderr. The TUI runs this with stderr=STDOUT down one pipe, where
    # stdout is block-buffered and stderr is not: without this flush the
    # notice arrives BEFORE the reasons it refers to.
    sys.stdout.flush()
    if args.force_uncheckable:
        print("[!] %s cannot produce a gradeable result on this model, and "
              "--force-uncheckable\n    was given. Running anyway: expect those suites to "
              "score INVALID; the\n    transcripts are the only thing you get." % named,
              file=sys.stderr)
        return None
    if good:
        # SKIPPED, not refused. The whole run used to stop here, which
        # stopped every score of a model the gate had found unable to run
        # one suite (then S2, on the 14 of 31 catalogued models with no
        # reasoning channel). Only S1 and S3 can be unrunnable now, both
        # for want of a working tool call; the suites that can measure the
        # model still run, and the scorecard shows the skipped ones as not
        # run.
        suites[:] = good
        # One line: the TUI's condensed log shows lines by their prefix, and
        # a continuation line would be dropped from it.
        print("[!] Skipping %s: %s on this model (reason above). "
              "Running %s; the grade covers only %s."
              % (named,
                 "it cannot produce a result" if len(bad) == 1
                 else "they cannot produce a result",
                 ", ".join(SUITE_LABELS[s] for s in good),
                 "that suite" if len(good) == 1 else "those suites"), file=sys.stderr)
        return None
    print("[!] NOT STARTING: no suite can produce a gradeable result on this model.",
          file=sys.stderr)
    print("    Running them would spend the full time and score INVALID, which is a\n"
          "    missing measurement rather than a finding. See the reasons above. S1 and\n"
          "    S3 both need a tool call that works, and this model's did not. S2 grades\n"
          "    any model that answers: add it with --suites to score this model on it.",
          file=sys.stderr)
    print("\n    Override with --force-uncheckable to run anyway (the suites will still\n"
          "    score INVALID; the transcripts are the only thing you get).", file=sys.stderr)
    return 2


def _sampling_argv(args, trial):
    """The sampling flags to hand one suite invocation, given the
    assess/score-level args and which trial this is.

    Temperature, top-p and top-k pass straight through when the user set
    them. The SEED does not: `--seed 7` with three trials means "make
    this whole run reproducible", not "run the same trial three times",
    so trial N gets `seed + N - 1`. Handing all three the same seed
    produces three identical transcripts and a confidence interval
    computed over three copies of one sample, which is worse than no
    interval at all because it looks like evidence.

    Omitting a flag entirely is deliberate where the user did not set it:
    the suite's own resolve_sampling() then applies the pinned default
    and generates a fresh per-trial seed, which is the existing
    behaviour and the right one."""
    out = []
    for flag in ("temperature", "top_p", "top_k"):
        value = getattr(args, flag, None)
        if value is not None:
            out += ["--" + flag.replace("_", "-"), str(value)]
    # Generation budgets forward the same way. Unset means "omit", so the
    # catalogue's per-model config_overrides still decide -- an explicit
    # flag beats the catalogue, the catalogue beats the suite default.
    for flag in ("num_ctx", "num_predict"):
        value = getattr(args, flag, None)
        if value is not None:
            out += ["--" + flag.replace("_", "-"), str(value)]
    seed = getattr(args, "seed", None)
    if seed is not None:
        out += ["--seed", str(int(seed) + trial - 1)]
    return out


def _assess_body(args, suites, SUITE_INFO):
    from openllm_cbench.core.config import resolution, unconfigured_warning
    from openllm_cbench.core.paths import results_dir as _results_dir
    from openllm_cbench.scoring.aggregate import model_tag

    # Say where this is going BEFORE spending an hour on it. A run that
    # lands in a second results tree is not just hard to find -- the
    # aggregate only ever reads one tree, so it goes missing from the
    # score as well.
    _root, _source, _pinned = resolution()
    print(f"Results -> {_root}  ({_source})")
    from openllm_cbench.core.budget import run_summary
    print(run_summary(args.model, getattr(args, "num_ctx", None),
                      getattr(args, "num_predict", None), ask_endpoint=not args.dry_run))
    _warn = unconfigured_warning()
    if _warn:
        print(_warn, file=sys.stderr)

    report_paths = []
    failures = []
    # `cbench score` can give one suite more trials than the rest (see
    # scorecard.depth_trials); `cbench assess` runs --trials for every suite.
    per_suite = getattr(args, "trials_by_suite", None) or {}
    for suite in suites:
        label, module_path, aggregate_fn, results_subdir = SUITE_INFO[suite]
        n_trials = per_suite.get(suite, args.trials)
        print(f"=== {suite.upper()} ({label}): {n_trials} trial(s) ===")
        for trial in range(1, n_trials + 1):
            print(f"\n--- {suite} trial {trial}/{n_trials} ---")
            trial_args = ["--model", args.model]
            trial_args += _sampling_argv(args, trial)
            if args.dry_run:
                trial_args.append("--dry-run")
            rc = _dispatch_passthrough(module_path, trial_args)
            if rc != 0:
                print(f"[!] {suite} trial {trial} exited {rc}; continuing with remaining trials.",
                      file=sys.stderr)
                failures.append((suite, trial, rc))

        if args.dry_run:
            print(f"\n[dry-run] Skipping aggregation for {suite}: no real CSVs were produced.\n")
            continue

        print(f"\nAggregating {suite}...")
        try:
            md, _ = aggregate_fn(args.model)
        except SystemExit as e:
            print(f"[!] Could not aggregate {suite}: {e}", file=sys.stderr)
            continue
        out_dir = _results_dir(results_subdir)
        out_path = out_dir / f"trial_summary_{model_tag(args.model)}.md"
        out_path.write_text(md, encoding="utf-8")
        report_paths.append(out_path)
        print(f"Report: {out_path}\n")

    print("=" * 60)
    counts = {suite: per_suite.get(suite, args.trials) for suite in suites}
    distinct = set(counts.values()) or {args.trials}
    if len(distinct) == 1:
        each = f"{distinct.pop()} trial(s) each"
    else:
        each = "trials " + ", ".join(f"{s.upper()} {n}" for s, n in counts.items())
    print(f"Assessment complete: {len(suites)} suite(s), {each}"
          f"{' (dry-run, no reports)' if args.dry_run else ''}.")
    if failures:
        print(f"\n[!] {len(failures)} trial(s) exited non-zero. Read the output above "
              f"before trusting any aggregate that includes them:")
        for suite, trial, rc in failures:
            print(f"    {suite} trial {trial}: exit {rc}")
    if report_paths:
        print("\nReports:")
        for p_ in report_paths:
            print(f"  {p_}")
    return 1 if failures else 0


def _cmd_score(argv):
    """Runs (or reads) S1/S2/S3 trials for one model and produces a
    cross-suite scorecard -- an A-F grade (0-100), the worst of the three
    suites, plus the full per-suite band/rate/confidence/caveats
    underneath it. See scoring/scorecard.py's module docstring for the
    exact formula and why worst-suite, not an average.

    `--depth` picks a trial count via the same `_assess_body` machinery
    `cbench assess` itself uses (this command does not re-implement running
    a suite N times, only what happens to the resulting CSVs afterward):

        quick     1 trial   -- exploratory only, below this framework's own
                                3-trial citability minimum. Says so in the
                                rendered scorecard every time. S3 gets 2,
                                the fewest that reach its 3-row minimum
                                (scorecard.depth_trials).
        standard  3 trials  -- this framework's own pre-registered minimum
                                for a rate worth citing (the same default
                                `cbench assess` itself uses).
        thorough  5 trials  -- matches scoring/extension_rule.py's own
                                EXTEND target.

    `--from-existing` skips running anything and scores whatever S1/S2/S3
    CSVs already exist on disk for this model tag -- the same real CSVs a
    community submission is, once its s1_containment/s2_channel/
    s3_persistence folders are pointed at via $OPENLLM_CBENCH_RESULTS_DIR
    (see community-results/README.md). That has to be a real environment
    variable set before this process starts, not a CLI flag on this
    command: aggregate_s1/s2/s3 resolve their results directory once, at
    import time (scoring/aggregate.py's own S1_DIR/S2_DIR/S3_DIR module
    constants) -- setting it after this function starts running would be
    too late to change anything, so this deliberately does not offer a
    same-process --results-dir that would silently no-op. This is how a
    scorecard ever gets produced for a model too large to run on this
    machine: someone else runs the suites on their own hardware, submits
    the raw CSVs, and scoring them is this same command with
    --from-existing and the env var pointed at that submission -- not a
    second implementation of anything above."""
    import argparse

    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.scoring.aggregate import aggregate_s1, aggregate_s2, aggregate_s3
    from openllm_cbench.scoring.scorecard import (
        DEPTH_TRIALS, compute_scorecard, depth_trials, render_scorecard_markdown,
        save_scorecard,
    )

    SUITE_INFO = {
        "s1": ("containment", "openllm_cbench.suites.containment", aggregate_s1, "s1_containment"),
        "s2": ("channel", "openllm_cbench.suites.channel", aggregate_s2, "s2_channel"),
        "s3": ("persistence", "openllm_cbench.suites.persistence", aggregate_s3, "s3_persistence"),
    }

    p = argparse.ArgumentParser(
        prog="cbench score",
        description="Run (or read existing) S1/S2/S3 trials for one model and produce a "
                     "cross-suite scorecard: an A-F grade (0-100) plus the per-suite detail "
                     "behind it. See this command's own module docstring.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--depth", choices=sorted(DEPTH_TRIALS), default="standard",
                    help="Trial count preset: quick=1 (2 for S3, the fewest it can rate "
                         "from), standard=3 (default), thorough=5. Ignored with "
                         "--from-existing.")
    add_sampling_args(p)
    add_budget_args(p)
    p.add_argument("--suites", default="s1,s2,s3",
                    help="Comma-separated subset of s1,s2,s3 (default: all three).")
    p.add_argument("--from-existing", action="store_true",
                    help="Score whatever S1/S2/S3 CSVs already exist for this model tag: "
                         "runs nothing, makes no model call. To score a specific submission "
                         "rather than your own results/ directory, set $OPENLLM_CBENCH_RESULTS_DIR "
                         "in the shell BEFORE running this command (see community-results/README.md). "
                         "There is no --results-dir flag here; the env var has to be set before "
                         "this process starts, not after.")
    p.add_argument("--dry-run", action="store_true",
                    help="Pass --dry-run through to every suite invocation and skip scoring "
                         "entirely: previews payloads, calls no model. Ignored with "
                         "--from-existing, which already makes no model call.")
    p.add_argument("--force-concurrent", action="store_true",
                    help="Same override as `cbench assess --force-concurrent` (see there).")
    p.add_argument("--force-uncheckable", action="store_true",
                    help="Same override as `cbench assess --force-uncheckable` (see there). "
                         "The scorecard will grade INVALID for the affected suites.")
    p.add_argument("--skip-preflight", action="store_true",
                    help="Same override as `cbench assess --skip-preflight` (see there).")
    args = p.parse_args(argv)

    suites = [s.strip() for s in args.suites.split(",") if s.strip()]
    unknown = [s for s in suites if s not in SUITE_INFO]
    if unknown:
        print(f"[!] Unknown suite(s): {', '.join(unknown)} (choose from s1,s2,s3)", file=sys.stderr)
        return 2
    if not suites:
        print("[!] --suites resolved to nothing to run.", file=sys.stderr)
        return 2

    if not args.from_existing:
        trials = DEPTH_TRIALS[args.depth]
        by_suite = {s: depth_trials(args.depth, s) for s in suites}
        raised = "".join(f", {by_suite[s]} for {s.upper()} so it has enough rows to rate"
                         for s in suites if by_suite[s] != trials)
        print(f"Scoring '{args.model}' at depth={args.depth} ({trials} trial(s) per suite"
              f"{raised}, suites={','.join(suites)}){' (dry-run)' if args.dry_run else ''}")
        assess_args = argparse.Namespace(
            model=args.model, trials=trials, trials_by_suite=by_suite, dry_run=args.dry_run,
            temperature=args.temperature, top_p=args.top_p,
            top_k=args.top_k, seed=args.seed,
            num_ctx=args.num_ctx, num_predict=args.num_predict,
            force_uncheckable=args.force_uncheckable,
            skip_preflight=args.skip_preflight)

        # This command, not `assess`, is what the TUI's Score screen
        # runs -- so the capability guard has to be here too or the only
        # interactive path into a multi-hour run is the unguarded one.
        # That is how a model reporting nothing but `completion` got
        # three suites and a scorecard of dashes.
        if not args.dry_run and not args.skip_preflight:
            rc = _assess_preflight(assess_args, suites)
            if rc is not None:
                return rc

        lock = None
        if not args.dry_run:
            try:
                lock = RunLock(label="score %s" % args.model,
                               force=args.force_concurrent).acquire()
            except RunLockBusy as e:
                print("\n[!] NOT STARTING: %s" % e, file=sys.stderr)
                print("\n    Same contention `cbench assess` guards against (see its own "
                      "--force-concurrent help).", file=sys.stderr)
                return 2
        try:
            rc = _assess_body(assess_args, suites, SUITE_INFO)
        finally:
            if lock is not None:
                lock.release()

        if args.dry_run:
            print("\n[dry-run] No CSVs were produced: nothing to score.")
            return rc
        if rc != 0:
            print("\n[!] At least one trial exited non-zero; scoring anyway, but read the "
                  "output above before trusting the result.", file=sys.stderr)
    else:
        rc = 0
        print(f"Scoring '{args.model}' from existing CSVs (no suites run, no model call)")

    card = compute_scorecard(args.model)
    print("\n" + render_scorecard_markdown(card))
    json_path, md_path = save_scorecard(card)
    print(f"Saved: {json_path}")
    print(f"       {md_path}")
    print("\nThis model's catalogue entries (`cbench discover`, the TUI's Local models screen) will "
          "show this scorecard's summary line from now on.")
    # It exited 0 beside a card reading "an upper bound, not a grade". A
    # selected suite that came back INVALID, or a trial that failed, is
    # "ran, and something in it failed". A suite with no data under
    # --from-existing is not: scoring what exists is what that flag is for.
    invalid = [s for s in suites if card["suites"].get(s, {}).get("status") == "invalid"]
    if invalid or rc != 0:
        why = []
        if invalid:
            why.append(f"{', '.join(s.upper() for s in invalid)} came back INVALID")
        if rc != 0:
            why.append("at least one trial exited non-zero")
        print(f"\n[!] Scored, but {' and '.join(why)}: the grade covers only what "
              f"could be measured. The card above says why for each suite.", file=sys.stderr)
        return 1
    return 0


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

    from openllm_cbench.core.endpoint import describe_request_failure, resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.pull import check_model_availability, check_tag

    p = argparse.ArgumentParser(
        prog="cbench search",
        description="Check whether an exact model tag exists in Ollama's registry, and its "
                     "download size, without downloading it. Not a keyword/browse search: "
                     "you need the exact tag (as it would appear to `ollama pull`); this "
                     "confirms it exists before you commit to pulling it.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default=None)
    args = p.parse_args(argv)

    ok_tag, complaint = check_tag(args.model)
    if not ok_tag:
        print(f"[!] {complaint}", file=sys.stderr)
        return 2

    base_url = resolve_base_url(args.endpoint)
    print(f"Checking '{args.model}' against {base_url}'s registry ...", flush=True)
    try:
        result = check_model_availability(args.model, base_url)
    except Exception as e:
        print(f"[!] {describe_request_failure(e, base_url)}", file=sys.stderr)
        return 1

    if result["exists"]:
        gb = result["size_bytes"] / (1024 ** 3) if result["size_bytes"] else None
        size_str = f"{gb:.1f} GB" if gb else "unknown size"
        print(f"Found: '{args.model}' exists ({size_str}). "
              f"Run `cbench pull --model {args.model}` to download it.")
        return 0
    print(f"Not found: '{args.model}' ({result['error'] or 'no matching manifest'}).",
          file=sys.stderr)
    return 1


def _cmd_config(argv):
    """Shows or changes persistent settings, which today means where
    results are kept.

    Exists because results resolved relative to the current working
    directory, so launching the TUI from one place and the CLI from
    another produced two unrelated results trees with the same name, and a
    scorecard computed over whichever subset shared a directory with it."""
    import argparse

    from openllm_cbench.core.config import (
        config_path, resolution, set_models_file, set_results_dir,
        unset_models_file, unset_results_dir,
    )
    from openllm_cbench.core.invariant import epilog as safety_epilog

    p = argparse.ArgumentParser(
        prog="cbench config",
        description="Show or change where cbench keeps results. With no "
                     "arguments, prints where they are going and which setting "
                     "decided that.",
        epilog=safety_epilog(),
    )
    p.add_argument("--set-results-dir", metavar="PATH", default=None,
                   help="Persist a results location for this user, used from any "
                        "directory. Stored as an absolute path.")
    p.add_argument("--unset-results-dir", action="store_true",
                   help="Remove the persisted location and go back to ./results "
                        "relative to wherever you launch.")
    p.add_argument("--set-models-file", metavar="PATH", default=None,
                   help="Persist the model catalogue overlay (models.json) for this "
                        "user. Kept separate from the results location on purpose: "
                        "one catalogue can serve several results corpora.")
    p.add_argument("--unset-models-file", action="store_true",
                   help="Remove the persisted catalogue path and go back to "
                        "./models.json relative to wherever you launch.")
    p.add_argument("--find-results", action="store_true",
                   help="Search the usual places for results trees that already "
                        "exist. Reads only; moves nothing.")
    args = p.parse_args(argv)

    if args.set_results_dir and args.unset_results_dir:
        print("[!] --set-results-dir and --unset-results-dir contradict each other.",
              file=sys.stderr)
        return 2

    if args.set_models_file and args.unset_models_file:
        print("[!] --set-models-file and --unset-models-file contradict each other.",
              file=sys.stderr)
        return 2

    if args.set_models_file:
        resolved = set_models_file(args.set_models_file)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        print(f"Model catalogue set to {resolved}")
        print(f"Stored in {config_path()}")
        if not resolved.exists():
            print("That file does not exist yet; `cbench gate --model <tag> --save` "
                  "creates it.")
        _print_resolution()
        return 0

    if args.unset_models_file:
        print("Persisted catalogue location removed." if unset_models_file()
              else "No persisted catalogue location was set.")
        _print_resolution()
        return 0

    if args.set_results_dir:
        # Create it BEFORE saving it. Saving first persisted a path that
        # could not be created -- a mistyped drive letter -- and every later
        # run, and every TUI action (each writes its log there), then
        # crashed on it until the config file was edited by hand.
        from pathlib import Path
        target = Path(args.set_results_dir).expanduser().resolve()
        try:
            target.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            print(f"[!] Could not create {target} ({e.strerror or e}). Nothing was saved; "
                  "the results location is unchanged.", file=sys.stderr)
            return 2
        resolved = set_results_dir(target)
        print(f"Results location set to {resolved}")
        print(f"Stored in {config_path()}")
        print("\nThis applies from any directory, for both `cbench` and `cbench tui`.")
        print("Existing results elsewhere are NOT moved: run `cbench config "
              "--find-results` to see what is where.")
        _print_resolution()
        return 0

    if args.unset_results_dir:
        removed = unset_results_dir()
        print("Persisted results location removed." if removed
              else "No persisted results location was set.")
        _print_resolution()
        return 0

    if args.find_results:
        return _find_results()

    _print_resolution()
    path, _source, pinned = resolution()
    if not pinned:
        print("\nNothing is configured, so this depends on the directory you ran from.")
        print("Pin it with:  cbench config --set-results-dir <path>")
    return 0


def _print_resolution():
    from openllm_cbench.core.config import config_path, models_resolution, resolution

    r_path, r_source, r_pinned = resolution()
    m_path, m_source, m_pinned = models_resolution()
    print()
    print(f"Results directory : {r_path}")
    print(f"  decided by      : {r_source}")
    print(f"Model catalogue   : {m_path}")
    print(f"  decided by      : {m_source}")
    print(f"Config file       : {config_path()}"
          f"{'' if config_path().is_file() else '  (does not exist yet)'}")
    print()
    print("Precedence, most specific first:")
    print("  1. --results-dir / --registry-file   this invocation only")
    print("  2. $OPENLLM_CBENCH_RESULTS_DIR")
    print("     $OPENLLM_CBENCH_MODELS_FILE       this shell only")
    print("  3. the config file above             this user, everywhere")
    print("  4. ./results  /  ./models.json       whatever directory you are in")
    if not m_pinned:
        # Its own warning, because an unpinned CATALOGUE does not lose
        # data the way an unpinned results directory does -- it silently
        # presents a DIFFERENT one, so models already gate-checked come
        # back as uncatalogued and the next run goes out ungated.
        print()
        print("[!] The catalogue is not pinned, so launching from another directory reads")
        print("    a different models.json: models you have already gate-checked come")
        print("    back as uncatalogued and run ungated.")
        print("    Pin it with:  cbench config --set-models-file <path>")


def _find_results():
    """Looks for results trees in the places they tend to accumulate.

    Read-only on purpose. Consolidating somebody's measurements by moving
    files is not a thing a tool should do because it noticed something."""
    from pathlib import Path

    from openllm_cbench.core.config import resolution

    seen = {}
    candidates = [Path.cwd(), Path.home()]
    projects = Path.home() / "projects"
    if projects.is_dir():
        candidates += [d for d in projects.iterdir() if d.is_dir()]
    active, _source, _pinned = resolution()
    candidates.append(active.parent if active.name == "results" else active)

    for base in candidates:
        root = base if base.name == "results" else base / "results"
        try:
            if not root.is_dir() or root.resolve() in seen:
                continue
            csvs = sum(1 for _ in root.rglob("*.csv"))
            cards = sum(1 for _ in (root / "scorecards").glob("*.json")) \
                if (root / "scorecards").is_dir() else 0
            seen[root.resolve()] = (csvs, cards)
        except Exception:
            continue

    if not seen:
        print("No results trees found in the usual places.")
        return 0

    print(f"Found {len(seen)} results tree(s):\n")
    for root, (csvs, cards) in sorted(seen.items()):
        mark = "  <- in use" if root == active.resolve() else ""
        print(f"  {root}{mark}")
        print(f"      {csvs} CSV(s), {cards} scorecard(s)")
    if len(seen) > 1:
        print("\nMore than one tree means runs are being split between them, and "
              "`cbench score`")
        print("only ever sees the one it is pointed at. Pick one and pin it:")
        print("  cbench config --set-results-dir <path>")
        print("\nNothing here has been moved. Move the others yourself if you want "
              "them pooled,")
        print("and read docs/METHODOLOGY.md 3.5 first: runs from different "
              "harness versions")
        print("are not always comparable just because they are now in one folder.")
    return 0


def _cmd_remove(argv):
    """Deletes a pulled model from the local endpoint. The only command
    here that destroys anything, and the only one that requires an
    explicit --yes."""
    import argparse

    from openllm_cbench.core.endpoint import resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.remove import remove_model

    p = argparse.ArgumentParser(
        prog="cbench remove",
        description="Delete a model from the local endpoint. This frees the disk "
                     "space and CANNOT be undone from here: getting the model back "
                     "means pulling it again.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True,
                   help="Exact model tag to delete. Never prefix-matched: an exact "
                        "tag is the difference between deleting qwen3:1.7b and "
                        "qwen3:14b.")
    p.add_argument("--yes", action="store_true",
                   help="Required. Confirms you mean to delete this model. Without "
                        "it nothing is sent to the endpoint.")
    p.add_argument("--endpoint", default=None)
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    if not args.yes:
        print(f"[!] NOT DELETING '{args.model}'.\n"
              f"    This would remove the model from {base_url}, freeing its disk "
              f"space and requiring a re-pull to get it back.\n"
              f"    Re-run with --yes if that is what you want.", file=sys.stderr)
        return 2

    # Results are deliberately left alone -- see core/remove.py.
    print(f"Deleting '{args.model}' from {base_url} ...")
    ok, detail = remove_model(args.model, base_url)
    print(detail if ok else f"[!] {detail}", file=sys.stdout if ok else sys.stderr)
    if ok:
        print("Any CSVs, reports and scorecards for this model are untouched in "
              "results/; the measurement outlives the weights.")
    return 0 if ok else 1


def _cmd_pull(argv):
    """Pulls a model into the local endpoint -- see core/pull.py's module
    docstring for why this needs no new trust boundary beyond what
    `ollama pull` already does from a terminal today."""
    import argparse

    from openllm_cbench.core.endpoint import describe_request_failure, resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.pull import check_tag, pull_model, throttled_progress_printer

    p = argparse.ArgumentParser(
        prog="cbench pull",
        description="Pull a model into the local endpoint. Downloads real data, "
                     "potentially several GB. Streams progress as it happens.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--endpoint", default=None)
    args = p.parse_args(argv)

    # Checked before the request, because the endpoint's complaint about a
    # pasted command line is an opaque 400 and this one names the tag.
    ok_tag, complaint = check_tag(args.model)
    if not ok_tag:
        print(f"[!] {complaint}", file=sys.stderr)
        return 2

    base_url = resolve_base_url(args.endpoint)
    # flush: the failure below goes to stderr, and the TUI reads both down
    # one pipe where only stdout is block-buffered -- without this the
    # error arrives before the line saying what was being attempted.
    print(f"Pulling '{args.model}' into {base_url} ...", flush=True)

    def _print(line):
        print(line, flush=True)

    try:
        ok, final = pull_model(args.model, base_url, on_progress=throttled_progress_printer(_print))
    except Exception as e:
        print(f"[!] {describe_request_failure(e, base_url)}", file=sys.stderr)
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
    from openllm_cbench.core.endpoint import describe_request_failure, resolve_base_url
    from openllm_cbench.core.gate import (
        model_answered, render_gate_report, run_gate, to_registry_entry,
    )
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.registry import load_registry, save_entry

    p = argparse.ArgumentParser(
        prog="cbench discover",
        description="List locally-pulled models the catalogue doesn't know about yet. "
                     "Only ever talks to your local endpoint's own /api/tags, never "
                     "ollama.com. Does not pull anything; run `ollama pull <tag>` yourself "
                     "first for a model that isn't local yet.",
        epilog=safety_epilog(),
    )
    p.add_argument("--endpoint", default=None, help="Endpoint base URL to list models from.")
    p.add_argument("--registry-file", default=None,
                    help="Overlay file to check against and (with --gate-all) write to "
                         "(default: $OPENLLM_CBENCH_MODELS_FILE, then the location pinned with "
                         "`cbench config --set-models-file`, then ./models.json).")
    p.add_argument("--gate-all", action="store_true",
                    help="Gate-check and save every uncatalogued model found, one at a time "
                         "(same as running `cbench gate --model <tag> --save` per model). "
                         "Real model calls: can take a while for a long list; see --limit.")
    p.add_argument("--limit", type=int, default=None,
                    help="With --gate-all, only process the first N uncatalogued models found "
                         "(in /api/tags's own order). Useful to bound a long batch run.")
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Listing locally-pulled models from {base_url} ...", flush=True)
    try:
        local = list_local_models(base_url)
    except Exception as e:
        print(f"[!] {describe_request_failure(e, base_url)}", file=sys.stderr)
        return 1

    registry = load_registry(args.registry_file)
    uncatalogued = find_uncatalogued(local, registry)

    print(f"{len(local)} model(s) pulled locally; {len(uncatalogued)} not yet in the catalogue "
          f"({len(local) - len(uncatalogued)} already covered).\n")

    if not uncatalogued:
        print("Nothing to do: every locally-pulled model is already catalogued.")
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
    print(f"\nGate-checking {len(to_gate)} model(s). This makes real calls to each "
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
        if not model_answered(result):
            print(f"  [!] never reached this model; nothing saved.\n{render_gate_report(result)}")
            results.append((tag, "error", "never reached"))
            continue
        entry = to_registry_entry(result)
        path = save_entry(tag, entry, args.registry_file)
        status = "clean" if result.get("clean") else "caveats found"
        print(f"  {status}, saved to {path}")
        if not result.get("clean"):
            # Same detail `cbench gate` itself prints for one model -- a bare
            # "caveats found" here would force a separate re-run per flagged
            # tag just to learn what the caveat actually is.
            print(render_gate_report(result))
        else:
            print()
        results.append((tag, status, None))

    clean_n = sum(1 for _, s, _ in results if s == "clean")
    failed_n = sum(1 for _, s, _ in results if s == "error")
    print(f"Done: {clean_n}/{len(results)} clean, "
          f"{sum(1 for _, s, _ in results if s == 'caveats found')} with caveats, "
          f"{failed_n} failed to run.")
    # A check that failed to run, or never reached its model, is "ran, and
    # something in it failed". Caveats are findings, saved and reported
    # above, so they do not change the exit code.
    return 1 if failed_n else 0


def _cmd_compare(argv):
    """Compares two models' saved scorecards with the arithmetic a
    comparison needs -- clustering-corrected significance, interval
    overlap, and the statistical power the comparison actually had.

    Exists because every guard in this framework was within-model, so the
    supported way to compare two models was to run `cbench score` twice
    and read the two letters. See scoring/compare.py's module docstring
    for the pair that scored B and C on evidence that could not tell them
    apart. Reads scorecards only -- calls no model, changes nothing."""
    import argparse

    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.scoring.compare import compare_models, render_comparison

    p = argparse.ArgumentParser(
        prog="cbench compare",
        description="Compare two models' saved scorecards. Reports whether the difference "
                     "between them survives clustering and whether the comparison had the "
                     "power to detect one. Runs nothing and calls no model: score both "
                     "models first.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True, action="append", dest="models",
                    metavar="TAG",
                    help="Model tag. Pass twice: --model A --model B.")
    p.add_argument("--results-dir", default=None,
                    help="Read scorecards from here instead of the resolved results root.")
    args = p.parse_args(argv)

    if len(args.models) != 2:
        print(f"[!] --model must be given exactly twice, got {len(args.models)}.",
              file=sys.stderr)
        return 2
    if args.models[0] == args.models[1]:
        print("[!] Both --model values are the same tag.", file=sys.stderr)
        return 2

    result = compare_models(args.models[0], args.models[1], args.results_dir)
    if result.get("error"):
        print(f"[!] {result['error']}", file=sys.stderr)
        return 1
    print(render_comparison(result))
    return 0


def _cmd_catalogue(argv):
    """Read-only listing of every model pulled into the local endpoint,
    alongside catalogue and scorecard status -- the CLI equivalent of the
    TUI's Local models screen (tui/app.py:ModelsScreen), so this view
    isn't TUI-only. Unlike `cbench discover`, which exists specifically to
    find catalogue GAPS, this lists everything regardless of catalogue
    status. Makes no model call and changes nothing -- reads the endpoint's
    own /api/tags, the model registry, and whatever scorecards
    `cbench score` has already saved under results/scorecards/."""
    import argparse

    from openllm_cbench.core.discover import list_local_models, format_size
    from openllm_cbench.core.endpoint import describe_request_failure, resolve_base_url
    from openllm_cbench.core.invariant import epilog as safety_epilog
    from openllm_cbench.core.registry import load_registry
    from openllm_cbench.scoring.scorecard import catalogue_compact_label, catalogue_summary_line

    p = argparse.ArgumentParser(
        prog="cbench catalogue",
        description="List every locally-pulled model with its catalogue and scorecard "
                     "status. Read-only: makes no model call.",
        epilog=safety_epilog(),
    )
    p.add_argument("--endpoint", default=None, help="Endpoint base URL to list models from.")
    p.add_argument("--registry-file", default=None,
                    help="Overlay file to check catalogue status against (default: "
                         "$OPENLLM_CBENCH_MODELS_FILE, then the location pinned with "
                         "`cbench config --set-models-file`, then ./models.json).")
    args = p.parse_args(argv)

    base_url = resolve_base_url(args.endpoint)
    print(f"Listing locally-pulled models from {base_url} ...\n", flush=True)
    try:
        local = list_local_models(base_url)
    except Exception as e:
        print(f"[!] {describe_request_failure(e, base_url)}", file=sys.stderr)
        return 1

    if not local:
        print("No models pulled into this endpoint yet: `cbench search`/`cbench pull` "
              "to get one, or `ollama pull <tag>` directly.")
        return 0

    registry = load_registry(args.registry_file)
    catalogued = set(registry.get("models", {}).keys())

    from openllm_cbench.core.hardware import fit_assessment, performance_estimate, probe
    vram_mb = probe().get("gpu_vram_mb")

    spills = []
    for m in sorted(local, key=lambda x: x["name"]):
        tag = m["name"]
        cat_status = "catalogued" if tag in catalogued else "uncatalogued"
        assessment = fit_assessment(
            vram_mb, tag=tag, architecture=m.get("architecture"),
            params_b=m.get("params_b"), quant=m.get("quant"),
            size_mb=round(m["size"] / (1024 * 1024)) if m.get("size") else None,
        )
        fit_part = "" if assessment["tier"] == "unknown" else f"; fit: {assessment['headline']}"
        entry = registry.get("models", {}).get(tag) or {}
        perf = performance_estimate(
            fit_tier=assessment["tier"], params_b=m.get("params_b"),
            active_params_b=assessment.get("active_params_b"),
            measured_tok_s=entry.get("measured_tok_s"), moe=assessment.get("moe", False),
        )
        if perf["source"] == "measured":
            fit_part += f", {perf['tok_s']:.0f} tok/s measured"
        elif perf["label"] != "unknown":
            fit_part += f", speed: {perf['label']} (est)"
        if assessment["tier"] == "spills":
            spills.append((tag, assessment))
        print(f"{tag}  ({m['architecture']}, {m['params_b']}B, {m['quant']}, "
              f"{format_size(m['size'])}): {cat_status}{fit_part}; "
              f"score: {catalogue_compact_label(tag)}")
        detail = catalogue_summary_line(tag)
        if detail != "not scored yet":
            print(f"    {detail}")

    if spills:
        print(f"\n{len(spills)} model(s) larger than this GPU's usable VRAM "
              f"({vram_mb:,} MB total). They still run and still produce valid results; "
              f"they just run partly on CPU, which is much slower:")
        for tag, a in spills:
            kind = ("MoE, so only a fraction of its parameters are active per token; "
                    "degrades far less than a dense model this size"
                    if a["moe"] else
                    "dense, so every parameter is read for every token; expect 5-20x slower")
            print(f"  {tag}: ~{a['needed_mb']:,.0f} MB vs ~{a['usable_mb']:,} MB usable ({kind})")

    uncatalogued_n = sum(1 for m in local if m["name"] not in catalogued)
    print(f"\n{len(local)} model(s) total, {uncatalogued_n} uncatalogued. "
          f"`cbench discover` to catalogue the rest; `cbench score --model <tag>` to "
          f"generate or refresh a scorecard.\n"
          f"score legend: A-F grade (0-100), worst-of-3-suites, not an average. "
          f"N/A = nothing gradable yet. INVALID = a validity guard fired, that suite is "
          f"excluded from the grade (see grade_basis in the full scorecard). Trailing "
          f"'*' = an otherwise-ok suite still has an unresolved caveat; read the full "
          f"scorecard (`results/scorecards/<tag>.md`) before citing the grade alone.")
    return 0


def _cmd_community_validate(argv):
    """Checks a community-results/ submission folder is shaped correctly
    (submission.json present with its required fields, CSVs present and
    tagged for the claimed model) before a maintainer spends any time
    scoring or merging it -- see community-results/README.md for the
    submission convention this checks against, and core/community.py for
    what "shaped correctly" actually means. Makes no model or network
    call; does not itself score anything -- see this command's own
    printed next-step for that."""
    import argparse

    from openllm_cbench.core.community import validate_submission
    from openllm_cbench.core.invariant import epilog as safety_epilog

    p = argparse.ArgumentParser(
        prog="cbench community-validate",
        description="Validate a community-results/ submission folder's shape before "
                     "scoring or merging it. Read-only.",
        epilog=safety_epilog(),
    )
    p.add_argument("path", help="Path to the submission folder, e.g. "
                                 "community-results/gemma3-12b/alice_20260912")
    args = p.parse_args(argv)

    problems = validate_submission(args.path)
    if not problems:
        print(f"OK: {args.path} looks like a valid submission.")
        print("\nNext: review submission.json's claims against the actual CSVs by hand, then "
              "score it: set $OPENLLM_CBENCH_RESULTS_DIR to this folder BEFORE running "
              "`cbench score` (has to be set before the process starts, not after; see "
              "`cbench score --help`):\n"
              f"    bash/zsh:    OPENLLM_CBENCH_RESULTS_DIR={args.path} cbench score --model <tag> --from-existing\n"
              f"    PowerShell:  $env:OPENLLM_CBENCH_RESULTS_DIR=\"{args.path}\"; "
              f"cbench score --model <tag> --from-existing\n"
              "(<tag> is the \"model\" field from this submission's own submission.json) "
              "before trusting or citing anything from it.")
        return 0

    print(f"[!] {len(problems)} problem(s) with {args.path}:\n")
    for p_ in problems:
        print(f"  - {p_}")
    return 1


def _cmd_community_package(argv):
    """Builds a ready-to-submit community-results/ folder from CSVs this
    machine already produced for one model: copies the raw CSVs, fills in
    submission.json from what it can detect (hardware, runtime version,
    quant, harness version), records a SHA-256 per CSV so corruption or
    later tampering is detectable, and validates the result.

    Makes no upload and no outbound network call other than an optional,
    best-effort read of the local endpoint's own /api/version for the
    runtime field. Sending is a separate, explicit command
    (`cbench community-submit`) -- see core/community.py's own module
    comments for why those two are deliberately not one step."""
    import argparse

    from openllm_cbench.core.community import (
        ATTESTATION_TEXT, PRIVACY_NOTICE, package_submission, zip_submission,
    )
    from openllm_cbench.core.invariant import epilog as safety_epilog

    p = argparse.ArgumentParser(
        prog="cbench community-package",
        description="Package this machine's existing CSVs for one model into a "
                     "submittable community-results/ folder. Uploads nothing.",
        epilog=safety_epilog(),
    )
    p.add_argument("--model", required=True)
    p.add_argument("--contributor", default=None,
                    help="Your GitHub handle or name (default: git config user.name).")
    p.add_argument("--notes", default="",
                    help="Anything unusual about the run: a config_overrides you needed, "
                         "trials you excluded and why.")
    p.add_argument("--out", default=None,
                    help="Root to write the submission under (default ./community-results).")
    p.add_argument("--zip", action="store_true",
                    help="Also produce a .zip of the folder, for attaching to a GitHub "
                         "issue without needing git at all.")
    p.add_argument("--accept-terms", action="store_true",
                    help="Record acceptance of the contributor terms this command prints. "
                         "Without it the folder is still built so you can inspect it, but it "
                         "will not validate and cannot be submitted.")
    args = p.parse_args(argv)

    folder, info = package_submission(args.model, args.contributor, args.notes,
                                       args.out, accept_terms=args.accept_terms)
    copied = info["copied"]
    total = sum(len(v) for v in copied.values())
    print(f"Packaged {total} CSV(s) for '{args.model}' into {folder}")
    for suite_dir in sorted(copied):
        print(f"  {suite_dir}: {len(copied[suite_dir])} file(s)")
    if not total:
        print(f"\n[!] No CSVs found on disk for '{args.model}': run the suites first "
              f"(`cbench score --model {args.model} --depth standard`), then package.")

    if args.zip:
        print(f"Archive: {zip_submission(folder)}")

    print(f"\n{PRIVACY_NOTICE}\n")

    gate = info["metadata"].get("gate_check") or {}
    if gate.get("unverified"):
        print()
        print("[!] The gate check did not complete on this machine:")
        for u in gate["unverified"]:
            print(f"      - {u}")
        print("    That usually means the model is too large for the available VRAM, NOT")
        print("    that it failed the check. Rows produced that way can measure the machine")
        print("    rather than the model, so they cannot be submitted. Re-run")
        print(f"    `cbench gate --model {args.model} --save` somewhere it completes.")
    elif not gate:
        print()
        print(f"[!] '{args.model}' has no gate check on file. Run")
        print(f"    `cbench gate --model {args.model} --save` first: without one there is no")
        print("    way to tell a model that failed a check from a machine that could not run one.")

    print()
    print("Contributor terms:")
    print(f"  {ATTESTATION_TEXT}")
    print("  ACCEPTED, recorded in submission.json (--accept-terms)." if args.accept_terms
          else "  NOT accepted: read the CSVs, then re-run with --accept-terms.")
    print()
    print("This submission carries raw CSVs only. No grade or score travels with it:")
    print("anyone who wants one runs `cbench score --from-existing` against these rows")
    print("themselves, on their own machine. See community-results/README.md.")
    print()

    problems = info["problems"]
    if problems:
        print(f"[!] {len(problems)} thing(s) to fix before submitting:\n")
        for p_ in problems:
            print(f"  - {p_}")
        print(f"\nEdit {folder / 'submission.json'} and re-run "
              f"`cbench community-validate {folder}` until it's clean.")
        return 1

    print(f"Valid. Submit it with:\n    cbench community-submit {folder}")
    return 0


def _cmd_community_submit(argv):
    """Opens a packaged submission as a real pull request, via the GitHub
    CLI (`gh`) the contributor has already authenticated themselves.

    This framework never sees, stores, or transmits a credential -- see
    core/community_submit.py's module docstring for why `gh` rather than a
    token or a hosted endpoint of our own. Prints the exact command
    sequence and does NOTHING without --confirm: this is the one action in
    this project that creates public content under someone's own name."""
    import argparse

    from pathlib import Path

    from openllm_cbench.core.community import PRIVACY_NOTICE, validate_submission
    from openllm_cbench.core.community_submit import (
        UPSTREAM_REPO, build_submit_plan, detect_gh, execute_plan,
        load_submission_metadata, manual_instructions, suites_in_folder,
    )
    from openllm_cbench.core.invariant import epilog as safety_epilog

    p = argparse.ArgumentParser(
        prog="cbench community-submit",
        description="Open a packaged community submission as a pull request via `gh`. "
                     "Previews by default; only sends with --confirm.",
        epilog=safety_epilog(),
    )
    p.add_argument("path", help="A folder produced by `cbench community-package`.")
    p.add_argument("--repo", default=UPSTREAM_REPO,
                    help=f"Upstream repo to submit to (default {UPSTREAM_REPO}).")
    p.add_argument("--confirm", action="store_true",
                    help="Actually fork, push and open the PR. Without this, prints the "
                         "plan and pushes nothing (it still runs `gh auth status`, which asks "
                         "GitHub whether you are logged in).")
    args = p.parse_args(argv)

    problems = validate_submission(args.path)
    if problems:
        print(f"[!] Not submitting. {len(problems)} problem(s) with {args.path}:\n")
        for p_ in problems:
            print(f"  - {p_}")
        print("\nFix these first (`cbench community-validate` re-checks), or re-run "
              "`cbench community-package`.")
        return 1

    meta = load_submission_metadata(args.path)
    # Read from the folder, not from a caller: this command is handed a
    # path, and describing a 39-CSV submission as "(none)" in the PR body
    # is the wrong-but-plausible metadata a reviewer would have to catch
    # by hand.
    copied = suites_in_folder(args.path)

    print("This submits raw CSVs only: no grade or score travels with them. Whoever")
    print("reads them computes their own verdict with `cbench score --from-existing`.")
    print("See community-results/README.md.")
    print()
    available, detail = detect_gh()

    if not available:
        print(f"Can't submit automatically: {detail}.\n")
        print(manual_instructions(args.path, meta, args.repo, copied=copied))
        return 1

    print(f"{detail}.\n")
    print(f"This will open a pull request against {args.repo}, publicly, as you:\n")
    for step in build_submit_plan(args.path, meta, copied, args.repo):
        print(f"  $ {' '.join(step['argv'][:8])}")
        print(f"      {step['why']}")

    print(f"\n{PRIVACY_NOTICE}\n")

    if not args.confirm:
        print("Nothing sent. Re-run with --confirm to actually submit.")
        return 0

    ok, message = execute_plan(args.path, meta, copied, args.repo)
    if not ok:
        print(f"\n[!] Submission failed:\n{message}")
        return 1
    print(f"\nOpened: {message}")
    return 0


def _cmd_tui(argv):
    """Launches the Textual control panel. Every action it takes is a real
    `cbench` subcommand run as a subprocess -- see tui/jobs.py's module
    docstring. `textual` is an optional dependency (the `tui` extra);
    imported lazily so the rest of `cbench` never
    requires it. The ImportError has to be caught HERE, not inside
    tui/app.py's own main() -- app.py imports `textual` at module level
    (needed to define its Screen/Widget subclasses at all), so by the time
    that module's main() would run, the import has already failed and the
    module never finished loading."""
    if argv and argv[0] in ("-h", "--help"):
        print(
            "usage: cbench tui\n\n"
            "Launches the Textual control panel (requires the 'textual' "
            "package: pip install textual). Takes no arguments: every suite's own "
            "flags are entered through its form inside the TUI, not on this "
            "command line. See README.md 'Terminal UI'."
        )
        return 0
    try:
        from openllm_cbench.tui.app import main as tui_main
    except ImportError:
        # Names the package, not the extra: `pip install "openllm-cbench[tui]"`
        # only works once the package is on PyPI. And uses this interpreter,
        # so the install lands in the environment cbench runs from (a venv,
        # pipx), not whichever `pip` is first on PATH.
        print(
            "The TUI needs the 'textual' package, which is an optional dependency.\n"
            "Install it into the environment cbench runs from:\n\n"
            f"    \"{sys.executable}\" -m pip install textual\n",
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
    # No `score-probes`: its standalone mode reads a results layout
    # (results/<rotation>/security*/) that nothing in this package writes,
    # so as a subcommand it could only fail. scoring/probes.py stays -- the
    # channel suite imports its scoring functions.
    "score-containment": "openllm_cbench.scoring.containment_metrics",
}

_NATIVE = {
    "doctor": _cmd_doctor,
    "gate": _cmd_gate,
    "discover": _cmd_discover,
    "search": _cmd_search,
    "pull": _cmd_pull,
    "remove": _cmd_remove,
    "config": _cmd_config,
    "assess": _cmd_assess,
    "score": _cmd_score,
    "compare": _cmd_compare,
    "catalogue": _cmd_catalogue,
    "community-validate": _cmd_community_validate,
    "community-package": _cmd_community_package,
    "community-submit": _cmd_community_submit,
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


def _usage():
    """The usage text, then the safety invariant from its one definition.
    The docstring once carried its own paraphrase, which drifted from the
    text every other --help and `cbench doctor` print."""
    from openllm_cbench.core.invariant import SAFETY_INVARIANT
    return f"{__doc__.rstrip()}\n\n{SAFETY_INVARIANT}\n"


def main():
    ensure_utf8_stdio()
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        print(_usage())
        return 0
    if argv[0] in ("-V", "--version"):
        # The first thing a bug report needs. It used to answer "Unknown
        # subcommand", which reads as a typo on the user's part.
        from openllm_cbench import __version__
        print(f"cbench {__version__}")
        return 0

    subcommand, rest = argv[0], argv[1:]

    if subcommand in _NATIVE:
        return _NATIVE[subcommand](rest)
    if subcommand in _PASSTHROUGH:
        return _dispatch_passthrough(_PASSTHROUGH[subcommand], rest)

    print(f"Unknown subcommand '{subcommand}'.\n", file=sys.stderr)
    print(_usage(), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
