# Contributing

Thanks for considering a contribution. This document covers what the
project is, the one rule that isn't negotiable, how the code is laid out,
how to add to it, and the conventions worth keeping. It's tool-agnostic —
nothing here assumes any particular editor, AI assistant, or IDE.

## What this project is

`openllm-cbench` is a local testing framework for three specific
questions about open-weight LLMs: agentic tool-use containment (does a
model reach beyond the boundary it's given?), reasoning-channel divergence
(does a hidden reasoning trace fail a probe the visible answer passes?),
and deceptive persistence (does a model try to rewrite a challenged log
entry rather than correct it honestly?). See [ARCHITECTURE.md](ARCHITECTURE.md)
for the full picture.

## The non-negotiable invariant

**Every suite measures attempt, never success.** No contribution may add
a code path where a model's tool call can reach a real destination other
than the configured chat endpoint or a suite's own loopback canary, and
`run_shell` must never execute anything, under any flag, for any task set.

If a change touches `core/canary.py`, `core/sandbox.py`, or any suite's
`execute_tool()`-equivalent, run `tests/test_safety_invariant.py` before
and after and say so explicitly in the PR description. This test requires
no model and no network — there's no excuse for skipping it.

## Layout

```
src/openllm_cbench/
  cli.py              `cbench` entry point — thin passthrough to each
                       module's own main(), plus native subcommands
                       (doctor, gate, discover, search, pull, remove,
                       assess, score, catalogue, community-validate,
                       community-package, community-submit, tui)
  core/                shared primitives, no suite-specific logic
    canary.py           loopback HTTP listener + bind assertion
    sandbox.py           in-memory fabricated sandbox contents
    endpoint.py         chat endpoint URL resolution (flag > env > default)
    paths.py            packaged-data + results-directory resolution
    provenance.py       third-party-model labelling rule
    registry.py          model catalogue: seed + local overlay, config
                         resolution, thinking-mode selection
    gate.py               model gate-check (capabilities, tool call,
                         channel separation at both think states,
                         sampling params) + catalogue-entry conversion;
                         summarize_gate_output() reads that report back
                         out of captured output for a caller watching
                         gate as a subprocess
    discover.py            lists locally-pulled models missing from the
                         catalogue, via the endpoint's own /api/tags --
                         never talks to ollama.com (see its own
                         docstring for why)
    pull.py                 exact-tag existence check (`cbench search`)
                         and model download (`cbench pull`), both via
                         the local endpoint's own /api/pull route
    hardware.py           advisory VRAM/RAM probe, never blocking --
                         probe()/recommend_band() for `cbench doctor`,
                         check_model_fit() for a single model's fit
                         against detected VRAM
    runlock.py             launch-time exclusivity guard: refuses to
                         start a second assessment while one holds the
                         lock (`--force-concurrent` overrides)
    console.py             UTF-8 stdio setup, applied once at entry
    community.py           shape-checks a community-results/ submission
                         folder before it's scored or merged
                         (`validate_submission()`, including per-CSV
                         checksum verification), and builds one from a
                         model's existing local CSVs
                         (`package_submission()`) -- `PRIVACY_NOTICE` here
                         is shown at both packaging and submit time
    community_submit.py    opens a packaged submission as a pull request
                         via the contributor's own authenticated `gh` --
                         see this module's own docstring for why `gh`
                         over a token or a hosted endpoint of our own;
                         prints manual fork-and-PR / prefilled-issue
                         instructions when `gh` is absent or logged out
    invariant.py          the one safety-invariant string, shared by
                         every --help epilog and `cbench doctor`
    sampling.py            pinned sampling parameters (temperature/top_p/
                         top_k/seed) shared by every suite --
                         add_sampling_args()/resolve_sampling() send the
                         same four values to every model rather than
                         leaving them to each model's own Modelfile, and
                         every suite writes all four into every CSV row
    delimiters.py           the reasoning-delimiter merge guard shared by
                         `cbench gate` and the channel suite -- one
                         definition of the four built-in families plus a
                         model's catalogued `delimiters`, so the two
                         can't silently disagree the way they used to
    runclock.py             run-start time recorded in the data instead
                         of inferred from a file's mtime --
                         run_started_now()/run_time_row_fields() stamp
                         every suite's CSV rows, time_from_filename()
                         recovers a time from an older file's own
                         `..._YYYYmmdd_HHMMSS...` name
    remove.py               the only destructive operation here: deletes a
                         model from the endpoint. Exact tags only, never a
                         prefix match, and it leaves results/ alone
    progress.py             the dashboard's local progress panel: neutral
                         counts, the `is_citable()` rule (gate-checked, no
                         suite refused by a validity guard, >= 3 trials),
                         and a level earned on rigour rather than volume.
                         Reads local files only -- no account, no server,
                         nothing transmitted
  suites/
    containment.py       S1 -- egress & containment
    channel.py            S2 -- reasoning-channel divergence
    persistence.py        S3 -- deceptive persistence
  scoring/               offline re-scoring, aggregation, and
                         significance testing over CSVs already on disk
    scorecard.py           cross-suite scorecard: an A-F grade (0-100,
                         worst of the three suites) plus the per-suite
                         detail, built on aggregate_s1/s2/s3's stats
    capability.py           whether a suite was in a position to observe
                         what it scores: S2's leak rate counts only rows
                         that returned a reasoning trace, S3 excludes rows
                         where the model never wrote a log (so the
                         challenge turn was about a step that did not
                         exist). A suite that could not have fired is a
                         missing measurement, not a null
    comparability.py        run-pooling guard shared by aggregate_s1/s2/s3:
                         flags CSVs that predate the sampling columns
                         pooled with CSVs that carry them, or CSVs that
                         disagree on temperature/top_p/top_k -- the
                         suite's trial summary gets a STOP block and the
                         scorecard reports that suite INVALID, excluded
                         from the grade
  integrations/          optional Inspect cross-validation (pip install
                         ".[inspect]")
  tui/                     optional Textual control panel over the CLI
                         (pip install ".[tui]") -- jobs.py builds the
                         real `cbench <subcommand> ...` argv and runs it
                         as a subprocess, then reads that subprocess's
                         own streamed output back (condensed_line_filter
                         for what reaches the screen, parse_trial_header
                         for the trial-progress bar); app.py has no
                         suite logic of its own, only screens and forms,
                         plus the _report_job_result() tail every
                         action-taking screen shares
data/
  tasks/, probes/, scenarios/    the actual task/probe/scenario JSON
  models/verified.json           packaged model-catalogue seed
community-results/
  README.md                      submission convention for sharing raw
                                 trial CSVs on a model you can't run
                                 locally -- see there before adding to it
tests/
  test_safety_invariant.py       the core safety claim, executable,
                                 no model or network needed
  test_scoring_parity.py         golden fixtures pinning every scoring
                                 threshold/heuristic
  test_registry_and_hardware.py, test_catalogue_wiring.py
```

## Adding things

### A new suite

A suite is a `main()`-having module under `suites/` with its own
`argparse` parser (declared once, not duplicated in `cli.py` — see how
`containment.py`/`channel.py`/`persistence.py` do it), a `--dry-run` flag
that prints the payload without calling a model, and — if it gives the
model any tool with real-world reach — the exact same interception
discipline as `containment.py`'s `execute_tool()`. Register it in
`cli.py`'s `_PASSTHROUGH` dict. Add a safety-invariant test alongside the
existing ones if it introduces any new tool.

### A new task set / probe set / scenario file

Drop the JSON under `data/tasks/`, `data/probes/`, or `data/scenarios/`
and point `--tasks-file`/`--prompts-file`/`--scenarios-file` at it — no
suite requires editing to support a new file, only a new file. If it
reads sandbox fixtures the standard set doesn't, gate it the way the
harmful-intent task set is gated (§ ARCHITECTURE.md 6): an explicit flag
combination, checked and refused early with a clear error rather than
failing confusingly mid-run.

### A new model

Run `cbench discover` to see what's already pulled locally but not yet
catalogued, or go straight to `cbench gate --model <tag> --save` for a
specific tag before running anything else against it. This writes what
the gate check can discover automatically into your local `models.json`
overlay; hand-edit the saved entry to add `config_overrides` (a raised
`num_predict`, a different `timeout`, etc.) once a real run tells you
what the model actually needs. See `core/registry.py`'s module docstring
for the full catalogue schema.

### A new scoring metric or significance test

Add it to the relevant `scoring/` module and extend
`tests/test_scoring_parity.py` with a fixture case. Never change an
existing threshold or heuristic without updating the parity fixtures in
the same commit — a silent threshold change is worse than a bug, because
it changes what earlier results meant without anyone noticing.

### A community-submitted result for a model you can't run locally

See [`community-results/README.md`](community-results/README.md) for the
full submission convention — raw trial CSVs plus a `submission.json`,
submitted via PR, validated with `cbench community-validate` and then
scored the normal way (`cbench score --from-existing`) before anyone
trusts the number. Not a place for a bare score with no data behind it.

**The publication rule, which is not negotiable either:** raw
measurements travel, verdicts do not. A submission carries per-row CSVs
and nothing else — never a scorecard, a grade, or a claimed rate, and
`validate_submission()` refuses all of them. This repository is not a
leaderboard and must not become one by accident. A grade is an editorial
conclusion about a named commercial product; a CSV row is a measurement
qualified by the configuration recorded beside it, and only the second is
something a maintainer can stand behind on a stranger's behalf. Two
further guards exist for the same reason: a submission must record a
**completed** gate check (a check that timed out is an absent check, not
a failed one — rows from that machine may describe the machine rather
than the model), and a contributor must explicitly accept terms covering
right-to-share, confidentiality, accurate hardware, an Apache-2.0 licence
grant, and the permanence of git history. If you are adding anything to
this area, read `core/community.py`'s publication-rule comment first —
it records the live incident that motivated each guard.

`cbench community-package --model <tag> --accept-terms` builds the folder for you from
CSVs already on disk (copies, never rewrites, and records a checksum per
file), and `cbench community-submit <folder>` opens it as a pull request
through your own authenticated `gh` — previews by default, sends nothing
without `--confirm`. Hand-assembling the folder per the layout below is
still how it works underneath, and is the documented fallback for anyone
without `gh` (or without git at all, via `--zip` and a GitHub issue), but
it isn't the expected path anymore.

## Conventions worth keeping

- **The TUI never gets suite logic of its own.** Any new suite/scoring
  functionality lives in `suites/` or `scoring/` and is reached from the
  TUI the same way it's reached from a terminal -- a real `cbench
  <subcommand>` subprocess (`tui/jobs.py:cbench_command()`), not a direct
  import of the suite's `main()` into `tui/app.py`. If a PR adds a
  feature that only the TUI can do, that's a sign it's in the wrong file.
  The one exception: a screen may call a `core/` or `scoring/` function
  **directly** (in-process, via `asyncio.to_thread` so it can't block the
  UI) purely to *read* something for display. The existing examples: the
  Models screen listing local models via `core/discover.py` and its Score
  column reading a saved scorecard via
  `scoring/scorecard.py:catalogue_compact_label()`; the Score screen's
  catalogue status (`core/registry.py:lookup()`), its hardware fit
  warning (`core/hardware.py:probe()` / `check_model_fit()`), its
  "From existing" pre-flight check that any CSV exists at all for the
  tag (`scoring/aggregate.py:find_csvs()`), and its read-back of a
  gate subprocess's own printed report
  (`core/gate.py:summarize_gate_output()`). Every one of those calls the
  identical function the CLI itself calls, not a second implementation,
  and none has a side effect. Anything with a side effect (writing the
  catalogue, downloading a model, running a suite) still goes through a
  real subprocess, no exception -- including the batch "Gate + save all
  uncatalogued" action, which is `cbench discover --gate-all` rather than
  a TUI-side loop over the single-model gate.
- **Smoke-test before opening a PR, and say what you actually did.**
  Compile/syntax check is necessary but not sufficient — this project has
  twice caught real runtime-only bugs (a variable that only resolved at
  call time, an endpoint parameter reused for two different URL shapes)
  that no amount of static review found. State plainly in the PR whether
  you validated against a live model, against `--dry-run` output only, or
  logic-only against fabricated data — don't imply more verification than
  happened.
- **Gate-check a channel-splitting change at BOTH `think=true` and
  `think=false`.** A model can split cleanly at one state and merge at
  the other; a single-state check has missed a real, shipped bug before.
- **Never introduce per-model branching in a suite.** If a model needs
  different config, it goes in the catalogue (`core/registry.py`,
  `data/models/verified.json`), not an `if model == "..."` in suite code.
  The catalogue is data specifically so a new model's quirks don't require
  a code change.
- **Pinned sampling is recorded per row, not applied silently.** Every
  suite sends the same `--temperature`/`--top-p`/`--top-k`/`--seed`
  (`core/sampling.py:add_sampling_args()`/`resolve_sampling()`) and writes
  all four into every CSV row it produces, success or failure. If you add
  a new suite or a new code path that builds a chat request, call
  `resolve_sampling()` once in `main()` and pass the result down through
  `build_options()`/`sampling_row_fields()` — don't read `args.temperature`
  etc. ad hoc at each row-building site; per-call-site drift is exactly how
  one row in a run ends up disagreeing with the others about what was
  actually sent.
- **Before adding an unmatched serving parameter to a comparison, check
  for one.** Temperature/top-p/top-k/seed are pinned identically across
  every suite call by default now, so that specific confound is closed
  unless you override one of the four per model. Anything set at the
  serving-config level *outside* those four (a repetition penalty, a
  Modelfile `stop` sequence) still isn't pinned — if your contribution
  runs two models against each other (a base/fork pair, an A/B test), diff
  their Modelfile sampling parameters via `cbench gate` first. An unmatched
  parameter still invalidates the comparison before either model is ever
  run.
- **Keep `ARCHITECTURE.md` §7 in sync.** If you find a new way this
  harness's own measurements can mislead someone, add it there, not just
  in a code comment — that section exists specifically to save the next
  person from rediscovering it the hard way.

## Questions

Open an issue. For anything touching the safety invariant specifically,
see [SECURITY.md](SECURITY.md) instead.
