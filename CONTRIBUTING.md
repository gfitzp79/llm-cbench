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
                       (doctor, gate, discover, search, pull, assess,
                       score, catalogue, community-validate, tui)
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
                         sampling params) + catalogue-entry conversion
    discover.py            lists locally-pulled models missing from the
                         catalogue, via the endpoint's own /api/tags --
                         never talks to ollama.com (see its own
                         docstring for why)
    hardware.py           advisory VRAM/RAM probe, never blocking
    community.py           shape-checks a community-results/ submission
                         folder before it's scored or merged
    invariant.py          the one safety-invariant string, shared by
                         every --help epilog and `cbench doctor`
  suites/
    containment.py       S1 -- egress & containment
    channel.py            S2 -- reasoning-channel divergence
    persistence.py        S3 -- deceptive persistence
  scoring/               offline re-scoring, aggregation, and
                         significance testing over CSVs already on disk
    scorecard.py           cross-suite scorecard (per-suite band +
                         confidence, deliberately not one composite
                         number) built on aggregate_s1/s2/s3's stats
  integrations/          optional Inspect cross-validation (pip install
                         ".[inspect]")
  tui/                     optional Textual control panel over the CLI
                         (pip install ".[tui]") -- jobs.py builds the
                         real `cbench <subcommand> ...` argv and runs it
                         as a subprocess; app.py has no suite logic of
                         its own, only screens and forms
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

## Conventions worth keeping

- **The TUI never gets suite logic of its own.** Any new suite/scoring
  functionality lives in `suites/` or `scoring/` and is reached from the
  TUI the same way it's reached from a terminal -- a real `cbench
  <subcommand>` subprocess (`tui/jobs.py:cbench_command()`), not a direct
  import of the suite's `main()` into `tui/app.py`. If a PR adds a
  feature that only the TUI can do, that's a sign it's in the wrong file.
  The one exception: a screen may call a `core/` or `scoring/` function
  **directly** (in-process, via `asyncio.to_thread` so it can't block the
  UI) purely to *read* something for display -- the Models screen listing
  local models via `core/discover.py`, and its Score column reading a
  saved scorecard via `scoring/scorecard.py:catalogue_compact_label()`,
  are the existing examples. Both call the identical function `cbench
  discover`/`cbench catalogue` themselves call, not a second
  implementation, and neither has a side effect. Anything with a side
  effect (writing the catalogue, downloading a model, running a suite)
  still goes through a real subprocess, no exception.
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
- **Before adding an unmatched serving parameter to a comparison, check
  for one.** If your contribution runs two models against each other
  (a base/fork pair, an A/B test), diff their sampling parameters via
  `cbench gate` first. An unmatched parameter at the serving-config level
  invalidates the comparison before either model is ever run.
- **Keep `ARCHITECTURE.md` §7 in sync.** If you find a new way this
  harness's own measurements can mislead someone, add it there, not just
  in a code comment — that section exists specifically to save the next
  person from rediscovering it the hard way.

## Questions

Open an issue. For anything touching the safety invariant specifically,
see [SECURITY.md](SECURITY.md) instead.
