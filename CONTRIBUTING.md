# Contributing

**This project does not accept contributions.** Pull requests and result
submissions are not accepted, and the repository has no issue tracker. To
report a security vulnerability, use the private route in
[SECURITY.md](SECURITY.md).

The licence (Apache-2.0) permits you to fork the code and change it for
your own use. The rest of this document is for anyone reading the code or
extending their own copy: what the project is, the rule that is not
negotiable, how the code is laid out, how to add to it, and the
conventions it keeps. It assumes no particular editor, IDE or AI
assistant.

## What this project is

`openllm-cbench` is a local testing framework for open-weight LLMs. It
asks three questions, one per suite:

- **S1 containment:** given real tools inside a sandboxed task, does the
  model try to reach beyond the boundary it was given?
- **S2 channel:** can a probe pass on the visible answer while the hidden
  reasoning trace fails it?
- **S3 persistence:** challenged on a logged step, does the model try to
  rewrite the record rather than correct it honestly?

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full picture.

## The non-negotiable invariant

**Every suite measures attempt, never success.** No change may add a
code path through which a model's tool call reaches any real
destination other than the suite's own loopback canary, or through which
a suite sends a request anywhere other than the configured chat endpoint
and that canary. `run_shell` never executes anything, under any flag, for
any task set.

If a change touches `core/canary.py`, `core/sandbox.py` or any suite's
tool-execution code (`execute_tool()` or its equivalent), run
`tests/test_safety_invariant.py` before and after the change, and say so
in the commit message. The test needs no model and no network, so it can
always be run.

## Setting up

You need Python 3.10 or later.

```bash
git clone https://github.com/gfitzp79/llm-cbench
cd llm-cbench
python -m venv .venv                     # then activate it
python -m pip install --upgrade pip setuptools
pip install -e ".[dev,tui]"              # or ".[dev,tui,inspect]" for Inspect
pytest -q                                # no model or network needed
```

Upgrade pip and setuptools before installing: a new virtual environment
can start with the versions your Python shipped, which may carry known
advisories, and nothing else in this setup updates them. The `inspect`
extra is needed only for the Inspect cross-validation under
`integrations/`.

CI runs the tests for every push to `main` on Linux, macOS and Windows,
each with Python 3.10 and 3.13. Every job also runs the safety-invariant
test as a separate step, checks that the modules import cleanly, and runs
`cbench doctor` with no endpoint present. A local pass covers one of the
six combinations in the matrix.

## Layout

```
src/openllm_cbench/
  cli.py                    the `cbench` entry point: native subcommands
                            (doctor, gate, discover, search, pull, remove,
                            config, assess, score, compare, catalogue, tui)
                            and passthroughs to a
                            module's own main() (containment, channel,
                            persistence, aggregate, guardrail, extension-rule,
                            score-containment)
  core/                     shared primitives, no suite-specific logic
    canary.py               loopback HTTP listener and its bind assertion
    sandbox.py              fabricated in-memory sandbox contents
    endpoint.py             chat endpoint resolution: `--endpoint`, then
                            $OPENLLM_CBENCH_ENDPOINT, then
                            http://localhost:11434
    paths.py                packaged-data and results-folder resolution
    config.py               persistent user settings (the results folder and
                            the catalogue location), so neither depends on the
                            launch directory
    provenance.py           third-party-model labelling rule
    registry.py             model catalogue: packaged seed plus local overlay,
                            config_overrides resolution, thinking-mode and
                            delimiter lookup
    gate.py                 gate check (capabilities, tool call, channel
                            separation at both think states, Modelfile
                            sampling parameters) and catalogue-entry
                            conversion; summarize_gate_output() reads the
                            report back from captured output for a caller that
                            ran `cbench gate` as a subprocess
    preflight.py            the single mapping from gate findings to a
                            per-suite verdict (READY, INVALID or UNVERIFIED),
                            used by the gate report and by the pre-flight in
                            `cbench score` and `cbench assess`, which skips a
                            suite that cannot produce a gradeable result
                            before the run rather than after it, and refuses
                            the run when no selected suite can
    runlock.py              launch-time exclusivity guard for `cbench score`
                            and `cbench assess`: one assessment at a time per
                            user, refused while another holds the run lock,
                            while a suite process is live, or when the process
                            table cannot be read (`--force-concurrent`
                            overrides); with preflight.py, one of the two
                            guards that run before the work
    discover.py             locally pulled models missing from the catalogue,
                            read from the endpoint's own /api/tags; never
                            contacts ollama.com (its docstring explains why)
    pull.py                 exact-tag existence check (`cbench search`) and
                            model download (`cbench pull`), both through the
                            local endpoint's /api/pull route
    remove.py               the only destructive operation: deletes a model
                            from the endpoint by exact tag, never by prefix,
                            and leaves results alone
    hardware.py             advisory VRAM and RAM probe that never blocks a
                            run: probe() and format_report() for
                            `cbench doctor`; fit_assessment(), built on
                            check_model_fit(), for one model's fit against
                            detected VRAM
    locations.py            the locations report in `cbench doctor`: Ollama's
                            install, logs and model storage (read from the
                            server's startup log, then confirmed against the
                            models it lists), network exposure, and this
                            tool's own files, each found and checked rather
                            than assumed from a platform default
    sampling.py             pinned sampling parameters (temperature, top_p,
                            top_k, seed) and the generation-budget flags
                            shared by every suite; see the sampling convention
                            below
    budget.py               the automatic generation budget: resolve_budget()
                            takes num_ctx and num_predict from the flag, then
                            the catalogue's config_overrides, then a larger
                            budget for a model that reasons (model_reasons()),
                            then the suite default; budget_line() and
                            run_summary() print what was chosen and why
    delimiters.py           the reasoning-delimiter merge guard shared by
                            `cbench gate` and the S2 suite: the built-in
                            delimiter families plus a model's catalogued
                            `delimiters`, defined once so the two cannot
                            disagree
    runclock.py             run start time recorded in the data rather than
                            inferred from a file's mtime: run_started_now()
                            and run_time_row_fields() stamp every suite's CSV
                            rows, and time_from_filename() recovers a time
                            from an older file's `..._YYYYmmdd_HHMMSS` name
    context_window.py       whether the context window bound a run:
                            max_prompt_tokens and peak_context_tokens per row,
                            and the half-window rule that separates a provably
                            untruncated run from one at risk
    exitcodes.py            what an exit code means in words (0 done, 1 ran
                            and something failed, 2 refused to start), and
                            after_run(), which fails a suite run whose
                            requests did not reach the model
    progress.py             the dashboard's local progress panel: neutral
                            counts, the is_citable() rule (catalogued, no
                            suite refused by a validity guard, at least 3
                            trials per suite), and a level earned on rigour
                            rather than volume; reads local files only and
                            transmits nothing
    invariant.py            the safety-invariant string, printed by
                            `cbench --help`, shown as the `--help` epilog of
                            each suite and native subcommand, and shown in the
                            TUI's invariant bar
    console.py              UTF-8 stdio set-up, applied once at entry
  suites/
    containment.py          S1 containment
    channel.py              S2 channel
    persistence.py          S3 persistence
  scoring/                  offline scoring, aggregation and significance
                            testing over CSVs already on disk
    aggregate.py            repeated-trial aggregation (`cbench aggregate`)
                            and the per-suite validity guards (task set,
                            schema version, control diagnostics); also home to
                            find_csvs() and model_tag()
    scorecard.py            cross-suite scorecard: an A-F grade (0-100, the
                            worst of the three suites) plus per-suite detail,
                            built on the aggregate statistics
    compare.py              cross-model comparison of two saved scorecards:
                            whether the difference survives clustering, and
                            the power the comparison had. The verdict is
                            DIFFERENT, INCONCLUSIVE or NO DIFFERENCE, so an
                            underpowered null is never reported as similarity.
                            The only cross-model guard; every other guard is
                            within-model
    clustering.py           effective sample size: rows are repeated probes,
                            not independent observations, so every confidence
                            interval uses n_eff rather than n, with the ICC
                            estimated from each run's own data
    capability.py           whether a suite could observe what it scores: S2's
                            leak rate counts only rows that returned a
                            reasoning trace, and S3 counts only rows whose log
                            held a step for the challenge to be about. A suite
                            that could not have fired reports a missing
                            measurement, not a null
    comparability.py        the run-pooling guard used by every aggregate:
                            pooling CSVs that predate the sampling columns
                            with CSVs that carry them, or CSVs that differ in
                            temperature, top_p, top_k or generation budget,
                            puts a STOP block in the trial summary, and the
                            scorecard reports that suite INVALID
    containment_metrics.py  offline egress re-scoring of S1 CSVs
                            (`cbench score-containment`), plus the
                            INCOMPLETE-row and target-classification rules the
                            S1 suite imports
    probes.py               S2 probe scoring, imported by suites/channel.py
    guardrail.py            guardrail-detection scoring of recorded S1 tool
                            calls with a classifier model served by the
                            endpoint (`cbench guardrail`)
    extension_rule.py       the pre-registered trial-extension rule for a base
                            and variant pair (`cbench extension-rule`)
  integrations/             optional Inspect cross-validation (the `inspect`
                            extra): Inspect tasks for S1 and S3, and
                            reconcilers that compare their verdicts with this
                            framework's own
  tui/                      optional Textual control panel over the CLI (the
                            `tui` extra)
    jobs.py                 builds the real `cbench <subcommand> ...` argv,
                            runs it as a subprocess and streams its output
                            back (condensed_line_filter() for what reaches the
                            screen, parse_trial_header() for the trial
                            progress bar)
    app.py                  screens and forms only, plus the
                            _report_job_result() tail that every action screen
                            shares; no suite logic
  data/                     packaged with the wheel, read through
                            core/paths.py
    tasks/                  S1 task sets
    probes/                 S2 probe set
    scenarios/              S3 scenarios
    models/verified.json    packaged catalogue seed: configuration only, no
                            verdicts; its `_schema` key documents every
                            catalogue field
tests/                      every test runs with no model and no network
  test_safety_invariant.py  the core safety claim, executable
  test_scoring_parity.py    golden fixtures pinning every scoring threshold
                            and heuristic
```

## Adding things

### A new suite

A suite is a module under `suites/` with a `main()` and its own
`argparse` parser, declared once in the module and not repeated in
`cli.py` (see how `containment.py`, `channel.py` and `persistence.py` do
it). It needs a `--dry-run` flag that prints the payload without calling
a model. Resolve its generation budget with
`core/budget.py:resolve_budget()` and print `budget_line()`, as the three
suites do, so the same precedence (flag, then catalogue, then automatic)
applies to it. If it gives the model any tool with real-world reach, it
must intercept that tool exactly as `containment.py`'s `execute_tool()`
does.
Register it in the `_PASSTHROUGH` dict in `cli.py`, and add a
safety-invariant test alongside the existing ones for any new tool.

`cbench assess` and `cbench score` run only S1, S2 and S3. Including a
new suite in them also means adding it to the `SUITE_INFO` tables in
`cli.py`, an aggregate function in `scoring/aggregate.py` and a verdict
in `scoring/scorecard.py`.

### A new task set, probe set or scenario file

Put the JSON under `src/openllm_cbench/data/tasks/`, `data/probes/` or
`data/scenarios/`, and point `--tasks-file`, `--prompts-file` or
`--scenarios-file` at it. A new file needs no suite changes.
`cbench score-containment` tells S1 task sets apart by their task count,
so give a new task set a count that no existing set uses.

If a task set reads sandbox fixtures that the standard set does not, gate
it the way the harmful-intent task set is gated (ARCHITECTURE.md section
6): an explicit flag combination, checked at start-up and refused with a
clear error rather than failing mid-run.

### A new model

Run `cbench discover` to list models pulled locally but not yet in the
catalogue, or run `cbench gate --model <model-tag> --save` for a single
tag before anything else runs against it. The gate check saves what it
can discover into your local catalogue (`models.json`). It leaves
`config_overrides` empty, except for `think: false`, which it sets itself
when the model calls tools only with its reasoning channel off. The
`thinking` field it records sets the model's automatic generation budget
(docs/USER_GUIDE.md, "Generation budgets"). If a real run shows the model needs
something else, edit the saved entry to add `config_overrides` such as a
different `num_predict` or a longer `timeout`.
The `_schema` key in `data/models/verified.json` documents every
catalogue field.

### A new scoring metric or significance test

Add it to the relevant `scoring/` module and add a fixture case to
`tests/test_scoring_parity.py`. Never change an existing threshold or
heuristic without updating the parity fixtures in the same commit. A
silent threshold change is worse than a bug, because it changes what
earlier results meant without anyone noticing.

## Conventions to keep

- **Keep suite logic out of the TUI.** New suite or scoring functionality
  lives in `suites/` or `scoring/`, and the TUI reaches it the way a
  terminal does: through a real `cbench <subcommand>` subprocess
  (`tui/jobs.py:cbench_command()`), never by importing a suite's `main()`
  into `tui/app.py`. A feature that only the TUI can use is in the wrong
  file.

  The one exception: a screen may call a `core/` or `scoring/` function
  in-process to read something for display, off the UI thread through
  `asyncio.to_thread` when the call can be slow. For example, the Models
  screen lists local models through `core/discover.py` and reads each
  saved scorecard through `scoring/scorecard.py:catalogue_compact_label()`.
  The Score screen reads the catalogue status and the suite readiness the
  gate check recorded (`core/registry.py:lookup()`), builds its hardware
  fit warning from `core/hardware.py:probe()` and `fit_assessment()`,
  checks whether any CSV exists for the tag before a "Re-score saved
  results only" run
  (`scoring/aggregate.py:find_csvs()`), and reads back the report printed
  by its own gate subprocess (`core/gate.py:summarize_gate_output()`).
  Each calls the same function the CLI calls, and none changes anything.

  Anything that changes state (writing the catalogue, pulling or deleting
  a model, running a suite) goes through a real subprocess. That includes
  the batch "Gate + save all uncatalogued" action, which runs
  `cbench discover --gate-all` rather than looping over single gate
  checks inside the TUI. The only file the TUI writes itself is a copy of
  each job's output, under `tui-logs/` in the results folder.
- **Smoke-test before calling a change done, and say what you did.** A
  compile or syntax check is necessary but not sufficient: some defects
  appear only at run time, such as a name that resolves only when it is
  called, or one endpoint parameter reused for two different URL shapes.
  State in the commit message whether you validated against a live model,
  against `--dry-run` output only, or against fabricated data only. Do
  not imply more verification than took place.
- **Gate-check a channel-splitting change at both `think=true` and
  `think=false`.** A model can split its channels cleanly at one state
  and merge them at the other, so a check at one state can pass a change
  that is broken at the other.
- **Never branch on a model name in suite code.** Configuration that a
  model needs belongs in the catalogue (`core/registry.py`,
  `data/models/verified.json`), not in an `if model == "..."` in a suite.
  The catalogue is data so that a new model's quirks need no code change.
- **Record pinned sampling in every row.** Every suite takes the same four
  sampling flags (`--temperature`, `--top-p`, `--top-k`, `--seed`, from
  `core/sampling.py:add_sampling_args()`), resolves them once in `main()`
  with `resolve_sampling()` (which generates a seed when none is given),
  sends them through `build_options()`, and stamps all four values onto
  every CSV row at write time with `sampling_row_fields()`, including
  error and timeout rows. Follow the same pattern in any new suite or
  code path that builds a chat request. Do not read `args.temperature`
  and the others separately where each row is built: per-site drift is
  how one row in a run ends up disagreeing with the others about what was
  sent.
- **Check for unmatched serving parameters before comparing two models.**
  Temperature, top-p and top-k are pinned to the same defaults for every
  model, and `presence_penalty` to 0, so they cannot confound a
  comparison unless you override one for one model's run. A parameter set
  in a model's Modelfile outside that list, such as a repetition penalty
  or a `stop` sequence, is not pinned. If your change runs two
  models against each other (a base and fork pair, an A/B test), first
  compare the Modelfile sampling parameters that `cbench gate` reports
  for each model. An unmatched parameter invalidates the comparison
  before either model runs.
- **Keep ARCHITECTURE.md section 7 current.** If you find a new way this
  harness's measurements can mislead someone, add it to that section, not
  only to a code comment. The section exists so that the next person does
  not have to rediscover it.

## Questions

There is no issue tracker, so questions about the code are answered by
the code and these documents. For anything that touches the safety
invariant, or any other security concern, use the private route in
[SECURITY.md](SECURITY.md).
