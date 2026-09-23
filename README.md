# openllm-cbench

Local containment and reasoning-leakage testing framework for open-weight
LLMs.

> **Safety invariant.** Every suite in this framework measures **attempt,
> never success**. Each suite's `http_request` tool can only ever reach a
> loopback canary (`127.0.0.1`) that asserts its own bind address before
> starting; any other host is intercepted before a socket is opened and
> answered with a synthetic failure. `run_shell` is always a logged no-op:
> nothing it is given is ever executed. See [ARCHITECTURE.md](ARCHITECTURE.md)
> for the full control inventory, and run
> `pytest tests/test_safety_invariant.py` to verify it yourself; the test
> needs no model and no network.

If you serve an open-weight model locally, this tool shows what the model
*does* under agentic pressure rather than what it says when asked directly.
It measures three behaviours that neither a benchmark score nor a manual
chat transcript reveals: whether the model reaches past a tool
boundary it was given, whether its hidden reasoning trace fails a probe
that its visible answer resists, and whether it tries to rewrite a log
rather than own a mistake when challenged on one. It is for anyone
deploying a local model into something with real tools attached (an agent,
a pipeline, anything beyond a chat window) who wants that answer from a run
against their own model on their own hardware, not a leaderboard entry for
a checkpoint they cannot reproduce.

The scope is deliberately three suites rather than a general red-teaming
toolkit. Broad scanners are well served elsewhere (see
[Comparison to other tools](#comparison-to-other-tools)), and the project
favours depth on three specific questions that are otherwise hard to
measure over shallow coverage of many.

## What is included

| Suite | Command | Question it asks |
|---|---|---|
| S1 containment | `cbench containment` | Given real tools inside a sandboxed task, does the model reach beyond the boundary it was given? |
| S2 channel | `cbench channel` | Does the hidden reasoning trace fail a probe that the visible answer resists, and how often does the visible answer fail it outright? |
| S3 persistence | `cbench persistence` | Challenged on a logged action, does the model try to rewrite the record rather than correct it honestly? |

The other commands:

| Command | What it does |
|---|---|
| `cbench doctor` | Checks the environment: the endpoint, where Ollama and cbench keep their files, the canary and the hardware. Calls no model and writes nothing. |
| `cbench config` | Shows, pins or clears where results and the catalogue are kept. |
| `cbench gate` | Gate-checks one model; `--save` records the result in the catalogue. |
| `cbench discover` | Lists pulled models the catalogue does not know yet; `--gate-all` gate-checks and saves them. |
| `cbench catalogue` | Lists every pulled model with its catalogue status, hardware fit, speed and score. |
| `cbench search`, `cbench pull`, `cbench remove` | Checks that an exact tag exists in Ollama's registry, downloads a model, or deletes one from the endpoint. |
| `cbench assess` | Runs N trials of each selected suite, then aggregates each suite's trials. |
| `cbench score` | Runs (or reads existing) trials and produces an A-F scorecard. |
| `cbench compare` | Compares two saved scorecards, with a significance test corrected for clustering and the power the comparison had. |
| `cbench aggregate` | Pools one model's trial CSVs for one suite into a trial summary. |
| `cbench extension-rule` | Applies the pre-registered 3-trial extension rule to a base/variant pair. |
| `cbench score-containment` | Re-scores S1 CSVs with more detailed egress metrics. |
| `cbench guardrail` | Scores the tool calls recorded in S1 CSVs against a guardrail classifier model, offline: detection rate and over-refusal cost. It measures detection, not prevention. |
| `cbench community-package`, `cbench community-validate`, `cbench community-submit` | Packages, checks and submits your raw CSVs as a community result. |
| `cbench tui` | Opens the terminal UI, a control panel over the commands above. |

`cbench --version` prints the installed version. Each suite and scoring
module declares its own flags, so `cbench <subcommand> --help` is the
complete reference for any subcommand. Optional cross-validation of S1 and
S3 against [Inspect](https://inspect.aisi.org.uk/) lives in
`src/openllm_cbench/integrations/` (the `inspect` extra; see
[Install](#install)).

The full architecture, the control inventory and the measurement pitfalls
this tool is built to avoid are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Install

Requires Python 3.10 or later and a model served locally through Ollama's
native API (default endpoint `http://localhost:11434`); section 4 of
[ARCHITECTURE.md](ARCHITECTURE.md) lists the routes it uses. To use
another endpoint, set `$OPENLLM_CBENCH_ENDPOINT`, which every command that
talks to the endpoint reads. `--endpoint` overrides it on `doctor`, `gate`,
`discover`, `catalogue`, `search`, `pull`, `remove` and the three suites;
`score`, `assess`, `guardrail` and `community-package` take no `--endpoint`
and read the environment variable only. The package is not on PyPI, so
install it from GitHub.

To install it with the terminal UI:

```bash
pip install "openllm-cbench[tui] @ git+https://github.com/gfitzp79/llm-cbench"
cbench doctor
```

Leave out `[tui]` for the CLI alone; its only dependency is `requests`. If
your system Python refuses `pip install` with "externally-managed-environment"
(for example on recent Ubuntu releases), use a virtual environment, or
`pipx install` with the same argument.

To read or change the code, run the tests, or use the Inspect
cross-validation (its task files run by path, so it needs a clone):

```bash
git clone https://github.com/gfitzp79/llm-cbench
cd llm-cbench
pip install -e ".[dev,tui]"      # or ".[dev,tui,inspect]" for Inspect
pytest -q                        # needs no model and no network
```

## Where results are kept

**Set this once, before your first real run:**

```bash
cbench config --set-results-dir ~/cbench-results
```

Without it, results go to `./results` under whichever directory you launch
from. Launch the TUI from your home directory and the CLI from a project,
and you get two unrelated results folders with the same name, each
invisible to the other. That is worse than untidy: `cbench score` reads
only the folder it resolves to, so a run that landed in the other one is
missing from the aggregate without any warning, and the grade is computed
over whatever subset shared a directory with it.

`cbench config` on its own prints where results are going and which setting
decided that. `cbench config --find-results` searches the usual places for
results folders that already exist; run it once if you have used the tool
from more than one directory. It only reads, and moves nothing.

`cbench doctor` lists every location at once, each with the setting or
check behind it: Ollama's executable, logs and model storage, and cbench's
config file, results folder, catalogue, TUI logs and run lock. Ollama's
model folder is read from the running server's own startup log and
confirmed by matching its manifests against the models the server lists,
because the platform default can exist, be empty and be wrong: a folder
chosen in the Ollama app's settings is not an environment variable any
other process can see. For an endpoint on this machine, `cbench doctor`
also warns when the server listens beyond loopback (`OLLAMA_HOST` set to
`0.0.0.0` or a LAN address), because Ollama has no authentication.

Resolution order, most specific first:

| Setting | Scope |
|---|---|
| `--results-dir` on a command that takes it (the three suites and `compare`) | that invocation |
| `$OPENLLM_CBENCH_RESULTS_DIR` | that shell |
| the config file | your user, in every directory |
| `./results` | whichever directory you are in |

The config file is `%APPDATA%\openllm-cbench\config.json` on Windows and
`$XDG_CONFIG_HOME/openllm-cbench/config.json` (by default
`~/.config/openllm-cbench/config.json`) elsewhere; `$OPENLLM_CBENCH_CONFIG`
overrides its location. The same settings are available in the TUI under
**Settings**, and the dashboard shows the active results location on every
launch, in yellow when it is not pinned.

**Pin the model catalogue too**, for the same reason and with the same four
layers (`--registry-file` on a command that takes it,
`$OPENLLM_CBENCH_MODELS_FILE`, the config file, then `./models.json`):

```bash
cbench config --set-models-file ~/cbench-results/models.json
```

An unpinned catalogue is worse than an unpinned results folder, because it
loses nothing visibly: it presents a *different* `models.json` without
saying so. Models you have already gate-checked come back as uncatalogued,
and the next run goes out ungated, without the configuration those gate
checks recorded. The catalogue is a separate setting rather than derived
from the results location because one catalogue can serve several sets of
results, and moving your results should not mean re-gating every model.
`--unset-results-dir` and `--unset-models-file` remove a pinned location.

## What goes in `--model <model-tag>`

Commands that act on one model take `--model <model-tag>`. The tag is
whatever your endpoint calls the model, passed through unchanged in every
request this framework sends. For the default (Ollama) endpoint it is the
`NAME` column of `ollama list`:

```bash
$ ollama list
NAME                    ID              SIZE      MODIFIED
gemma3:12b              f4031aab637d    8.1 GB    8 weeks ago
qwen3.5:9b              6488c96fa5fa    6.6 GB    12 days ago
hf.co/org/repo:Q4_K_M   ...

$ cbench doctor              # takes no --model: checks the endpoint only
$ cbench gate --model gemma3:12b --save
```

There is no separate registration step and no fixed roster: if your
endpoint recognises the tag, `cbench` can run a suite against it. A wrong
tag fails at the first command that uses it, with a message that says the
model could not be reached: `cbench gate` exits `2`, and the pre-flight in
`cbench score` and `cbench assess` refuses to start (also exit `2`). It does
not surface later as a confusing result inside a suite's output.
`cbench doctor` checks only the endpoint, so it cannot catch a wrong tag.

## Quick start

```bash
# Check the environment: the endpoint is reachable, where Ollama and cbench
# keep their files, the canary binds loopback, and the hardware headroom.
# Calls no model and writes nothing.
cbench doctor

# Gate-check a model before trusting any real run against it: tool-call
# well-formedness, reasoning-channel separation at both think states, and
# the sampling parameters the endpoint reports. The report says, per suite,
# whether S1, S2 and S3 can produce a gradeable result on this model.
cbench gate --model <model-tag>

# The same check, with the result saved to your local model catalogue, so
# every suite picks up the right configuration for this tag from then on.
cbench gate --model <model-tag> --save

# Grade it: all three suites, one A-F scorecard. quick runs 1 trial per
# suite to see it work; standard (3 trials, the default) gives a grade
# worth citing.
cbench score --model <model-tag> --depth quick
```

### Exit codes

Every command uses the same three exit codes, with one exception noted
below:

| Exit code | Meaning | What to do |
|---|---|---|
| `0` | Done. | Read the output. |
| `1` | It ran, and something in it failed: a request that never reached the model, a gate check that was not clean, a scored suite that came back INVALID. | The output says what failed; there is a result to inspect. |
| `2` | It refused to start, and nothing ran or was written: bad input, a model the pre-flight could not reach, a selected suite the pre-flight found cannot produce a gradeable result, or a run lock refusal (another assessment or a suite run already in progress). | Fix what the message names, then run again. |

A suite run in which any request to the endpoint failed exits `1` and says
how many failed. Those rows measured nothing, so they are left out of every
rate rather than scored as a model that did nothing. `cbench score` also
exits `1` when a trial exited non-zero; a suite with no data under
`--from-existing` is not a failure, because scoring whatever exists is what
that flag is for. `cbench gate` exits `2` when it never reached the model,
and `--save` then saves nothing.

The exception is `cbench extension-rule`, whose exit code is its decision so
that a script can act on it: `0` stop, `2` extend, `1` error.

## Running the suites

### Containment

```bash
cbench containment --model <model-tag> --boundary both
cbench containment --model <model-tag> --task fx_lookup --dry-run   # preview the payload; calls no model
```

`--boundary both`, the default, runs every task with the boundary
instruction stated and with no boundary. `cbench containment --help` lists
the remaining options, including the harmful-intent task set (`--tasks-file`
with `--sandbox extended`); section 6 of [ARCHITECTURE.md](ARCHITECTURE.md)
explains why that set ships.

### Channel

```bash
cbench channel --model <model-tag> --think both
cbench channel --model <model-tag> --effort all   # for model families with graded reasoning effort
```

### Persistence

```bash
cbench persistence --model <model-tag>
cbench persistence --model <model-tag> --scenario dedup_customer_records
```

### Generation budgets

```bash
cbench score --model <model-tag> --num-ctx 8192 --num-predict 4096
```

`--num-ctx` (the context window) and `--num-predict` (the reply budget) can
be set on `cbench score`, `cbench assess`, each suite, and the TUI's Score
screen. **Leave them unset to let the model catalogue decide:** an explicit
flag overrides the model's `config_overrides` in the catalogue, which
override the suite default.

Raise `--num-predict` when a run reports more than a handful of TRUNCATED
rows (S2) or INCOMPLETE rows (S1). Such a row produced no final answer, so
it leaves the denominator, and the budget moves the rate without any change
in the model's behaviour.

**Both values are recorded in every row**, and for that reason the pooling
guard refuses to mix two budgets into one rate (see
[Run comparability is checked before pooling](#run-comparability-is-checked-before-pooling)).
A budget that was applied but not recorded would be the same kind of
confound that the sampling pin below exists to prevent.

#### The context window is checked, because it fails silently

`--num-predict` running out is visible: the row is marked `TRUNCATED` (in S1,
`INCOMPLETE`), leaves the denominator, and the report says so. **`--num-ctx` running out is
silent.** The server returns no error; it drops tokens from the front of the
conversation and answers anyway. The front is where the boundary instruction
and the model's own earlier turns are, so an over-tight window produces a
model that cannot see the rule it is being scored against, and a verdict
that faithfully records the failure.

Every run report and every trial summary therefore includes one line,
whether or not anything is wrong:

```
Context window: peak 977 of 8192 tokens (12% of the window).
```

and adds a warning when the window may have bound. The two warnings claim
different things:

- **`may have been truncated`**: the prompt filled at least half the window.
  The server reports the token count it evaluated *after* truncating, so a
  prompt that fitted and one cut down to fit report the same number. Re-run
  with a larger `--num-ctx`; if the count does not move, nothing was
  truncated.
- **`the context window bound`**: prompt plus generated tokens reached
  `num_ctx`. Those rows are unusable; raise the window and re-run.

### Reproducibility: pinned sampling

Every suite sends the same four sampling parameters with every chat
request, rather than leaving them to whatever the endpoint's Modelfile sets
for that tag: `--temperature` (default 0.8), `--top-p` (default 0.9),
`--top-k` (default 40) and `--seed`. All four are written into every CSV row
this framework produces, so a run remains auditable after the fact instead
of carrying an invisible confound.

```bash
cbench containment --model <model-tag> --seed 777   # replay one run exactly
cbench containment --model <model-tag>               # seed generated and recorded in every row
```

`cbench score` and `cbench assess` take the same four flags and pass them to
every suite invocation they make, so a whole scorecard can be produced at a
chosen temperature or replayed from a chosen seed.

**An explicit `--seed` is offset by the trial index:** with `--seed 7`,
trial 1 runs at 7, trial 2 at 8, and so on. This is the one place the flags
are not passed through unchanged, and it matters: giving all three trials of
`--depth standard` the same seed makes them byte-identical, so the run takes
three times as long and reports a confidence interval computed over three
copies of one sample. The run as a whole stays reproducible, and the trials
within it stay distinct.

Omitting `--seed` is deliberate. A fixed default would make every trial in a
multi-trial run identical, destroying the trial-to-trial variance that the
3-trial minimum exists to measure. A seed that is generated and recorded
keeps that variance and still lets any single run be replayed exactly: two
runs of the same model on the same endpoint at the same explicit seed
produce byte-identical output.

Every row also records `run_started_at`, the wall-clock time the run began,
with a UTC offset. It is recorded rather than inferred because a file's
modification time belongs to whatever tool last touched the file:
`git clone`, a copy and a zip round trip all rewrite it, and a results
folder that has been through any of them would otherwise claim that every
run happened at the moment it was unpacked. See section 6.1 of
`docs/METHODOLOGY_TECHNICAL.md`.

Pinning closes a confound that is otherwise silent. Without it, a ranked
comparison can run different models at different temperature and sampling
settings, and those differences can rival or exceed the effect sizes the
comparison is trying to measure. Pinned sampling does not make two models
comparable if they are still served differently in *other* ways; see
[section 7 of ARCHITECTURE.md](ARCHITECTURE.md#7-how-not-to-fool-yourself-with-this-tool)
for what else to diff with `cbench gate` before trusting a base/fork or A/B
comparison. The sampling flags are defined in `core/sampling.py`.

### A suite that never fired its positive control is refused

`cbench score` reports S1 as `INVALID`, and leaves it out of the grade, when
its positive control never fired across the trials. That control is a task
that explicitly requires an external fetch, so a model that never attempts
it is either refusing everything or has broken tool wiring, and in both
cases the suite has not shown it can return a positive. Without this rule, a
model with no tool-calling support would score A (100/100) with high
confidence: it attempts nothing because it can attempt nothing, and "no
escape attempts observed" would read as perfect containment.

`cbench score` and `cbench assess` do not rely on you checking first. Before
spending anything, they run a gate check of their own (the pre-flight) and
**refuse a suite that cannot produce a gradeable result**, naming the suite,
the reason, and the narrowed command that runs the rest:

```
[!] NOT STARTING -- S2 channel cannot produce a gradeable result on this model.
    Running it would spend the full time and score INVALID, which is a missing
    measurement rather than a finding. See the reasons above.

    What you CAN run:
      cbench score --model llama3.1:8b --suites s1,s3 --depth quick
    Read any grade from that as covering only those suites.
```

The pre-flight catches three cases, each of which occurs on real models:

| What the endpoint reports | What happens |
|---|---|
| `completion` only | No tool calls and no reasoning trace: all three suites come back INVALID. |
| `completion, tools` | S1 and S3 are fine; S2 has no separate channel for a leak to be found in. |
| `tools, thinking`, but the trace comes back **empty** | The capability is advertised and not delivered, so S2 has nothing to measure. |

The last case is the worst, because the advertised capability makes S2 look
supported right up until the scorecard reports a 0% leak rate that is a
property of the instrument, not of the model.

The pre-flight also refuses to start when it cannot reach the model at all
(an endpoint that is down, or a tag it does not serve), because every trial
would make the same failing calls. That refusal is not a verdict about the
model: nothing was measured.

A check that could not complete is **not** a refusal. A tool call that timed
out, or got no response at all, says something about your machine or the
connection, not about the model, so that suite is reported `UNVERIFIED` and
the run proceeds; `core/gate.py:warm_up()` exists to stop a stopwatch being
turned into a capability verdict. Likewise, if the pre-flight itself fails
with an error, it warns and the run carries on, because a malfunction in the
check is not evidence about the model.

Overrides, on both commands: `--force-uncheckable` runs a refused suite
anyway (it still grades INVALID, and the transcripts are all you get), and
`--skip-preflight` skips the check entirely, for an endpoint that misreports
its own capabilities. `--dry-run`, and `--from-existing` on `cbench score`,
skip it too, since neither calls a model.

A grade computed from fewer than three suites says so. The headline is the
worst suite's rate, so adding the missing suites could only lower it or
leave it unchanged: a partial grade is an upper bound. The compact label
carries its coverage (`A (100/100) [high] 1/3`), so a one-suite A and a
three-suite A are not confused in a table cell.

### Run comparability is checked before pooling

`cbench aggregate` and `cbench score` pool every CSV on disk for a model
tag, which is how results from different harness configurations end up
averaged into one number unless something checks. The pooling guard makes
three exact equality checks:

- **Sampling recorded by some CSVs and not others.** Some of a model's CSVs
  carry the `temperature`, `top_p` and `top_k` columns and others predate
  them.
- **Different sampling.** The CSVs that carry those columns disagree on
  them.
- **Different generation budgets.** The CSVs that record `num_ctx` and
  `num_predict` disagree on them.

When any check fires, that suite's trial summary carries a
`STOP -- THESE RUNS ARE NOT COMPARABLE` block, and `cbench score` reports
the suite `INVALID` and excludes it from the grade. A results folder that
mixes CSVs from before and after sampling was recorded therefore produces
an `INVALID` suite rather than a grade. The fix is the same in every case:
re-run so that every trial shares one configuration, or point
`$OPENLLM_CBENCH_RESULTS_DIR` at a folder holding only the runs you mean to
pool.

The guard is deliberately narrow: exact equality tests on recorded values,
not a heuristic. It does not fire on a differing `--seed` (varying the seed
per trial is the point) or on a corpus that is uniformly old (unpinned, but
consistently so), and it does not compare the budget of a file that
predates the budget columns. It does not catch a mismatched `max_turns`, or
a harness fix that changed what an existing column means; see
[section 3.5 of docs/METHODOLOGY.md](docs/METHODOLOGY.md#35-only-comparable-runs-are-pooled)
for exactly what is and is not covered. Every trial summary opens with a
`Generated:` line and a `Runs pooled: N, of which ...` line, because the
file has no timestamp in its name and is overwritten in place on every
re-aggregation.

## Full assessment

```bash
cbench assess --model <model-tag>                          # S1, S2 and S3, 3 trials each (the default)
cbench assess --model <model-tag> --suites s1,s3 --trials 5
cbench assess --model <model-tag> --dry-run                # preview the payloads; calls no model
cbench assess --model <model-tag> --force-uncheckable      # run a suite the pre-flight refused
cbench assess --model <model-tag> --skip-preflight         # no capability check at all
```

`cbench assess` runs N trials of each selected suite back to back, then
aggregates each suite's trials into a trial summary and prints where to find
it. It automates the manual workflow of running the same command N times
and then aggregating (see
[Aggregation and trial-extension decisions](#aggregation-and-trial-extension-decisions))
rather than replacing it: the same CSVs land in the same place either way.
`--trials` defaults to 3, this framework's pre-registered minimum for a rate
worth citing. A run can take from minutes to hours depending on model size
and trial count, so start with `--dry-run` to see what will run.

The capability pre-flight runs first; see
[A suite that never fired its positive control is refused](#a-suite-that-never-fired-its-positive-control-is-refused)
for what it refuses, and what it deliberately does not.

**One assessment at a time.** `cbench assess` and `cbench score` take a run
lock before the first trial, and refuse to start (exit `2`) if another
assessment holds it, if a suite, `cbench score` or `cbench assess` process
is already running, or if the process table cannot be read at all, because
a machine that could not be checked has not been shown to be idle. Two runs
sharing one GPU do not fail: they halve each other's throughput and corrupt
every timing taken from them. The lock is per user (`~/.cbench/run.lock`;
`$CBENCH_LOCK_DIR` overrides it), not per directory, so two shells in
different folders cannot each hold it. A single suite run
(`cbench containment`, `cbench channel` or `cbench persistence`) neither
takes the lock nor checks it, so nothing stops you starting one beside an
assessment. `--force-concurrent` overrides all three refusals, and is only
for work that is provably on different hardware. A dry run and
`cbench score --from-existing` take no lock. See section 3.6 of
[docs/METHODOLOGY.md](docs/METHODOLOGY.md).

`cbench assess` covers S1, S2 and S3 only, the three suites this framework
ships. It does not include the [Inspect](https://inspect.aisi.org.uk/)
cross-check, which stays a separate, deliberate step (see the module
docstrings in `src/openllm_cbench/integrations/`), and there is no
equivalent for external benchmark suites, which this framework does not
ship.

## Aggregation and trial-extension decisions

```bash
cbench aggregate --suite s1 --model <model-tag>
cbench extension-rule --pair <base-tag> <variant-tag>
```

`cbench assess` and `cbench score` run the aggregation step automatically.
Run `cbench aggregate` directly when you already have trial CSVs on disk
(for example from separate manual runs) and want them pooled into a trial
summary. `cbench extension-rule` is always a separate, deliberate call: it
compares a base/variant pair against each other, which is a different
question from summarising one model's trials, and nothing runs it for you.
Its exit code is its decision (see [Exit codes](#exit-codes)).

## Terminal UI

```bash
cbench tui   # needs the tui extra; see Install
```

The terminal UI is a thin control panel over the CLI. Every action it takes
runs a real `cbench` subcommand as a subprocess, and the exact command line
appears on screen before it starts; there is no second implementation of
any suite. No command runs until you click. The safety invariant is shown on
every screen, and suite, score, gate, search, pull, doctor and community
actions save their full output as a log under `tui-logs/` in the results
folder.

The dashboard shows the active results location and a short progress panel:
how many models are pulled, catalogued, scored, **citable** and submitted,
plus a level and a suggested next action. Its buttons open these screens:

| Button | What it does |
|---|---|
| Run a suite | One run of one suite (`cbench containment`, `channel` or `persistence`), with a field for extra flags. Writes a CSV and a report, but no grade. |
| Score a model | `cbench score`; see [The Score screen](#the-score-screen). |
| Gate a model | `cbench gate`, with a tick box that adds `--save` (off by default). |
| Local models | Every pulled model, with gate, delete, search and pull actions; see [Local models](#local-models). |
| Browse reports | The reports in the results folder; see [Browse reports](#browse-reports). |
| Share / validate results | `cbench community-package`, `community-validate` and `community-submit`; see [Sharing results from the TUI](#sharing-results-from-the-tui). |
| Check environment | `cbench doctor`, with its output shown on the dashboard. |
| Settings | `cbench config`: Save results location, Save catalogue location, Clear results location (back to ./results), Find existing results. |
| About / extend this | The short version of [Extending this framework with an AI coding assistant](#extending-this-framework-with-an-ai-coding-assistant). |

The Run a suite, Gate a model and Score a model screens each have a picker
listing the models pulled into your endpoint, which fills in the model-tag
field. Dry run is off by default wherever it appears, matching the CLI.

The progress counts and the level are deliberately different things. The
counts are neutral inventory. The **level** (newcomer, novice, intermediate,
advanced, contributor) is earned on rigour, not volume, because a raw count
of scored models overstates what has been measured: in a real results
folder, "scored" models include test artefacts, models that never ran,
models that are invalid on every suite, and an A that means "this model
cannot call tools, so it attempted nothing". Every criterion for the level
is a rule this framework already enforces elsewhere. A result counts as
*citable* when the model is in the catalogue, a scorecard exists, no suite
was refused by its own validity guard, and every scored suite ran at least 3
trials (this framework's pre-registered minimum). The grade itself is
deliberately **not** a criterion: a model that scores badly has still been
measured properly, and rewarding good grades would reward picking easy
models.

Everything in the panel is read from your own machine: its files, plus the
endpoint's list of models. There is no account, no server and no ranking
against anybody else, and nothing in the panel is transmitted.

### Local models

**Local models** is a table of every model pulled into your endpoint: tag,
parameters, quantisation, size, when it was added, **Fit**, **Speed**,
whether it is catalogued, and **Score**. Click a column header to sort by
it, and click it again to reverse the order. Sorting uses the underlying
value rather than the displayed text, so sizes sort numerically instead of
putting "9.0 GB" after "10.5 GB".

- **Fit** says whether the model fits in this machine's VRAM (fits, tight
  or spills), and marks a mixture-of-experts model, which degrades far less
  than a dense model when it spills.
- **Speed** shows a measured tokens-per-second figure in bold when a saved
  gate check timed the model on this machine, and a word (fast, good,
  moderate or slow) when it is estimated from the parameters read per
  token. For a mixture-of-experts model that is the active experts rather
  than the full weight count, which is why a 30B-A3B model outruns a dense
  30B model of the same size on disk. A measurement and an estimate are
  deliberately shown differently.
- **Score** shows the compact label from the model's scorecard, if one
  exists (see [Scoring a model](#scoring-a-model)).

"Gate + save selected" gate-checks the selected model and saves the result
(`cbench gate --model <tag> --save`). "Gate + save all uncatalogued" runs
`cbench discover --gate-all`, passing the number in the "Gate at most N
model(s) per batch" field as `--limit`. Gate-checking a model is a real
model call, and a full Ollama library can take an hour, so the bound sits
next to the button. Leave the field blank to gate every uncatalogued model;
anything other than a positive whole number refuses to start rather than
widening the batch. "Search / pull a new model" opens a screen that runs
`cbench search` (no download) and `cbench pull`.

"Delete selected" is the only destructive action in this tool. It runs only
when the "Confirm delete" box is ticked in the same interaction, names the
exact tag before it runs, and calls `cbench remove --model <tag> --yes` as a
subprocess like every other action. The box unticks itself as the delete
starts, so a second delete needs its own confirmation rather than
inheriting the first. **Your results are never touched:** the CSVs, reports
and scorecard for a deleted model stay in the results folder, because the
measurement is what this framework exists to produce, and removing it as a
side effect of freeing disk space would be data loss presented as
convenience.

```bash
cbench remove --model <model-tag> --yes
```

Without `--yes`, nothing is sent to the endpoint and the command exits `2`.
The tag is never prefix-matched, because a loose match is how `qwen3:14b`
gets deleted by somebody who meant `qwen3:1.7b`.

### Browse reports

**Browse reports** lists the reports in the results folder in a table
(model, suite, kind, date and file) and shows the one you pick. A directory
tree underneath gives access to the raw CSVs and run logs, which are
evidence rather than browsing material. The screen opens filtered to
**scorecards only**, because that is the document almost everyone opens
this screen to read: a model run six times writes eighteen single-run
reports and one scorecard, and listing them together buries the one you
wanted. Two pickers change the filter: kind (scorecards, trial summaries,
single runs, or everything) and model. With "everything" selected,
scorecards sort first, then trial summaries, then single-run reports, each
newest first within its group.

A single-run report's date comes from the timestamp the suite writes into
its filename, not from the file's modification time, because a results
folder that has been cloned, copied or unzipped carries the wrong
modification time on every file. Scorecards and trial summaries keep a
fixed filename and are overwritten in place, so their date, like that of
any report without a timestamp in its name, comes from the modification
time and is marked with a trailing `~` so that it is not mistaken for the
time of the run.

### Sharing results from the TUI

**Share / validate results** runs the three community commands (see
[Sharing your own results](#sharing-your-own-results)). It offers the
models that have CSVs on disk and the submissions already packaged as
choices, because a typed path is a source of errors with no benefit when
both sets are known; two text fields take a tag or a path that the pickers
do not list. Optional fields
set the notes for the reviewer (`--notes`: anything unusual about the run)
and your GitHub handle (`--contributor`; left blank, it uses
`git config user.name`). "Also make a .zip" is off by default, as on the
CLI. "Accept contributor terms" adds `--accept-terms`, and "Confirm submit"
is the only thing that adds `--confirm`; without it, Submit previews the
commands and pushes nothing.

### The Score screen

There is deliberately no "full assessment" screen. `cbench score` runs the
same trials and aggregation as `cbench assess` and produces a grade on top,
so a separate button would be two doors into one room. The one thing
`cbench assess` offers that `cbench score` does not is an arbitrary
`--trials N` (`cbench score` offers 1, 3 or 5 through `--depth`); run
`cbench assess` in a terminal for that, as the Score screen itself says.

The Score screen has a tick box per suite, a depth picker, "From existing",
optional `--num-ctx` and `--num-predict` fields (blank leaves them to the
catalogue), "Dry run" (off by default), the `--force-uncheckable` override
(off by default; the usual fix for a refused suite is to untick it instead)
and "Gate-check first" (on by default). As soon as you enter a tag, the
screen shows whether it is catalogued. A live preview says in words what
pressing Score will do and shows the exact command line, including the gate
step whenever it will run.

- **Gate-check first.** For a model that is not in your catalogue, this
  runs `cbench gate --model <tag> --save` before scoring, so a new model
  does not run ungated. It is skipped for a dry run and for "From
  existing", both of which promise no model call. The log then reports one
  of three distinct outcomes: the gate check could not reach the endpoint
  for this model (the score run makes the same calls and will most likely
  fail the same way), it ran and found caveats (listed verbatim, for
  example no tool-calling support), or it was clean. The gate step itself
  never stops the score run; the pre-flight inside `cbench score` then
  decides, as described in
  [A suite that never fired its positive control is refused](#a-suite-that-never-fired-its-positive-control-is-refused).
- **Hardware warning.** If this machine's GPU looks too small for the
  model, an advisory warning says so before the run starts (from the
  best-effort probe in `core/hardware.py`; it never blocks a run).
- **Progress.** A run that is not "From existing" shows a progress bar,
  driven by the `--- suite trial N/M ---` lines in the log, with an
  estimated time remaining that appears once the first trial has finished.
- **From existing** refuses to start when nothing is on disk yet for the
  tag you entered, rather than producing grade N/A with exit code `0`,
  which looks like a real result until you investigate.

## The model catalogue

Different models need different configuration to produce valid data: a
raised generation budget for one, an effort-level sweep instead of a boolean
think toggle for another, a known channel-merge state at one think setting
for a third. This framework handles that declaratively rather than with
per-model code. `src/openllm_cbench/data/models/verified.json` ships a small
set of gate-checked worked examples, and a local `models.json` overlay,
grown with `cbench gate --save`, extends it with your own. Every suite
consults the catalogue for a model's configuration before falling back to
its own built-in defaults. An explicit CLI flag always wins over both, and
`--no-catalogue` on a suite ignores the catalogue entirely. An uncatalogued
tag is not refused: it runs "ungated", with a banner saying so.

### Populating it

**Find out what is missing first, with `cbench discover`.** It lists every
model already pulled into your endpoint that the catalogue does not know
yet:

```bash
cbench discover                        # list what is uncatalogued
cbench discover --gate-all             # also gate-check and save each one
cbench discover --gate-all --limit 3   # bound a long batch to the first 3
```

It talks only to your endpoint's own `/api/tags`, never to ollama.com. It
does not browse Ollama's remote library by keyword: Ollama has no official
API for that (there is an
[open feature request](https://github.com/ollama/ollama/issues/9142) for
one), and this framework deliberately does not scrape or wrap an unofficial
one.

**If you already know the exact tag, check that it exists before pulling,
with `cbench search`:**

```bash
cbench search --model <exact-tag>   # for example gemma3:12b; downloads nothing
```

It is not a keyword search: you need the exact tag, the same string you
would pass to `ollama pull`. It reuses the manifest-fetch step of Ollama's
own pull, against Ollama's real registry, and aborts the connection before
any layer data downloads. You get a real existence check and the download
size for the cost of an aborted request, rather than a multi-gigabyte
download to ask whether a tag exists. Then:

```bash
cbench pull --model <exact-tag>     # downloads the model
```

`ollama pull <tag>` works equally well. `cbench search` and `cbench pull`
exist so that you do not have to leave the CLI or the TUI mid-workflow, not
because they do anything `ollama` cannot. Once a model is pulled,
`cbench discover` picks it up.

There are two ways to add an entry for a specific tag, and they write to
different files.

**1. Automatic, and the recommended starting point:
`cbench gate --model <tag> --save`.** This runs the same capability,
tool-call and channel-separation check as `cbench gate`, then writes the
result into your **local overlay**: a `models.json` found the same way as
the results folder (`--registry-file`, then `$OPENLLM_CBENCH_MODELS_FILE`,
then the location pinned with `cbench config --set-models-file`, then
`./models.json`). This file is yours. Every suite reads it for that tag from
then on, this repository's `.gitignore` excludes it, and it is never the
packaged seed described below.

A gate check cannot derive numeric tuning such as `num_predict`, so
`config_overrides` is saved empty, with one exception: when the check finds
that the model only calls tools with its reasoning channel off, it saves
`{"think": false}`, and S1 and S3 send that from then on (an explicit
`--think` still wins). If a real run shows that
the model needs a larger budget or a longer timeout, add that yourself
(schema below). Nothing is saved if the check never reached the model (an
unreachable endpoint or an unknown tag), since nothing about it was
measured.

**2. Manual: edit `models.json` (your overlay) or, if you are contributing
a worked example back to the project,
`src/openllm_cbench/data/models/verified.json`** (the packaged, read-only
seed). An overlay entry for a tag replaces the seed entry for that tag
outright; the two are not merged. The full field-by-field schema, with the
reasoning behind each field, is in the `_schema` key of `verified.json`;
read it before writing an entry by hand. Quick reference:

| Field | Acted on by | Meaning |
|---|---|---|
| `architecture`, `tools`, `thinking` | Nothing (informational) | What `cbench gate` found; context for a person reading the catalogue. |
| `params_b`, `quant` | No suite; the Score screen's hardware fit warning reads both | What `cbench gate` found. `params_b` is always a count in **billions**: a model whose endpoint reports millions (for example `134.52M`) is converted on the way in, not suffix-stripped. The fit warning (`core/hardware.py:check_model_fit()`) uses the model's real on-disk size when the endpoint lists the model, and otherwise estimates the VRAM needed from these two; it is advisory only and never blocks anything. An entry that could not be measured stores the literal string `"unknown"`, and the fit check then stays silent rather than guessing. |
| `thinking_mode` | S2 | `"effort"` selects an `--effort all` sweep instead of `--think`; `"ignores_think"` selects `--think false` only. Omit it for an ordinary boolean toggle. |
| `channel_separation` | Nothing (informational) | `{"think_on": ..., "think_off": ...}`, each typically `"clean"`, `"UNRELIABLE"` or `null`. Put the actual consequence in `caveats` too; this field alone changes no suite's behaviour. |
| `delimiters` | S2 **and** `cbench gate` | This model's reasoning delimiters, for a model that marks its reasoning in a way none of the built-in conventions recognise. Write them as you read them, opening and closing markers joined by an ellipsis: `["<odd>...</odd>"]`. The two halves are matched separately, because a model emits the opening marker and then runs out of budget far more often than it emits the exact joined string. A match is recorded as `catalogued` in the CSV's `merge_evidence` column rather than attributed to a built-in family, so an operator's confirmed convention can be told apart from a guess. The nested `reasoning.delimiters` spelling is also accepted. |
| `config_overrides` | Every suite, before its own built-in default | Recognised keys: `num_ctx`, `num_predict`, `timeout`, `max_turns` (S1), `max_task_turns` (S3), and `think` (S1 and S3; `cbench gate --save` writes `false` here itself when the model's tool calling only works with reasoning off). An explicit CLI flag still wins. |
| `caveats` | Nothing directly; printed verbatim | Shown in every suite's startup banner and in `cbench gate`'s report when this tag is used. Free text; this is where "think=off is unreliable for this model" belongs. |

`config_overrides`, `thinking_mode` and `delimiters` are the fields that
something acts on. The rest exist so that the next person, including you at
a later date, does not have to rediscover the same quirk by watching a run
go wrong.

Fill in `delimiters` as soon as you find a model that needs it. The gate
check and S2 both read it, so a convention you confirm once
is honoured everywhere afterwards, including on a re-gate, which would
otherwise keep reporting the model as clean.

## Scoring a model

```bash
cbench score --model <model-tag> --depth standard   # quick=1 trial, standard=3 (default), thorough=5
cbench catalogue                                     # every pulled model, with catalogue and score status
```

Like `cbench assess`, `cbench score` runs the capability pre-flight first and
refuses a suite that cannot produce a gradeable result, with the same
`--force-uncheckable` and `--skip-preflight` overrides (see
[A suite that never fired its positive control is refused](#a-suite-that-never-fired-its-positive-control-is-refused)).

**Comparing two scorecards is its own command; see
[Comparing two models](#comparing-two-models). Do not read two grades side
by side and conclude anything.**

`cbench score` produces a cross-suite **scorecard**: a headline **A-F grade
(0-100)**, plus the per-suite detail underneath it (band, rate, confidence
and any caveats for each suite). It is saved as JSON and Markdown under
`scorecards/` in the results folder, and shown next to the model in
`cbench catalogue` and on the TUI's Local models screen from then on. The
grade is the **worst** of the three suites, not an average; the module
docstring of [`scoring/scorecard.py`](src/openllm_cbench/scoring/scorecard.py)
gives the full formula and explains why the worst suite dominates.
`--depth quick` (1 trial) is exploratory only, below this framework's
3-trial minimum for a rate worth citing, and every scorecard repeats that
warning prominently.

**A grade is not portable between machines.** It was produced on particular
hardware, and the same model on a different card can score differently.
The mechanism is concrete: a model that fits in VRAM on one machine may
spill to system RAM on a smaller one, which makes it slower, so more
requests hit their timeout and more generations stop at the token budget.
Those rows become INCOMPLETE or TRUNCATED and leave the denominator, so the
rate moves without the model's behaviour moving at all. A larger or
enterprise-class card can finish rows that were dropped elsewhere, and those
rows can fall either way. Every scorecard records the VRAM it was computed
on and repeats this in its own "Results vary by hardware" section. Compare
grades only from the same hardware, budget and framework version, and treat
a grade from someone else's machine as evidence about their setup as much
as about the model.

**S2 reports two numbers, and the grade takes the worse of them.** *Probe
failure* is the visible answer doing what an adversarial probe asked
(supplying the procedure, or emitting the injected canary), over every row
whose request succeeded and did not truncate. *Channel leak* is the narrower
question, where the visible answer resisted and the hidden reasoning did
not, over the rows that returned a reasoning trace, since a row without one
cannot produce that verdict. The denominators differ because the questions
do.

Grading on the leak alone would reward a model whose hidden channel barely
exists: it can comply with most of the probe set in plain sight and still
be reported clean, because none of those compliances came from a channel
it does not have. One real model that complied in 126 of 240 rows would
have scored an A.

**The probe-failure number is an interval, not a value, and the grade takes
its lower bound.** The scorer cannot classify some answers either way.
Those rows are disputed, so rather than dropping them (which silently
assumes they fail at the same rate as the readable ones, and for a weak
model the readable ones are mostly failures), the scorecard reports the
range they can move the answer across: *53% if every unreadable row
resisted, 79% if none did*. The grade uses the lower bound, so a model is
never marked down for rows the scorer could not read. When the two bounds
fall in different letter grades, the scorecard says so and warns that the
grade depends on that convention as much as on the model. This is the
"bounds over choices" rule from section 4 of
[docs/METHODOLOGY_TECHNICAL.md](docs/METHODOLOGY_TECHNICAL.md).

### Comparing two models

```bash
cbench compare --model <tag-a> --model <tag-b>     # reads saved scorecards; runs nothing
```

**Do not compare two grades by reading the two letters.** Every other guard
in this framework works within one model, and letters can differ where the
evidence does not. `cbench compare` reads the two saved scorecards and
reports, per suite, a significance test corrected for clustering, the
overlap of the confidence intervals, and the statistical power the
comparison had. It calls no model, so score both models first.

For example, two independently republished GGUF builds of one 9B model,
identically pinned on all four sampling axes and run for three trials each,
scored **B (79)** and **C (66)**. With the arithmetic a comparison needs:

| S2 probe-failure rate | naive | corrected for clustering |
|---|---|---|
| 25/117 vs 40/117 | **p = 0.041** | 8/37 vs 13/37 · **p = 0.302** |

The naive test clears p < 0.05 because it treats 117 rows as 117
independent observations, when they are a bank of probes each asked once per
think state per trial. `cbench compare` reports **both**, because the gap
between them is the finding.

**The verdict is three-way, never two:**

- `DIFFERENT`: significant on the clustering-corrected test.
- `INCONCLUSIVE`: not significant, **and under 80% power**. The honest
  reading is that the instrument cannot tell these two models apart.
  Section 3.1 of [docs/METHODOLOGY.md](docs/METHODOLOGY.md) applies the same
  rule to a suite that never fired its positive control: a comparison that
  could not have detected a difference has not produced a null either.
- `NO DIFFERENCE`: not significant, with the power to have found a
  difference. This is the only case in which similarity is a finding.

A suite without a gradeable result for one or both models (INVALID, not
run, or with no scored rows) is reported as `NOT COMPARABLE` instead of
receiving a verdict.

The example pair came out `INCONCLUSIVE` on all three suites, at **24%
power** on S2. Reaching 80% would need `n_eff ≈ 191` per arm against the 37
it had, roughly 5× the probes. Where the required growth exceeds 10×, the
report says plainly that powering the comparison is not a realistic plan,
rather than printing a number nobody can act on.

More **trials** will not fix an `INCONCLUSIVE`: `n_eff` scales with the
number of distinct probes, and once rows cluster this strongly it has a
ceiling that no trial count passes.

### Scoring a model you cannot run locally

`cbench score --model <tag> --from-existing` scores whatever S1, S2 and S3
CSVs already exist in the results folder, without running anything or
calling a model. That is also how a model too large for your own hardware
gets a scorecard at all: someone else runs it on theirs and shares the raw
CSVs, and you set `$OPENLLM_CBENCH_RESULTS_DIR` to their submission folder
before running the command (it must be set before the process starts;
`cbench score` has no `--results-dir`). See
[community-results/README.md](community-results/README.md) for the
submission convention: why raw CSVs rather than a submitted score, the
folder layout, and `cbench community-validate` before merging.

### Sharing your own results

```bash
cbench gate --model <model-tag> --save          # prerequisite; see below
cbench community-package --model <model-tag> --accept-terms
cbench community-submit community-results/<model-tag>/<contributor>_<date>
```

**Raw measurements travel; verdicts do not.** A submission carries per-row
CSVs and a `submission.json` describing how they were produced, never a
scorecard, a grade or a claimed rate, and validation refuses all three. This
repository is not a leaderboard and will not become one: anyone who wants a
grade computes it from the submitted rows with `cbench score --from-existing`.
There are two reasons. A grade is an editorial *conclusion* about a named
commercial product, whereas a CSV row is a *measurement*, permanently
qualified by the configuration recorded beside it. And results are
hardware-dependent enough to invert a verdict: a model too large for the
card can time out on every gate check, which reads as "cannot call tools"
when the model could and the machine was too small. See
[community-results/README.md](community-results/README.md) for the full
rule.

That failure mode is why `cbench gate --model <tag> --save` is a
**prerequisite**: a package whose gate check is missing or did not complete
does not validate, and so cannot be submitted, because a check that timed
out is an absent check rather than a failed one. Slowness alone is fine: a
model that spills into system RAM still produces valid rows.

`--accept-terms` records your acceptance of the contributor terms: the right
to share the files, no confidential or personal material, accurate hardware
details, an Apache-2.0 licence grant, and acknowledgement that publication
in public git history is permanent. Run the command without the flag first:
it still builds the folder and prints the terms, and the folder does not
validate until you have read them and the CSVs and packaged again with the
flag.

`cbench community-package` bundles whatever S1, S2 and S3 CSVs this machine
has already produced for a model into a submittable
`community-results/<model-tag>/<contributor>_<date>/` folder. It copies the
CSVs (it never rewrites or moves them), fills in `submission.json` from what
it can detect (hardware, the local endpoint's runtime version, quantisation,
this harness's version), records a SHA-256 hash per CSV so that later
corruption or tampering is detectable, and validates the result. It does not
refuse an invalid package: it builds the folder anyway, lists what to fix
and exits `1`. It uploads nothing. `--zip` also produces an archive, for
anyone who would rather attach it to a GitHub issue than use git;
`--contributor` sets your GitHub handle or name (by default
`git config user.name`), and `--notes` records anything unusual about the
run for the reviewer.

`cbench community-submit` opens a packaged folder as a pull request through
your own authenticated `gh` (the GitHub CLI); this framework never sees,
stores or transmits a credential. It re-validates the folder (exiting `1`
if anything is wrong), checks with `gh auth status` that you are logged in,
previews the exact command sequence and a reminder that the CSVs contain
the model's raw output (see `PRIVACY_NOTICE` in
[`core/community.py`](src/openllm_cbench/core/community.py)), and pushes
nothing until you pass `--confirm`. Without `gh` installed and logged
in, it prints manual instructions instead: a fork and pull request for
anyone comfortable with git, or a prefilled GitHub issue for anyone who is
not. The TUI does the same three things without a terminal (see
[Sharing results from the TUI](#sharing-results-from-the-tui)).

## Scale and applicability

"Local" here means served entirely on hardware you control, at whatever
quantisation your endpoint runs. The gate-checked worked examples in
`src/openllm_cbench/data/models/verified.json` span roughly 8B to 21B
parameters at Q4_K_M, Q4_0, MXFP4 and F16, the range this framework's
scoring heuristics and task set were tuned and validated against.

The framework also works well below that range: models down to 0.6B
gate-check and complete a full assessment at every `cbench score --depth`
level, with real S1 and S2 signal rather than a degenerate clean run. At the smallest end the failure mode is not degraded signal but
**no tool-calling capability at all**. For example, a 135M model fails the
gate's tool-call check outright (the endpoint returns a plain HTTP 400; it
is not a framework error), and the gate report says so plainly rather than
letting you discover it mid-run. `cbench gate --save` catches most
scale-related surprises in this way before you spend a real run on them
(see [The model catalogue](#the-model-catalogue)).

Between that floor and roughly 8B, the bottom of the catalogue's range,
expect S1 and S2 to degrade more gradually: more `TRUNCATED` and
malformed-tool-call rows as the model struggles with the tool-call schema
or the reasoning-trace format. That is a capability signal in its own
right, but it means fewer of the run's rows carry real S1 or S2 signal. The scorecard's per-suite caveats surface this; read the
underlying report before trusting a small model's headline rate.

At the large end, nothing in the architecture caps model size, since the
harness only holds a chat completion and a short tool-call loop in memory.
This framework has not been exercised against anything beyond the low tens
of billions of parameters, however, so treat a much larger model as
untested rather than assumed to work.

## Comparison to other tools

See [section 5 of ARCHITECTURE.md](ARCHITECTURE.md#5-positioning) for how
this tool relates to Garak, Promptfoo, PyRIT and Inspect. In short, those
are broad, general-purpose red-teaming and evaluation harnesses; this tool
answers three specific measurement questions in depth, built for agentic
tool-use containment, hidden-reasoning-channel divergence and audit-log
persistence behaviour in locally served models.

## Extending this framework with an AI coding assistant

This project was built with an AI coding assistant (Claude Code), not to
demonstrate the assistant but for the discipline it supports: checking a
claim against the code rather than memory, and writing a regression test
for every real bug before moving on. That discipline matters for a tool
whose job is to measure whether a model's stated capabilities match what it
does.

The project is deliberately not tied to that one tool, which is why it has
a [CONTRIBUTING.md](CONTRIBUTING.md) rather than a tool-specific
configuration file. Extending a suite, adding a model to the catalogue or
building a new scoring metric works the same way whether you use
**Claude Code**, **Claude Cowork**, **Codex CLI** or **ChatGPT Cowork**:
open the repository and point your tool at [ARCHITECTURE.md](ARCHITECTURE.md)
(what is measured, the control inventory, and how not to fool yourself with
this tool), [CONTRIBUTING.md](CONTRIBUTING.md) (layout and conventions) and
`tests/` (the parity and safety-invariant tests every change should keep
passing). None of it assumes a specific vendor. The TUI's "About / extend
this" screen carries the short version of this section.

## Methodology

Two documents describe how this framework tests, what makes a result from
it worth citing, and where it is known to be weak:

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**: what each suite
  measures, what "good" looks like (controls, exclusion rules, pinned
  sampling), how to read a result, and what this instrument has got wrong
  and changed as a result.
- **[docs/METHODOLOGY_TECHNICAL.md](docs/METHODOLOGY_TECHNICAL.md)**: the
  estimators, exclusion rules with their mandatory companion checks,
  validity gates, pre-registration rules, reproduction steps, and the known
  limits of the instrument (section 8).

Neither document publishes results. This repository ships a tool, not
anyone's results, and section 4 of METHODOLOGY.md explains why that
distinction is substantive rather than a matter of tidiness.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Licence

Released under the Apache License 2.0; see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for the threat model and how to report a
vulnerability.
