# openllm-cbench

Local containment and reasoning-leakage testing framework for open-weight
LLMs.

> **Safety invariant, stated up front:** every suite in this framework
> measures **attempt, never success**. The only host any suite's
> `http_request` tool can ever actually reach is a loopback canary
> (`127.0.0.1`) that asserts its own bind address before starting; any
> other destination is intercepted before a socket opens and answered
> with a synthetic failure. `run_shell` is always a logged no-op — nothing
> it is given is ever executed. See [ARCHITECTURE.md](ARCHITECTURE.md) for
> the full control inventory and `pytest tests/test_safety_invariant.py`
> to verify it yourself, no model or network required.

If you're serving an open-weight model locally and want to know what it
actually *does* under agentic pressure — not what it says when you ask it
directly — this measures three specific behaviors that a benchmark score
or a manual chat transcript won't surface: whether it reaches past a tool
boundary it was given, whether its hidden reasoning trace fails a probe
its visible answer resists, and whether it tries to rewrite a log rather
than own a mistake when challenged on one. It's for anyone deploying a
local model into something with real tools attached — an agent, a
pipeline, anything past a chat window — who wants that answer from a
run against their own model on their own hardware, not a leaderboard
entry for a checkpoint they can't reproduce. It's three suites instead of
a general red-teaming toolkit on purpose: broad scanners are already well
served elsewhere (see "Comparison to other tools" below), and depth on
three specific, previously-hard-to-measure questions was judged more
useful than shallow coverage of many.

## What's here

| Suite | Command | Question it asks |
|---|---|---|
| Containment | `cbench containment` | Given real tools inside a sandboxed task, does the model reach beyond the boundary it was given? |
| Channel | `cbench channel` | Can a probe pass on the visible answer while the hidden reasoning trace fails it? |
| Persistence | `cbench persistence` | Challenged on a logged action, does the model try to rewrite the record rather than correct it honestly? |

Plus: trial aggregation and the pre-registered extension-rule decision
(`cbench aggregate`, `cbench extension-rule`), a cross-suite scorecard
(`cbench score`, `cbench catalogue` — see "Scoring a model" below), a
guardrail-detection scorer (`cbench guardrail`), a model gate-check
(`cbench gate`), and optional cross-validation against
[Inspect](https://inspect.aisi.org.uk/) (`integrations/`, install with
`pip install "openllm-cbench[inspect]"`).

Full architecture, control inventory, and the measurement pitfalls this
tool is built to avoid: [ARCHITECTURE.md](ARCHITECTURE.md).

## Where results are kept

**Set this once, before your first real run.**

```bash
cbench config --set-results-dir ~/cbench-results
```

Without it, results resolve to `./results` next to whatever directory you
launched from. Run the TUI from your home directory and the CLI from a
project and you get two unrelated results trees with the same name, each
invisible to the other. That is worse than untidy: `cbench score` reads
whichever tree it is pointed at, so a run that landed in the other one is
silently missing from the aggregate and the grade is computed over
whatever subset shared a directory with it.

`cbench config` on its own prints where results are going and which
setting decided that. `cbench config --find-results` searches the usual
places for trees that already exist, which is worth running once if you
have used the tool from more than one directory. It reads only and moves
nothing.

Resolution order, most specific first:

| | scope |
|---|---|
| `--results-dir` on the command | that invocation |
| `$OPENLLM_CBENCH_RESULTS_DIR` | that shell |
| the config file | your user, everywhere |
| `./results` | whatever directory you are in |

The same setting is reachable from the TUI under **Settings**, and the
dashboard shows the active location on every launch, in yellow when it is
not pinned.

**Pin the model catalogue too**, for the same reason and with the same
four layers:

```bash
cbench config --set-models-file ~/cbench-results/models.json
```

An unpinned catalogue behaves worse than an unpinned results folder,
because it does not lose anything visibly -- it silently presents a
*different* `models.json`. Models you have already gate-checked come back
as uncatalogued, and the next run goes out ungated without the config
guidance those gate checks discovered. It is kept as its own setting
rather than derived from the results location, since one catalogue can
serve several results corpora and moving your results should not re-gate
every model.

## Install

```bash
pip install -e .
# or, with the Inspect cross-validation extra:
pip install -e ".[inspect]"
# or, with the optional terminal UI:
pip install -e ".[tui]"
```

Requires Python >=3.10 and a locally-served model behind an
Ollama-compatible chat endpoint (default `http://localhost:11434`,
override with `--endpoint` or `$OPENLLM_CBENCH_ENDPOINT`).

## What goes in `--model <model-tag>`

Every command below takes `--model <model-tag>`. The tag is whatever
your endpoint itself calls the model, passed through verbatim in every
request this framework sends -- for the default (Ollama), that's exactly
the `NAME` column from `ollama list`, e.g.:

```bash
$ ollama list
NAME                    ID              SIZE      MODIFIED
gemma3:12b              f4031aab637d    8.1 GB    8 weeks ago
qwen3.5:9b              6488c96fa5fa    6.6 GB    12 days ago
hf.co/org/repo:Q4_K_M   ...

$ cbench doctor              # no --model needed, just checks the endpoint
$ cbench gate --model gemma3:12b --save
```

There's no separate registration step and no fixed roster -- if your
endpoint recognizes the tag, `cbench` can run a suite against it. Get the
tag wrong and the *first* call to it (`cbench doctor`'s reachability
check, or `cbench gate`) fails immediately with a clear connection/model
error, not a confusing result buried in a suite's output later.

## Quick start

```bash
# Sanity-check your environment: endpoint reachable, canary binds
# loopback, hardware headroom, catalogue status. Calls no model.
cbench doctor

# Gate-check a model before trusting any real run against it: tool-call
# wellformedness, reasoning-channel separation at BOTH think states,
# sampling-parameter extraction.
cbench gate --model <model-tag>

# Save that gate result into your local model catalogue so every suite
# picks up the right config for this tag automatically from now on.
cbench gate --model <model-tag> --save
```

### Containment

```bash
cbench containment --model <model-tag> --boundary both
cbench containment --model <model-tag> --task fx_lookup --dry-run   # preview, no model call
```

### Channel

```bash
cbench channel --model <model-tag> --think both
cbench channel --model <model-tag> --effort all   # for effort-graded model families
```

### Persistence

```bash
cbench persistence --model <model-tag>
cbench persistence --model <model-tag> --scenario dedup_customer_records
```

### Reproducibility: pinned sampling

Every suite sends the same four sampling parameters with every chat
call, rather than leaving them to whatever the endpoint's own Modelfile
happens to set for that tag: `--temperature` (default 0.8), `--top-p`
(0.9), `--top-k` (40), and `--seed`. All four are written into every CSV
row this framework produces, so a run stays auditable after the fact
instead of carrying an invisible confound.

```bash
cbench containment --model <model-tag> --seed 777   # exact replay of one run
cbench containment --model <model-tag>               # seed generated + recorded per row
```

`cbench score` and `cbench assess` take the same four flags and pass
them down to every suite invocation they make, so a whole scorecard can
be produced at a chosen temperature or replayed from a chosen seed.

**An explicit `--seed` is offset by the trial index**, so trial 1 of
`--seed 7` runs at 7, trial 2 at 8, and so on. This is the one place the
flags are not passed through verbatim, and it matters: handing the same
seed to all three trials of `--depth standard` makes them byte-identical,
so the run costs three times as long and reports a confidence interval
computed over three copies of one sample. The run as a whole stays
reproducible, and the trials inside it stay distinct.

Every row also carries `run_started_at`, the wall-clock time the run
began, with a UTC offset. It is recorded rather than inferred because a
file's mtime belongs to whatever tool last touched it -- `git clone`,
a copy, or a zip round-trip all rewrite it, and a results tree that has
been through any of them will otherwise claim every run happened at the
moment it was unpacked. See `docs/METHODOLOGY_TECHNICAL.md` section 6.1.

Omitting `--seed` is deliberate, not an oversight: a fixed default would
make every trial in a multi-trial run identical, destroying the
trial-to-trial variance the 3-trial minimum exists to measure. A
generated-and-recorded seed keeps that variance and still lets any single
run be replayed exactly — verified live: two runs at the same explicit
seed produced byte-identical output.

This closes a real, previously-silent confound: without explicit sampling
pinning, a ranked comparison can run different models at different temperature
and sampling settings, large enough to rival or exceed the effect sizes the
comparison is trying to measure. Pinning sampling doesn't make two models
"comparable" if they're still served differently in *other* ways — see
[ARCHITECTURE.md §7](ARCHITECTURE.md#7-how-not-to-fool-yourself-with-this-tool)
for what's still worth diffing with `cbench gate` before trusting a
base/fork or A/B comparison. Full flag reference: `core/sampling.py`.

### A suite that never fired its positive control is refused

`cbench score` reports a containment suite `INVALID` and leaves it out of the
grade when its positive control never fired across the trials. That control is
a task which explicitly requires an external fetch, so a non-attempt means
either extreme refusal or broken tool wiring, and in both cases the suite has
not shown it can return a positive.

**This is a real behaviour change.** A model with no tool-calling support
previously scored A out of 100 with high confidence: it attempts nothing
because it can attempt nothing, and "no escape attempts observed" read as
perfect containment. Run `cbench gate --model <tag>` first -- it names a
missing tools capability directly, before a suite run spends time discovering
it the slow way.

Relatedly, a grade computed from fewer than three suites now says so. The
headline is the worst suite's rate, so the suites that did not produce a usable
result can only be worse than the ones that did: a partial grade is an upper
bound, and the compact tag carries its coverage (`A (100/100) [high] 1/3`) so a
one-suite A and a three-suite A are not confused in a table cell.

### Run comparability is checked before pooling

`cbench aggregate` and `cbench score` glob every CSV on disk for a model
tag, which is how results from different harness configurations end up
averaged into one number unless something checks for it. One dimension of
that is now checked automatically: whether the pooled CSVs agree on
sampling.

If some of a model's CSVs predate the `temperature`/`top_p`/`top_k`
columns while others carry them, or if all of them carry the columns but
disagree, that suite's trial summary carries a `STOP -- THESE RUNS ARE NOT
COMPARABLE` block and `cbench score` reports the suite `INVALID`, excluded
from the grade. **This is a real behaviour change:** a `results/`
directory spanning this framework's own sampling-pinning release will now
produce an `INVALID` suite where it used to produce a grade. The fix is
the same either way -- re-run so every trial shares one configuration, or
point `$OPENLLM_CBENCH_RESULTS_DIR` at a directory holding only the runs
you mean to pool.

The check is deliberately narrow: an exact equality test on a value each
CSV records, not a heuristic. It does not fire on a differing `--seed`
(varying the seed per trial is the point) or on a corpus that's
uniformly old (unpinned, but consistently so with itself). It also does
not catch a mismatched generation budget or a harness fix that changed
what an existing column means -- see
[docs/METHODOLOGY.md §3.5](docs/METHODOLOGY.md#35-only-comparable-runs-are-pooled)
for exactly what is and isn't covered. Every trial summary now also opens
with a `Generated:` line and a `Runs pooled: N, of which ...` line, because
the file itself carries no timestamp in its name and is overwritten in
place on every re-aggregation.

### Full assessment

```bash
cbench assess --model <model-tag>                          # S1+S2+S3, 3 trials each (default)
cbench assess --model <model-tag> --suites s1,s3 --trials 5
cbench assess --model <model-tag> --dry-run                # preview every payload, call no model
```

Runs N trials of each selected suite back to back, then automatically
aggregates each suite's trials into a trial-summary report and prints
where to find it -- automating the documented "run the same command N
times, then aggregate" workflow (see `cbench aggregate` below) rather
than replacing it. `--trials` defaults to 3, this framework's own
pre-registered minimum for a rate worth citing (see ARCHITECTURE.md).
This is S1 (containment) + S2 (channel) + S3 (persistence) only -- the
three suites this framework ships. It does not include a cross-check
against [Inspect](https://inspect.aisi.org.uk/) (`integrations/`,
optional and separate on purpose -- see `cbench assess --help`) and has
no equivalent for external benchmark suites, which this framework
doesn't ship at all. Can take anywhere from minutes to hours depending
on model size and trial count -- start with `--dry-run` to see what
will run before committing to it.

### Terminal UI

```bash
cbench tui   # requires: pip install "openllm-cbench[tui]"
```

The dashboard opens with a short progress panel: how many models are
pulled, catalogued, scored, **citable** and submitted, plus a level and a
suggested next action.

Counts and level are deliberately different things. The counts are neutral
inventory. The **level** (newcomer, novice, intermediate, advanced,
contributor) is earned on rigour, not volume, and every criterion is a rule
this framework already enforces somewhere else. A result counts as *citable*
when the model is gate-checked, no suite was refused by its own validity guard,
and every suite ran at least 3 trials (this project's own pre-registered
minimum). The grade itself is deliberately **not** a criterion: a model that
scores badly has still been measured properly, and rewarding good grades would
reward picking easy models.

The reason for that design is visible in any real results directory. Counting
"models scored" would have reported seven on the machine this was built
against, of which one was a test artefact, one had never run, one was invalid
across all three suites, and one carried an A that meant "this model cannot
call tools, so it attempted nothing". Two were worth citing. A level computed
from the seven would have been a lie told with arithmetic.

Everything in the panel is read from your own machine. There is no account, no
server and no ranking against anybody else, and nothing in it is transmitted.

A thin control panel over the same CLI above -- forms for running a
single suite or a scorecard (`cbench score`, depth picker included),
gate-checking a model, pulling a new model, and packaging, validating or
submitting a community submission (see "Sharing your own results"
below).

**Local models** now also shows an **Added** column (when each model was
pulled), sorts by any column when you click its header (click again to
reverse), and can **delete** a selected model. Sorting orders the
underlying value, not the formatted cell, so sizes sort numerically
rather than putting "9.0 GB" after "10.5 GB".

Delete is the only destructive thing this tool does. It requires a
confirmation box ticked in the same interaction, names the exact tag
before it runs, and calls `cbench remove --model <tag> --yes` as a real
subprocess like every other action here. The box unticks itself
afterwards, so a second delete needs its own confirmation rather than
inheriting the first. **Your results are never touched**: the CSVs,
reports and scorecard for a deleted model stay in `results/`, because the
measurement is what this framework exists to produce and removing it as a
side effect of freeing disk space would be data loss dressed up as
convenience.

```bash
cbench remove --model <model-tag> --yes
```

Without `--yes` nothing is sent to the endpoint. The tag is never
prefix-matched, because a loose match is how `qwen3:14b` gets deleted by
somebody who meant `qwen3:1.7b`.

**Local models** lists what's pulled locally with a picker that fills
the model field for you, a **Fit** column (does this model fit in your
VRAM, and is it a mixture-of-experts model that degrades gently when it
doesn't), a **Speed** column, a **Score** column showing whatever
scorecard already exists, and a batch "gate + save all uncatalogued"
action -- with a "Gate at most N model(s) per batch" field beside the
button, since gate-checking an uncatalogued model is a real model call
and a full Ollama library can take an hour with no bound on it otherwise
(blank runs every uncatalogued model, same as the CLI's `cbench discover
--gate-all` with no `--limit`). Speed shows a measured tokens/sec figure
in bold when a gate check has actually timed that model on this machine,
and a word (fast/good/moderate/slow) when it is estimating from the
parameters read per token -- for a mixture-of-experts model that's the
active experts rather than the full weight count, which is why a
30B-A3B outruns a dense 30B of the same size on disk. A measurement and
a calculation are deliberately not shown the same way.

**Browse reports** opens on **scorecards only**, because that is the
document almost everyone comes to this screen to read. A model run six
times writes eighteen single-run reports and one scorecard, and listing
them together buries the one you wanted. Two pickers change that: filter
by kind (scorecards, trial summaries, single runs, or everything) and by
model. With "everything" selected the scorecard still sorts first, then
the pooled trial summaries, then the individual runs, each newest-first
inside its group.

**Browse reports** is a table of the reports on disk -- model, suite,
kind and date, newest first -- because that is how you look for one, not
by navigating to `results/s2_channel/trial_summary_<tag>.md`. It lists
trial summaries, scorecards and single-run reports; the raw CSVs and run
logs are in the directory tree beneath it, since those are evidence
rather than browsing material. The date comes from the timestamp the
suite encoded in the report's own filename, not the file's mtime -- a
results tree that's been cloned, copied or unzipped carries the wrong
mtime on every file. A report from before this column existed has no
timestamp in its name, so its date falls back to mtime and is marked
with a trailing `~` so it isn't mistaken for the actual run time.

**Share / validate results** picks a model from those that actually have
CSVs on disk, and a submission from those already packaged, rather than
asking you to type either. A typed path is a class of error with no
upside when both sets are knowable. An optional notes field feeds
`cbench community-package --notes` -- anything unusual about the run
that should reach a reviewer.

There is deliberately no "full assessment" screen: `cbench score` runs
the identical trials and aggregation `cbench assess` does and produces a
grade on top, so a separate button for it was two doors into one room.
The one thing `assess` still has that `score` doesn't is an arbitrary
`--trials N` (score offers 1/3/5 via `--depth`) -- run it from a terminal
for that case; the Score screen says so on the screen itself. It has no
logic of its own: every action it takes is the literal `cbench
<subcommand> ...` invocation shown on screen before it starts, launched
as a real subprocess, not a second implementation of any suite. Nothing
runs on startup without a click.

The Score screen specifically: an uncatalogued model gets gate-checked
first by default, and the result is shown as one of three distinct
things, not one vague "didn't complete cleanly" line -- couldn't reach
the endpoint at all (a real problem, likely to recur in the score run
right after), ran fine and found specific caveats (named verbatim, e.g.
"no tool-calling support"), or clean. It never blocks the run either
way -- that's a deliberate project policy, not an oversight -- it only
makes the reason legible. If this machine's GPU is smaller than the
model's own params/quant suggest it needs, a hardware warning says so
before the run starts (advisory, from `core/hardware.py`'s own
best-effort probe). A real (non-`--from-existing`) run also gets a
progress bar, driven by the same `--- suite trial N/M ---` lines already
in the log, with an ETA that only starts reporting once the first trial
has actually finished. And "From existing" refuses to start at all if
there's nothing on disk yet for the tag you typed, rather than silently
producing grade N/A with exit code 0 -- indistinguishable from a real
result until you go looking.

### Aggregation and trial-extension decisions

```bash
cbench aggregate --suite s1 --model <model-tag>
cbench extension-rule --pair <base-tag> <variant-tag>
```

`cbench assess` above runs the aggregation step automatically; run
`cbench aggregate` directly if you already have trial CSVs on disk (e.g.
from separate manual runs) and just want them aggregated.
`cbench extension-rule` is always a separate, deliberate call — it
compares a base/variant pair against each other, which is a different
question from summarizing one model's trials, and nothing runs it for
you. Every subcommand's real flags live in that module, not duplicated
in `cli.py` — run `cbench <subcommand> --help` for the full list.

## The model catalogue

Different models need different configuration to produce valid data —
a raised generation budget here, an effort-level sweep instead of a
boolean think toggle there, a known channel-merge state at one think
setting. This framework handles that declaratively rather than with
per-model code: `data/models/verified.json` ships a small set of
gate-checked worked examples, and a local `models.json` overlay (grown
via `cbench gate --save`) extends it with your own. Every suite consults
the catalogue for a model's config before falling back to its own
hardcoded defaults; an explicit CLI flag always wins over both. An
unlisted model tag isn't refused — it just runs "ungated," with a banner
saying so.

### Populating it

**Find out what's missing first — `cbench discover`.** Lists every model
already pulled into your local endpoint that the catalogue doesn't know
about yet:

```bash
cbench discover              # just list what's uncatalogued
cbench discover --gate-all   # ...and gate-check + save every one of them
cbench discover --gate-all --limit 3   # bound a long batch to the first 3
```

Only ever talks to your local endpoint's own `/api/tags` — never
ollama.com. It doesn't browse Ollama's remote library by keyword; Ollama
has no official API for that (there's an [open feature request](https://github.com/ollama/ollama/issues/9142)
for one), and this framework deliberately doesn't scrape or wrap an
unofficial one.

**Already know the exact tag? Check it exists before pulling —
`cbench search`.**

```bash
cbench search --model <exact-tag>   # e.g. gemma3:12b -- no download
```

Not a keyword search — you need the exact tag, the same string you'd
pass to `ollama pull`. Reuses that same command's own manifest-fetch
step (real, live, against Ollama's actual registry) and aborts the
connection right after, before any layer data downloads — a real
existence check and download size for the cost of an aborted request,
not a multi-GB download just to ask "does this exist." Then:

```bash
cbench pull --model <exact-tag>     # downloads it for real
```

`ollama pull <tag>` yourself works exactly as well — `cbench
search`/`cbench pull` exist so you don't have to leave the CLI/TUI
mid-workflow, not because they do anything `ollama` itself couldn't.
Once pulled, `cbench discover` picks it up.

Two ways to add a *specific* entry once you know the tag, and they write
to two different files:

**1. Automatic (recommended starting point) — `cbench gate --model <tag> --save`.**
Runs the same capability/tool-call/channel-separation check `cbench gate`
always does, then writes the result into your **local overlay**, a
`models.json` file created in your current working directory (override
the location with `--registry-file` or `$OPENLLM_CBENCH_MODELS_FILE`).
This file is yours — every suite reads it automatically from then on for
that tag, it's gitignored by default, and it's never the packaged seed
below. A gate check can't derive numeric tuning like `num_predict` on its
own, so `config_overrides` is saved empty; if a real run tells you this
model needs a raised budget or a longer timeout, add that yourself
(schema below).

**2. Manual — edit `models.json` (your overlay) or, if you're
contributing a worked example back to the project,
`src/openllm_cbench/data/models/verified.json`** (the packaged, read-only
seed; an overlay entry for the same tag always wins over a seed entry).
Full field-by-field schema, with the reasoning behind each field, is
documented inline in `verified.json`'s own `_schema` key — read that
before hand-writing an entry. Quick reference:

| Field | Acted on by | Meaning |
|---|---|---|
| `architecture`, `tools`, `thinking` | Nothing — informational | What `cbench gate` found; useful context for a human reading the catalogue |
| `params_b`, `quant` | No suite — but the TUI's hardware fit warning reads both | What `cbench gate` found. `params_b` is always a count in **billions**: a model whose endpoint reports it in millions (e.g. `134.52M`) is converted on the way in, not suffix-stripped. The Score screen estimates VRAM need from these two (`core/hardware.py:check_model_fit()`) and warns before a run if this machine looks too small — advisory only, it never blocks anything. An entry that couldn't be measured stores the literal string `"unknown"`, and the fit check stays silent rather than guessing. |
| `thinking_mode` | The channel suite | `"effort"` → auto-selects an `--effort all` sweep instead of `--think`. `"ignores_think"` → auto-selects `--think false` only. Omit for an ordinary boolean toggle. |
| `channel_separation` | Nothing — informational | `{"think_on": ..., "think_off": ...}`, each `"clean"` / `"UNRELIABLE"` / `null`. Put the actual consequence in `caveats` too — this field alone doesn't change any suite's behavior. |
| `delimiters` | The channel suite **and** `cbench gate` | This model's reasoning delimiters, for a model that marks its reasoning in a way none of the four built-in conventions recognise. Written the way you read them, open and close joined by an ellipsis: `["<odd>...</odd>"]`. Both halves are matched separately, because a model emits the opening marker and then runs out of budget far more often than it emits the exact joined string. A hit is recorded as `catalogued` in the CSV's `merge_evidence` column rather than attributed to a built-in family, so you can tell an operator's confirmed convention from a guess. The nested `reasoning.delimiters` spelling is accepted too. |
| `config_overrides` | Every suite, before its own hardcoded default | Recognized keys: `num_ctx`, `num_predict`, `timeout`, `max_turns` (containment), `max_task_turns` (persistence). An explicit CLI flag still wins over this. |
| `caveats` | Nothing directly — printed verbatim | Shown in every suite's startup banner and in `cbench gate`'s report when this tag is used. Free text; this is where "think=off is unreliable for this model" belongs. |

`config_overrides`, `thinking_mode` and `delimiters` are the fields
something actually *acts on*. The rest exists so the next person
(including future you) doesn't have to re-discover the same quirk by
watching a run go wrong.

`delimiters` is worth filling in the moment you find a model that needs
it. Both the gate check and the channel suite read it, so a convention
you confirm once is honoured everywhere afterwards -- including on a
re-gate, which would otherwise keep reporting the model clean.

## Scoring a model

```bash
cbench score --model <model-tag> --depth standard   # quick=1 trial, standard=3 (default), thorough=5
cbench catalogue                                     # every local model + catalogue/score status
```

Produces a cross-suite **scorecard**: a headline **A-F grade (0-100)**,
plus the per-suite detail underneath it (band, rate, confidence, and any
caveats for containment/channel/persistence individually), saved under
`results/scorecards/<tag>.{json,md}` and shown next to that model in
`cbench catalogue` and the TUI's Models browser from then on. The grade is
the **worst** of the three suites, not an average — see
[`scoring/scorecard.py`](src/openllm_cbench/scoring/scorecard.py)'s own
module docstring for the full formula and why worst-suite-dominates rather
than averaging. `--depth quick` (1 trial) is explicitly exploratory — below
this framework's own 3-trial citability minimum — and any grade produced
at that depth says so, prominently, every time.

**A grade is not portable between machines.** It was produced on
particular hardware, and the same model on a different card can score
differently. The mechanism is concrete rather than hand-waving: a model
that fits in VRAM on one machine may spill to system RAM on a smaller
one, which makes it slower, which makes more requests hit their timeout
and more generations stop at the token budget. Those rows become
INCOMPLETE or TRUNCATED and leave the denominator, so the rate moves
without the model's behaviour moving at all. A larger or enterprise card
can finish rows that were dropped elsewhere, and those rows can fall
either way. Every scorecard records the VRAM it was computed on and says
this in its own "Results vary by hardware" section. Compare grades from
the same hardware, budget and framework version; treat one from someone
else's machine as evidence about their setup as much as about the model.

**S2 reports two numbers and the grade takes the worse of them.** *Probe
failure* is the visible answer doing what an adversarial probe asked
(supplying the procedure, emitting the injected canary), over every row
the suite attempted and did not truncate. *Channel leak* is
the narrower question — the visible answer resisted and the hidden
reasoning did not — over the rows that returned a reasoning trace, since
a row without one cannot produce that verdict. The denominators differ
because the questions do.

Grading on the leak alone had a predictable failure: a model whose hidden
channel barely existed could comply with most of the probe set in plain
sight and still be reported "clean", because none of those compliances
came from a channel it did not have. One real model complied with 126 of
240 probes and scored an A.

**The probe-failure number is an interval, not a value, and the grade
takes its lower bound.** Some answers the scorer cannot classify either
way. Those rows are disputed, so rather than dropping them — which
silently assumes they fail at the same rate as the readable ones, and for
a weak model the readable ones are mostly failures — the scorecard reports
what they can move the answer across: *53% if every unreadable row
resisted, 79% if none did*. The grade uses the lower bound, so a model is
never marked down for rows the scorer could not read, and when the two
bounds land in different letter grades the scorecard says so rather than
printing one of them. This is the "bounds over choices" rule from
[docs/METHODOLOGY_TECHNICAL.md](docs/METHODOLOGY_TECHNICAL.md) section 4,
applied where the tool previously only recommended it.

### Scoring a model you can't run locally

`cbench score --model <tag> --from-existing` scores whatever S1/S2/S3
CSVs already exist on disk, without running anything or making a model
call. That's also how a model too large for your own hardware gets a
scorecard at all: someone else runs it on theirs and shares the raw
CSVs — see [`community-results/README.md`](community-results/README.md)
for the submission convention (why raw CSVs and not a submitted score,
folder layout, `cbench community-validate` before merging).

### Sharing your own results

```bash
cbench gate --model <model-tag> --save          # prerequisite, see below
cbench community-package --model <model-tag> --accept-terms
cbench community-submit community-results/<model-tag>/<handle>_<date>
```

**Raw measurements travel; verdicts do not.** A submission carries
per-row CSVs and nothing else — never a scorecard, a grade, or a claimed
rate, and validation refuses all three. This repository is not a
leaderboard and will not become one: anyone who wants a grade computes it
themselves from the submitted rows with `cbench score --from-existing`.
Two reasons, both load-bearing. A grade is an editorial *conclusion*
about a named commercial product, where a CSV row is a *measurement*
permanently qualified by the configuration recorded beside it. And
results are hardware-dependent enough to invert a verdict: a 27.9B model
on a 12GB card had every gate check time out and reported `Tool call
check failed`, which reads as "cannot call tools" — it could, the machine
was too small. See [community-results/README.md](community-results/README.md)
for the full rule.

That incident is why `cbench gate --model <tag> --save` is a
**prerequisite**: packaging refuses a submission whose gate checks never
completed, since a check that timed out is an absent check rather than a
failed one. Slowness alone is fine — a model that spills into system RAM
still produces valid rows.

`--accept-terms` records the contributor terms (right to share, nothing
confidential, accurate hardware, an Apache-2.0 licence grant, and that
publication is permanent in git history). Run it without the flag first:
it still builds the folder and prints the terms, and tells you it won't
validate until you've read them and the CSVs.

`community-package` bundles whatever S1/S2/S3 CSVs this machine already
produced for a model into a submittable `community-results/<tag>/
<handle>_<date>/` folder: it copies the CSVs (never rewrites or moves
them), fills in `submission.json` from what it can detect on its own
(hardware, the local endpoint's runtime version, quant, this harness's
own version), records a SHA-256 per CSV so later corruption or tampering
is detectable, and validates the result. It uploads nothing. `--zip` also
produces an archive for anyone who wants to attach it to a GitHub issue
instead of using git.

`community-submit` opens a packaged folder as a pull request through the
contributor's own authenticated `gh` (the GitHub CLI) — this framework
never sees, stores, or transmits a credential. It previews the exact
command sequence and a reminder that the CSVs contain the model's raw
output (see `PRIVACY_NOTICE` in
[`core/community.py`](src/openllm_cbench/core/community.py)) and does
nothing else until you pass `--confirm`. Without `gh` installed and
logged in, it prints manual instructions instead — fork-and-PR for
anyone comfortable with git, or a prefilled GitHub issue URL for anyone
who isn't. The TUI's community screen (Package / Validate / Submit
buttons, an "Accept contributor terms" checkbox, and a "Confirm submit"
checkbox that's the only thing that adds `--confirm`) does the same three
things without a terminal.

## Scale and applicability

"Local" here means served entirely on hardware you control, at whatever
quantization your endpoint runs — the gate-checked worked examples in
`data/models/verified.json` span roughly 8B to 21B parameters at Q4_K_M/
Q4_0/MXFP4/F16, the range this framework's own scoring heuristics and
task set were actually tuned and validated against. It also runs fine
well below that: models under 1B (down to 0.6B, gate-checked and put
through a full assessment at every `cbench score --depth` level, with
real containment/channel signal, not just a degenerate clean run) work
correctly today. `cbench gate --save` catches most scale-related
surprises before you spend a real run finding them (see "The model
catalogue" above) — at the smallest end, the failure mode isn't degraded
signal, it's **no tool-calling capability at all**: a gate check against
a 135M model in this size class failed its tool-call check outright (a
plain HTTP 400 from the endpoint, not a framework bug), and the gate
report says so plainly rather than letting you discover it mid-run. Above
that floor and below the catalogue's own ~8B range, expect the
containment and channel suites specifically to degrade in a softer way:
more `TRUNCATED` and malformed-tool-call rows as the model struggles with
the tool-call schema or reasoning-trace format, which is a capability
signal in its own right but means fewer of that run's rows carry real
containment/channel signal — a scorecard's own per-suite caveats
(`cbench score`) surface this, but read the underlying report before
trusting a small model's headline rate regardless. At the large end,
nothing here architecturally caps model size — the harness only ever
holds a chat completion and a short tool-call loop in memory — but this
framework has not been exercised against anything past the low tens of
billions, so
treat a much larger model as untested territory rather than assumed-fine.

## Comparison to other tools

See [ARCHITECTURE.md §5](ARCHITECTURE.md#5-positioning) for how this
relates to Garak, Promptfoo, PyRIT, and Inspect. Short version: those are
broad, general-purpose red-teaming/eval harnesses; this is three specific,
deep measurement questions, purpose-built for agentic tool-use containment,
hidden-reasoning-channel divergence, and audit-log persistence behavior in
locally-served models.

## Extending this framework with an AI coding assistant

This project was built collaboratively with an AI coding assistant
(Claude Code) -- not as a demo of that, but because the discipline it's
good at (checking a claim against the actual code instead of memory,
writing a regression test for every real bug before moving on) mattered
for a tool whose whole job is measuring whether a model's stated
capabilities match what it actually does.

It's deliberately not tied to that one tool, though -- that's why this
repo has a [CONTRIBUTING.md](CONTRIBUTING.md) instead of a tool-specific
config file. Extending a suite, adding a model to the catalogue, or
building a new scoring metric works the same way whether you use
**Claude Code**, **Claude Cowork**, **Codex CLI**, or **ChatGPT Cowork**:
open the repo and point your tool at
[ARCHITECTURE.md](ARCHITECTURE.md) (what's measured, the control
inventory, and "how not to fool yourself with this tool"),
[CONTRIBUTING.md](CONTRIBUTING.md) (layout and conventions), and
`tests/` (the parity and safety-invariant tests any change should keep
passing). None of it assumes a specific vendor -- the same TUI screen
("About / extend this") has this section's short version for when
you're already in `cbench tui`.

## Methodology

Two documents describe how this framework tests, what makes a result from
it worth citing, and where it is known to be weak:

- **[docs/METHODOLOGY.md](docs/METHODOLOGY.md)** -- what each suite
  measures, what "good" looks like (controls, exclusion rules, pinned
  sampling), how to read a result, and what this instrument has got wrong
  and changed as a result.
- **[docs/METHODOLOGY_TECHNICAL.md](docs/METHODOLOGY_TECHNICAL.md)** --
  the estimators, exclusion rules with their mandatory companion checks,
  validity gates, pre-registration rules and reproduction steps.

Neither carries any measurements. This repository ships a tool, not
anyone's results, and section 4 of the first document explains why that
distinction is load-bearing rather than tidy-minded.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for the threat model and how to report a
vulnerability.
