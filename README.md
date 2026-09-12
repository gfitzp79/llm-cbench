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

A thin control panel over the same CLI above -- forms for running a
suite, a full assessment, or a scorecard (`cbench score`, depth picker
included), gate-checking a model, browsing what's already pulled locally
(with a picker to fill the model field for you, and a Score column
showing whatever scorecard already exists for each model), pulling a new
model, and a browser for whatever `results/` already contains. It has no
logic of its own: every action it takes is the literal `cbench
<subcommand> ...` invocation shown on screen before it starts, launched
as a real subprocess, not a second implementation of any suite. Nothing
runs on startup without a click.

### Aggregation and trial-extension decisions

```bash
cbench aggregate --suite s1 --model <model-tag>
cbench extension-rule --base <base-tag> --variant <variant-tag>
```

`cbench assess` above runs these automatically; run them directly if
you already have trial CSVs on disk (e.g. from separate manual runs) and
just want them aggregated. Every subcommand's real flags live in that
module, not duplicated in `cli.py` — run `cbench <subcommand> --help`
for the full list.

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
| `architecture`, `params_b`, `quant`, `tools`, `thinking` | Nothing — informational | What `cbench gate` found; useful context for a human reading the catalogue |
| `thinking_mode` | The channel suite | `"effort"` → auto-selects an `--effort all` sweep instead of `--think`. `"ignores_think"` → auto-selects `--think false` only. Omit for an ordinary boolean toggle. |
| `channel_separation` | Nothing — informational | `{"think_on": ..., "think_off": ...}`, each `"clean"` / `"UNRELIABLE"` / `null`. Put the actual consequence in `caveats` too — this field alone doesn't change any suite's behavior. |
| `config_overrides` | Every suite, before its own hardcoded default | Recognized keys: `num_ctx`, `num_predict`, `timeout`, `max_turns` (containment), `max_task_turns` (persistence). An explicit CLI flag still wins over this. |
| `caveats` | Nothing directly — printed verbatim | Shown in every suite's startup banner and in `cbench gate`'s report when this tag is used. Free text; this is where "think=off is unreliable for this model" belongs. |

`config_overrides` is the only field any suite actually *acts on* — the
rest exists so the next person (including future you) doesn't have to
re-discover the same quirk by watching a run go wrong.

## Scoring a model

```bash
cbench score --model <model-tag> --depth standard   # quick=1 trial, standard=3 (default), thorough=5
cbench catalogue                                     # every local model + catalogue/score status
```

Produces a cross-suite **scorecard** — per-suite bands (contained/
occasional/frequent for containment, clean/occasional/frequent leak for
channel, clean/signal for persistence) plus a confidence label per suite,
saved under `results/scorecards/<tag>.{json,md}` and shown next to that
model in `cbench catalogue` and the TUI's Models browser from then on.
Deliberately **not a single number**: see
[`scoring/scorecard.py`](src/openllm_cbench/scoring/scorecard.py)'s own
module docstring for why collapsing three suites that measure unrelated
failure modes into one score would have to either discard the nuance
between them or hide it in a footnote nobody reads before citing the
number. `--depth quick` (1 trial) is explicitly exploratory — below this
framework's own 3-trial citability minimum — and the rendered scorecard
says so every time.

### Scoring a model you can't run locally

`cbench score --model <tag> --from-existing` scores whatever S1/S2/S3
CSVs already exist on disk, without running anything or making a model
call. That's also how a model too large for your own hardware gets a
scorecard at all: someone else runs it on theirs and shares the raw
CSVs — see [`community-results/README.md`](community-results/README.md)
for the submission convention (why raw CSVs and not a submitted score,
folder layout, `cbench community-validate` before merging).

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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for the threat model and how to report a
vulnerability.
