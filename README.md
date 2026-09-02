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

TODO: one paragraph — what problem this solves, who it's for, why it
exists as three specific suites rather than a general red-teaming
toolkit.

## What's here

| Suite | Command | Question it asks |
|---|---|---|
| Containment | `cbench containment` | Given real tools inside a sandboxed task, does the model reach beyond the boundary it was given? |
| Channel | `cbench channel` | Can a probe pass on the visible answer while the hidden reasoning trace fails it? |
| Persistence | `cbench persistence` | Challenged on a logged action, does the model try to rewrite the record rather than correct it honestly? |

Plus: trial aggregation and the pre-registered extension-rule decision
(`cbench aggregate`, `cbench extension-rule`), a guardrail-detection
scorer (`cbench guardrail`), a model gate-check (`cbench gate`), and
optional cross-validation against [Inspect](https://inspect.aisi.org.uk/)
(`integrations/`, install with `pip install "openllm-cbench[inspect]"`).

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

### Terminal UI

```bash
cbench tui   # requires: pip install "openllm-cbench[tui]"
```

A thin control panel over the same CLI above -- forms for running a
suite, gate-checking a model, browsing what's already pulled locally
(with a picker to fill the model field for you), pulling a new model,
and a browser for whatever `results/` already contains. It has no logic
of its own: every action it takes is the literal `cbench <subcommand>
...` invocation shown on screen before it starts, launched as a real
subprocess, not a second implementation of any suite. Nothing runs on
startup without a click.

### Aggregation and trial-extension decisions

```bash
cbench aggregate --suite s1 --model <model-tag>
cbench extension-rule --base <base-tag> --variant <variant-tag>
```

Every subcommand's real flags live in that module, not duplicated in
`cli.py` — run `cbench <subcommand> --help` for the full list.

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
ollama.com. It doesn't pull anything or search Ollama's remote library;
Ollama has no official API for that (there's an [open feature request](https://github.com/ollama/ollama/issues/9142)
for one). `ollama pull <tag>` a model yourself first, then `cbench
discover` picks it up.

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

## Scale and applicability

TODO: paragraph on what capability tiers this has actually been exercised
against, what "local" means in practice (quantization levels, parameter
ranges), and what to expect running it against a model well outside that
range.

## Comparison to other tools

See [ARCHITECTURE.md §5](ARCHITECTURE.md#5-positioning) for how this
relates to Garak, Promptfoo, PyRIT, and Inspect. Short version: those are
broad, general-purpose red-teaming/eval harnesses; this is three specific,
deep measurement questions, purpose-built for agentic tool-use containment,
hidden-reasoning-channel divergence, and audit-log persistence behavior in
locally-served models.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 — see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for the threat model and how to report a
vulnerability.
