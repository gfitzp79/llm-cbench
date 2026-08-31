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

A thin control panel over the same CLI above -- forms for running a suite
or a gate check, live-streamed output, and a browser for whatever
`results/` already contains. It has no logic of its own: every action it
takes is the literal `cbench <subcommand> ...` invocation shown on screen
before the run starts, launched as a real subprocess, not a second
implementation of any suite. Nothing runs on startup without a click.

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
