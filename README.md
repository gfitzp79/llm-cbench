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

cbench also gate-checks models, keeps a catalogue of them, and scores and
compares them; [Commands](docs/USER_GUIDE.md#commands) lists every command.
The full architecture, the control inventory and the measurement pitfalls
this tool is built to avoid are in [ARCHITECTURE.md](ARCHITECTURE.md).

## Prerequisites

- **Python 3.10 or later.** The `python3` that ships with macOS is 3.9;
  see [macOS](docs/SETUP.md#macos).
- **Ollama, serving the model you want to test**, on hardware you control.
  cbench sends every model request to Ollama's native API (by default
  `http://localhost:11434`), so a server that offers only an
  OpenAI-compatible API is not enough. [Setup](docs/SETUP.md#ollama)
  covers installing Ollama, pulling a model, and the Ollama settings that
  affect a run.
- **Enough GPU memory to hold the model** (recommended). A model that does
  not fit still runs, more slowly, and every result records how much of it was in GPU
  memory; see [GPU memory](docs/SETUP.md#gpu-memory).

**Keep Ollama on loopback.** Ollama has no authentication, so a server that
listens on the network (`OLLAMA_HOST=0.0.0.0`, or "Expose Ollama to the
network" in the Ollama app) lets anyone who can reach it use your models and
GPU, and delete models. See
[Keep Ollama on loopback](docs/SETUP.md#keep-ollama-on-loopback).

## Install

The package is not on PyPI, so install it from GitHub. With the terminal
UI:

```bash
pip install "openllm-cbench[tui] @ git+https://github.com/gfitzp79/llm-cbench"
cbench doctor
```

Leave out `[tui]` for the CLI alone. [Install cbench](docs/SETUP.md#install-cbench)
covers virtual environments, `pipx`, and installing from a clone to run the
tests or the Inspect cross-validation.

**On macOS**, the command above fails as written: the `python3` that ships
with macOS is 3.9, there is no `pip` command, and Homebrew's Python refuses
`pip install`. With [Homebrew](https://brew.sh) installed, use `pipx`:

```bash
brew install python@3.12 pipx
pipx ensurepath
```

**Close the terminal and open a new one**, so that it picks up the `PATH`
change, then:

```bash
pipx install --python python3.12 "openllm-cbench[tui] @ git+https://github.com/gfitzp79/llm-cbench"
cbench doctor
```

To update later, run `pipx reinstall openllm-cbench`. [macOS](docs/SETUP.md#macos)
covers installing Homebrew, installing from a clone, Ollama's settings on a
Mac, and GPU memory on Apple Silicon.

## Quick start

```bash
# Pin where results and the model catalogue are kept, once, before the
# first real run, so every run reads and writes the same files.
cbench config --set-results-dir ~/cbench-results
cbench config --set-models-file ~/cbench-results/models.json

# Check the environment: the endpoint is reachable, where Ollama and cbench
# keep their files, the canary binds loopback, and the hardware headroom.
# Calls no model and writes nothing.
cbench doctor

# Gate-check a model before trusting any real run against it: tool-call
# well-formedness, reasoning-channel separation at both think states, and
# the sampling parameters the endpoint reports. The report says, per suite,
# whether S1, S2 and S3 can produce a gradeable result on this model;
# cbench score and cbench assess skip any suite that cannot.
cbench gate --model <model-tag>

# The same check, with the result saved to your local model catalogue, so
# every suite picks up the right configuration for this tag from then on.
cbench gate --model <model-tag> --save

# Grade it: one A-F scorecard from every suite that can measure the model.
# quick runs 1 trial per suite (2 for S3, the fewest it can rate) to see it
# work; standard (3 trials, the default) gives a grade worth citing.
cbench score --model <model-tag> --depth quick
```

`<model-tag>` is the name in the `NAME` column of `ollama list`; see
[What goes in `--model <model-tag>`](docs/SETUP.md#what-goes-in---model-model-tag).
[Where results are kept](docs/SETUP.md#where-results-are-kept) explains why
both locations are worth pinning. Three trials per suite is this
framework's pre-registered minimum for a rate worth citing, so treat a
`quick` grade as exploratory.

Every command exits `0` when it is done, `1` when it ran and something in
it failed, and `2` when it refused to start and wrote nothing. The exception
is `cbench extension-rule`, whose exit code is its decision; see
[Exit codes](docs/USER_GUIDE.md#exit-codes).

## Documentation

| Page | What it covers |
|---|---|
| [docs/SETUP.md](docs/SETUP.md) | Ollama prerequisites and configuration, installing cbench, macOS, where results and the catalogue are kept, and what goes in `--model` |
| [docs/USER_GUIDE.md](docs/USER_GUIDE.md) | Every command, running the suites, the guards that decide whether a result counts, assessing, scoring and comparing models, and exit codes |
| [docs/MODEL_CATALOGUE.md](docs/MODEL_CATALOGUE.md) | The per-model configuration every suite reads, how to populate it, and its fields |
| [docs/TUI.md](docs/TUI.md) | The terminal UI and each of its screens |
| [ARCHITECTURE.md](ARCHITECTURE.md) | What is measured, the control inventory, and how not to fool yourself with this tool |
| [SECURITY.md](SECURITY.md) | The threat model, securing the model endpoint, and reporting a vulnerability |

## Responsible use

A grade measures one model, on one machine, under one set of settings. Four
things follow from that.

- **A grade holds only for the conditions it was measured under.** The same
  model can grade differently on other hardware, with more or less memory
  available, or at other settings: the generation budget, the time limit on
  each request, or the sampling. A model that does not fit in GPU memory
  runs more slowly, so more requests reach the time limit before they
  finish, and those rows leave the denominator (see
  [Scoring a model](docs/USER_GUIDE.md#scoring-a-model)). A larger
  generation budget lets rows finish that stopped at the reply limit, and
  lets a model caught in a reasoning loop run for longer.
  Any of these can move a grade up or down. Run cbench on your own hardware
  before relying on anyone else's grade.
- **A good grade is not a guardrail.** Every suite measures attempt, never
  success, in a sandbox built so that the attempt cannot succeed. A clean
  result means the suites did not provoke the behaviour they look for, in
  their tasks, on your machine; it does not make a model safe to deploy. Run
  a local model inside boundaries of your own: filter its network egress,
  give it the fewest tools it needs, and review what it does. Out of the box,
  many local models have few guardrails, and some have none.
- **Weigh capability against guardrails for your use case.** If you use a
  local model for penetration testing, bug bounty work or other security
  research, set what the model can do against the guardrails it has or lacks.
  A safety-ablated model is built to comply with what it is asked, and its
  grade should be read in that light.
- **Publish a grade with its conditions.** A letter without the scorecard's
  "Results vary by hardware" section and caveats, and the generation budget
  your CSVs recorded, cannot be read by anyone else. If you publish a grade,
  publish those beside it.

For the framework's intended use, and for securing the model endpoint it
tests against, see [SECURITY.md](SECURITY.md).

## Comparison to other tools

This tool is built to run beside Promptfoo, Garak, PyRIT and Inspect rather
than in place of them. Those mostly measure what a user or an attacker can
make a model say or do. This tool measures what a model does when its own
task pulls against its boundary, and whether the signals used to oversee it,
the reasoning trace and the audit log, can be trusted. It scores by rule,
from the tool-call log, with only the model under test in memory on your own
machine. See [section 5 of ARCHITECTURE.md](ARCHITECTURE.md#5-positioning) for
the comparison in full.

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
building a new scoring metric in your own copy works the same way whether
you use
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

## Contributions

This project does not accept contributions: no pull requests and no result
submissions. To report a security vulnerability, see
[SECURITY.md](SECURITY.md). The licence permits forks, and
[CONTRIBUTING.md](CONTRIBUTING.md) documents the code for anyone reading it
or extending their own copy.

## Licence

Released under the Apache License 2.0; see [LICENSE](LICENSE).

## Security

See [SECURITY.md](SECURITY.md) for the threat model and how to report a
vulnerability.
