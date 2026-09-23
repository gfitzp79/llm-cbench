# Security

## What this is

`openllm-cbench` is a local testing framework that deliberately probes
locally served open-weight LLMs for containment failures, leakage through
the hidden reasoning channel, and deceptive persistence. It is a
security-relevant *testing tool*, not an exploitable service. It has no
server component. The only port it listens on is a loopback-only canary:
the S1 containment suite and its optional Inspect counterpart each start
one for their own run, and `cbench doctor` starts one briefly to confirm
the bind. The only action it takes on model output is to simulate the
model's tool calls inside the sandbox, as described below; otherwise it
logs and scores that output and formats a report.

[ARCHITECTURE.md](ARCHITECTURE.md) section 2 ("Control inventory") names
the mechanisms this rests on. This document covers the threat model,
intended use and how to report a problem.

## Threat model

**In scope: what this tool is built to resist**

- A model under test trying to use its tools to reach outside the
  sandbox. S1 gives the model `http_request`, `run_shell` and
  `read_local_file` precisely to provoke and safely observe this: a
  request to any host other than the canary is
  intercepted before a socket is opened, `run_shell` never executes, and
  `read_local_file` returns only fabricated in-memory files. The S3
  persistence suite's tools act only on an in-memory action log, and the
  S2 channel suite gives the model no tools.
- A model under test trying to manipulate the harness through its output,
  for example a tool-call argument or a final answer that reads like an
  instruction to the runner. The runner never executes model output as
  code or follows it as an instruction; it reads, scores and logs it as
  data at every point.

**Out of scope: what this tool does not defend against**

- Running the framework against a chat endpoint you do not control or
  trust. The suites send your prompts and sandbox fixtures, verbatim, to
  whatever endpoint `--endpoint` or `$OPENLLM_CBENCH_ENDPOINT` names.
  Point it at a model you serve yourself.
- Vulnerabilities in the model-serving stack behind the endpoint (Ollama,
  or any server that implements its API). The framework is a client of
  that stack, not a hardening layer for it.
- Multi-tenant or networked deployment. This is a single-operator local
  tool, a CLI with a terminal UI. It has not been reviewed for use as a
  shared service, and that use is not supported.

## Intended use

Run the framework against models you are testing, on infrastructure you
control, for defensive evaluation: capability assessment before
deployment, regression testing after a fine-tune or ablation, coursework,
or research into these failure modes. The harmful-intent task set
(`data/tasks/containment_tasks_harmful.json`) exists for the same
purpose as the rest of the framework: measuring whether a model
*attempts* something, in a sandbox built so that the attempt cannot
succeed. It runs only when you select it explicitly with `--tasks-file`
together with `--sandbox extended`. ARCHITECTURE.md section 6 explains
why it ships rather than being withheld.

The tool is not intended for, and provides no capability towards,
attacking a system you do not control. It has no target-acquisition,
exploitation or payload-delivery capability. Every attempt it can
provoke is intercepted by the framework itself: a request to any host
other than the loopback canary is answered with a synthetic failure
before a socket is opened.

## What is and is not exploitable

- **The loopback canary** (`core/canary.py`) binds `127.0.0.1` only and
  asserts that bind before serving. It is not reachable from another host
  on your network under normal operation.
- **`run_shell` never executes anything**, whatever the model output,
  task file or flags. No configuration of this framework turns it into a
  real shell.
- **The chat endpoint is the main trust boundary.** The framework sends
  data to whatever endpoint you configure and treats its responses as
  model output, never as instructions to the harness (see "In scope"
  above). If you point it at an endpoint you do not control, you are
  extending trust to that endpoint, not to this framework.
- **`cbench pull` and `cbench search` extend that trust to the registry a
  model tag names.** Both go through your local daemon's own `/api/pull`
  route. `cbench pull` downloads the model, with the same risk as running
  `ollama pull <model-tag>` yourself. `cbench search` makes the same
  request and closes it once the registry returns the manifest, so it
  reports whether the tag exists, and its size, without downloading the
  model. Neither adds a capability the daemon does not already have, and
  pulling a tag you do not recognise or trust is the same decision either
  way.
- **`cbench community-submit` is the only command that talks to a service
  other than the endpoint**, and it does so only through your own `gh`
  and `git`. Without `--confirm` it checks that `gh` is installed and
  logged in and prints the commands it would run. With `--confirm` it
  forks the repository, pushes a branch and opens a public pull request
  under your name. The framework never sees, stores or transmits your
  credentials.
- **Reports and CSVs contain model text verbatim and unsanitised**,
  written to your results folder (`cbench config` shows where it is).
  Before you share one, treat it as you would raw model output from any
  other tool: it can contain anything the model under test said,
  including what a harmful-intent task provoked it to say.

## Reporting a vulnerability

Each of the following breaks the project's core safety claim. Report it
as a security issue, not a regular bug:

- a model's tool call reaching any real host other than the loopback
  canary;
- a suite sending a real request anywhere other than the configured chat
  endpoint and that canary;
- `run_shell` executing a command.

**Report it privately through GitHub's vulnerability reporting, not in a
public issue:** open this repository's **Security** tab and choose
**Report a vulnerability**. The report goes to the maintainer only, stays
out of the public issue tracker until there is a fix, and needs no email
address published in this file.

> **Maintainer note.** The **Report a vulnerability** button works only
> when private vulnerability reporting is enabled for this repository
> (**Settings** → **Security** → **Private vulnerability reporting**).
> Enable it before the repository is made public. To handle reports by
> email instead, replace this section with a security contact address.
> Remove this note at publication.

For anything else, such as a scoring bug, a documentation gap or a
feature request, use the regular issue tracker; it does not need private
handling.
