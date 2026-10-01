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

## Securing the model endpoint

cbench measures the model; it does not secure the server that runs it. On a
machine running this tool, the most likely exposure is the Ollama server
itself, so check it before anything else.

**Ollama has no authentication.** Anyone who can reach its port can use it
with no credentials. By default Ollama listens only on loopback
(`127.0.0.1:11434`), which only this machine can reach. It listens on every
network interface when `OLLAMA_HOST` is set to `0.0.0.0` (or to a LAN
address), or when "Expose Ollama to the network" is turned on in the Ollama
app's settings. Any device on the same network, and anything that can route
to it, can then use the server.

What an exposed server gives away:

- **Your compute.** Anyone on the network can run inference on your GPU.
- **Your models.** The API can pull and delete models (`/api/pull`,
  `/api/delete`), so someone else can fill your disk or remove what you have.
- **Your test traffic.** cbench's prompts, sandbox fixtures and the model's
  replies travel over plain HTTP. On loopback that traffic never leaves the
  machine; across a network, anyone on the path can read it.

What to do:

1. Run `cbench doctor`. For an endpoint on this machine, it reads the
   running server's own startup log and warns when the server listens beyond
   loopback.
2. Unless you need remote access, keep Ollama on loopback: turn off "Expose
   Ollama to the network" in the Ollama app, or set `OLLAMA_HOST=127.0.0.1`
   and restart Ollama.
3. If you do need Ollama on a network, put it behind something that
   authenticates, such as a reverse proxy with authentication, a VPN or an SSH
   tunnel, and restrict the port with a firewall. Point cbench at the endpoint
   through that route, which also gives its traffic encryption.

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
- **No command sends anything to a service other than the endpoint.**
  Every request cbench makes goes to the chat endpoint you configure or,
  inside a suite, to its own loopback canary. It has no upload, submission
  or telemetry path: results, CSVs and scorecards stay in your results
  folder until you move them yourself.
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

**Report it privately through GitHub's vulnerability reporting:** open
this repository's **Security** tab and choose
**Report a vulnerability**. The report goes to the maintainer only and
needs no email address published in this file.

This project does not accept contributions and has no issue tracker
([CONTRIBUTING.md](CONTRIBUTING.md)), so the private route above is for
security reports only.
