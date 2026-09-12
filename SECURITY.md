# Security

## What this is

`openllm-cbench` is a local testing framework that deliberately probes for
containment failures, hidden-reasoning-channel leakage, and deceptive
persistence behavior in locally-served open-weight LLMs. It is a
security-relevant *testing tool*, not itself an exploitable service — there
is no server component, no listening port beyond a loopback-only canary
each suite starts for itself, and no code path that acts on model output
beyond logging it and formatting a report.

Read [ARCHITECTURE.md](ARCHITECTURE.md) §2 ("Control inventory") for the
specific mechanisms this rests on. This document covers the threat model,
intended use, and how to report a problem.

## Threat model

**In scope — what this tool is built to resist:**

- A model under test attempting to use its tool access (`http_request`,
  `run_shell`, `read_local_file`) to reach outside its sandbox. This is the
  entire point of the containment and persistence suites — they exist to
  provoke and safely observe exactly this behavior.
- A model under test attempting to manipulate the harness itself through
  its responses — e.g. text in a tool result or final answer that looks
  like an instruction to the runner. The runner never re-parses model
  output as code or as a new instruction; it is read, scored, and logged
  as data at every point.

**Out of scope — what this tool does not defend against:**

- Running this framework against a chat endpoint you do not control or
  trust. The suites send your prompts and sandbox fixtures to whatever
  endpoint `--endpoint` / `$OPENLLM_CBENCH_ENDPOINT` points at, verbatim.
  Point it at a local model you're serving yourself.
- Vulnerabilities in the model-serving stack (Ollama, llama.cpp, vLLM,
  etc.) itself. This framework is a client of that stack, not a hardener
  of it.
- Multi-tenant or networked deployment. This is a single-operator, local
  CLI tool. It has never been reviewed for running as a shared service,
  and doing so is not a supported use case.

## Intended use

Run this against models you are testing, on infrastructure you control,
for defensive evaluation purposes — capability assessment before
deployment, regression testing after a fine-tune or ablation, coursework,
or research into these specific failure modes. The harmful-intent task
set (`data/tasks/containment_tasks_harmful.json`, gated behind
`--sandbox extended`) exists for the same purpose as the rest of the
framework: measuring whether a model *attempts* something, in a sandbox
built so the attempt cannot succeed. See ARCHITECTURE.md §6 for the full
rationale on why it ships rather than being withheld.

This tool is not intended for, and provides no capability toward, attacking
a system you do not control. It has no target-acquisition, exploitation,
or payload-delivery capability of any kind — every "attempt" it can
provoke is one this framework itself intercepts before it leaves the
process.

## What is and isn't exploitable

- **The loopback canary** (`core/canary.py`) binds `127.0.0.1` only and
  asserts that bind before serving. It is not reachable from another host
  on your network under normal operation.
- **`run_shell` never executes anything**, regardless of model output,
  regardless of task file, regardless of flags. There is no configuration
  of this framework that turns it into a real shell.
- **The chat endpoint is the main real trust boundary.** This framework
  sends data to whatever endpoint you configure and trusts its responses
  as model output (not as instructions to the harness itself — see
  "in scope" above). If you point `--endpoint` at something you don't
  control, you are extending trust to that endpoint, not to this
  framework.
- **`cbench pull` extends the same trust decision to whatever model tag
  you give it.** It downloads real data from Ollama's public registry via
  your local daemon's own `/api/pull` — identical risk to running `ollama
  pull <tag>` yourself, not a new capability this framework adds. Pulling
  a tag you don't recognize or don't trust is the same decision either way.
- **Report/CSV output** contains model text verbatim, unsanitized, written
  to your local `results/` directory (gitignored by default). Treat a
  report the same way you'd treat raw model output from any other tool
  before sharing it — it can contain whatever the model under test said,
  including anything it was provoked into saying by a harmful-intent probe.

## Reporting a vulnerability

If you find a way for this framework to make a real outbound connection
to somewhere other than the configured chat endpoint or its own loopback
canary, or a way for `run_shell` to actually execute a command, that is a
genuine break of this project's core safety claim and should be reported
as a security issue, not a regular bug.

**Report it privately via GitHub's own vulnerability reporting**, not a
public issue: open this repo's **Security** tab → **Report a
vulnerability**. That routes the report to the maintainer only, keeps it
out of the public issue tracker until there's a fix, and needs no email
address published in this file.

(Maintainer note: this requires "Private vulnerability reporting" to be
turned on for this repository — Settings → Security → Private
vulnerability reporting — before the button above will do anything. Turn
it on before this repo goes public, and swap this section for a security
contact email instead if you'd rather handle reports that way.)

For anything else — a scoring bug, a documentation gap, a feature
request — use the regular issue tracker; it doesn't need private handling.
