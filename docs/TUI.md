# Terminal UI

```bash
cbench tui   # needs the tui extra; see docs/SETUP.md, Install cbench
```

The terminal UI is a thin control panel over the CLI. Every action it takes
runs a real `cbench` subcommand as a subprocess, and the exact command line
appears on screen before it starts; there is no second implementation of
any suite. No command runs until you click. The safety invariant is shown on
every screen, and suite, score, gate, search, pull and doctor actions save
their full output as a log under `tui-logs/` in the results
folder.

The dashboard shows the active results location and a short progress panel:
how many models are pulled, catalogued, scored and **citable**, plus a
level and a suggested next action. Its buttons open these screens:

| Button | What it does |
|---|---|
| Run a suite | One run of one suite (`cbench containment`, `channel` or `persistence`), with a field for extra flags. Writes a CSV and a report, but no grade. |
| Score a model | `cbench score`; see [The Score screen](#the-score-screen). |
| Gate a model | `cbench gate`, with a tick box that adds `--save` (off by default). |
| Local models | Every pulled model, with gate, delete, search and pull actions; see [Local models](#local-models). |
| Browse reports | The reports in the results folder; see [Browse reports](#browse-reports). |
| Check environment | `cbench doctor`, with its output shown on the dashboard. |
| Settings | `cbench config`: Save results location, Save catalogue location, Clear results location (back to ./results), Find existing results. |
| About / extend this | The short version of [Extending this framework with an AI coding assistant](../README.md#extending-this-framework-with-an-ai-coding-assistant). |

The Run a suite, Gate a model and Score a model screens each have a picker
listing the models pulled into your endpoint, which fills in the model-tag
field. "Preview only", the TUI's name for `--dry-run`, is off by default
wherever it appears, matching the CLI.

The progress counts and the level are deliberately different things. The
counts are neutral inventory. The **level** (newcomer, novice, intermediate,
advanced) is earned on rigour, not volume, because a raw count
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

## Local models

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
  exists (see [Scoring a model](USER_GUIDE.md#scoring-a-model)).

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

## Browse reports

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

## The Score screen

There is deliberately no "full assessment" screen. `cbench score` runs the
same trials and aggregation as `cbench assess` and produces a grade on top,
so a separate button would be two doors into one room. The one thing
`cbench assess` offers that `cbench score` does not is an arbitrary
`--trials N` (`cbench score` offers 1, 3 or 5 through `--depth`); run
`cbench assess` in a terminal for that.

The Score screen's form has five sections:

- **Model:** a picker listing the models pulled into your endpoint, or a
  field to type a tag. As soon as you enter a tag, the screen says whether
  the model is in your catalogue.
- **Suites:** a tick box per suite. Whenever the model changes, every
  suite is ticked except S1 and S3 when the model's catalogue entry records
  them as unable to run on it (both need working tool calls), and a
  one-line reason under the boxes says why. S2 stays ticked, because it
  grades any model that answers. You can tick S1 and S3 back, but the
  pre-flight then skips them anyway unless "Also run suites that cannot
  measure this model" is ticked (see
  [A suite that never fired its positive control is refused](USER_GUIDE.md#a-suite-that-never-fired-its-positive-control-is-refused)).
- **Depth:** Quick (1 trial per suite, 2 for S3), Standard (3 trials per
  suite, the minimum for a grade worth citing, and the default) or
  Thorough (5 trials per suite).
- **Generation budget (optional):** "Context window in tokens" and "Reply
  limit in tokens", sent as `--num-ctx` and `--num-predict`. Leave both
  blank for the automatic budget (see
  [Generation budgets](USER_GUIDE.md#generation-budgets)).
- **Options**, in this order:
  - "Gate-check the model first if it is not in your catalogue
    (recommended)", ticked by default.
  - "Re-score saved results only (runs nothing and calls no model)", which
    adds `--from-existing`; the depth and budget settings are then ignored.
  - "Preview only: show the requests without calling the model", which
    adds `--dry-run`; unticked by default.
  - "Also run suites that cannot measure this model (they grade INVALID;
    only useful for their transcripts)", which adds `--force-uncheckable`;
    unticked by default.

The form scrolls; the Score button, the preview and the log stay in place
below it. The preview says in words what pressing Score will do and shows
the exact command line, including the gate step whenever it will run.

- **Gate-check the model first.** For a model that is not in your
  catalogue, this runs `cbench gate --model <tag> --save` before scoring,
  so a new model does not run ungated. It is skipped with "Preview only"
  and with "Re-score saved results only", both of which promise no model
  call. The log then reports one of three distinct outcomes: the gate
  check could not reach the endpoint for this model (the score run makes
  the same calls and will most likely fail the same way), it ran and found
  caveats (listed verbatim, for example no tool-calling support), or it was
  clean. The gate step itself never stops the score run; the pre-flight
  inside `cbench score` then decides, as described in
  [A suite that never fired its positive control is refused](USER_GUIDE.md#a-suite-that-never-fired-its-positive-control-is-refused).
- **Hardware warning.** If this machine's GPU looks too small for the
  model, an advisory warning says so before the run starts (from the
  best-effort probe in `core/hardware.py`; it never blocks a run).
- **Progress.** A run that is not a re-score shows a progress bar, driven
  by the `--- suite trial N/M ---` lines in the log, with an estimated time
  remaining that appears once the first trial has finished.
- **Re-score saved results only** refuses to start when no results are
  saved yet for the tag you entered, rather than producing grade N/A with
  exit code `0`, which looks like a real result until you investigate.
- **A refused run says why.** When `cbench score` refuses to start (exit
  code `2`), the last two lines of the log are a `Why:` line, taken from
  the command's own `NOT STARTING` line, and the verdict, so the reason
  stays on screen after the rest of the output has scrolled away.
