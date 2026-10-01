# The model catalogue

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

## Populating it

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

A gate check does not write numeric tuning such as `num_predict`; the
`thinking` field it records is what sets a model's automatic generation
budget (see [Generation budgets](USER_GUIDE.md#generation-budgets)). `config_overrides`
is saved empty, with one exception: when the check finds that the model
only calls tools with its reasoning channel off, it saves
`{"think": false}`, and S1 and S3 send that from then on (an explicit
`--think` still wins). If a real run shows that the model needs a
different budget or a longer timeout, add that yourself (schema below).
Nothing is saved if the check never reached the model (an unreachable
endpoint or an unknown tag), since nothing about it was measured.

**2. Manual: edit `models.json`** (your overlay), never
`src/openllm_cbench/data/models/verified.json` (the packaged, read-only
seed). An overlay entry for a tag replaces the seed entry for that tag
outright; the two are not merged. The full field-by-field schema, with the
reasoning behind each field, is in the `_schema` key of `verified.json`;
read it before writing an entry by hand. Quick reference:

| Field | Acted on by | Meaning |
|---|---|---|
| `architecture` | The Score screen's hardware fit warning | The model family `cbench gate` found. |
| `tools` | Nothing (informational) | Whether the endpoint reported a `tools` capability; context for a person reading the catalogue. |
| `thinking` | Every suite's automatic generation budget | Whether the endpoint reported a `thinking` capability when `cbench gate` checked the model. `true` gives the model the larger automatic budget unless reasoning is switched off (see [Generation budgets](USER_GUIDE.md#generation-budgets)). |
| `suite_readiness` | The TUI's Score screen | The per-suite verdict and reason from the gate check (the same verdicts the pre-flight uses). The Score screen unticks S1 and S3 when they are recorded as unable to run on the model, and ignores a recorded S2 verdict. |
| `params_b`, `quant` | No suite; the Score screen's hardware fit warning reads both | What `cbench gate` found. `params_b` is always a count in **billions**: a model whose endpoint reports millions (for example `134.52M`) is converted on the way in, not suffix-stripped. The fit warning (`core/hardware.py:check_model_fit()`) uses the model's real on-disk size when the endpoint lists the model, and otherwise estimates the VRAM needed from these two; it is advisory only and never blocks anything. An entry that could not be measured stores the literal string `"unknown"`, and the fit check then stays silent rather than guessing. |
| `thinking_mode` | S2 | `"effort"` selects an `--effort all` sweep instead of `--think`; `"ignores_think"` selects `--think false` only. Omit it for an ordinary boolean toggle. |
| `channel_separation` | Nothing (informational) | `{"think_on": ..., "think_off": ...}`, each typically `"clean"`, `"UNRELIABLE"` or `null`. Put the actual consequence in `caveats` too; this field alone changes no suite's behaviour. |
| `delimiters` | S2 **and** `cbench gate` | This model's reasoning delimiters, for a model that marks its reasoning in a way none of the built-in conventions recognise. Write them as you read them, opening and closing markers joined by an ellipsis: `["<odd>...</odd>"]`. The two halves are matched separately, because a model emits the opening marker and then runs out of budget far more often than it emits the exact joined string. A match is recorded as `catalogued` in the CSV's `merge_evidence` column rather than attributed to a built-in family, so an operator's confirmed convention can be told apart from a guess. The nested `reasoning.delimiters` spelling is also accepted. |
| `config_overrides` | Every suite, ahead of the automatic generation budget and the suite's own defaults | Recognised keys: `num_ctx`, `num_predict`, `timeout`, `max_turns` (S1), `max_task_turns` (S3), and `think` (S1 and S3; `cbench gate --save` writes `false` here itself when the model's tool calling only works with reasoning off). An explicit CLI flag still wins. |
| `caveats` | Nothing directly; printed verbatim | Shown in every suite's startup banner and in `cbench gate`'s report when this tag is used. Free text; this is where "think=off is unreliable for this model" belongs. |

`config_overrides`, `thinking`, `thinking_mode` and `delimiters` change
what a run does, and `suite_readiness`, `params_b` and `quant` inform the
Score screen. The rest exist so that the next person, including you at
a later date, does not have to rediscover the same quirk by watching a run
go wrong.

Fill in `delimiters` as soon as you find a model that needs it. The gate
check and S2 both read it, so a convention you confirm once
is honoured everywhere afterwards, including on a re-gate, which would
otherwise keep reporting the model as clean.

A catalogue entry records the gate check as it was when the entry was
saved, and updating cbench does not rewrite it. The gate check never
records S2 as unable to run, so an entry that does, or that carries a
caveat saying S2 will come back INVALID, is out of date: the Score screen
ignores that S2 verdict, but every suite's startup banner still prints the
caveat. Re-gate the model to refresh the entry:

```bash
cbench gate --model <model-tag> --save
```

`--save` replaces the whole entry, so add back any field you edited by
hand, such as `delimiters` or `config_overrides`.
