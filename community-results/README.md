# Community-submitted results

Not every model fits on every machine. If you cannot run a model
yourself, someone who can may already have shared results for it, and
you can share results for the models you run. This folder holds those
results: **raw trial CSVs, submitted by pull request and scored by each
reader with this framework's own scoring before anyone relies on a
number.**

## Why raw CSVs, not a submitted score

A submitted score with no data behind it cannot be checked: anyone could
type in any number. A submitted CSV is the per-row output that this
framework's suites write, in the format `cbench aggregate` and
`cbench score` read. As a result:

- Anyone can run the same scoring code
  (`cbench score --model <model-tag> --from-existing`) against the
  submitted CSVs and get the same scorecard, independently of the
  contributor.
- Every validity guard the framework applies to your own results applies
  to a submission in the same way: the task-set and CSV schema-version
  checks, the checks for mixed or unrecorded sampling and generation
  budgets across pooled CSVs, and the positive- and negative-control
  diagnostics. An incomparable submission is flagged, not averaged in.
- The S2 channel and S3 persistence suites use heuristic matching, so a
  row they flag still needs its transcript read before it counts as
  confirmed. Community data does not change that.

## The publication rule

**Raw measurements travel. Verdicts do not.**

A submission carries per-row CSVs and nothing else. It never carries a
scorecard, a grade or a claimed rate. `cbench community-validate`
refuses any file whose path contains `scorecard` or `trial_summary`, in a
folder name or a file name, and any `submission.json` that sets a `grade`,
`score`, `band` or `rate` field.

This repository is **not a leaderboard and will not become one.** It
publishes no table of model scores. Anyone who wants a grade computes it
locally (the PowerShell form is under "How a maintainer reviews one"):

```bash
OPENLLM_CBENCH_RESULTS_DIR=community-results/<model-tag>/<contributor>_<date> \
  cbench score --model <model-tag> --from-existing
```

There are three reasons, and none of them is a matter of style:

1. **A grade is a conclusion about a named commercial product; a CSV row
   is a measurement taken on one machine.** The first is an editorial
   claim that can be wrong in public and harm a real company. The second
   records what happened, permanently qualified by the configuration
   recorded beside it. Only the second is safe for a stranger to hand a
   maintainer, and only the second is something a maintainer can stand
   behind.

2. **Results depend on hardware in ways that can invert a verdict.** On a
   12 GB card, every gate check of a 27.9B model timed out, and the
   report read `Tool call check failed`, which reads as "this model
   cannot call tools". The model could call tools; the machine was too
   small. A submission built from that run would have published a false
   claim about a named product, in good faith, produced by this
   framework's own tooling. That is why validation refuses a submission
   whose gate check did not complete (see the next section).

3. **A grade discards what makes it defensible:** the confidence
   interval, the caveats, the validity guards and the "attempt, never
   success" framing. The nuance is the rigorous part; the letter is the
   part that travels badly and gets quoted alone.

## What a submission must carry

`cbench community-package` writes all of this for you.

- **The required metadata.** `submission.json` needs non-empty `model`,
  `contributor`, `date`, `hardware_summary` and `endpoint` fields, plus the
  `gate_check` and `attestation` records described in the next two items.
  Packaging fills the five fields from the tag, `--contributor` (or
  `git config user.name`), the packaging date, a hardware probe and the
  endpoint's `/api/version`. `date` is the day the submission was
  packaged; each CSV row records when its own run started. Packaging also
  records the optional `quant` (from your catalogue), `cbench_version` and
  `notes` (your `--notes`). A required field it cannot detect, such as
  `endpoint` when the endpoint is unreachable during packaging, stays
  empty and fails validation until you fill it in.
- **A completed gate check.** `cbench gate --model <model-tag> --save`
  must have run to completion on the machine that produced the rows.
  Packaging copies what your catalogue records into `gate_check`, and
  validation refuses a submission with no gate check or with any check
  that did not complete. A check that *timed out* is not a failed check
  but an **absent** one, and rows produced on a machine that could not
  complete the checks may describe the machine rather than the model.
  Slowness alone is fine: a model that spills into system RAM still
  produces valid rows, only more slowly.
- **Contributor terms, explicitly accepted** (`--accept-terms`, or the
  "Accept contributor terms" checkbox in the TUI). You confirm that you
  have the right to share the files; that they contain no confidential,
  personal or proprietary material, and that you have read the raw CSVs
  and not only the summary; and that the recorded hardware, runtime and
  model tag are accurate. You grant this project a perpetual, irrevocable
  licence to publish and redistribute the files under the repository's
  Apache-2.0 licence. You also acknowledge that the submission is raw
  data claiming no score, and that publication is **permanent**: git
  history keeps a file after it is deleted. Packaging prints the full
  text. `attestation.accepted` is `false` until you package with
  `--accept-terms`, and validation refuses a submission whose terms were
  not accepted, or were accepted in an older version.
- **A SHA-256 per CSV**, so that corruption in transit or an edit after
  review is detectable, and a merged submission can be checked against
  what was reviewed. A hand-assembled submission may omit these; see
  "Folder layout".

## If you want something removed

Open an issue or contact the maintainer. Submissions live in git history,
so removal means rewriting history rather than deleting a file: it is
possible, but it is not instant, and it needs a maintainer. Decide what
you are comfortable publishing **before** you submit. That is why the
packaging step prints the raw-output warning and asks you to accept the
terms, rather than leaving them in a document.

If you believe a submission misrepresents a model you are responsible
for, say so in an issue, with specifics. The raw CSVs let a disputed
claim be re-examined against the actual rows rather than argued in the
abstract, which is the other reason data travels and verdicts do not.

## What it does not do

This is a folder in a git repository, reviewed by pull request. It is
**not** a hosted leaderboard, a live scoring service or an automatic
trust mechanism. A merged submission means that a maintainer reviewed it
(`cbench community-validate`, then a manual read of `submission.json`
against the CSVs, then `cbench score --from-existing` to see the
resulting scorecard) and judged it worth keeping on file. It does not
mean that its numbers are independently verified. Apply the caveats you
would apply to your own results before citing anything from here.

## Folder layout

```
community-results/
  <model-tag>/
    <contributor>_<YYYYMMDD>/
      submission.json          <- required; see SUBMISSION_TEMPLATE.json
      s1_containment/          <- include whichever suites you ran
        containment_<model-tag>_<timestamp>.csv
      s2_channel/
        channel_<model-tag>_<timestamp>.csv
      s3_persistence/
        persistence_<model-tag>_<timestamp>.csv
```

In folder and file names, `<model-tag>` stands for the model tag with
`:` and `/` replaced by `-`, the transform
(`scoring/aggregate.py:model_tag()`) the framework applies to every file
name it writes: `gemma3:12b` becomes `gemma3-12b`. In a `--model`
argument it is the tag itself, as recorded in the `model` field of
`submission.json`. `<contributor>` is the `--contributor` value, or
`git config user.name`, with spaces and `/` replaced by `-` (`anon` when
neither is set, and validation then reports the empty `contributor`
field), and the date is the day the folder was packaged.
`cbench community-package` writes the folder under `./community-results`
in the current directory; `--out` changes that root.

Keep the CSV file names that `cbench containment`, `cbench channel` and
`cbench persistence` gave them. `cbench community-validate` checks that
each name contains the model tag from `submission.json`.

`submission.json` may also carry a `checksums` key: `{relative CSV path:
sha256}` for every CSV in the submission. `cbench community-package`
fills it in. A hand-assembled submission can leave it out entirely, which
is not an error but leaves the files unverifiable. If the key is present,
it must be complete and current: `cbench community-validate` reports a
checksum that does not match the file on disk, a listed file that is
missing, and a CSV that is present but not listed.

You do not need all three suites. A submission with only
`s3_persistence/` is still useful: the scorecard shows the suites you did
not run as "not run", not as failures.

## How to submit

1. Gate-check the model and save the result:

   ```bash
   cbench gate --model <model-tag> --save
   ```

   This is a prerequisite. Validation refuses a submission whose gate
   checks did not complete, for the reason given under "The publication
   rule".

2. Run the suites you can against your own local endpoint, as you would
   for yourself. `cbench score --model <model-tag> --depth standard` runs
   3 trials per suite, the framework's pre-registered minimum for a rate
   worth citing; `cbench assess` and the individual suite commands also
   work. You can submit fewer trials, but say so plainly: the framework's
   own trial summaries call a rate from fewer than 3 trials "a
   wider-uncertainty version of the single-shot number, not a settled
   rate".

3. Package the CSVs:

   ```bash
   cbench community-package --model <model-tag> --accept-terms
   ```

   This builds `community-results/<model-tag>/<contributor>_<date>/`
   from the CSVs already on disk for that model. It copies them (never
   moving or rewriting them), fills in `submission.json` with what it can
   detect, records a SHA-256 per CSV and validates the result. It does not
   refuse an invalid package: it builds the folder anyway, prints anything
   that still needs a hand edit before the submission is valid, and exits
   `1` while anything is left to fix. It uploads nothing. Add `--contributor <your-handle>` if
   `git config user.name` is not the name you want on the submission, and
   `--notes "<text>"` for anything a reviewer should know, such as a
   `config_overrides` you needed or trials you excluded.

   Run it without `--accept-terms` first. It still builds the folder,
   prints the raw-output warning and the contributor terms, and reports
   that the submission will not validate until you accept them. Read the
   CSVs, then run it again with `--accept-terms`. Every run rewrites
   `submission.json`, so keep `--accept-terms` on any later run,
   including one that adds `--zip`.

4. Open the pull request:

   ```bash
   cbench community-submit community-results/<model-tag>/<contributor>_<date>
   ```

   This validates the folder, refuses it if anything is wrong, and opens
   it as a pull request through your own authenticated `gh` (the GitHub
   CLI). The framework never sees, stores or transmits your credentials.
   Without `--confirm` it prints the exact command sequence and the
   raw-output reminder, and pushes nothing; run it again with `--confirm`
   to submit. If `gh` is not installed or not logged in, it prints two
   manual routes instead:

   - **Fork and pull request**, for anyone with git: fork the repository,
     clone your fork, copy the packaged folder in at the path shown,
     commit and push it, and open the pull request yourself.
   - **Attach to an issue**, for anyone without git: run the packaging
     command again with `--zip` added (keeping `--accept-terms`) to get
     `<contributor>_<date>.zip` beside the folder, then open the
     prefilled issue URL that `community-submit` printed and attach the
     zip. This needs only a GitHub account and a browser.

5. Whichever route you use, say in the pull request or issue which model
   and runtime you used and roughly what you saw. A maintainer runs
   `cbench score --from-existing` on it regardless, but knowing what to
   expect speeds up review.

The TUI's **Share / validate results** screen covers steps 3 and 4
without a terminal. It has **Package**, **Validate** and **Submit**
buttons; optional notes and contributor fields (a blank contributor
means `git config user.name`); and three checkboxes, all off by default:
"Also make a .zip", "Accept contributor terms", and "Confirm submit",
which is the only thing that adds `--confirm`.

If you would rather not use the commands, the folder layout above is
exactly what they produce. Copy `SUBMISSION_TEMPLATE.json` to
`submission.json` and fill it in, copy the CSVs from the
`s1_containment/`, `s2_channel/` and `s3_persistence/` folders in your
results folder into place, and run `cbench community-validate` before
opening the pull request yourself. Validation applies the same rules to
a hand-assembled submission, including the `gate_check` and
`attestation` records.

## How a maintainer reviews one

```bash
cbench community-validate community-results/<model-tag>/<contributor>_<date>
```

This checks that the submission is shaped correctly and follows the
publication rule: the required `submission.json` fields are present, the
CSVs are present and named for the claimed model, a `checksums` manifest
(if there is one) matches the files, no verdict file or result field is
present, the gate check completed, and the terms were accepted (see
`core/community.py`). A missing manifest is not a problem in itself; a
mismatched or incomplete one is. The command does **not** run a suite or
judge whether the numbers are believable. That is the next step:

```bash
# bash or zsh
OPENLLM_CBENCH_RESULTS_DIR=community-results/<model-tag>/<contributor>_<date> \
  cbench score --model <model-tag> --from-existing
```

```powershell
# PowerShell
$env:OPENLLM_CBENCH_RESULTS_DIR = "community-results/<model-tag>/<contributor>_<date>"
cbench score --model <model-tag> --from-existing
```

Use the `model` field from `submission.json` as the `--model` value, and
set the variable before `cbench score` starts: `score` has no
`--results-dir` flag. The result is the same scorecard `cbench score`
produces for a local run. Read it the same way, caveats included, before
merging.

`cbench score` saves its scorecard in the results folder, which here is
the submission itself: it writes `scorecards/<model-tag>.json` and
`scorecards/<model-tag>.md` beside the CSVs. A submission carries no
verdicts, so `cbench community-validate` refuses a folder that contains
them. Delete that `scorecards/` folder before merging, and re-validate.
