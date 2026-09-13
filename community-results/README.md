# Community-submitted results

Not every model fits on every machine. If you can't run a model locally to
score it yourself, someone who *can* run it may already have — or you can
run one you have and share it for others who can't. This folder is where
those results live: **raw trial CSVs, submitted via pull request, re-run
through this framework's own scoring before anyone trusts the number.**

## Why raw CSVs, not a submitted score

A submitted "security score" with no data behind it can't be checked
against anything — anyone could type in whatever number they want. A
submitted CSV is the actual per-row output this framework's own suites
produce, in the same format `cbench aggregate`/`cbench score` already
know how to read. That means:

- Anyone can re-run the exact same scoring code (`cbench score --model
  <tag> --from-existing`) against the submitted CSVs and get the exact
  same scorecard, independently of whoever submitted it.
- Every guard this framework already has (task-set mismatch, CSV
  schema-version mismatch, the positive/negative control-diagnostic
  checks) runs on a submission exactly like it runs on your own local
  results — a bad or incomparable submission gets flagged, not silently
  averaged in.
- A submission's own `heuristic keyword/behaviour matching` results (S2,
  S3) still say "read the transcript before treating this as confirmed",
  same as they always have. Nothing about accepting community data changes
  that discipline.

## What it doesn't do

This is a folder in a git repo, reviewed via pull request — **not** a
hosted leaderboard, not a live-scoring service, and not an automatic
trust mechanism. A merged submission means a maintainer looked at it
(`cbench community-validate`, then a manual read of `submission.json`
against the actual CSVs, then `cbench score --from-existing` to see the
resulting scorecard) and judged it worth keeping on file — not that its
numbers are independently verified true. Read the same caveats you'd read
on your own results before citing anything from here.

## Folder layout

```
community-results/
  <model-tag, sanitized>/
    <your-handle>_<YYYYMMDD>/
      submission.json          <- required, see SUBMISSION_TEMPLATE.json
      s1_containment/          <- optional -- include whichever suites you ran
        containment_<tag>_<timestamp>.csv
      s2_channel/
        channel_<tag>_<timestamp>.csv
      s3_persistence/
        persistence_<tag>_<timestamp>.csv
```

`submission.json` may also carry a `checksums` key — `{relative CSV path:
sha256}` for every CSV in the submission. `cbench community-package`
fills this in for you; a hand-assembled submission can leave it out
entirely (that's not an error, just unverifiable) but must not include a
partial or stale one — `cbench community-validate` treats a checksum that
doesn't match the file on disk, or a listed file that isn't there, as a
real problem, same as any other shape mismatch.

`<model-tag, sanitized>` is the model tag with `:` and `/` replaced by
`-` (the same transform `scoring/aggregate.py:model_tag()` already
applies to every filename this framework writes) — e.g. `gemma3:12b`
becomes `gemma3-12b`. The CSV filenames themselves are whatever
`cbench containment`/`cbench channel`/`cbench persistence` already named
them; don't rename them by hand, `cbench community-validate` checks that
each one's filename actually contains the model tag from your
`submission.json`.

You don't need all three suites. A submission with just `s3_persistence/`
is still useful — the scorecard will show the suites you didn't run as
"not run", not as a failure.

## How to submit

1. Run whatever suites you can against your own local endpoint, the same
   way you would for yourself: `cbench score --model <tag> --depth
   standard` (or `cbench assess`, or the individual suites directly) is
   this framework's own pre-registered 3-trial minimum for a rate worth
   citing — fewer trials than that is still acceptable to submit, but say
   so plainly, the same way this framework's own aggregate reports flag a
   sub-3-trial result as "wider-uncertainty ... not a settled rate."
2. ```bash
   cbench community-package --model <tag>
   ```
   builds `community-results/<model-tag>/<your-handle>_<date>/` from the
   CSVs step 1 just produced: it copies them (never rewrites or moves
   them), fills in `submission.json` with whatever it can detect on its
   own (hardware, the local endpoint's runtime version, quant, this
   harness's own version — `--contributor`/`--notes` for the rest), and
   records a SHA-256 per CSV. It prints exactly what, if anything, still
   needs a hand-edit before the result is valid, and runs
   `cbench community-validate` on the result for you. It uploads nothing.
3. ```bash
   cbench community-submit community-results/<model-tag>/<your-handle>_<date>
   ```
   opens that folder as a pull request through your own authenticated
   `gh` (the GitHub CLI) — this framework never sees, stores, or
   transmits a credential of yours. It previews the exact command
   sequence and the reminder that these CSVs contain the model's raw
   output, and sends nothing until you re-run it with `--confirm`.
   Without `gh` installed and logged in, it prints the fallback by hand
   instead:
   - **fork-and-PR**, for anyone with git: clone your fork, copy the
     packaged folder in at the path shown, `git add community-results &&
     git commit && git push`, open the PR yourself.
   - **attach-to-an-issue**, for anyone without git: re-run
     `cbench community-package --model <tag> --zip` for an archive, then
     open the prefilled issue URL it gives you and drag the zip on —
     needs nothing but a GitHub account and a browser.

   The TUI's community screen (Package / Validate / Submit buttons, a
   "Confirm submit" checkbox that's the only thing that adds `--confirm`)
   does the same three steps without a terminal.
4. Whichever route you used, say in the PR (or issue) description what
   model, what endpoint/runtime, and roughly what you saw — a maintainer
   will run `cbench score --from-existing` on it regardless, but a
   heads-up on what to expect speeds up review.

If you'd rather not use either command, the folder layout above is exactly
what they produce — you can still copy `SUBMISSION_TEMPLATE.json` and the
CSVs from `results/s1_containment/` (etc) into place by hand and validate
with `cbench community-validate` before opening the PR yourself. That's
just the harder way to reach the same folder now.

## How a maintainer reviews one

```
cbench community-validate community-results/<model-tag>/<handle>_<date>
```

checks the submission is shaped correctly (required `submission.json`
fields present, CSVs present and tagged for the claimed model, and — when
a `checksums` manifest is present — that every listed CSV is still
byte-identical to what was packaged) — see `core/community.py`. A missing
manifest is not itself a problem; a mismatched or incomplete one is. This
command does **not** run any suite or judge whether the numbers are
believable; that's the next step:

```
# bash/zsh
OPENLLM_CBENCH_RESULTS_DIR=community-results/<model-tag>/<handle>_<date> \
  cbench score --model <tag> --from-existing

# PowerShell
$env:OPENLLM_CBENCH_RESULTS_DIR = "community-results/<model-tag>/<handle>_<date>"
cbench score --model <tag> --from-existing
```

This produces the exact same scorecard `cbench score` would for a local
run — read it the same way, caveats and all, before merging.
