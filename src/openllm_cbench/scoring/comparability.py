"""
Whether a set of trial CSVs may be pooled into one rate.

WHY THIS EXISTS. `find_csvs()` globs every CSV on disk for a model tag.
That is convenient and it is also how runs from different harness
versions, different sampling configurations and different generation
budgets end up silently averaged into a single number. The framework's
stated rule has always been that results collected before a correctness
fix are not pooled with results collected after it; until now nothing
checked, and the documentation said so in as many words.

It stopped being hypothetical the first time the tool was pointed at a
real results directory: one model's CSVs spanned three weeks and four
correctness fixes, including a canary-matching fix that changed what
`escape_attempt` MEANS and a sampling pin that changed what the model was
asked. Only the task-set guard noticed anything. Two suites pooled the
lot and reported a confident rate.

WHAT IT CHECKS, AND WHY BOTH CHECKS ARE EXACT. Every condition here is an
equality test on a value the CSV records, never a threshold or a
heuristic. That is deliberate. A detector that guesses produces false
positives, and a validity guard that cries wolf is worse than no guard,
because operators learn to page past it -- this project has already seen
one guard produce 68 findings of which 67 were false, and has already
shipped one guard that false-invalidated the suite carrying most of its
signal. A guard that can only fire on a genuine, recorded difference
cannot do either.

  1. MIXED INSTRUMENTATION. Some CSVs carry the sampling columns and
     others do not. The ones that do not were run before sampling was
     pinned, at whatever each model's own Modelfile happened to set, and
     that value is not recoverable now. A rate pooled across the two is
     part measurement and part configuration, with no way to separate
     them afterwards.

  2. DIFFERENT PINNED SAMPLING. All CSVs carry the columns but disagree
     on temperature, top-p or top-k. These runs asked the model different
     questions.

WHAT IT DELIBERATELY DOES NOT CHECK. A corpus that is uniformly OLD does
not fire. Those runs are unpinned but they are unpinned in the same way
as each other, so they remain internally comparable, and invalidating
every result anyone collected before today would be a guard that fires on
correct data. Nor does differing SEED fire: varying the seed across
trials is the intended behaviour and is what makes a multi-trial rate
mean anything.
"""

SAMPLING_KEYS = ("temperature", "top_p", "top_k")


def _fingerprint(row):
    """The sampling configuration a row was produced under, or None if
    this CSV predates the columns entirely."""
    if not any(k in row for k in SAMPLING_KEYS):
        return None
    return tuple(str(row.get(k, "")).strip() for k in SAMPLING_KEYS)


def file_provenance(rows):
    """Summarises one CSV. `rows` is the already-parsed list of dicts.

    Returns {"sampling": tuple|None, "started": str|None}. `sampling` is
    None for a pre-pin file. `started` is that run's recorded wall-clock
    start, which is the value that survives a checkout where an mtime
    does not (see core/runclock.py)."""
    if not rows:
        return {"sampling": None, "started": None}
    first = rows[0]
    started = str(first.get("run_started_at", "") or "").strip() or None
    return {"sampling": _fingerprint(first), "started": started}


def pooling_problems(provenance_by_file):
    """Returns a list of human-readable problems, empty when the files may
    be pooled.

    `provenance_by_file` maps a display name to a file_provenance() dict.
    A single file is trivially self-consistent and never a problem."""
    if len(provenance_by_file) < 2:
        return []

    problems = []
    pinned = {n: p["sampling"] for n, p in provenance_by_file.items()
              if p["sampling"] is not None}
    unpinned = sorted(n for n, p in provenance_by_file.items()
                      if p["sampling"] is None)

    if pinned and unpinned:
        problems.append(
            "Some of these runs recorded their sampling parameters and some "
            "predate that column. The older ones ran at whatever the model's "
            "own Modelfile set, which cannot be recovered now, so this rate is "
            "part measurement and part configuration. Unrecorded: "
            + ", ".join(f"`{n}`" for n in unpinned)
        )

    distinct = sorted(set(pinned.values()))
    if len(distinct) > 1:
        shown = "; ".join(
            "/".join(f"{k}={v}" for k, v in zip(SAMPLING_KEYS, fp))
            for fp in distinct
        )
        problems.append(
            "These runs used different sampling settings, so they asked the "
            f"model different questions: {shown}."
        )

    return problems


def run_window(provenance_by_file):
    """(earliest, latest, n_stamped) across these files.

    n_stamped is returned and not inferred, because a window computed from
    a subset must never be described as though it covered everything. In a
    real mixed corpus exactly one file of eight carried a timestamp, and
    reporting "all started <that time>" would have asserted a fact about
    seven files whose run time is simply unknown.

    Reported alongside a pooled rate because the span is the first thing
    that tells a reader whether pooling was reasonable. Three trials over
    four minutes is one run; three trials over three weeks is a question."""
    stamps = sorted(p["started"] for p in provenance_by_file.values() if p["started"])
    if not stamps:
        return None, None, 0
    return stamps[0], stamps[-1], len(stamps)


def render_block(provenance_by_file, generated_at):
    """The markdown every trial summary carries: when it was generated,
    what window the pooled runs span, and any pooling problem.

    `generated_at` is written into the DOCUMENT because a trial summary
    has a fixed filename and is overwritten in place -- it carries no
    timestamp in its name, so without this line a copied or cloned summary
    cannot be dated at all."""
    total = len(provenance_by_file)
    lines = [f"Generated: {generated_at}"]
    first, last, n_stamped = run_window(provenance_by_file)

    if n_stamped == 0:
        lines.append(
            f"Runs pooled: {total}. None of these CSVs carry the "
            "`run_started_at` column, so when they ran cannot be read from the "
            "data. A file's mtime is not an answer: it belongs to whatever tool "
            "last touched the file."
        )
    else:
        # Say what the window covers. A span derived from some of the files
        # must not be phrased as though it described all of them.
        if n_stamped == total:
            scope = "all " + str(total)
        else:
            scope = f"{n_stamped} of {total}"
        when = (f"started {first}" if first == last
                else f"started between {first} and {last}")
        lines.append(f"Runs pooled: {total}, of which {scope} recorded a start time: {when}.")
        if n_stamped < total:
            lines.append(
                f"The remaining {total - n_stamped} predate the `run_started_at` "
                "column and cannot be dated from the data at all."
            )
    lines.append("")

    problems = pooling_problems(provenance_by_file)
    if problems:
        lines += [
            "> **STOP -- THESE RUNS ARE NOT COMPARABLE. This aggregate pools "
            "incompatible runs.**",
            ">",
        ]
        for p in problems:
            lines.append(f"> - {p}")
        lines += [
            ">",
            "> Re-run so that every trial shares one configuration, or point "
            "`OPENLLM_CBENCH_RESULTS_DIR` at a directory holding only the runs "
            "you mean to pool. This framework does not choose for you which "
            "runs belong together.",
            "",
        ]
    return lines, bool(problems)
