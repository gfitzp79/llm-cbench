"""
Pinned sampling parameters, shared by every suite.

WHY THIS EXISTS
===============
Before this, the options payload carried only `num_ctx`, `num_predict` and
`presence_penalty`. Temperature, top_p, top_k and seed were left to
whatever the model's own Modelfile happened to set -- which differs per
model, and is exactly the kind of uncontrolled variable that invalidates a
comparison without announcing itself.

This is measured, not theoretical. An audit of one ranked comparison
found FOUR DIFFERENT TEMPERATURES in force across it, with the effect
size of that confound estimated at 10-25 percentage points against the
3-13pp gaps the comparison was trying to resolve. The ranking was
measuring its own configuration. Worse, nothing in the output said so,
because the settings in force were never recorded anywhere.

Two consequences, both load-bearing:

  PIN IT. Every suite sends the same sampling triple unless told
  otherwise, so two models are asked the same question the same way.

  RECORD IT. All four values are written into every CSV row. A generation
  parameter that is applied but not recorded is the worst of both worlds:
  it changes the result and leaves no trace that it did. This framework's
  channel suite carries a schema-version guard for exactly that reason: an
  earlier build did not record `num_predict`, which became an invisible
  confound that invalidated comparisons. Once a column is missing, there
  is no way to recover the value afterwards. Recording the parameter at
  the time it runs is cheaper than detecting its absence later.

THE SEED IS GENERATED, NOT FIXED
================================
`--seed` unset means "generate one and write it down", not "don't use
one". A fixed default would make every trial in a multi-trial run
identical, destroying the trial-to-trial variance the 3-trial minimum
exists to measure. A generated-and-recorded seed keeps the variance and
still lets any single run be replayed exactly.
"""

import random

DEFAULT_TEMPERATURE = 0.8
DEFAULT_TOP_P = 0.9
DEFAULT_TOP_K = 40

# Written into every CSV row by every suite. Order is fixed so the column
# order is stable across suites and across versions.
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "seed")


def add_sampling_args(p):
    """Adds the four pinned-sampling flags to a suite's parser."""
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"Sampling temperature (default {DEFAULT_TEMPERATURE}, pinned so two "
                         f"models are asked the same way).")
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P,
                    help=f"Nucleus sampling cutoff (default {DEFAULT_TOP_P}).")
    p.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                    help=f"Top-k cutoff (default {DEFAULT_TOP_K}).")
    p.add_argument("--seed", type=int, default=None,
                    help="Seed for this run. Omit to generate one and record it, which keeps "
                         "trial-to-trial variance while leaving any single run replayable. "
                         "Pass an explicit value to reproduce a specific run exactly.")


def resolve_sampling(args):
    """Returns the sampling dict this run will use, generating a seed if
    none was supplied. Call once, in main(), and pass the result down --
    deliberately NOT module-level mutable state, because a suite's rows
    are built in several places and per-call-site drift is a known way
    for one of them to quietly disagree with the others."""
    return {
        "temperature": getattr(args, "temperature", DEFAULT_TEMPERATURE),
        "top_p": getattr(args, "top_p", DEFAULT_TOP_P),
        "top_k": getattr(args, "top_k", DEFAULT_TOP_K),
        "seed": args.seed if getattr(args, "seed", None) is not None
                else random.randint(1, 2**31 - 1),
    }


def build_options(num_ctx, num_predict, sampling=None):
    """The options payload for every chat call in every suite.

    `presence_penalty` stays pinned at 0: a nonzero value was OBSERVED to
    change refusal phrasing, which moves a keyword-matched verdict without
    changing the behaviour being measured. Stated as a fact rather than a
    possibility on purpose -- this is the sentence standing between the
    pin and someone who wants to unpin it.

    `sampling=None` yields the historical payload with no sampling keys,
    so a caller that has not been migrated behaves exactly as before
    rather than silently acquiring new defaults."""
    opts = {"num_ctx": num_ctx, "num_predict": num_predict, "presence_penalty": 0}
    if not sampling:
        return opts
    opts["temperature"] = sampling["temperature"]
    opts["top_p"] = sampling["top_p"]
    opts["top_k"] = sampling["top_k"]
    if sampling.get("seed") is not None:
        opts["seed"] = sampling["seed"]
    return opts


def sampling_row_fields(sampling):
    """The four values as CSV cells. Empty dict-safe: an unmigrated or
    failed run writes blanks rather than raising, and a blank column is
    itself the signal that the run predates the pin."""
    sampling = sampling or {}
    return {k: sampling.get(k, "") for k in SAMPLING_FIELDS}
