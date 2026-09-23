"""Generation budgets chosen from the model, not left to the user.

A reasoning model spends part of every reply thinking before it answers.
At the budget sized for a model that does not reason (2,048 reply tokens)
it often ran out mid-thought. Measured on qwen3:4b at quick depth: all four
S3 rows ended with an empty log and S3 came back INVALID; at 8,192 every
row finished and S3 was rated. Most models in a typical catalogue reason,
so a fixed default failed the common case and left the fix to a flag.

The budget is decided once per model, before the run, from whether the
model reasons, and it is recorded in every row as before. Deterministic on
purpose: the same model always gets the same budget, so its runs pool.
Raising the budget only for rows that ran out would give more room to
exactly the rows that failed, which biases the rate, and would split one
run across two budgets that the pooling guard refuses to combine.

Precedence is unchanged: an explicit flag, then the model's
`config_overrides` in the catalogue, then this.
"""

REASONING_NUM_PREDICT = 8192

# The window has to hold the conversation AND the reply. At an 8,192-token
# reply, qwen3:4b's S3 already peaked at 5,811 tokens of an 8,192 window, and
# a reply that pushes the start of the conversation out of the window fails
# silently: the model loses the rule it is scored against. Raising the reply
# budget without the window is the fix that does not take
# (METHODOLOGY_TECHNICAL.md section 6).
REASONING_NUM_CTX = 16384


def _endpoint_reports_thinking(model, base_url=None):
    """True or False from the endpoint's own capability list, or None when
    it cannot be asked. One read of the model-info route; no model call."""
    try:
        from openllm_cbench.core.endpoint import show_url
        from openllm_cbench.core.gate import fetch_show_info
        info = fetch_show_info(model, endpoint=show_url(base_url), timeout=10)
    except Exception:
        return None
    caps = info.get("capabilities")
    if not isinstance(caps, list):
        return None
    return "thinking" in caps


def model_reasons(model, entry=None, base_url=None, think=None, ask_endpoint=True):
    """Whether this model will reason during the run: True, False, or None
    when that cannot be told.

    `think` is the caller's reasoning setting for this run: False when it
    is switched off, True when the caller knows it is on (S2 does not
    apply the catalogue's {"think": false}), None to read that catalogue
    setting. Switched off means no thinking to budget for. Otherwise the
    catalogue entry's `thinking`, which `cbench gate` measured, and for an
    uncatalogued model the endpoint's capability list."""
    overrides = (entry or {}).get("config_overrides") or {}
    if think is False or (think is None and overrides.get("think") is False):
        return False
    if entry is not None and entry.get("thinking") is not None:
        return bool(entry["thinking"])
    if not ask_endpoint:
        return None
    return _endpoint_reports_thinking(model, base_url)


def resolve_budget(flag_ctx, flag_predict, overrides, suite_ctx, suite_predict, reasoning):
    """(num_ctx, num_predict, source) for one run.

    Each value takes the first of: the flag, the catalogue's
    config_overrides, the automatic value for a reasoning model, the
    suite's own default. `source` names where the budget came from, for
    the line every run prints."""
    overrides = overrides or {}
    auto_ctx, auto_predict = suite_ctx, suite_predict
    if reasoning:
        auto_ctx = max(suite_ctx, REASONING_NUM_CTX)
        auto_predict = max(suite_predict, REASONING_NUM_PREDICT)

    def pick(flag, key, auto):
        if flag is not None:
            return flag, "flag"
        if key in overrides:
            return overrides[key], "catalogue"
        return auto, "automatic"

    def describe(src):
        if src == "automatic":
            return ("automatic, sized for a reasoning model" if reasoning
                    else "automatic, this suite's default")
        return {"flag": "set on the command line",
                "catalogue": "from the model's catalogue entry"}[src]

    num_ctx, ctx_from = pick(flag_ctx, "num_ctx", auto_ctx)
    num_predict, predict_from = pick(flag_predict, "num_predict", auto_predict)
    if ctx_from == predict_from:
        source = describe(ctx_from)
    else:
        source = f"context {describe(ctx_from)}; reply {describe(predict_from)}"
    return num_ctx, num_predict, source


def run_summary(model, flag_ctx=None, flag_predict=None, base_url=None, ask_endpoint=True):
    """One line, printed before a multi-suite run starts, saying which
    budget the suites will use and why. Each suite prints its own exact
    values as it starts; this is the answer before any of them do."""
    from openllm_cbench.core.registry import load_registry, lookup
    entry = lookup(model, load_registry())
    overrides = (entry or {}).get("config_overrides") or {}
    if flag_ctx is not None or flag_predict is not None:
        return "Generation budget: set on the command line."
    if "num_ctx" in overrides or "num_predict" in overrides:
        return "Generation budget: from this model's catalogue entry."
    reasoning = model_reasons(model, entry, base_url, ask_endpoint=ask_endpoint)
    if reasoning:
        return (f"Generation budget: automatic, {REASONING_NUM_PREDICT} reply tokens in a "
                f"{REASONING_NUM_CTX}-token context window, because this model reasons.")
    if reasoning is False:
        return "Generation budget: automatic, each suite's default (this model does not reason)."
    return ("Generation budget: each suite's default (could not tell whether this model "
            "reasons).")


def budget_line(num_ctx, num_predict, source):
    """The one line every suite run prints about its budget. Worded apart
    from run_summary's "Generation budget:" so the TUI's condensed log,
    which keeps that prefix, shows the summary once rather than this line
    once per trial."""
    return (f"Generation budget for this suite: {num_predict} reply tokens in a "
            f"{num_ctx}-token context window ({source}). Override with --num-predict "
            f"and --num-ctx.")
