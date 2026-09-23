"""
Local progress: what this machine has assessed, and what is worth doing
next.

WHAT THIS COUNTS, AND WHY IT MATTERS THAT IT IS NOT VOLUME. The obvious
version of this panel counts models pulled, models gated and models
scored, and awards a tier for accumulating them. That version rewards
running many shallow scans, which is the opposite of what a crowd-sourced
security benchmark needs: a corpus full of single-trial results that the
framework's own rules say are below its minimum for a citable rate is not
a bigger corpus, it is a noisier one that looks like data.

The evidence for that is sitting in any real results directory. On the
machine this was written against, seven models had scorecards. One was a
test artefact, one had never run, one was invalid across all three
suites, and one carried an A/100 that meant "this model cannot call tools
at all, so it attempted nothing". Of seven "scored" models, two were
worth citing. A tier computed from the seven would have been a lie told
with arithmetic.

So the panel is two separate things, deliberately:

  COUNTS are neutral inventory. How many models are pulled, catalogued,
  scored. No judgement, no tier, nothing to chase. They answer "where am
  I" and nothing else.

  The TIER is earned on rigour, and every criterion below is a rule this
  framework already states and already enforces somewhere else. Nothing
  new is invented here; this surfaces compliance with rules that exist.

A result counts as CITABLE when all of these hold:

  - the model is in the catalogue, so its gate caveats are known
    (README "The model catalogue");
  - a scorecard exists for it;
  - no suite was refused by its own validity guard, which covers the
    task-set, comparability, schema-version and positive-control rules;
  - every scored suite ran at least DEPTH_TRIALS["standard"] trials, this
    framework's own pre-registered minimum for a rate worth citing
    (METHODOLOGY.md 3.1, scoring/extension_rule.py).

Deliberately NOT a criterion: the grade itself. A model that scores badly
has still been measured properly, and tiering on the grade would reward
picking easy models.

This is a LOCAL panel. There is no identity and no server: it reads this
machine's own files and reports on this machine's own work. It is not a
ranking against anyone else, and nothing here is transmitted.
"""

import json
from pathlib import Path

from openllm_cbench.core.paths import results_dir
from openllm_cbench.scoring.scorecard import DEPTH_TRIALS

MIN_CITABLE_TRIALS = DEPTH_TRIALS["standard"]

# Ordered worst to best. A tier is the highest one whose rule is met.
TIERS = ("newcomer", "novice", "intermediate", "advanced", "contributor")

ADVANCED_CITABLE = 3


def _load_scorecards(root=None):
    """Every scorecard on disk, keyed by model tag. Never raises: a
    malformed file is skipped rather than taking the dashboard down."""
    d = Path(root) if root else results_dir("scorecards")
    out = {}
    if not d.is_dir():
        return out
    for p in sorted(d.glob("*.json")):
        try:
            card = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        model = card.get("model")
        if model:
            out[model] = card
    return out


def is_citable(card, catalogued_tags):
    """Whether one scorecard is a result worth citing. See this module's
    docstring for why each clause is here, and note that the grade is
    deliberately not one of them."""
    if not isinstance(card, dict):
        return False
    if card.get("model") not in catalogued_tags:
        return False
    suites = card.get("suites") or {}
    if not suites:
        return False
    for s in suites.values():
        if not isinstance(s, dict):
            return False
        if s.get("status") != "ok":
            return False
        if (s.get("n_trials") or 0) < MIN_CITABLE_TRIALS:
            return False
    return True


def _tier(catalogued, citable, validated_submissions):
    if validated_submissions:
        return "contributor"
    if citable >= ADVANCED_CITABLE:
        return "advanced"
    if citable >= 1:
        return "intermediate"
    if catalogued >= 1:
        return "novice"
    return "newcomer"


def _next_action(local, catalogued, scored, citable, uncitable_scored):
    """The single most useful thing to do next, or None when there is
    nothing obvious.

    This is the part of the panel most likely to actually help somebody.
    A new user does not need a badge, they need to know which of the
    twelve buttons in front of them is the one that moves their work
    forward."""
    if local == 0:
        return ("Pull a model to test", "No models found at the endpoint.")
    if catalogued == 0:
        return ("Gate-check a model",
                "Nothing is catalogued yet. A gate check finds the tool-calling "
                "and channel gaps that would otherwise silently invalidate a run.")
    if scored == 0:
        return ("Score a catalogued model",
                f"{catalogued} model(s) catalogued, none scored yet.")
    if citable == 0 and uncitable_scored:
        return ("Re-score at standard depth",
                f"{uncitable_scored} scorecard(s) exist but none are citable yet: "
                f"they need {MIN_CITABLE_TRIALS} trials per suite and no suite refused "
                f"by a validity guard.")
    if catalogued > scored:
        return ("Score another model",
                f"{catalogued - scored} catalogued model(s) have no scorecard.")
    return None


def compute_progress(local_models=None, registry=None, scorecards_root=None,
                     submissions_root=None):
    """The whole panel, as data. Every argument is injectable so this is
    testable without an endpoint, a results tree or a network call.

    `local_models` is the endpoint's model list, or None when the endpoint
    could not be reached -- in which case the local count is reported as
    None rather than 0, because "we could not ask" and "you have none" are
    different facts and showing the second for the first is a lie the
    dashboard would tell every time Ollama was stopped."""
    from openllm_cbench.core.registry import load_registry

    reg = registry if registry is not None else load_registry()
    catalogued_tags = set((reg or {}).get("models", {}))

    local_tags = None
    if local_models is not None:
        local_tags = {m.get("name") for m in local_models if m.get("name")}

    cards = _load_scorecards(scorecards_root)
    # Only count scorecards for models actually present locally, when we
    # know what is present. A scorecard for a model that has since been
    # removed is history, not progress.
    if local_tags is not None:
        cards = {k: v for k, v in cards.items() if k in local_tags}

    citable_models = sorted(m for m, c in cards.items()
                            if is_citable(c, catalogued_tags))

    validated = []
    try:
        from openllm_cbench.core.community import (
            list_packaged_submissions, validate_submission,
        )
        for sub in list_packaged_submissions(submissions_root):
            try:
                if not validate_submission(sub["path"]):
                    validated.append(sub)
            except Exception:
                continue
    except Exception:
        validated = []

    catalogued_local = (len(catalogued_tags & local_tags)
                        if local_tags is not None else len(catalogued_tags))
    scored = len(cards)
    citable = len(citable_models)

    return {
        "local": len(local_tags) if local_tags is not None else None,
        "catalogued": catalogued_local,
        "scored": scored,
        "citable": citable,
        "submissions": len(validated),
        "citable_models": citable_models,
        "tier": _tier(catalogued_local, citable, validated),
        "next_action": _next_action(
            len(local_tags) if local_tags is not None else 1,
            catalogued_local, scored, citable, scored - citable),
        "min_citable_trials": MIN_CITABLE_TRIALS,
    }


def render_line(progress):
    """One line of plain counts, for a dashboard or a CLI footer. The
    counts carry no judgement; the tier is reported separately because it
    means something different."""
    local = progress["local"]
    local_txt = "endpoint unreachable" if local is None else f"{local} local"
    return (f"{local_txt} · {progress['catalogued']} catalogued · "
            f"{progress['scored']} scored · {progress['citable']} citable · "
            f"{progress['submissions']} submitted")
