"""
Deleting a pulled model from the local endpoint.

THE ONLY DESTRUCTIVE OPERATION IN THIS FRAMEWORK. Everything else here
reads, measures or writes into `results/`; this removes gigabytes from
disk and cannot be undone from inside the tool. Getting the model back
means re-pulling it, which for a quantised 30B is a long download and for
a tag that has since been deleted upstream may not be possible at all.

Three consequences, all deliberate:

  - Nothing here runs without an explicit confirmation from the caller.
    `cbench remove` requires `--yes`, and the TUI requires a checkbox
    ticked in the same interaction, not a remembered one.

  - The tag is never guessed or matched loosely. `lookup`-style prefix or
    substring matching is exactly how someone deletes `qwen3:14b` while
    meaning `qwen3:1.7b`. This sends the tag it was given and nothing
    else, and the endpoint either has that exact tag or the call fails.

  - Results are NOT touched. A model's CSVs, reports and scorecard stay
    on disk after the weights are gone, because the measurement is the
    thing this framework exists to produce and deleting it as a side
    effect of freeing disk space would be a data-loss bug wearing a
    convenience feature's clothes. `cbench discover` will simply stop
    listing the tag.
"""

import requests

from openllm_cbench.core.endpoint import resolve_base_url


def remove_url(base_url=None):
    return resolve_base_url(base_url).rstrip("/") + "/api/delete"


def remove_model(tag, base_url=None, timeout=60):
    """Deletes one model tag from the local endpoint.

    Returns (ok, detail). Never raises for an ordinary failure: the
    caller is usually a UI that needs to report what happened rather than
    crash, and "could not delete" is information, not an exception.

    A blank tag is refused rather than sent. An endpoint asked to delete
    "" may or may not do something surprising, and finding out is not
    worth it."""
    tag = (tag or "").strip()
    if not tag:
        return False, "no model tag given -- refusing to send a delete with an empty tag"
    try:
        resp = requests.delete(remove_url(base_url), json={"model": tag}, timeout=timeout)
    except Exception as e:
        return False, f"request failed: {e}"

    if resp.status_code == 404:
        return False, (f"'{tag}' is not present at this endpoint (404). Nothing was "
                       f"deleted. Check the exact tag with `cbench discover`.")
    if not resp.ok:
        return False, f"endpoint returned {resp.status_code}: {resp.text[:200]}"
    return True, f"deleted '{tag}' from the local endpoint"
