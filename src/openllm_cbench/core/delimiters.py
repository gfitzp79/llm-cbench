"""
Reasoning delimiters: how this framework recognises that a model's
reasoning trace has merged into its visible answer.

WHY THIS IS IN core/ AND NOT IN THE CHANNEL SUITE. It used to be in the
suite, and the gate check carried its own copy -- a hard-coded `<think>`
test whose docstring claimed it was "the same signature the channel
suite's own runtime guard uses". It was, once. When the suite's guard
grew from one marker to the family below, the gate's copy did not, and
the two quietly disagreed for every model that delimits its reasoning
any other way. That is the worst possible place for the disagreement to
live: the gate is what decides whether a model is catalogued as CLEAN,
so a model the suite would flag on every row could be gated clean first
and only contradict itself later.

One definition, imported by both. A duplicated predicate is a defect
waiting for someone to fix half of it.

WHY A FAMILY AND NOT JUST <think>: a model that delimits its reasoning
any other way slips straight past a single-marker guard, and the scorer
then grades the reasoning trace as if it were the answer, marking the
model worse than it behaved. At least one shipping model family uses
`[BEGIN FINAL RESPONSE]`, and harness-level formats like `<|channel|>`
appear in others.

MEASURED BEFORE SHIPPING, on a 2970-row corpus: extending to this family
flags 78 of 1067 thinking-empty rows against the single-marker guard's
30, a 2.6x recall increase, at 2 false positives in 1903
demonstrably-split rows, i.e. 0.11%. The regression test pins that
ceiling, because a guard that starts firing on split rows destroys the
metric it protects.

Deliberately NOT included: a generic phrase heuristic ("we must", "let
me", ... over a length threshold). Measured on the same corpus it caught
8 further rows at 7 false positives, roughly break-even, and it is
English-specific, which is the wrong bet for a tool whose users mostly
run non-English models. A near-break-even signal folded into the same
boolean would make the flag mean less than it does now.
"""

MERGE_DELIMITERS = (
    ("think_tag", ("<think>", "</think>")),
    ("final_response_marker", ("[begin final response]", "[end final response]")),
    ("channel_marker", ("<|channel|>",)),
    ("reasoning_tag", ("<reasoning>", "</reasoning>")),
)


def parse_catalogued_delimiters(raw):
    """Turns a catalogue entry's `delimiters` list into markers this
    module can match.

    The catalogue writes a delimiter the way a human reads it, as a
    single string with the open and close joined by an ellipsis:
    `"<think>...</think>"`, `"[BEGIN FINAL RESPONSE]...[END FINAL
    RESPONSE]"`. Both halves are matchable markers, so the pair is split
    rather than searched for literally -- a model emits the open marker
    and then runs out of budget far more often than it emits the exact
    joined string, which is never.

    Tolerates a bare marker with no ellipsis, a non-list, and junk
    entries, because this is user-editable data read at run time and a
    malformed catalogue must not stop a run."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, (list, tuple)):
        return ()
    markers = []
    for item in raw:
        if not isinstance(item, str):
            continue
        for part in item.split("..."):
            part = part.strip().lower()
            if part:
                markers.append(part)
    return tuple(dict.fromkeys(markers))


def merge_evidence(content, thinking, extra_markers=()):
    """Returns the delimiter-family label that fired, or "" if none did.

    Separate from the boolean so the CSV can record WHICH branch fired.
    The boolean says only that something is wrong; a run full of
    `final_response_marker` hits and a run full of `think_tag` hits are
    different problems with different fixes, and collapsing them loses
    exactly the information needed to tell them apart.

    `extra_markers` carries this model's catalogued delimiters. They are
    checked FIRST and reported under their own label, because a
    catalogued delimiter is something an operator gate-checked and wrote
    down for this specific model, and it should not be silently
    attributed to whichever built-in family happens to overlap with it."""
    if thinking.strip():
        return ""
    c = content.lower()
    if any(m in c for m in extra_markers):
        return "catalogued"
    for label, markers in MERGE_DELIMITERS:
        if any(m in c for m in markers):
            return label
    return ""


def merge_suspected(content, thinking, extra_markers=()):
    """True if the API returned no separate thinking trace but the content
    still carries a reasoning delimiter -- a sign this model/runtime pair
    isn't actually splitting the two channels, which would make every
    CLEAN/CHANNEL_LEAK verdict from this run unreliable. Most relevant for
    a GGUF pulled directly from a third-party repository rather than
    served from the runtime's own built-in template library, where
    channel-splitting behavior isn't guaranteed the way it is for a
    natively supported model."""
    return bool(merge_evidence(content, thinking, extra_markers))
