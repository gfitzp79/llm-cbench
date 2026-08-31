"""
Third-party-model provenance labelling, shared by containment.py,
channel.py and persistence.py (was duplicated identically across all
three in the original lab; consolidated here in the extraction with no
change in behaviour).
"""


def provenance_note(model):
    """Empero's Qwen3.8-9B distillation and any Huihui community fine-tune
    must always be labeled. Found missing from one runner during the
    original lab's development; the other two already had it -- this
    shared version prevents that class of gap by construction."""
    m = model.lower()
    if "empero" in m:
        return "> Third-party distillation (Empero), unaffiliated with Alibaba.\n"
    if "huihui" in m:
        return ("> Community safety-ablated fine-tune (Huihui), not an Alibaba, "
                "Qwen-team, Google, or OpenAI release.\n")
    return ""
