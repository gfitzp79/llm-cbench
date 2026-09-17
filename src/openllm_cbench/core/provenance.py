"""
Third-party-model provenance labelling, shared by containment.py,
channel.py and persistence.py. Consolidating this into one place prevents
inconsistencies across suite implementations.
"""


def provenance_note(model):
    """Empero's Qwen3.8-9B distillation and any Huihui community fine-tune
    must always be labeled. This was duplicated across three runners and
    was missing from one of them -- the other two had it, so nothing looked
    wrong. One definition removes that class of gap by construction rather
    than by vigilance."""
    m = model.lower()
    if "empero" in m:
        return "> Third-party distillation (Empero), unaffiliated with Alibaba.\n"
    if "huihui" in m:
        return ("> Community safety-ablated fine-tune (Huihui), not an Alibaba, "
                "Qwen-team, Google, or OpenAI release.\n")
    return ""
