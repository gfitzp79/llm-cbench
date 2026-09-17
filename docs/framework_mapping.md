# Framework mapping -- probe/suite → OWASP / MITRE ATLAS / NIST AI RMF

Source of truth for how this framework's suites map to the three frameworks
of record. Depth on a small number of items rather than breadth across a
checklist -- see the reference section below for what's in scope and what
isn't.

## Suite-level mapping

| Suite | OWASP Top 10 for LLMs | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| S1 -- egress & containment (`containment.py`) | **LLM06 (Excessive Agency)** | Exfiltration and unauthorized-access tactics, demonstrated in an agentic tool-use loop rather than a single-turn probe | MANAGE (the boundary-adherence measurement itself), GOVERN (the stated-vs-none boundary comparison as a policy-effectiveness question) |
| S2 -- channel divergence (`channel.py`) | LLM01, LLM07 -- extended to the reasoning channel; a probe that passes on `content` but fails on `thinking` is still a real LLM01/LLM07 instance, just invisible to a content-only scorer | -- (channel divergence isn't a named ATLAS technique; it's a measurement-methodology finding, not an attack technique) | MEASURE (the channel-split scoring itself), MANAGE (the `merged_channel_suspected` harness-integrity guard) |
| S3 -- deceptive persistence (`persistence.py`) | LLM06 (Excessive Agency) -- the log-tampering behavior is a form of unauthorized self-modification of audit state, same root category as S1 | Persistence tactics | GOVERN -- this is fundamentally an audit-integrity question, not a measurement one |
| Inspect cross-validation (`integrations/inspect_*.py`) | Doesn't add new OWASP coverage -- validates S1/S3's existing LLM06 coverage against a second, externally-trusted implementation | Doesn't add new ATLAS coverage -- same reasoning | MEASURE (independent verification of the harness's own scoring), GOVERN (an assessment methodology that can cite external-framework agreement is stronger governance evidence than a single-harness claim) |
| Provenance axis (third-party model tracking, cross-cutting) | **LLM03 (Supply Chain)** | -- | GOVERN -- model provenance (vendor-official, third-party distillation, community fine-tune) is a governance/procurement question |

## Framework coverage, honestly stated

Depth on a few items, not breadth across a checklist.

- **OWASP Top 10 for LLMs**: LLM06 (Excessive Agency) is the deepest-covered
  item across S1 and S3, with Inspect cross-validation providing corroboration. LLM03 (Supply
  Chain) is covered via the provenance axis. LLM01/LLM07 (Prompt Injection,
  System Prompt Leakage) are covered by S2 with an emphasis on the reasoning
  channel. This is four items with real depth -- not all ten with a paragraph
  each, and that's a deliberate scoping choice, not a gap.
- **NIST AI RMF**: MEASURE from S2 and Inspect cross-validation; MANAGE from
  S1's boundary contrast; GOVERN from S3, the provenance axis, and Inspect
  cross-validation's corroborating role. All four RMF functions (GOVERN/MAP/MEASURE/MANAGE) aren't equally
  represented -- MAP in particular has no dedicated suite output here, which
  is worth naming as a real gap rather than stretching an existing suite's
  mapping to cover it artificially.
- **MITRE ATLAS**: exfiltration and unauthorized-access tactics (S1),
  persistence tactics (S3). Narrower than a full ATLAS technique sweep by
  design -- this framework demonstrates a small number of techniques in a
  real agentic loop rather than surveying the full matrix at single-turn
  depth.

## What this file does not claim

- It does not map every ATLAS technique or every OWASP item -- only the
  ones this framework's suites actually produce evidence for. An unmapped
  item means "not tested here," not "not applicable."
- It does not assign a maturity or coverage score to *this framework's own
  compliance-mapping coverage* -- that belongs in a write-up alongside the
  actual numbers, not in a static mapping table. That's a different claim
  from `cbench score` (see `scoring/scorecard.py`), which does grade a
  *specific model's* S1/S2/S3 results (an A-F letter grade, worst-suite-
  dominates) -- grading one model's measured behaviour is not the same
  thing as grading this file's own OWASP/ATLAS/NIST coverage, and the two
  shouldn't be conflated.
- This mapping covers the three suites this framework ships (containment,
  channel, persistence) plus the optional Inspect cross-validation. It
  does not attempt to map anything outside this framework's own scope.
