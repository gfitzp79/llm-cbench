# Framework mapping: probe/suite → OWASP / MITRE ATLAS / NIST AI RMF

This file is the source of truth for how the framework's suites map to three
frameworks of record: the OWASP Top 10 for LLM Applications (2025 edition),
MITRE ATLAS and the NIST AI Risk Management Framework (AI RMF). It covers a
small number of items in depth rather than a checklist in breadth; "What this
file does not claim", below, sets out what is in scope and what is not.

## Suite-level mapping

| Suite | OWASP Top 10 for LLMs | MITRE ATLAS | NIST AI RMF |
|---|---|---|---|
| S1 containment (`suites/containment.py`) | **LLM06 (Excessive Agency)** | Exfiltration and unauthorised-access tactics, demonstrated in an agentic tool-use loop rather than a single-turn probe | MANAGE (the boundary-adherence measurement itself), GOVERN (the `stated` versus `none` boundary comparison as a policy-effectiveness question) |
| S2 channel (`suites/channel.py`) | LLM01 and LLM07, extended to the reasoning channel. A probe that passes on `content` but fails on `thinking` is still a real LLM01 or LLM07 instance, but a content-only scorer cannot see it. | The probe bank tags its probes in the `framework` field of `data/probes/eval_prompts.json`: AML.T0051 (LLM prompt injection, with the indirect sub-technique AML.T0051.001), AML.T0054 (LLM jailbreak) and AML.T0056 (system prompt extraction). Channel divergence itself is not a named ATLAS technique; it is a measurement-methodology finding, not an attack technique. | MEASURE (the channel-split scoring itself), MANAGE (the `merged_channel_suspected` harness-integrity guard) |
| S3 persistence (`suites/persistence.py`) | LLM06 (Excessive Agency). Log tampering is a form of unauthorised self-modification of audit state, the same root category as S1. | Persistence tactics | GOVERN: fundamentally an audit-integrity question, not a measurement one |
| Inspect cross-validation (`integrations/inspect_*.py`) | Adds no OWASP coverage of its own. It validates the existing LLM06 coverage of S1 and S3 against a second, externally trusted implementation. | Adds no ATLAS coverage, for the same reason | MEASURE (independent verification of the harness's own scoring), GOVERN (an assessment methodology that can cite agreement with an external framework is stronger governance evidence than a single-harness claim) |
| Provenance axis (third-party model tracking, cross-cutting) | **LLM03 (Supply Chain)** | None | GOVERN: model provenance (vendor-official, third-party distillation, community fine-tune) is a governance and procurement question |

## Framework coverage, honestly stated

Depth on a few items, not breadth across a checklist.

- **OWASP Top 10 for LLMs**: LLM06 (Excessive Agency) is the most deeply
  covered item, across S1 and S3, with Inspect cross-validation providing
  corroboration. LLM03 (Supply Chain) is covered through the provenance axis.
  LLM01 and LLM07 (Prompt Injection, System Prompt Leakage) are covered by S2,
  with an emphasis on the reasoning channel. That is four items covered in
  depth rather than all ten covered by a paragraph each, and it is a deliberate
  scoping choice, not a gap.
- **NIST AI RMF**: MEASURE from S2 and Inspect cross-validation; MANAGE from
  S1's boundary-adherence measurement and S2's merge guard; GOVERN from S1's
  `stated` versus `none` comparison, S3, the provenance axis and the
  corroborating role of Inspect cross-validation. The four RMF functions
  (GOVERN, MAP, MEASURE and MANAGE) are not equally represented. MAP in
  particular has no dedicated suite output here, which is worth naming as a
  real gap rather than stretching an existing suite's mapping to cover it
  artificially.
- **MITRE ATLAS**: exfiltration and unauthorised-access tactics (S1),
  persistence tactics (S3), and the prompt injection, jailbreak and system
  prompt extraction techniques that S2's probes exercise (AML.T0051, AML.T0054
  and AML.T0056). This is narrower than a full ATLAS technique sweep by design:
  the framework exercises a small number of techniques in depth (S1 and S3 in
  a real agentic loop, S2 on both the visible answer and the reasoning trace)
  rather than surveying the full matrix.

## What this file does not claim

- It does not map every ATLAS technique or every OWASP item, only those the
  framework's suites produce evidence for. An unmapped item means "not tested
  here", not "not applicable".
- It does not assign a maturity or coverage score to *this framework's own
  compliance-mapping coverage*; that belongs in a write-up alongside the actual
  numbers, not in a static mapping table. `cbench score` (see
  `scoring/scorecard.py`) makes a different claim: it grades a *specific
  model's* S1, S2 and S3 results with an A-F letter grade set by the worst
  suite. Grading one model's measured behaviour is not the same as grading this
  file's OWASP, ATLAS or NIST coverage, and the two should not be conflated.
- The mapping covers the three suites the framework ships (S1 containment, S2
  channel and S3 persistence), the optional Inspect cross-validation and the
  provenance axis. It does not attempt to map anything outside the framework's
  own scope.
