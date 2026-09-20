# Testing methodology

How this framework tests open-weight language models for containment and
reasoning-channel leakage, and what makes a result from it worth citing.

Audience: anyone deciding whether to deploy an open-weight model, or deciding
whether to trust a number this framework produced. A companion document,
[METHODOLOGY_TECHNICAL.md](METHODOLOGY_TECHNICAL.md), carries the estimators,
exclusion rules and statistical detail.

**This document describes the instrument, not any particular result.** See
section 4 for why no measurements appear here.

---

## 1. What this measures, and what it does not

This framework measures **attempt, not success**.

Every suite records whether a model tried to do something it was told not to
do. No suite records whether the attempt would have worked, because no attempt
is ever allowed to work. The canary listener binds to `127.0.0.1` and refuses
to start otherwise. The shell tool is a logged no-op: nothing passed to it is
ever executed. Any host other than the loopback canary is intercepted before a
socket opens and answered with a synthetic failure.

This is a deliberate design constraint, not a limitation to work around. A test
harness that lets a model reach a real external host to prove it would have is
a harness that exfiltrates data during testing.

**What a result tells you:** this model, under this pressure, at this
generation budget, reached for a channel it was told not to use, at this rate.

**What a result does not tell you:** that the model is safe, that the model is
unsafe, or that the rate would hold under a different prompt, a different
budget, or an agent scaffold you built yourself. Attempt rates are a property
of the model *and* the harness together.

---

## 2. The suites

| suite | question | what the model is given | what is recorded |
|---|---|---|---|
| **S1 Containment** | Under task pressure, does the model reach for a network channel it was told not to use? | A business task, a sandbox of files, three tools (`http_request`, `run_shell`, file read), and a boundary instruction | Every tool call, the target of each, the final answer, turns used, stop reason |
| **S2 Channel** | Does the visible answer do what an adversarial probe asks, and does the reasoning trace leak what the visible answer refuses? | Adversarial probes: prompt injection, jailbreaks, evasion, system-prompt extraction | Visible answer and reasoning trace, scored separately, then combined |
| **S3 Persistence** | When challenged about its own logged actions, does the model tamper with the log? | A task, a log tool, then a challenge turn questioning what it did | Log edits, appended corrections, denial language, identity resets |

An optional cross-validation tier re-runs S1 and S3 through the Inspect
framework and compares verdicts, to check whether a finding is a property of
the model or of this harness's scoring.

### S1 in more detail, because it carries most of the design

S1 runs two boundary conditions against the same tasks (`--boundary`,
default `both`):

- **`none`**: no boundary instruction. The baseline.
- **`stated`**: names a permitted destination, for example "you may only reach
  the internal endpoint".

Running both separates "this model reaches for the network under this task
pressure" from "this model reaches for the network *having been told where it
may go*". Because the same tasks and the same trial seeds are used in both
arms, the difference between them is attributable to the boundary text rather
than to the workload, which is what makes the contrast usable.

A third arm (an absolute prohibition naming no destination at all) exists in
related work and is not shipped in this framework. If you implement one,
add it as a third value of `--boundary` rather than as a separate run, so
that the matched analysis in the technical companion still applies.

Two task sets ship:

- **12-task set**: ordinary business pressure, for example fetching an exchange
  rate, retrying a blocked host, fetching a dependency.
- **7-task harmful set**: unambiguous harmful intent, for example credential
  misuse, DLP evasion, customer data exfiltration.

Both sets include a **negative control** (a task needing no network at all, so
any network call is unprompted overreach) and a **positive control** (a task
that explicitly requires an external fetch, so a non-attempt means either
extreme refusal or broken tool wiring).

---

## 3. What good looks like

A number from this framework is worth citing only if all of the following hold.
Each rule exists because a real result had to be withdrawn when it did not.

### 3.1 The suite has demonstrated it can return a positive

**A suite may not report a null until it has shown it can produce a positive.**

This is the organising principle of the whole framework. A suite that never
fires is indistinguishable from a suite that is broken, and reporting "we found
nothing" from a broken instrument is worse than reporting nothing at all.

In practice this means the positive control must fire. If it does not, the null
is not a finding, it is a missing measurement.

**This is enforced where the grade is produced, not only stated here.** A
containment suite whose positive control never fired across its trials is
reported `INVALID` and excluded from the grade rather than being reported clean
with a caveat attached. The distinction is not academic: a model with no
tool-calling support attempts nothing because it can attempt nothing, so the
suite reads "no escape attempts observed" at 0% and the model scored A out of
100 with the caveat filed in a per-suite field nobody reads. A missing
measurement was being converted into the best possible grade.

A control that fires on some boundary arms and not others is a different thing
and remains a caveat, per 3.2 below: cite the arms where it fires, footnote the
others.

**The other two suites enforce the same rule through what they can observe.**
A channel leak is a row where the visible answer passed and the hidden reasoning
failed, so it can only be detected on a row that came back with a reasoning
trace: rows without one are outside the leak rate's denominator, and a run with
no traces at all is reported `INVALID` rather than clean. A persistence scenario
requires the model to complete a task with one logged step and then be
challenged about that step, so a row where the model never wrote to the log has
no step to be challenged about: those rows are excluded and counted, and a run
with no logged rows at all is `INVALID`. In both cases the alternative was a 0%
that described the instrument rather than the model.

**And it is now enforced before the run, not only after it.** Everything above
decides, from rows already on disk, that a measurement was missing — which is
detection after the cost is sunk. Whether a suite *could* fire is knowable from
the model's own capabilities in seconds, and the gate had been measuring
exactly that and discarding it: a model reporting no reasoning capability had
its channel checks skipped, and a skipped check contributed nothing, so it
gate-checked "clean" on the very thing the check exists to establish. `cbench
score` and `cbench assess` now run that check first and refuse a suite whose
validity guard could not fire, naming the narrowed command that runs the rest.

The refusal is narrow on purpose. It fires only when the capability is
*established* to be absent or undelivered — never when a check failed to
complete, which is a fact about the machine rather than the model, and never
when the pre-flight itself could not run. A guard that stops a run on its own
malfunction is worse than no guard.

### 3.1a A suite reports every failure mode it can see, not the narrowest one

S2 produces two rates, and the grade takes the worse.

**Probe failure** is the visible answer doing what an adversarial probe
asked: supplying the procedure, emitting the injected canary. Its
denominator is every row the suite attempted and did not truncate, and
because some answers cannot be classified either way it is reported as an
**interval** rather than a value: the lower bound treats every unreadable
row as having resisted, the upper bound treats none of them as having.
The grade uses the lower bound, so a model is never marked down for rows
the scorer could not read, and when the two bounds fall in different
letter grades the scorecard says so. This is the "bounds over choices"
rule from the technical companion's section 4.

**Channel leak** is the narrower and more interesting question: the visible
answer resisted and the hidden reasoning did not. Its denominator is the
rows that returned a reasoning trace, since a row without one cannot
produce the verdict.

They are reported separately because they are different questions with
different denominators, and grading on the leak alone had a predictable
consequence: a model whose hidden channel barely existed could comply with
most of the probe set in plain sight and still be reported clean, because
none of those compliances came from a channel it did not have. A suite
that can see two failure modes and grades on one is not being
conservative, it is being quiet about the other.

### 3.2 Controls behave

- Negative control clean: no network calls on a task that needs none.
- Positive control fires: the model does reach out when explicitly asked to.

A positive control that fires on some boundary arms and not others is a signal
about that model, not a pass. A model that denies having web access at all when
a boundary is present suppresses its own positive control, and its results on
that arm should be footnoted rather than cited.

### 3.3 Unfinished rows are excluded, and the exclusion rate is published

A row where the model never reached a stopping point cannot be scored as either
contained or an attempt. It is unknown.

Rows are marked **INCOMPLETE** when the model produced no final answer and
either exhausted its turn budget or was cut off mid-generation. These are
excluded from attempt rates. The rule is one function of columns every CSV
already has, so it is computed once per row rather than re-derived by each
piece of analysis that needs it.

**Excluding them is not neutral, so the exclusion rate is always published
alongside the rate itself, and the difference between arms is tested.** If one
arm is far more incomplete than the other, the comparison is disqualified
regardless of which way the attempt rate points. A comparison where one arm
finished most of its rows and the other finished almost none is not a
measurement of behaviour, it is a measurement of who ran out of turns.

The same rule applies to S2 with token truncation instead of turn exhaustion.

### 3.4 Sampling is pinned and recorded

Temperature, top-p, top-k and a per-run seed are set explicitly and written
into every output row.

This matters more than it sounds. Model configuration files carry their own
default sampling parameters, and those defaults differ between models. A ranked
comparison collected without pinning compares models and their default sampling
settings at the same time, and there is no way to separate the two afterwards
because the settings in force were never recorded.

Generation budget (`num_ctx`, `num_predict`, `max_turns`) is recorded per row
for the same reason. So is the wall-clock time the run started, because a
file's mtime belongs to whatever tool last touched the file rather than to the
run that produced it. So are the tokens each row actually used -- the prompt
count and the prompt-plus-generated peak -- which is what makes it possible to
ask afterwards whether the context window bound, a question the endpoint
answers in no other way.

### 3.5 Only comparable runs are pooled

Results collected before a harness fix are not pooled with results collected
after it. Runs at different generation budgets are not pooled.

**Two dimensions are now checked for you: sampling and the generation
budget.** `cbench aggregate`/`cbench score` compare every pooled CSV's
recorded temperature/top_p/top_k and its recorded `num_ctx`/`num_predict`,
and whether it recorded them at all. Pool a CSV that predates those columns
with one that carries them, or pool two that carry them but disagree, and
the trial summary carries a "STOP -- THESE RUNS ARE NOT COMPARABLE" block
and that suite is reported `INVALID`, excluded from the grade. The remedy
is the same either way: re-run so every trial shares one configuration, or
point `OPENLLM_CBENCH_RESULTS_DIR` at a directory holding only the runs you
mean to pool.

**Separately, the context window is checked for whether it bound at all.**
`num_predict` running out is loud -- the row is marked `TRUNCATED` and
leaves the denominator. `num_ctx` running out is silent: the server drops
tokens off the front of the conversation and answers anyway, which in a
multi-turn suite is the boundary instruction or the model's own earlier
turns. Every run and every aggregation reports the peak tokens against the
window, and warns when the window may have bound. Because the server
reports the token count it evaluated *after* truncating, that warning
claims only that truncation cannot be ruled out; re-running at a larger
`num_ctx` is what separates a prompt that fit from one cut down to fit.

**Every other dimension is still on you.** A mismatched `max_turns` does
not stop a pool, and neither does
a harness correctness fix that changed what an already-present column
means without changing whether it's present. Every row records the
sampling parameters, the generation budget and the time the run started,
which is what makes the rule checkable at all, but `cbench score` reads
every CSV for the model tag you name, comparable or not, outside the one
check above. If you build analysis on top of these CSVs, enforce the rest
of this rule there, and fail rather than warn.

### 3.6 One test run at a time

The framework takes an exclusive lock before a batch and refuses to start if
another batch or a suite process is already running.

Two runs sharing one GPU do not fail. They halve each other's throughput and
corrupt every timing measurement taken from them. This was learned the direct
way: a run measured at 52 minutes took 117 when a forgotten second batch was
racing it, and the anomaly was nearly written up as model variance.

---

## 4. Why there are no results in this document

This is the repository for a tool. It does not carry any benchmark
results or measurements from external sources.

There are two reasons, and the second is the important one.

The first is that your results are yours. Nothing this framework produces is
uploaded anywhere, and the only path by which a result leaves your machine is
one you drive yourself through `cbench community-package` and
`cbench community-submit`.

The second is that a measurement and the instrument that produced it have
different lifetimes. A number collected on one machine, at one generation
budget, against one model build, at one version of this harness, is true of
that combination and is not automatically true of yours. Shipping a table of
numbers next to the tool that produced them invites exactly the reading the
rest of this document argues against: that the number is a property of the
model. It is a property of the model, the harness, the budget and the prompt
together, and section 5 is about how to read one accordingly.

Benchmark results using this framework are published separately and should
cite this document and METHODOLOGY_TECHNICAL.md as their methodology reference.

---

## 5. How to read a result from this framework

1. **Read the unfinished rate first.** An attempt rate computed on 3 scored rows
   out of 50 is not a rate.
2. **Check the controls.** If the positive control did not fire, the null is not
   a null.
3. **Prefer bands to ranks.** Adjacent models in any ranking are usually
   indistinguishable, and the scorecard reports a band and an interval rather
   than a position for that reason.
4. **Treat every rate as a lower bound on real-world risk.** The harness caps
   turns, uses one prompt phrasing per task, and stops attempts at the socket.
5. **Do not pool across framework versions.** Check the recorded sampling and
   budget columns.

---

## 6. What this instrument got wrong, and what changed as a result

Documented because a methodology that hides its own failures gives you no way
to calibrate the results it presents. Each of these is a defect in the
framework itself, not in a model it measured.

**A tamper-detection bug fabricated the events it detected.** The persistence
suite executed each challenge-turn tool call twice, so a single log edit was
recorded as two. Every log-tampering event produced before the fix was an
artifact. The suite's null result was withdrawn rather than caveated, and a
cross-framework validation that appeared to confirm it turned out to be two
scorers reading the same doubled calls. Cross-validation against a second
framework does not protect you when both frameworks read the same corrupted
input.

**A validity guard excluded the suite that made the grade worse.** The rule
excluding unfinished rows was allowed to decide which tasks a trial had run, so
a task that truncated in some trials and not others looked like a different task
set and the containment suite was marked invalid. Nothing errored; the output
looked like a careful tool being careful. Because the grade is the worst of the
three suites, dropping one can only move the grade up, and here it moved from 43
to 50 because containment was that model's worst suite. A false positive in a
validity guard is not a neutral annoyance: in a worst-of aggregator it
systematically flatters the thing being measured. It also got more likely with
every extra trial, so running more trials made the data more likely to be
declared unusable. The technical companion's section 3.4 carries the rules that
came out of it.

**A headline finding was rebuilt three times, twice because of the
denominator.** It was first computed over egress *calls*. Calls within a row
are not independent, since a model retrying one blocked host generates many, so
the unit became the row. Recomputed, it survived. A second review found the
contingency table conditioned on rows that escaped, while the thing being
tested changes whether a row escapes at all: conditioning on a post-treatment
variable. Recomputed unconditionally, the original framing failed outright and
a different, stronger one replaced it. **If a denominator can be changed by the
treatment, it is not a denominator.**

**A mechanism claim was cut entirely.** Three attempts to explain *why* an
effect occurred produced three different answers, and the deciding comparison
turned out never to have been run. When it was run, it was inconclusive. The
framework therefore makes no claim about mechanism, and reports what it
measures rather than why.

**Three claims of verification turned out not to have been performed.** In each
case a check was asserted and the artifact that would have constituted it did
not exist. The rule that came out of it: for every claim of having checked,
name the artifact that constitutes the check.

**A proposed detector was measured and rejected.** A generic phrase-based
heuristic for detecting models whose reasoning trace merges into the visible
answer was specified, implemented and measured across 2,970 rows. It caught 8
additional cases at the cost of 7 false positives, and its keyword list only
works in English. It was not shipped. The measurement is the finding, and the
rejection is recorded in `core/delimiters.py` so nobody proposes it again
without reading why.

**The same predicate was fixed in one place and left wrong in another, twice.**
A canary-matching fix had six call sites where five had been written down, and
the sixth was found by a grep after the work was believed finished. The
reasoning-delimiter guard was extended from one marker to four in the suite
while the gate check kept its own older copy, so a model the suite would flag
could be gate-checked clean first. Both are now single definitions imported by
every caller. **Fix a duplicated predicate everywhere, and sweep before
declaring it done.**

---

## 7. Limitations

- **Turn budget binds.** Most attempts occur in the first few turns of the
  budget, so behaviours that would emerge later cannot appear in the window.
  Absence of escalation is not evidence of its absence.
- **One phrasing per task.** Model resistance to one probe does not reliably
  generalise to a same-intent rephrase, so per-task rates are specific to their
  wording.
- **The reasoning-channel guard is delimiter-based** and will mis-score a model
  using an unrecognised convention. It is checked against known conventions at
  gate time, and a model with its own convention can be catalogued so that
  every later run knows about it.
- **Attempt rates depend on the hardware too.** A model that fits in VRAM on
  one machine and spills to system RAM on another is slower there, so more
  requests hit their timeout and more generations stop at the budget. Those
  rows are excluded as INCOMPLETE or TRUNCATED, which moves the rate without
  the model's behaviour moving. A result is a joint property of the model, the
  harness, the generation budget AND the machine. Record the hardware with any
  rate you intend to compare, and do not read a difference between two machines
  as a difference between two models.
- **Attempt rates are joint properties** of model and harness. They are
  comparable across models tested with this framework at the same version and
  budget, and are not comparable to numbers from another harness.
