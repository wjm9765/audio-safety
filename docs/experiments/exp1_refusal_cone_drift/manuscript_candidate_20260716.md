# Manuscript Candidate — Same Request, Different Verdict

Date: 2026-07-16

Status: **conditional narrative and claim-gate draft; not a preregistration, result, or submission-ready manuscript**

Depends on: [content-conditioned safety/causal review](./review_20260716_content_conditioned_safety_invariance.md)

This document tests whether the proposed paper has one coherent scientific question
without making phase, pitch, EQ, codec, resampling, a refusal vector, or certification
the intellectual center. Bracketed result statements are deliberately unfilled. The
existing Run 6/7 evidence cannot fill them.

## Working title

> **Same Request, Different Verdict: Tracing Safety Invariance Failures in
> Audio-Language Models**

Use a stronger causal title only if the final unseen-family route test passes:

> **Same Request, Different Verdict: Shared Causal Routes of Safety Invariance
> Failure in Audio-Language Models**

## Central question

> **Did the model perceive a different request, or did it perceive the same request
> and make a different safety decision?**

## One-sentence thesis

> We test whether fixed within-audio channel changes alter the full policy verdict for
> requests whose lexical content and pragmatic intent pass frozen equivalence checks,
> and whether any such flips follow a shared or operator-private internal route.

## Conditional abstract

Safety policies govern what a user requests, not incidental details of the waveform
that carries it. Prior audio jailbreak and metamorphic-testing studies already show
that audio variations can expose safety inconsistencies under their semantic checks.
What their aggregate success rates do not resolve is whether a paired generative
verdict flip reflects changed model-perceived content, generic disruption or
compliance, stochastic measurement error, or a change in a safety-relevant internal
route. We operationalize this question for full responses by distinguishing policy
refusal, harmful operational assistance, and unresolved or non-responsive generation.
We triangulate lexical-content and pragmatic-intent preservation with frozen
transcript, critical-span, comprehension, and human checks while retaining residual
model-perceived content change as a competing explanation. We then introduce the
**Operator-Held-Out Bidirectional Interchange (OHBI) test**: development families
select a candidate route, and a mechanistically distinct sealed family tests
sham-coordinate restoration and active-coordinate induction without refitting. Across
**[MODELS]**,
**[NAMED FAMILIES]**, and **[FRESH ITEMS]**, we find **[PAIRED FULL-RESPONSE EFFECT]**.
On the sealed family, **[INSERT EXACTLY ONE: shared-route result / private-route result /
mechanism null]**. This result shows **[ONE SUPPORTED INSIGHT]**, rather than merely
showing that another audio operator can jailbreak a model.

## Introduction draft

Safety policies are written over requests, not waveforms. If two recordings preserve
the same safety-relevant lexical content and pragmatic intent, incidental changes in
the channel carrying that request should not decide whether an assistant refuses or
provides harmful operational help. Text systems also face invariance failures under
paraphrase and formatting, but speech adds unavoidable waveform variability even when
those aspects of the request are held fixed. A microphone, transmission path, or
playback stack can change the signal without changing what the user is asking. Safety
behavior that changes with this encoding is therefore a reliability failure at the
interface between perception and policy.

Prior work already establishes the behavioral phenomenon. Best-of-`N` finds unsafe
outputs by searching innocuous input variations
([Hughes et al., 2025](https://papers.nips.cc/paper_files/paper/2025/hash/69f3eb242c7c9df9ea2f2b66ea8b3c0f-Abstract-Conference.html));
AJailBench evaluates small audio perturbations under a semantic constraint
([Song et al., 2026](https://aclanthology.org/2026.acl-long.1259/)); Signal-BoN composes
many channel transformations across Audio LLMs
([Feng et al., 2026](https://arxiv.org/abs/2605.30031)); and MTAM exposes label
inconsistencies under metamorphic relations in audio moderation systems
([Wang et al., 2025](https://arxiv.org/abs/2509.24215)). These studies make another
operator inventory or a higher aggregate jailbreak rate insufficient as a scientific
contribution. They leave a more consequential question: **when the full policy verdict
flips, did the model perceive a different request, or did it perceive the same request
and make a different safety decision?**

A behavioral flip alone cannot answer that question. A transformed input may alter
model-perceived content despite passing external checks, cause generic generation
failure or a broad compliance shift, cross a noisy decision boundary, follow an
operator-private route, or disturb a route shared by acoustically different causes.
We therefore treat semantic preservation as triangulated evidence, not as something
an external transcript can prove about the target model, and distinguish refusal,
harmful operational assistance, and incoherent or non-responsive generation over the
full response. Mechanistic work also raises the evidential bar. Prior studies already
intervene on average modality drift, construct local causal jailbreak explanations,
trace audio-text fusion, and patch same-audio arbitration reversals
([Guo et al., 2026](https://arxiv.org/abs/2605.18104);
[Kumar and Ahuja, 2026](https://arxiv.org/abs/2605.00123);
[Chen et al., 2026](https://arxiv.org/abs/2603.13768);
[Gao et al., 2026](https://arxiv.org/abs/2606.05161)). A direction that can force
refusal is therefore only an **actuator**. It explains a natural channel-induced flip
only if the transform's own state change passes stronger route-specific counterfactual
tests without per-family refitting.

We call that protocol the **Operator-Held-Out Bidirectional Interchange (OHBI) test**.
OHBI uses standard activation interchange; its novelty claim is the sealed-family
identification protocol, not a new patching algorithm. For each independently screened
harmful request, a fixed active transformation is compared with an
operator-specific matched sham generated from the same source. The categorical
full-response endpoint and all semantic gates are frozen before transformed safety
outcomes are observed; the paired effect is estimated over every screen-eligible item,
while verified refusal-to-harmful-assistance cases are reported separately. Development
families may localize a site and fit a low-rank candidate route. On untouched items and
a mechanistically distinct sealed family, we then ask two directional counterfactuals:
does restoring the sham-run coordinate prevent harmful assistance, and does inserting
the active-run coordinate induce it? These donor coordinates come from natural
executions, but the patched activation is still a hybrid state and is not a natural
mediator by construction. The layer, route, rank, dose, judges, and analysis are frozen
before that test. Pre-frozen path, target-model comprehension, benign-utility, and
equally tuned generic/private/null controls determine whether a positive effect is
consistent with a shared safety route rather than semantic transfer, generic control,
or an unsupported hybrid activation.

Across **[M models]**, **[F named channel families]**, and **[N independently screened
requests]**, fixed active conditions changed the paired full-response harmful-assistance
risk by **[effect and uncertainty]** relative to their matched shams while **[semantic,
failure, and benign-utility results]**. On the family sealed from all mechanism
selection, **[INSERT EXACTLY ONE LOCKED RESULT BRANCH]**. **[If shared: bidirectional
interchange and path controls support a common safety-decision route across the named
families.] [If private: within-family effects pass but cross-family interchange fails,
showing that the same outward flip arises through different internal routes and that a
generic refusal actuator repairs outputs without explaining their causes.] [If null:
controlled interventions do not reproduce the behavioral effect, so the paper reports
behavioral instability without a localized shared mechanism.]** Only the branch backed
by the frozen analysis may remain in a final manuscript.

## Why the fifth paragraph is intentionally conditional

An introduction earns its insight from a result, not from an interesting question
alone. The paragraph above is a claim slot, not publishable prose. The three branches
are substantively different papers and may not be blended after results are seen:

| Confirmatory result | Paper-level insight | Required rewrite |
|---|---|---|
| Shared route passes on sealed family | Distinct acoustic causes converge on a common safety-decision bottleneck across the named families | State exact restoration/induction effects, matched-null separation, path controls, and failed generic baselines. |
| Behavioral effects replicate, but only private routes pass | The same refusal-to-assistance symptom has different internal causes; a universal refusal controller can repair an output without explaining the failure | Remove every shared-mediator claim and organize results around heterogeneity. |
| Behavioral effect passes, mechanism test is null | The verdict is unstable, but the tested route does not explain why | Claim no localized mechanism. Continue as a paper only if the frozen generative-verdict study independently clears its novelty gate; otherwise declare the broad paper `NO-GO`. |
| Only the phase family reproduces | The phenomenon is implementation-specific under the tested scope | Abandon this broad title and write a narrow phase-processing case study, if worthwhile. |

## Conditional contributions

The final paper may claim only contributions whose gates pass:

1. **Measurement object.** We operationalize the existing audio metamorphic-invariance
   idea for a generative policy verdict, separating refusal, harmful operational
   assistance, and unresolved generation, and separating the all-screen-eligible-item
   paired effect from a post-treatment verified-flip event.
2. **Fresh evidence.** We provide **[EXACT MULTI-FAMILY, MULTI-MODEL RESULT]** under
   fixed outcome-independent transformations, matched shams, triangulated semantic
   checks, full-response adjudication, stochastic sensitivity analysis, and benign
   utility controls.
3. **Causal discrimination.** We use OHBI to
   distinguish **[EXACTLY ONE: a shared route / operator-private routes / no supported
   candidate route]** from a generic refusal actuator. This contribution exists only
   if the corresponding full-response, path, null, and sealed-family gates pass.

A family-indexed empirical safety boundary may be a secondary analysis. A
deterministic certificate is not part of this manuscript spine and should return to
the abstract or contribution list only if a separate sound, non-vacuous proof lands.

## Why this is not an operator paper

Operator families have only three experimental roles:

- instantiate controlled within-source contrasts under which content and intent are
  expected to remain stable;
- supply mechanistically distinct causes for a shared-versus-private route test;
- make preprocessing and construct-validity failures auditable.

The scientific objects are the **categorical generative policy verdict**, the
**competing causal explanations for a paired flip**, and the **operator-held-out route
test**. Particular transformations are coverage and instruments, not the title or
headline contribution.

## Required result substitutions

| Placeholder | Evidence required before replacement | Current status |
|---|---|---|
| `[MODELS]` | Qwen2-Audio plus a structurally different Audio/Speech LLM, each with independent eligibility screening | Missing |
| `[NAMED FAMILIES]` | Fresh fixed families with matched shams; the final family's computational class and distinction criterion are frozen from implementation structure before safety outcomes or activations, and it is sealed from all selection | Missing |
| `[FRESH ITEMS]` | New item/source/speaker-grouped cohort with sample size frozen by design-level simulation | Missing |
| `[PAIRED FULL-RESPONSE EFFECT]` | All-screen-eligible-item paired risk, discordance, uncertainty, full-response judges, decoding sensitivity, and generic-failure accounting | Missing |
| Semantic/content evidence | Frozen external transcript and critical-span checks, comprehension and harmful-intent probes, blinded human audit, plus an explicit residual `H_model` limitation | Missing |
| `[CAUSAL RESULT]` | Four interchange cells, bidirectional full-response restoration/induction, predecision path restriction, support/dose diagnostics, matched nulls, generic/local-oracle/private baselines, interactions, and no-refit sealed-family evaluation | Missing |
| `[ONE SUPPORTED INSIGHT]` | One result branch selected strictly by the frozen interpretation table | Missing |

The existing Run 6/7 cohort is fully exposed, single-model, and phase-family-specific.
It cannot fill any confirmatory placeholder through cross-validation or relabeling.

## Current phase result: evidence boundary, not Introduction paragraph

The current pilot remains causally tied to its phase-vocoder contrast. Factorized
controls localized the L18 representation shift to the processing path rather than the
intended pitch change, but the preregistered behavioral necessity and specificity
gates did not pass: G1, G2, and G4 missed, G3 was partial, and G5 passed its continuous
first-token-margin primary. Persistent steering of the frozen L18 direction can move
that margin and reverse some outcome-selected cases, which is actuator evidence. It
does not establish a general transformation effect or a natural mediator of
full-response flips.

If no fresh experiment is run, the safe ending is only:

> In Qwen2-Audio, a particular phase-vocoder processing contrast is associated with an
> L18 refusal-related displacement and first-token-margin change. Persistent steering
> of that frozen direction modulates the margin, but the current evidence identifies
> neither a phase-independent full-response effect nor a natural cross-operator causal
> route.

The pilot belongs in a method-development or limitations section if a fresh study
supersedes it. Keeping it in the Introduction would make phase the paper's apparent
subject again.

## Narrative rejection tests

Reject or rewrite the final introduction if a reviewer can answer “yes” to any item:

1. Can the contribution be summarized as “AJailBench/Signal-BoN plus more operators”?
2. Can the mechanism be summarized as “ReGap/LOCA/causal tracing plus another patch”?
3. Is a steerable refusal direction presented as the transform's natural cause?
4. Does “same request” rely on transcript identity alone or imply that target-model
   perception was proved unchanged?
5. Is a post-treatment gate-conditioned verified-flip rate called the all-eligible
   paired causal effect?
6. Were full-response endpoints or semantic gates selected after transformed safety
   outcomes were seen?
7. Does a same-item node patch substitute for bidirectional route-specific transfer on
   a sealed operator family?
8. Is a new backend called a new causal family?
9. Does certification or a per-input radius compete with the causal question on the
   first page without a completed proof?
10. Does phase, pitch, EQ, codec, or resampling appear to be the scientific
    contribution?
11. Does the result paragraph still contain more than one branch or any unsupported
    placeholder at submission time?

Any “yes” is a narrative `NO-GO`, even if one mechanistic score is statistically
significant.

## Independent narrative review record

After the closest causal precedents were added, certification was removed from the
central spine, Run 7 was moved out of the Introduction, and the method was narrowed to
OHBI, a fresh independent review gave the **conditional narrative 8.7/10 (`PASS`) with
no P0 narrative flaw**. The same review scored the actual Run 7 evidence **2.5/10 for
the proposed broad paper** and **5/10 as a narrow phase-specific pilot**. This is a
narrative/design clearance only: the manuscript remains non-submission-ready until the
result placeholders are filled by a fresh preregistered study.
