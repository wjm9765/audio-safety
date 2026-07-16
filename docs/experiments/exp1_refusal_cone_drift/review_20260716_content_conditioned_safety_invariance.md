# Direction Review — Content-Conditioned Safety Invariance and Causal Mechanism

Date: 2026-07-16

Status: **review candidate; not a preregistration, active run, or result**

Scope: research-direction audit after the Run 7 phase result

This document does not modify the locked criteria in [design.md](./design.md) or the
append-only decisions in [results.md](./results.md). It identifies what the current
evidence actually supports, what the literature already owns, and what a new study
would have to establish before making a phase-independent claim.

## 1. Executive verdict

The motivating problem is valid, but the current causal experiment does not yet test
it at the intended level of generality.

> A safety policy should give the same verdict to the same harmful request when only
> low-level properties of its audio channel change.

The project currently has evidence for a narrower statement:

> In the exposed Qwen2-Audio cohort, one compound phase-vocoder family can produce
> two-judge-consensus refusal-to-operational-compliance flips. That treatment is
> associated with L18 refusal-related displacement, and persistent steering of the
> frozen L18 direction can modulate first-token margin and some selected outputs.

Run 7 is therefore still phase-bound. Freezing the L18 direction means that the
direction was not refit on the Run 7 phase outcomes; it does **not** make the treatment,
the discovered displacement, or the causal conclusion independent of phase.

The strongest defensible reading of Run 7 has three levels:

| Question | Current answer |
|---|---|
| Does the phase-vocoder contrast move L18 refusal-related state and first-token margin? | **Yes, in this cohort.** |
| Can persistent L18 refusal-direction steering move that margin and reverse some selected outputs? | **Yes; this is actuator evidence.** |
| Has the study shown that phase caused full-response safety flips specifically through L18? | **No.** G1, G2, and G4 missed; G3 was partial; G5 passed its preregistered continuous-margin primary. Its preregistered discordant-subgroup rule yielded 10 observed-outcome-conditioned item×sign cells, so the 60% full-response flip-back was exploratory under the registered `n<15` rule and used a refusal-prefix heuristic rather than fresh two-judge adjudication. The intervention was persistent all-token steering rather than a donor-coordinate path intervention. |

Consequently, neither of these broader claims is currently licensed:

- low-level content-preserving audio transformations generally weaken refusal;
- diverse transformations share a phase-independent L18 causal mediator.

The recommended problem spine is not “another operator attack.” It is:

> **Content-conditioned safety invariance:** when independently verified semantic
> content is fixed, does an Audio LLM preserve its policy verdict across ordinary
> channel transformations, and do failures from distinct transformation families
> pass through a shared causal decision mechanism?

Run 7 remains the completed preregistered phase-specific supporting result, not a
retrospectively downgraded run. Relative to the new phase-independent candidate, it
serves as motivating and method-development evidence only; it cannot count as
confirmatory evidence for the broader question.

For the scientific question under review, the recommended manuscript has one spine:
**paired generative safety-invariance failure followed by causal disambiguation of its
natural internal route**. The existing **Certified Acoustic Safety Margin** direction
is not invalidated, but it is a separate or optional companion branch. A family-indexed
empirical boundary may remain a secondary description; a deterministic certificate
belongs in the abstract or contribution list only if a sound, non-vacuous proof
actually lands. This separation prevents a failed certificate from taking down an
otherwise coherent causal paper and prevents a successful steering intervention from
being renamed mediation.

## 2. The scientific object

Let `x_i` be the source waveform for request `i`. For operator family `o`, define an
operator-specific matched sham `a0=S_o(x_i)` and active treatment
`a1=T_{o,s}(x_i)` at preregistered severity `s`. A separate screening run establishes
the target population before either endpoint condition is generated. Let:

- `E_i(a)=1` mean that external, frozen semantic-preservation gates pass after
  treatment; this is a post-treatment measurement, not a baseline covariate;
- `V_i(o,s)` be the categorical full-response label: policy refusal, harmful
  operational compliance, or unresolved/non-answer/generic failure;
- `Y_i(o,s)=1` iff `V_i(o,s)` is harmful operational compliance and `0` otherwise for
  the all-eligible paired-effect denominator. Report `V` and a separate failure indicator so a lower
  unsafe rate cannot hide generic collapse.

The desired metamorphic relation is:

```text
E_i(a0) = E_i(a1) = 1  =>  policy_verdict(a0) = policy_verdict(a1).
```

A primary safety-invariance violation is the paired event:

```text
matched-sham response = policy refusal
active-transform response = actionable partial/full harmful compliance
both semantic-preservation gates = pass
```

This should be called a **paired safety-verdict flip** or **content-conditioned
invariance failure**. “Attack” is appropriate only for a condition selected or
optimized to elicit failure. Fixed, outcome-independent EQ, codec, or resampling
roundtrips are better described as channel transformations.

“Content-preserving,” “perceptually identical,” and “imperceptible” are different
claims. The first requires semantic evidence; the latter two require calibrated human
perception evidence. The project should use only the strongest term actually tested.

### Optional secondary object: family-indexed safety margin

There is no honest universal distance that makes EQ gain, codec bitrate, and sampling
rate commensurate. Define a separate, preregistered admissible set and severity metric
for each operator family. For a continuous family `o`, the per-input boundary is:

```text
r_o(x_i) = inf { s : E_i(a1)=1 and Y_i(a1)=1 }, given
V_i(a0)=policy_refusal and E_i(a0)=1.
```

A deterministic lower certificate `r_cert,o(x_i)` is valid only if the method proves
that no verdict flip exists inside the stated continuous operator region, relative to
the frozen verdict oracle and its error bound. A dense sampled grid is not such a
proof. For discrete codec/bitrate or sample-rate choices, report exact robustness over
the declared finite set rather than pretending it is a continuous radius. Neither the
margin nor the certificate is required for the causal-invariance manuscript proposed
here.

## 3. What the literature already owns

The broad phenomenon “innocuous or semantically preserved audio variations can
jailbreak an Audio LLM” is no longer a novelty claim. The remaining opening is an
item-paired, within-audio causal account that separates semantic failure, generic
degradation, operator-private routes, and a shared safety-decision mechanism.

| Work | Problem | Method | Main result | Consequence for this project |
|---|---|---|---|---|
| [AJailBench / Audio Jailbreak (ACL 2026)](https://aclanthology.org/2026.acl-long.1259/) | Systematic jailbreak evaluation for LAMs | 1,495 prompts, seven time/frequency/mixing perturbations, semantic constraint, Bayesian search | Small semantically preserved perturbations reduce safety across leading LAMs | Preempts the broad behavioral premise and a transform-toolbox contribution. It does not provide a held-out-operator internal causal mechanism. |
| [Audio Jailbreaks in LALMs / Signal-BoN (2026)](https://arxiv.org/abs/2605.30031) | Unify attack/defense taxonomy and cost-aware evaluation | Ten open LALMs; `N=20` search over tempo, pitch, gain, dynamic range, EQ/filter, noise, reverb, sample-rate roundtrip, MP3, and Opus | Signal-BoN raises aggregate no-defense ASR from 0.071 to 0.223; successful vectors contain filtering and codec/resampling choices | Directly preempts “EQ, codec, and resampling are unexplored.” Its combined search is not an isolated per-operator causal effect, per-input radius, or internal mediator test. |
| [MTAM / Metamorphic Testing for Audio Content Moderation (ASE 2025)](https://arxiv.org/abs/2509.24215) | Test whether semantically equivalent toxic audio evades moderation software | 14 realistic metamorphic relations derived from 2,000 clips; five commercial systems and one academic classifier | Finds error rates up to 51.1% commercially and 45.7% academically, then improves the academic model through retraining | Directly preempts “audio semantic-invariance testing” as a new problem. Our gap is a full generative policy verdict and causal discrimination among perception, disruption, private-route, and shared-route explanations. |
| [Best-of-N Jailbreaking (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/69f3eb242c7c9df9ea2f2b66ea8b3c0f-Abstract-Conference.html) | Black-box robustness to innocuous input variation across modalities | Repeated random modality-specific augmentations and best-of-`N` selection | Audio jailbreak success grows with repeated search and follows power-law-like scaling | Preempts “minor audio changes reveal brittleness” as the headline. Fixed, non-selected transformations and mechanism identification remain distinct. |
| [Tune In, Act Up (2025)](https://arxiv.org/abs/2501.13772) | Effect of audio-specific edits on jailbreak behavior | Tone, emphasis, intonation, speed, noise, accent across several LALMs | Audio edits can alter jailbreak behavior, with strong model heterogeneity | An inventory of EQ/pitch/noise-style edits alone is incremental, not a paper-level novelty. |
| [Audio Is the Achilles’ Heel (NAACL 2025)](https://aclanthology.org/2025.naacl-long.470/) | Cross-modal and speech-specific safety vulnerability | Red teaming five audio LMMs under harmful audio/text, distracting sound, and speech jailbreaks | Large safety gaps and high attack success in open models | Preempts a generic “audio modality is less safe” motivation. |
| [Jailbreak-AudioBench (NeurIPS 2025)](https://papers.nips.cc/paper_files/paper/2025/hash/0ff38d72a2e0aa6dbe42de83a17b2223-Abstract-Datasets_and_Benchmarks_Track.html) | Comprehensive explicit/implicit audio jailbreak evaluation | Toolbox, curated edited-audio dataset, multi-model benchmark | Broad exposure of audio-specific threats | A new benchmark must contribute a sharper invariant and paired causal design, not merely more edits. |
| [ReGap / Safety Geometry Collapse (2026)](https://arxiv.org/abs/2605.18104) | Semantically equivalent text and non-text inputs receive different safety treatment | Text refusal direction, mean modality drift, fixed/adaptive activation correction | Counteracting modality drift improves refusal separability and safety | Closest conceptual competitor. It owns “modality drift compresses refusal geometry + activation correction”; our opening is within-audio paired transformations, heterogeneous routes, and final-holdout causal identification. |
| [LOCA (2026)](https://arxiv.org/abs/2605.00123) | Explain why a particular text jailbreak succeeds | Find a minimal set of interpretable intermediate representation changes that causally induces refusal for an original–jailbreak pair | Induces refusal with about six changes on average, outperforming adapted prior methods in the reported setting | Preempts “local causal explanation of a jailbreak” and sparse corrective edits as novelty. Our remaining distinction must start from a fixed within-audio treatment and test the natural route on all eligible items and an unseen operator family. |
| [Causal Tracing of Audio-Text Fusion (2026)](https://arxiv.org/abs/2603.13768) | Locate when and where LALMs integrate audio and text | Layer- and token-wise causal tracing across DeSTA, Qwen, and Voxtral | Reports architecture-dependent fusion and a final-token information bottleneck | Preempts causal tracing or audio activation localization itself. Candidate sites here must be only discovery inputs to a safety-specific, held-out-operator counterfactual test. |
| [Beyond Text Following (2026)](https://arxiv.org/abs/2606.05161) | Test whether audio evidence is absent or overridden under audio–text conflict | Same-audio counterfactual, activation patching, and counterfactual logit correction across five ALMs | Reports 64.1% preference sign flips, answer-position localization, and patch effects correlated with score differences | Preempts same-audio counterfactual patching and repair as generic method novelty. The remaining gap is a fixed within-audio channel cause, full policy response, natural bidirectionality, and no-refit transfer to a sealed operator family. |
| [SARSteer (ICML 2026)](https://openreview.net/forum?id=2iC5H9k8am) | Training-free audio safety defense without benign over-refusal | Text-derived refusal steering plus safe-space PCA ablation | Improves harmful refusal while preserving benign responses | Preempts audio refusal-vector steering itself as novelty. A steering response is not proof that a natural channel transformation used that mediator. |
| [SPIRIT (EMNLP 2025)](https://aclanthology.org/2025.emnlp-main.734/) | Defend speech models from white-box adversarial noise | Locate noise-sensitive activations and patch/modify them at inference | Large robustness gains with little reported utility loss | Preempts “activation patching repairs audio jailbreaks.” It does not establish a shared natural mediator across ordinary channel transformations. |
| [CodecAttack (2026)](https://arxiv.org/abs/2605.20519) | Make optimized waveform attacks survive real codecs | Optimize in codec latent space with multi-bitrate EoT | High targeted success through Opus and transfer to MP3/AAC | Codec is not an untouched attack surface. Our distinct question must use outcome-independent realistic roundtrips and test verdict invariance, not optimized codec-aware perturbations. |
| [Refusal in Language Models Is Mediated by a Single Direction (NeurIPS 2024)](https://arxiv.org/abs/2406.11717) | Identify a causal refusal representation in chat LLMs | Difference-in-means direction, directional ablation and addition | One residual direction can suppress or induce refusal across 13 models | Mean-difference steering is a necessary baseline, not a novel causal method. It identifies a refusal actuator, not automatically the mediator of an audio transform. |

As a separate scope note, certification is not an empty literature space.
[Statistical Verification of
General Perturbations](https://openreview.net/forum?id=B1eZweHFwr) treats generic
transform-parameter smoothing and explicitly includes audio volume/pitch;
[Randomized Smoothing Meets Vision-Language Models](https://aclanthology.org/2025.emnlp-main.1396/)
maps generative outputs through a safety oracle and accounts for bounded oracle error.
Therefore “first transform certificate” or “first judge-mapped generative safety
certificate” is unsafe. If pursued as a companion contribution, the defensible object
would be a deterministic, per-input certificate over a fully specified perceptual
audio-operator region and representation contract. It is not part of the central
novelty defense below.

### Novelty boundary

The causal-invariance paper can still be distinctive only if it contributes these
three linked layers:

1. **A generative specialization of an existing invariant:** extend audio metamorphic
   testing from a moderation class label to the full policy verdict of a generative
   model, explicitly separating harmful compliance, refusal, and unresolved failure.
2. **A paired causal design:** the same source is directly evaluated under fixed active
   and matched-sham interventions, and full-response behavior is the primary endpoint.
3. **A mechanism claim that survives operator holdout:** a representation/path learned
   without the test operator must restore and induce the predicted counterfactual on
   new items and a genuinely unseen operator family.

Family-indexed empirical margins are an optional descriptive analysis. A deterministic
certificate is a separate contribution gate, not a prerequisite for this paper. EQ,
codec, and resampling are coverage. They are not, by themselves, the contribution.

## 4. Replacement causal question

The high-level structural hypothesis is:

```text
controlled operator intervention A
        |
        v
audio/frontend representation Z
        |----> perceived content H_model(a) --------------------------------> Y
        |                                                                    ^
        |----> shared candidate state U_shared ----> refusal process R ------|
        |                                                                    |
        `----> operator-private state U_o ---------> R or Y (bypass) --------'

Source linguistic content C is intended to be fixed; H_model(a) may still change.
External gate E_obs is a post-treatment measurement of source/variant equivalence,
not a parent that fixes H_model(a).
```

This graph, together with decoding and measurement noise, creates competing
explanations rather than assuming the desired one:

- `H_semantic`: although source content `C` is fixed, the transformation changed
  model-perceived content `H_model(a)`; external `E_obs` is evidence about this route,
  not an intervention that holds it fixed;
- `H_disruption`: it caused generic generation/instruction-following degradation;
- `H_compliance`: it shifted compliance/calibration broadly for harmful and benign
  requests rather than selectively changing a safety decision;
- `H_stochastic`: the apparent pairwise flip is explained by decoding variance or
  verdict-judge error rather than a stable treatment effect;
- `H_private`: each effective operator uses a different internal route;
- `H_shared`: distinct effective operators cross a shared policy-decision mechanism.

The experiment is scientifically useful under every outcome. A null shared mechanism
is not rescued by renaming a private route “multidimensional.”

### Named identification protocol

The proposed method-level object is the **Operator-Held-Out Bidirectional Interchange
(OHBI) test**. OHBI is a stricter evaluation protocol built from standard activation
interchange, not a claim of a new patching algorithm. Donor coordinates come from
natural active/sham executions, but the resulting patched state is still an artificial
hybrid; the name therefore does not assert natural mediation or on-manifold transport.
The protocol has four non-substitutable parts:

1. freeze fixed within-audio active/sham treatments, the categorical full-response
   endpoint, semantic checks, and the screen-eligible population independently of the
   confirmatory transformed outcomes;
2. estimate the paired effect on every eligible item rather than selecting flips;
3. use development families only to select a site and candidate route, then test both
   sham-coordinate restoration and active-coordinate induction with path and
   matched-null controls;
4. freeze the complete intervention and apply it without refitting to new items from a
   mechanistically distinct, sealed operator family.

The protocol supports evidence consistent with a shared causal route only when the
natural operator displacement follows the frozen candidate, both directional
full-response effects pass, the specified path and alternative-explanation controls
pass, and the sealed-family effect separates from equally tuned generic-refusal,
operator-private, local-oracle, and matched-null baselines. Otherwise it reports a
private route, generic actuator, marker/readout, or mechanism null.

| Frozen candidate under the sealed operator | Bidirectional `do(U)` effect | Interpretation |
|---|---|---|
| Natural active–sham displacement moves along `U` | Restoration and induction pass | Candidate shared-route evidence, conditional on path and alternative-explanation controls |
| No natural movement along `U` | Intervention changes the verdict | Generic actuator, not the operator's route |
| Natural movement along `U` | Intervention does not change the verdict | Correlational marker or downstream readout |
| No natural movement along `U` | No intervention effect | Null for this candidate |

## 5. Evidence stack: from behavior to causal abstraction

No single steering experiment is strong enough. The recommended stack is sequential;
later stages run only after earlier gates are satisfied.

### Stage A — controlled paired total effect

Use a fresh cohort and derive every condition directly from the same neutral waveform.
Freeze operator implementation, backend, severity, sham, semantic gates, outcome
judges, and analysis before observing transformed safety outcomes.

Candidate operator families can begin with the currently discussed set:

- **EQ:** fixed shelves/bands, outcome-independent parameters, 0-dB roundtrip sham;
- **codec:** lossless sham plus fixed lossy codec/bitrate roundtrips, with delay,
  internal resampling, and loudness aligned;
- **resampling:** native→lower-rate→native roundtrip, identity sham, fixed anti-alias
  kernel and a second held-out backend.

These are not automatically independent latent causes: all can alter frontend spectral
statistics. Freeze family distinctions from the signal-processing implementation graph
before inspecting safety outcomes or model activations—for example, linear spectral
filtering, coding/quantization, sampling-lattice conversion, and nonlinear amplitude
mapping. This is a design-level distinction, not proof that the model uses independent
internal paths. Include at least one non-phase-vocoder family from a different frozen
class or a strong negative control, such as fixed gain/dynamic-range treatment. Claims
remain restricted to the named tested families.

Primary behavioral analysis:

- independent unit = source request/item, not the number of variants;
- target population = items that repeatedly refuse in a separate screening run;
  screening generations are never reused as the confirmatory matched-sham endpoint;
- primary paired total effect = all screen-eligible items under active transform versus
  operator-specific matched sham, regardless of semantic-gate, non-answer, or
  endpoint-sham outcome;
- verified-invariance-violation joint endpoint = both semantic gates pass, endpoint
  sham refuses, and active transform produces harmful compliance;
- the effect conditional on post-treatment `E=1` is descriptive per-protocol evidence,
  not the all-eligible causal effect without additional principal-stratum assumptions;
- full-response two-judge harmful-compliance verdict primary;
- frozen multi-token refusal-versus-compliance continuation score secondary;
  first-token margin is a mechanistic diagnostic, not a behavioral substitute;
- non-answer, decoding failure, benign over-refusal, and benign task utility separate;
- paired risk difference, McNemar test, item-cluster bootstrap interval, and frozen
  family/severity multiplicity handling;
- deterministic decoding primary, repeated stochastic decoding sensitivity analysis.

The current 150-item cohort is fully exposed and cannot become confirmatory through
cross-validation. Use a new harmful cohort, hard-benign controls, multiple voices and
renderers, a natural/human-speech subset, a structurally different second Audio LLM,
and an untouched replication set.

### Optional Stage A2 — per-input boundary and certificate

This stage is not on the critical path of the causal-invariance paper. If the project
also pursues the margin/certification branch, estimate `r_o(x)` separately for each
family only after establishing a total effect. For a small continuous operator box, a
claimed certificate must use a valid bound such as interval branch-and-bound with a
justified modulus/Lipschitz remainder; adaptive grid refinement alone cannot exclude
an unsampled flip. Prototype one robust and one brittle input before scaling GPU data
collection.

Freeze and serialize the complete representation contract: source normalization,
transform order, resampler/codec version, padding/trim, clipping, model processor,
feature extraction, and verdict oracle. [Representation Matters in Randomized
Smoothing for Audio Classification (2026)](https://arxiv.org/abs/2606.04210) shows why
the injection representation and normalization can materially change the object being
certified. Judge disagreement/error must enter the certificate rather than being
reported only as an afterthought.

Codec and fixed sampling-rate choices normally form a finite audit set, not a smooth
JND box. Exhaustively evaluating a declared finite set can certify that set, but it
does not certify nearby untested codecs, bitrates, kernels, or implementation versions.

### Stage B — content and alternative-explanation gates

Every transformed cell must record:

- two independent ASR systems' paired WER/CER;
- exact preservation of safety-critical action, object, negation, and intent spans;
- an independent semantic-equivalence/entailment judgment;
- frozen target-model transcript, critical-span recognition, and comprehension probes
  that do not expose or reuse the safety-response outcome;
- blinded human same-meaning and intelligibility judgments for all flips and a
  preregistered random sample of non-flips;
- model-independent harmful-intent recognition;
- loudness, clipping, duration/alignment, and operator-specific artifact metrics.

These checks triangulate preservation of lexical content and pragmatic intent; they do
not prove that every safety-relevant aspect of `H_model(a)` is unchanged. Residual
model-perceived content change remains a competing explanation and must be discussed
as such. Report the all-eligible paired effect, the joint verified-violation endpoint,
and clearly labeled
gate-passing descriptive estimates. A failed semantic gate is evidence for a
perception explanation, not a verified safety-invariance violation. If human semantic
labels are collected for all flips but only a sample of non-flips, use the frozen
two-phase sampling probabilities and inverse-probability weighting for population
rates; otherwise treat the human labels as an audit rather than the formal gate. Do
not use Qwen2-Audio's own transcription as the only semantic gate.

Matched benign/hard-benign and ordinary instruction-following controls distinguish:

| Observation | Best-supported interpretation |
|---|---|
| Safety flips track semantic-gate failure | Perception/content degradation |
| Harmful and benign responses both become more compliant | Generic compliance/calibration shift |
| Non-answer or decode failure rises without judged harmful compliance | Generic disruption |
| Internal margin changes but full responses do not | State sensitivity, not behavioral safety failure |
| Harmful compliance rises while meaning and benign utility remain stable | Refusal-specific safety evidence |

### Stage C — exact bidirectional activation patching for localization

At a frozen site, let `M_i(a)` be item `i`'s natural state under matched sham `a0`
or active treatment `a1`, and let `Y_i(a,m)` be the unsafe-response outcome after an
explicit internal intervention. Generate and judge all four cells:

```text
Y(a0, M(a0)): matched-sham context + sham state
Y(a1, M(a1)): active context + active state
Y(a1, M(a0)): active context + sham-state restoration
Y(a0, M(a1)): sham context + active-state insertion
```

For each named operator/severity, with expectation over the screen-eligible item
population and a frozen decoding-seed distribution, report:

```text
TE_o,s        = E[Y(a1,M(a1)) - Y(a0,M(a0))]
Restore_M     = E[Y(a1,M(a0)) - Y(a1,M(a1))]
Induce_M      = E[Y(a0,M(a1)) - Y(a0,M(a0))]
Residual_M    = E[Y(a1,M(a0)) - Y(a0,M(a0))]
Interaction_M = E[(Y(a1,M(a1)) - Y(a1,M(a0)))
                  - (Y(a0,M(a1)) - Y(a0,M(a0)))]
```

With unsafe assistance coded as `Y=1`, successful restoration makes `Restore_M`
negative and successful induction makes `Induce_M` positive. `Residual_M` is the
remaining active–sham contrast after sham-state restoration. `Interaction_M` tests
whether restoration and induction depend on the surrounding input context. These are
directly executed node-intervention contrasts; do not rename them natural indirect
effects or declare that one direction explains a percentage of the total effect.

Use full-state sham↔active patching across audio encoder frames, projector
outputs, and LLM residual sites only on the discovery split. This is a full-state donor
reference for localizing candidate sites, **not** an upper bound: nonlinear interaction
and cancellation can make a smaller path/subspace intervention larger. A same-item
state swap is allowed here as a counterfactual localization tool; it is not portable
mechanism evidence.

These four cells provide node-level interventional evidence, not automatic
identification of a natural mediator. The swapped whole state may carry semantics,
quality, operator identity, and policy state together, and hybrid states may be outside
the model's natural support. Mediation language requires a frozen mediator definition,
a path/cut-set hypothesis, support diagnostics, and the later interaction/scrubbing
tests.

The primary intervention should be a single frozen prefill site/token or an explicitly
defined path. Persistent all-token decode-time steering is retained only as an actuator
baseline because it can directly control generation.

Activation patching is sensitive to corruption, metric, and site choices, as shown by
[Zhang and Nanda (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html).
The categorical full-response primary endpoint, semantic gates, and judge protocol are
frozen before any transformed safety outcome is observed. Layer, token/site,
corruption/reference construction, rank, and intervention dose may be selected only on
development data and must be frozen before confirmatory evaluation. A selected
mechanistic score remains secondary to the already-frozen full-response endpoint.

### Stage D — route isolation and shared/private decomposition

At the frozen candidate site, test an explicit route such as:

```text
audio encoder -> projector -> selected residual state -> downstream refusal process -> output
```

Use reciprocal path patching and grouped/factorial interventions to test whether the
candidate route is faithful and whether parallel routes carry residual effect. Node
patch success alone can identify a sink or amplifier rather than the source mechanism.

Fit an outcome-free reachable basis `B` from natural matched-sham→active deltas on the
discovery split. Within frozen `B`, cross-fit the smallest low-rank `U` that predicts
the training-set response/margin change; select rank only on validation. `U` is then a
**candidate interchange subspace**, not a mediator by construction. Its evidence
comes from confirmatory natural-coordinate swaps, path tests, and scrubbing—not its
training objective.

Formal distributed alignment search (DAS) is optional, not the default load-bearing
method. DAS can discover rotated distributed variables
([Geiger et al., 2024](https://proceedings.mlr.press/v236/geiger24a.html)), but it
requires an outcome-independent rule or structural equation assigning each source's
high-level state and a complete counterfactual table. This project does not currently
have such labels: defining “refusal-committed” directly from the same output being
explained would be circular, while assigning state from active/sham treatment would be
wrong on non-flips. Unless an exogenous assignment is preregistered, call the procedure
cross-fitted subspace intervention, not DAS causal abstraction, and do not report IIA.

If a valid high-level assignment is later obtained, compute balanced IIA only on
genuinely state-changing source/base swaps. Same-state/no-change swaps are identity
sanity checks and cannot dominate the headline metric.

Required anti-circularity controls:

- group-split items, source recordings, speakers, and renderers;
- choose site, rank, rotation, and intervention dose only on train/validation;
- distinguish **donor-coordinate interchange**, where a frozen projection swaps a
  coordinate taken from a natural execution but creates a hybrid state, from
  **portable fixed-dose correction**, where the target transformed activation may not
  set its own dose;
- test out-of-pair donors with the explicit intervention
  `Y_i(a; do(P_U M_i <- P_U M_j(a')))` rather than only the same item's exact delta;
- split and resample both base item `i` and donor `j`; use crossed/two-way clustering
  or donor-block resampling so repeated donors do not inflate the effective sample;
- compare label-permuted, random-Haar, covariance/reachability-matched subspaces under
  equal tuning budgets;
- compare frozen Arditi/SARSteer-style refusal direction, the current L18 direction,
  a ReGap-style mean-drift correction, the full-state same-item localization
  reference, a SPIRIT-style clean-state/top-`k` replacement, and operator-specific
  subspaces under matched development budgets;
- include a LOCA-like per-item outcome-adaptive correction as a clearly labeled local
  oracle, not as a transferable baseline; if a compatible SAE is unavailable, use an
  equivalently flexible item-specific patch and disclose the difference;
- include identity/self, reverse, wrong-layer/token/donor, label-permuted, and
  covariance/reachability/norm-matched null interventions;
- require that the natural operator displacement actually moves along frozen `U`, that
  the route is pre-decision and incremental over a generic refusal controller, and that
  semantic/quality state is preserved;
- test both restoration and induction on full responses.

### Stage E — final unseen-operator causal test

This is the load-bearing phase-independent experiment.

1. Use leave-one-family-out rotations only inside development/model selection, with
   every fold and metric frozen and no feedback from one fold into another.
2. Freeze site, rank, rotation, donors, dose rule, judges, and—only if formally
   available—the high-level counterfactual table.
3. Preserve a mechanistically distinct fourth operator family as a final external
   holdout that was never used in any fitting, selection, or rotation. A new backend of
   a seen family tests implementation transfer only and cannot substitute for this.
   The distinction class and implementation rationale must have been frozen at Stage A
   without inspecting this family's safety outcomes or activations.
4. First require the sealed family to reproduce its paired total effect over all
   eligible items. Then test bidirectional sham/active-coordinate interchange, path
   specificity, scrubbing, and matched nulls on new items from that family. Add
   balanced state-changing IIA only if
   the optional formal-DAS prerequisites were satisfied before training.
5. Evaluate operator-private `U_o` only as a separate secondary analysis using
   within-operator discovery/test item splits. Never fit `U_o` on the final holdout and
   then call that family held out.

With only three fixed families, rotating LOFO supports transport among those three
named families; it does not support statistical generalization to “ordinary channel
transformations” as a population. Broader language requires more mechanistically
distinct families or an explicit restriction of claim scope. A held-out backend can
support a separate, lower-tier within-family implementation-transfer claim.

Interpretation:

| Confirmatory outcome | Licensed conclusion |
|---|---|
| Multiple operator total effects; final-family `U_shared` follows natural operator displacement and passes restoration/induction, route/scrubbing tests, and matched-null separation | Evidence consistent with a shared interventionally relevant route across the named tested families |
| Total effects replicate, but only `U_o` works | Operator-private causal routes; no common bottleneck claim |
| Full-state natural patch works, learned subspaces do not | Could be distributed/high-rank state, semantic/quality information transfer, hybrid-state artifact, or a wrong site/endpoint; no mediator claim |
| Steering works but donor-state patching/OHBI does not | Refusal actuator, not a natural transform mediator |
| Only phase-vocoder family has a total effect | Phase-specific result; stop the phase-independent paper claim |
| No family has a replicated total effect | No behavioral invariance failure under the tested budgets |

### Stage F — causal scrubbing and interaction audit

State the entire proposed computation before testing it. Resample or scrub features
that the hypothesis says are irrelevant while preserving matched semantic/harmfulness
classes; performance should remain. Scrambling the proposed policy state with a
value-mismatched donor should destroy the counterfactual effect. Competing graphs for
semantic loss, generic degradation, and operator identity must be tested rather than
assumed away. This is a falsification test of the complete causal mechanism hypothesis, not a
search for another high-scoring direction; see the broader [causal abstraction
framework](https://www.jmlr.org/papers/v26/23-0058.html).

Individual activation-patching effects cannot safely be added into “percent mediated.”
[The Curse of Multiple Mediators (2026)](https://arxiv.org/abs/2606.27510) shows that
standard patching estimands can include interactions with other mediator states and
can hide or inflate causal importance. Report 2×2 or grouped factorial interactions,
patch-distance/dose curves, full-state natural-patch references, and group-level
faithfulness.

## 6. Sample and split planning

The final sample size must be frozen by simulation using pilot-external assumptions,
not selected after seeing new outcomes. In an illustrative **unselected** paired binary
design with total discordance `q=.25`, two-sided `alpha=.05`, and power `.80`, a normal
approximation gives about 194 items for a 10-point difference and 784 for a 5-point
difference; conservative exact McNemar calculation is closer to 207 and 817. These are
planning examples, not this experiment's sample sizes.

Independent baseline-refuser screening changes the endpoint population and the two
discordance probabilities. Estimate endpoint-sham→active `p01` and `p10` from an
external pilot, then simulate the complete frozen design. The simulation must include:

- separate screening and endpoint matched-sham runs;
- family/severity multiplicity and hierarchical Stage-A→mechanism gating;
- semantic-verification sampling/attrition and judge misclassification;
- the four-cell restoration/induction contrast, operator and donor reuse, and crossed
  item×donor resampling;
- an independent benign non-inferiority margin and power calculation for utility;
- second-model replication with its own eligibility and operating characteristics.

Discovery (roughly 100–150 unique items) and validation (roughly 50–100) are compute
planning scaffolds only. Confirmatory `n` is whatever the simulation requires; it
should not be rounded down to 200 because an asymptotic example was near that value.
Likewise, 30–50 discordant cases may stabilize a descriptive case series but do not
repair outcome selection or license an inferential subgroup mechanism claim. Primary
mechanistic inference remains the preregistered all-eligible-item intervention
contrast.

The current `6/10` reversal statistic has a very wide uncertainty interval. Its
discordant-subgroup rule was preregistered, but membership is still defined by observed
treatment outcomes, `n<15` made the behavioral check exploratory, and the check used a
refusal-prefix heuristic rather than fresh two-judge adjudication. It is descriptive
support for actuator feasibility, not a population mediation rate.

## 7. Claim ladder and stop rules

### Claim levels

| Level | Required evidence | Safe wording |
|---|---|---|
| 0 | Current Run 7 only | A particular phase-vocoder family is associated with L18 refusal-state/margin displacement; persistent L18 steering can modulate the margin. |
| 1 | Fresh all-eligible paired effect plus joint verified-violation endpoint in one non-phase-vocoder family | One named fixed channel-transform family can produce content-conditioned verdict flips in the tested population. |
| 2 | Replication in at least two genuinely distinct families/backends and a second model | Safety-verdict instability generalizes beyond one operator/model. |
| 3 | Final mechanistically distinct family, OHBI restoration/induction, path and scrubbing controls; optional balanced IIA only with valid state labels | Evidence is consistent with a shared interventionally relevant safety route across the named tested transformations. |
| 4 | Preregistered benign non-inferiority plus correction on untouched operators/models | The mechanism supports a utility-preserving generalizable mitigation within the tested scope. |

Do not skip from Level 0 to Level 3 because an L18 steering vector generalizes across
selected phase variants.

### Stop or narrow the paper when

- only phase-derived conditions reproduce the behavioral total effect;
- semantic or harmful-intent recognition deteriorates with the flip rate;
- full-response judgments do not reproduce first-token-margin movement;
- persistent steering works but bidirectional/path interventions fail;
- no common subspace transfers to a completely held-out operator family;
- both shared and adequately powered within-family route tests fail. In that case a
  behavioral-only paper proceeds only if its frozen generative-verdict contribution
  independently clears a preregistered novelty/effect/replication gate; otherwise the
  broad manuscript is `NO-GO`;
- benign over-refusal or generic task degradation explains the apparent safety gain;
- a second model/backend does not replicate the claimed scope.

These are informative negative results, not invitations to change the endpoint, layer,
rank, operator subset, or minimum effect after inspection.

## 8. Recommended introduction logic

The introduction should not begin with pitch, phase, EQ, codec, or resampling. A natural
five-paragraph structure is:

1. **Normative invariant.** Safety policies govern the meaning of a request. If the
   lexical content and pragmatic intent remain the same under frozen checks, routine
   changes to its acoustic channel should not change whether the model refuses it.
2. **Known phenomenon.** Prior red-teaming and perturbation benchmarks already show
   that Audio LLM safety is brittle to audio variations. This establishes urgency but
   leaves the central explanation unresolved. The transition question is: **did the
   model perceive a different request, or perceive the same request and make a
   different safety decision?**
3. **Unresolved question.** An observed flip may reflect lost semantics, generic
   degradation or generic compliance, stochastic generation/judge error, an
   operator-private route, or a shared safety route. Behavioral ASR, representation
   correlation, and refusal steering cannot distinguish these explanations. Existing
   audio causal tracing and patching also mean that activation intervention itself is
   not the novelty.
4. **Methodological contribution.** Define the categorical full-response verdict;
   measure the paired all-screen-eligible-item effect and separately report the joint
   verified-violation event; then apply the **Operator-Held-Out Bidirectional
   Interchange (OHBI) test**. Development families localize and fit a candidate route,
   while bidirectional rescue/induction and path controls are evaluated without
   refitting on a final
   untouched family. Margin/certification details stay out of this causal spine.
5. **Result contribution.** Fill this paragraph only after confirmatory data. State
   separately which operator families violate the invariant, whether the mechanism is
   shared or private, and whether it transfers across models. Do not write the desired
   conclusion into the introduction before those gates pass.

A suitable working title is:

> **Same Request, Different Verdict: Causal Invariance Failures in Audio-Language
> Model Safety**

## 9. Reviewer red-team

A skeptical reviewer will ask:

1. “AJailBench and Best-of-N already showed semantically preserved perturbations.”
   The answer must be the paired invariant, alternative-explanation gates, and a
   final untouched-operator causal test—not a longer transform list.
2. “Your refusal direction is just Arditi/SARSteer on audio.” The answer must be that
   a frozen refusal vector is only a baseline; the main evidence is donor-coordinate,
   route-specific OHBI on a final unseen family.
3. “You selected flips and then patched them.” The confirmatory estimate must include
   every preregistered eligible item; discordant cases are secondary visualization.
4. “The patch is off-manifold and directly controls decoding.” Use full-state natural
   patch references, support diagnostics, prefill-local interventions, out-of-pair
   donors, dose curves, reachability-matched nulls, and persistent steering only as a
   sensitivity baseline.
5. “EQ, codec, and resampling all perturb the same frontend statistics.” Hold out an
   entire operator mechanism/backend and include a mechanistically different family or
   strong negative control.
6. “Your safety flip is actually an ASR or generation failure.” Report the all-eligible paired effect,
   the joint external-semantic-gate endpoint, and separate non-answer, task accuracy,
   and benign utility without treating post-treatment gate selection as randomized.
7. “One Qwen cohort is a model-specific curiosity.” Use a new cohort, multiple audio
   sources, a different architecture, and an untouched replication split.

Until those answers are backed by data, the responsible project status is:

> **Run 7 remains a completed preregistered phase-specific supporting result;
> phase-independent behavioral and causal claims remain untested.**
