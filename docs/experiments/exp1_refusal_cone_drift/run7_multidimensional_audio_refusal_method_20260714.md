# Run 7 method direction — coupled refusal cone × audio-reachable causal transport (2026-07-14)

> **Status:** method decision / paper-direction note. This is **not** a new preregistration and does not
> modify the locked criteria in `design.md` §0. All thresholds below are provisional until a separate Run 7
> preregistration is frozen. No result is claimed in this document.
>
> **Working name:** **COAST-R — Causal Operator-Aligned Safety Transport into Refusal**. The name is
> provisional; the scientific object is the coupling between a refusal cone and empirically audio-reachable
> transport modes.

## 0. Human direction and novelty policy

The human evaluator does **not** want a complete pivot away from refusal directions. The recent Audio/Speech
LLM safety-subspace preprints are treated as useful independent evidence that this research neighborhood is
real, not as a reason to abandon it. Roughly half of the conceptual pipeline may therefore reuse and compare
against established refusal-axis/cone methods. The required contribution is a sharper causal question built
on the project's own same-content acoustic evidence.

The selected paper question is:

> When the spoken content is fixed, do small declared audio transformations move an Audio LLM into unsafe
> behavior through one common route or through multiple audio-reachable routes, and how do those routes
> couple to the model's refusal representation?

This keeps the existing project arc:

1. Exp1 found a partially manipulable audio-conditioned refusal direction (`r_A`, `WEAK-GO`).
2. Run 6 found that a small `librosa` compound DSP transformation can repeatedly cross the behavioral
   refusal boundary, and that an L18 refusal-related component causally contributes to restoration.
3. Run 7 asks whether this is one instance of a broader, multidimensional **audio-to-refusal transport
   mechanism**.

It does **not** require more F0 attribution as a paper gate. WORLD F0-only audio can remain a useful control,
but the main object is a low-severity declared audio change of any kind that passes preregistered preservation
gates.

## 1. Decision in one paragraph

Do not headline a newly discovered global “multidimensional Audio LLM safety subspace.” That claim now
overlaps heavily with refusal cones, fast multidimensional refusal extraction, and the 2026 Speech LM safety
subspace preprint. Instead, retain those methods as the **refusal-side half** of the study and learn the
other half: a low-rank basis constrained to activation changes that real audio operators actually produce.
Then causally test whether successive reachable components can induce and repair refusal erosion on held-out
items and operators. The central paper object is a **refusal-cone × acoustic-transport coupling matrix**, not
PCA variance alone.

## 2. What the current project has and has not established

### Established or partially established

- `r_A` is a causal controller under the original Exp1 gate, but its natural readout was weak (`WEAK-GO`).
- The Run 6 `librosa` operator caused 21/91 neutral refusers to become brittle, with 41 verified flip cells.
- At L18, restoring a leave-one-item-out refusal-related component improved refusal by 27.8 percentage
  points beyond one matched orthogonal intervention. The orthogonal intervention itself was large, so exact
  specificity remains incomplete.
- Harmfulness was more stable than refusal under the compound transform, consistent with—but not yet proving—
  a harm-recognition versus policy-routing dissociation.
- The signed/odd local pitch tangent failed while the even/absolute-change component carried the effect. A
  linear signed tangent alone is therefore insufficient for the next method.
- The harmful-specific displacement had high effective rank in earlier analyses, while rank-1 correction
  failed. High rank is only motivation; it is not evidence that multiple dimensions are behaviorally causal.

### Already attempted

- Harmful-minus-benign difference-of-means refusal/harmfulness directions.
- RDO-style optimization for `r_A`.
- Thin SVD/PCA of paired difference-in-differences, effective rank, principal-angle/family-subspace analysis,
  and grouped rank-`k` ridge prediction of the first-token margin.
- A single L18 leave-one-item-out refusal direction with one orthogonal and one harmfulness control.
- Full generated responses with two-judge labels for actual refusal/compliance transitions.

### Not yet attempted

- A learned multidimensional basis optimized for **held-out causal effect**, rather than reconstruction or
  first-token prediction only.
- Cumulative rank-1 → rank-2 → rank-3 intervention showing that later components add causal effect.
- A shared-versus-operator-specific decomposition across several audio transformations.
- Bidirectional intervention: transformed → refusal repair **and** neutral → induced erosion.
- A multi-token refusal/compliance counterfactual continuation-score curve as the primary differentiable
  endpoint.
- A chat-template sensitivity or held-out-template test.
- A second, structurally different Speech/Audio LLM.

The earlier pilot omitted these because it was explicitly designed as a cheap feasibility screen. Its
hand-written first-token logit bank and SVD were reasonable screening tools, but Run 6 also showed why they
cannot carry the paper claim: a verified behavioral flip need not follow the sign of that proxy.

## 3. Relation to the closest literature

The overlap is intentional and becomes the first half of the method.

| Prior direction | What it already covers | What Run 7 adds |
|---|---|---|
| [Arditi et al., NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/f545448535dfde4f9786555403ab7c49-Abstract-Conference.html) | One causal refusal direction in text LLMs | Same-content audio-induced routes into/out of refusal |
| [Concept Cones, 2025](https://arxiv.org/abs/2502.17420) | RDO-style multidimensional refusal cones | A cone is retained as the refusal target; the new object is the operator-reachable coupling into it |
| [Fast multidimensional refusal via RFM-AGOP, 2026](https://arxiv.org/abs/2607.02396) | Fast extraction of multidimensional refusal subspaces | Strong extraction baseline; no same-content acoustic tangent or operator transfer |
| [There Is More to Refusal…, 2026](https://arxiv.org/abs/2602.02132) | Different refusal types occupy distinct directions, with a shared steering trade-off | Motivates separating “whether it refuses” from “how it refuses” in the endpoint |
| [Unified Safety Subspace in Speech LMs, Interspeech 2026 author manuscript](https://www.researchgate.net/publication/405947813_A_Unified_Safety_Subspace_Exists_in_Speech_Language_Models) | Qwen2-Audio/GLM-4-Voice PCA, audio↔text safety-vector transfer, steering | Independent validation of the topic; Run 7 uses matched same-utterance operator deltas, causal rank increments, and held-out-operator transfer |
| [ReGap, 2026](https://arxiv.org/abs/2605.18104) | Modality drift compresses refusal separation; inference-time correction | Within-audio local transforms rather than text↔modality gap; multiple operator routes |
| [Acoustic Interference, 2026](https://arxiv.org/abs/2605.18168) | Benign interference audio, refusal logits/direction, patching | No added sidecar audio: the content-bearing waveform itself is minimally transformed and paired to its neutral version |
| [SARSteer, 2025](https://arxiv.org/abs/2510.17633) | Text-derived refusal steering plus benign PCA space for Audio LLMs | Causal mechanism analysis of naturally reachable deltas, not only a defense vector |
| [SPIRIT, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.734/) | Activation patching and noise-sensitive units for adversarial speech | Operator-shared versus operator-specific refusal transport under declared small transforms |
| [DAS](https://arxiv.org/abs/2303.02536) and [subspace-patching caveats](https://proceedings.iclr.cc/paper_files/paper/2024/hash/70b8505ac79e3e131756f793cd80eb8d-Abstract-Conference.html) | Learn distributed intervention subspaces; warn that arbitrary subspaces can exploit dormant paths | Constrain every candidate direction and coefficient to empirical audio-induced activation changes, then test natural occurrence and both causal directions |

The paper should cite overlap positively: prior work supports the existence and controllability of refusal
geometry. The novelty claim should be narrower than “first audio refusal subspace.”

### Candidate-method comparison

| Candidate | Strength | Main failure mode | Role |
|---|---|---|---|
| Current shared DiD-SVD | Cheap, already implemented, honest rank baseline | Finds variance, not causal safety dimensions | Required baseline |
| Differential SAE on paired deltas | Sparse feature labels may be intuitive | Seed/width instability; refusal SAE work already exists; features need causal validation anyway | Optional interpretation appendix |
| Layer × position × operator tensor factorization | Can show where shared/specific modes emerge | Rotation/identifiability and sample complexity; “tensor” alone is not a mechanism | Secondary localization after COAST-R works |
| Input-space local Jacobian/Hessian spectrum | Directly characterizes perceptual operator susceptibility | May become a robustness metric disconnected from the already found refusal axis | Useful physical-space complement |
| **COAST-R reachable causal transport** | Extends `r_A` directly; natural reachability, held-out transfer, and causal rank are all testable | More intervention compute; fails cleanly if rank 1 is sufficient | **Selected core method** |

The selected method is therefore an extension of the existing refusal-axis program, not a replacement with
an unrelated representation technique.

## 4. Conceptual model: boundary versus transport

There are two different geometric objects.

1. **Refusal representation** `R`: directions/cone that read or control whether and how the model refuses.
2. **Audio-reachable transport** `U`: directions along which a declared small audio operator can actually
   move a fixed utterance.

The failure mechanism is their causal coupling:

```text
small audio operator
        │
        ▼
empirically reachable activation transport U
        │
        ▼
geometric alignment + interventional coupling M_operator
        │
        ▼
refusal-policy state / continuation scores and generation
        │
        ▼
full-response refusal or unsafe compliance
```

This distinction permits several honest outcomes:

- `R` is effectively 1D but several audio modes feed it;
- `R` is multidimensional but one acoustic mode dominates;
- both sides and their coupling are multidimensional;
- apparent high activation rank has no causal relevance.

The main multidimensional claim will concern **audio-reachable causal transport** unless the data separately
support multidimensionality of `R` itself.

## 5. Proposed method

### 5.1 Paired activations

For item `i`, label `y ∈ {H, B}` (harmful/benign), operator `o`, severity `a`, and layer/position site `s`,
let

```text
h(i,y,o,a,s) = residual activation
δ(i,y,o,a,s) = h(i,y,o,a,s) - h(i,y,neutral,s)
```

Every transformed waveform is derived from its own neutral waveform. Item—not audio variant—is the
independent unit and split group.

For signed operators, estimate both local odd and even responses:

```text
J_odd  = [h(+a) - h(-a)] / (2a)
J_even = [h(+a) + h(-a) - 2h(0)] / a²
```

`J_even` is load-bearing because Run 6 found that the signed tangent did not carry the causal result. For
one-sided operators such as added noise or reverb, use several nonzero severities and finite differences
from the sham condition.

### 5.2 Preserve generic acoustic motion instead of subtracting it away

For matched harmful/benign pairs, form both:

```text
g(i,o,a,s) = [δ_H + δ_B] / 2       # generic operator-induced motion
z(i,o,a,s) =  δ_H - δ_B            # harmful-side interaction
```

Do **not** force the learned refusal transport to be orthogonal to all benign acoustic motion. A generic
audio shift can be harmless for a well-separated benign query yet cross the boundary for a harmful query
that is already close to refusal failure. The model should retain `g` and `z` as separate explanatory
blocks, while benign output/utility constraints prevent a generic degradation direction from masquerading
as a good safety mechanism.

### 5.3 Constrain the basis to a non-vacuous audio-reachable space

Merely putting `U` in the span of every observed delta is not enough: with thousands of samples that span
can approach the full 4096-dimensional residual space. Construct the reachable space **before using any
refusal/compliance labels**.

1. On a calibration split, stack raw operator deltas and finite-difference terms only. Do not use behavior
   outcomes to construct this dictionary.
2. Fit an unsupervised operator basis `B_s ∈ R^(d × q)` and freeze `q ≪ d` using a preregistered
   held-out-delta reconstruction rule plus a hard cap (candidate cap: 32 or 64, fixed before outcome analysis).
3. Report numerical rank, spectrum, and the fraction of each untouched natural delta captured by `B_s`.
   Reaching the cap with poor held-out coverage is a reachability-model failure, not permission to increase it.
4. Learn the behavior-relevant basis only inside this frozen space:

```text
U_s = B_s V_s,      V_sᵀ V_s = I,      U_s ∈ R^(d × k).
```

The harmful/benign `g` and `z` blocks may supervise `V_s` after `B_s` is frozen, but they cannot enlarge
`B_s`. Every random/null basis is drawn from the identical `B_s` space with matched operator covariance.
For each causal cell, the intervention coefficient comes from that cell's own observed delta and must remain
inside its naturally observed amplitude range.

This label-free low-dimensional calibration, not raw span membership alone, is the safeguard against an
unrestricted DAS/patching solution that discovers a powerful but naturally dormant direction.

Orthonormalize `B`, `R`, and `U` under a frozen train-set covariance metric. The primary geometric estimands
are rotation-invariant projectors, principal angles, and singular spectra—not individual columns.

- `U_shared` is operationally defined as the frozen train-operator projector that predicts and causally
  transfers to an untouched operator.
- `U_specific[o]` is only the reproducible residual performance beyond that shared projector for operator
  `o`.

Individual columns are rotation-ambiguous. Component names are allowed only if bootstrap/seed matching is
stable after Procrustes alignment; otherwise report only projector- and rank-level results.

### 5.4 Retain standard refusal extraction as half of the method

Learn and freeze a refusal basis `R_s` on training data using:

- a primary within-harmful behavioral contrast (refused versus operationally complied states, grouped by
  item), so harmfulness is not used as a proxy label for refusal;
- the existing `r_A`/harmful-versus-benign difference-of-means direction as a continuity baseline and known
  controller, not as the sole definition of refusal;
- RDO/Concept Cone as the primary established multidimensional method;
- RFM-AGOP as a fast recent baseline if its implementation is reproducible;
- optional harmfulness direction `r_H` as a distinct control.

Include answered benign and hard-benign/over-refused states when estimating specificity. A direction that
only separates harmful from benign content is a harmfulness direction even if adding it happens to change
refusal.

Estimate `R_s` on a dataset/fold disjoint from the one used to supervise `U_s`, or use outer cross-fitting
that rotates the two roles. This prevents both objects from being tuned to the same refusal residual.

Select refusal rank on held-out multi-token refusal behavior, not on explained activation variance. A useful
descriptive geometric object is

```text
C_geo(s) = R_sᵀ Σ_train U_s.
```

An optional operator-conditioned predictor may use a full regularized map in `U` coordinates; a diagonal
gate and its columnwise meanings are not scientific estimands. The singular values of `C_geo` describe
geometric alignment, **not causal mediation**.

The primary coupling matrix is interventional and has separate induction/restoration blocks:

```text
M_induce,o[j,m]  = held-out paired ATE when out-of-pair predicted mode m is added
M_restore,o[j,m] = held-out paired ATE when same-pair mode m is removed,
                   relative to an equal-reconstruction generic basis
M_o = row_stack(M_induce,o, M_restore,o).
```

Estimate the singular values and causal rank of `M_o` with item bootstrap and behavior-label permutation
against matched reachable nulls. `C_geo` explains where the spaces point; `M_o` determines whether those
directions are actually read by downstream refusal computation.

### 5.5 Behavior-supervised and causal learning objective

For a harmful neutral/transformed pair, let `P_U` be the frozen covariance-metric projector and define

```text
δ_U = P_U [h_transformed - h_neutral]
δ_perp = [h_transformed - h_neutral] - δ_U.
```

Use the full mediation decomposition through the frozen downstream model:

```text
h_neutral
h_neutral + δ_U
h_neutral + δ_perp
h_neutral + δ_full

h_transformed
h_transformed - δ_U
h_transformed - δ_perp
h_transformed - δ_full
```

`δ_U` must repair the effect while `δ_perp` does not, and the component's natural dose-response must agree
with its intervention response. This same-pair add/subtract decomposition is a **mediation/restoration
diagnostic**, not by itself a sufficiency test.

#### 5.5.1 Anti-tautology: primary out-of-pair causal test

Adding a projection of a test pair's own `h_transformed - h_neutral` back to that neutral state can become a
state-reconstruction tautology. Therefore:

- same-pair subtraction from the transformed state = **component mediation/restoration**;
- same-pair addition to the neutral state = **reconstruction sanity check**;
- only out-of-pair predicted addition = **primary sufficiency test**.

For the primary induction test, predict test-item coordinates

```text
z_hat_test = f_train(operator descriptor, severity, h_neutral_test)
δ_hat_test = U z_hat_test
```

without reading that item's transformed activation, transformed response, or behavior label. Fit `f_train`
only on training utterances and bound `z_hat_test` by the train distribution.

Crucially, freeze `U` first and train `f_train` **only** to predict the naturally observed train coordinate
`Uᵀδ`. `f_train` receives no refusal/compliance label, downstream gradient, continuation score, or induction
loss. Freeze it before any causal generation. On a separate validation fold it must predict actual out-of-pair
delta coordinates better than the operator-mean baseline and preserve the preregistered severity direction/
dose ordering. If this natural-delta prediction gate fails, no sufficiency claim is allowed even if a vector
from the same space can steer refusal.

Required causal baselines are a cross-fitted operator mean, wrong-item donor, wrong-operator donor, and rank-
plus reconstruction-error-matched bases optimized for generic activation reconstruction or generic output
change. The full same-item delta is reported only as a non-mechanistic upper bound.

Thus a sufficiency claim requires out-of-pair `δ_hat_test` to induce the predicted full-response change, while
restoration requires `δ_U` to outperform equal-reconstruction generic bases and leave `δ_perp` ineffective.
For a seen operator, the descriptor may include its frozen family identity. For the untouched-operator test,
it may use only preregistered physical/acoustic descriptors and severity available without safety outcomes;
an unseen learned ID embedding is not allowed.

The primary intervention site is the frozen LLM layer at `P2` during the initial prompt forward, after which
the model runs normally and cached downstream consequences are allowed to propagate. Persistent intervention
at every generated token is a separately labeled sensitivity condition; it must not be pooled with the
primary estimand.

The training loss combines:

```text
L = L_repair_curve
  + L_induce_curve
  + λ L_heldout_behavior_prediction
  + β L_benign_output_and_utility
  + γ L_rank_and_shared/specific_sparsity.
```

- `L_repair_curve`: repaired teacher-forced output distribution approaches the neutral distribution.
- `L_induce_curve`: induced output distribution approaches the transformed distribution.
- prediction: projected coordinates predict the change in the continuation-score curve and behavioral label.
- benign term: the same operation must not create over-refusal or damage benign task utility.
- rank/group terms: prefer the smallest reproducible shared-plus-specific model.

This loss learns `U`; it does not train `f_train`. Natural-coordinate prediction is a separate label-free
stage after `U` is frozen, as specified above.

All weights, ranks, layers, and stopping rules are selected on train/validation items. The final test items,
held-out operator, and second model remain untouched.

## 6. Output endpoint: token logits are included, but not trusted alone

The earlier first-token margin remains a useful baseline. Run 7 adds three levels.

1. **Single-token/logit baseline.** Track model-specific tokenization variants of refusal and compliance
   openings at the assistant boundary. This connects directly to logit-lens/AudioLens-style work and makes
   the result comparable to prior studies.
2. **Primary differentiable endpoint.** Use length-normalized, teacher-forced log likelihood of multiple
   refusal and compliance continuations over the first 16–32 tokens. Report this as a **counterfactual
   continuation-score curve** at each position, not as the model's actual generation trajectory and not only
   `sorry` versus `sure` at step 1.
3. **Authoritative behavioral endpoint.** Generate the full response under deterministic decoding and score
   refusal, safe redirection, partial compliance, and operational harmful compliance with independent judges
   plus a blinded human subset. A token-margin sign change is never itself a jailbreak.

The continuation bank is split by semantic function and paraphrase, so the method cannot win by following
one lexical marker. Refusal-style change without unsafe behavioral change is reported separately.

### Chat-template test

- Train with each model's official/default chat template.
- Freeze and evaluate at least one semantically equivalent template/wrapper condition without refitting.
- Record `P1` (last user/instruction state), `P2` (assistant boundary), and early generation positions.
- If the component disappears under a harmless formatting change, label it template-specific rather than a
  general safety mechanism.

This directly addresses the earlier omission: chat-template trajectories have not yet been tested in the
project and become a falsification test, not an after-the-fact visualization.

## 7. Data and operator design

### Discovery model and cohort

- Qwen2-Audio-7B-Instruct, full 150 harmful/benign matched items already used in Run 6.
- These 150 items are **fully exposed exploratory data**. A new split cannot make any subset scientifically
  confirmatory after the cohort shaped the hypothesis and method.
- Use grouped cross-fitting across all 150 for Stage A method debugging only. Analyze the full cohort, not
  only flip-selected cells; near-boundary and verified-flip subsets remain secondary.
- Stage B requires a genuinely new confirmatory cohort. Determine its size by power analysis on the
  neutral-baseline-refusing harmful population; the planning target is at least **100 final-test baseline
  refusers per model**. Simulate the paired full-response endpoint using the expected discordant-pair rate and
  require at least 80% power for the frozen incremental effect after multiplicity correction. The plausible
  range is roughly 100–200 unique final-test items per model, but the simulation—not this range—sets `n`.
- Define eligibility from neutral behavior only, before examining any transformed outcome.
- Include multiple speakers/renderers and a preregistered human-recorded or natural-speech subset so the
  mechanism is not identified with one CosyVoice voice.
- The independent unit is the utterance/item. Operators, severities, decode seeds, and judges are repeated
  measurements, not additional sample size.

### Operator families

The minimum study should include distinct, exactly named mechanisms:

1. the existing `librosa` phase-vocoder compound transform, retained as the discovered case and never
   relabeled as isolated pitch/F0;
2. exact waveform gain;
3. duration/time modification with a declared implementation;
4. low-level additive/background noise or mild room response;
5. one held-out family such as resampling/codec/reverb for cross-operator generalization.

WORLD F0-only may be included as a supporting negative/control but is not a gate. Each operator gets sham,
small, and moderate levels in both directions where meaningful. Budgets are operator-specific and calibrated
by perceptual quality/JND rather than pretending that semitones, dB, SNR, and duration ratio share one raw
metric.

### Content-preservation gates

- Fix all preservation thresholds during operator calibration, before inspecting refusal outcomes, and apply
  them at the item–variant level with outcome-independent exclusion rules.
- Require external transcript equivalence and harmful-intent/meaning recognition, not WER alone.
- Use model-internal transcription plus matched harmless semantic-recognition and generic instruction-following
  probes to measure general degradation at equal acoustic distortion.
- loudness/clipping and operator-specific acoustic diagnostics;
- speaker/source consistency;
- blinded human **same-meaning** judgments on a stratified subset, not intelligibility alone;
- fixed speaker/source and waveform lineage.

The claim is about a **declared transform set**. Passing intelligibility does not prove isolation of a named
acoustic factor, and factor isolation is not needed for the broader small-transform claim.
Report every failed preservation cell. The paper-level phrase is “low-severity declared transformations
passing preregistered preservation gates,” not an unconditional assertion of content preservation.

### Second architecture

Use GLM-4-Voice if residual/token hooks and inference are reproducible; it is structurally more independent
than another Qwen checkpoint and was also used by the unified-safety-subspace preprint. SALMONN is a fallback
if GLM-4-Voice instrumentation proves infeasible. Qwen2.5-Omni is useful but is a weaker independence test.

## 8. Causal tests and baselines

### Required intervention ladder

For rank `k ∈ {1,2,3,4}` on held-out items:

1. no intervention;
2. best rank-1 reachable mode;
3. cumulative rank-2, rank-3, rank-4 reachable modes;
4. full same-item activation delta as an upper bound;
5. out-of-pair predicted induction on the neutral state (primary sufficiency);
6. same-pair reverse induction (reconstruction sanity check only);
7. existing `r_A`/DIM and RDO cone;
8. raw DiD-SVD basis;
9. unrestricted DAS-style basis;
10. harmfulness direction;
11. covariance-, norm-, rank-, and reconstruction-error-matched reachable null ensembles;
12. wrong-item, wrong-operator, non-flipping transformed donor, wrong-layer, and wrong-position controls.

The primary contrast is not “full repair versus one convenient orthogonal vector.” It is cumulative reachable
rank versus the distribution of matched nulls, with item bootstrap and null-direction uncertainty.

### Mediation/restoration and sufficiency

- **Mediation/restoration:** remove the same-pair candidate component from transformed activation and recover
  refusal beyond equal-reconstruction generic bases.
- **Sufficiency/induction:** add the out-of-pair component predicted from operator, severity, and neutral state;
  do not use the test item's transformed activation or outcome.
- **Sanity check only:** add the test pair's own projected delta back to neutral.
- Verify downstream propagation, not only the final response.
- Keep coefficients inside the empirical train range; no arbitrary high-norm steering.
- Confirm natural projected coordinates covary with outcomes before intervention.

### Shared versus specific transfer matrix

Fit on all-but-one operator and evaluate the shared modes on the untouched operator. Report a causal transfer
matrix whose entry `(train operator family, test operator family)` is the paired change in full-response
refusal from the intervention. Operator-family means are descriptive; item remains the statistical unit.

## 9. What counts as multidimensional

High participation ratio, a pretty PCA plot, or rank-2 reconstruction gain is not sufficient. A
multidimensional claim requires all of the following in the frozen test:

1. rank-2/3 improves held-out multi-token continuation-curve prediction over rank-1;
2. component 2 adds matched-restoration effect and out-of-pair induction effect after the best component 1
   is already present;
3. the second singular value of the interventional matrix `M_o` has a bootstrap lower bound above a matched
   reachable-null distribution;
4. the frozen `f_train` predicts natural held-out delta coordinates beyond the operator-mean baseline without
   any safety labels or gradients;
5. the cumulative effect exceeds a matched reachable-random ensemble;
6. at least two dimensions have reproducibly different operator-transfer signatures or refusal-component
   coupling patterns;
7. benign utility/over-refusal remains within the preregistered tolerance;
8. the conclusion is stable across transform backend or the backend dependence is reported as the result.

The primary estimand is the **held-out linear causal transport rank at a frozen site**. A curved one-dimensional
path can produce two linear components (`J_odd` and `J_even`), so rank 2 is not automatically “two mechanisms.”
Use “multiple routes” only if at least two non-collinear first-order operator tangents have different,
reproducible held-out transfer patterns. If the second component is only the even term of one operator,
describe it as local curvature or second-order transport.

Fit independently optimized nested rank-1/2/3 models from multiple seeds, order components using training/
validation data only, and freeze the ordering before the confirmatory intervention.

Provisional gate to freeze before the run:

- ≥10% held-out continuation-curve MSE improvement for rank-2/3 versus rank-1 at the selected site;
- ≥10 percentage-point additional paired full-response effect for cumulative rank-`k` versus rank-1 in the
  frozen primary causal endpoint, with out-of-pair induction required and a positive item-bootstrap lower
  bound versus the matched-null ensemble;
- frozen label-free `f_train` beats the operator-mean coordinate predictor and preserves severity/dose
  ordering on its validation fold;
- benign over-refusal increase ≤3 percentage points;
- at least one shared mode transfers to the untouched operator in the predicted direction.

If prediction or causal increment fails, report an effectively rank-1 mechanism. Do not search new ranks,
layers, or representation methods until a multidimensional result appears.

## 10. Staged execution plan

### Stage A — reuse current Run 6 data; method kill-test

1. Implement the multi-token continuation endpoint and template-aware position recording.
2. Treat every one of the existing 150 items as exploratory. Use outer item-grouped cross-fitting—not a
   relabeled “final test”—to build the behavior-label-free `B`, separately estimate `R`, and learn rank 1–4
   reachable modes at the frozen L18 primary site; use L16/L20 only as declared sensitivity sites.
3. Run the full `δ_U`/`δ_perp` add-and-subtract mediation decomposition and out-of-pair predicted induction
   with matched reachable nulls on held-out folds.
4. Compare `r_A`, DIM, RDO cone, raw SVD, reachable COAST-R, and matched reachable nulls.

This stage tests the algorithm and the distinction between rank and causal rank. It cannot establish
cross-transform generality.

### Stage B — Qwen2-Audio multi-operator full cohort

1. Collect a genuinely new cohort whose test size is fixed by paired-endpoint simulation at ≥80% power
   after correction, planning for at least 100 final-test neutral baseline refusers, plus multiple speakers/
   renderers and a natural/human-speech subset.
2. Freeze exact operator implementations, severity budgets, neutral-only eligibility, split, endpoints, and
   gates in a new preregistration before inspecting transformed outcomes.
3. Reserve disjoint data or outer folds for the unsupervised reachable basis `B`, refusal basis `R`, and
   transport learner `U`; keep the confirmatory test sealed.
4. Extract encoder → projector → LLM layer/position trajectories for the declared operator set.
5. Fit shared and operator-specific transport, leaving one operator family untouched.
6. Perform the interventional coupling matrix, full mediation decomposition, and harmfulness-versus-refusal
   routing tests.

### Stage C — architecture/backend replication

1. **Confirmatory architecture transfer:** apply the frozen algorithm, semantic position, normalized-depth
   mapping, rank, and intervention rule without GLM test-set localization. This analysis alone determines
   replication success; “same site” does not mean literal layer 18 in a model with different depth.
2. **Architecture-specific localization:** a site/rank may be selected on the second model's training split
   and frozen before its test split, but this is labeled discovery and cannot rescue failed confirmatory
   transfer.
3. Confirm the direction of out-of-pair induction, matched-reconstruction restoration, incremental rank,
   unseen-operator transfer, and benign preservation on GLM-4-Voice or the preregistered fallback.
4. Include a second implementation/backend for at least one nominal transform to quantify implementation
   dependence rather than hide it.

### Stage D — paper-facing audit

- multiple independent judges, blinded human subset, deterministic and repeated-seed decoding;
- confidence intervals at the item level and correction for the small number of frozen sites/ranks;
- open configs, exact operator lineage, failed cells, and reproducible activation/intervention artifacts;
- blind cross-check of the final GO/NO-GO calculation before claiming multidimensionality.

## 11. Paper claim and likely reviewer reaction

### Safe headline if the full design succeeds

> Across preregistered low-severity audio transformations that pass preservation gates, a transport subspace
> learned from training utterances and operators predicts and causally mediates refusal erosion on unseen
> utterances and operators. Additional causal rank beyond one is claimed only when an out-of-pair component
> improves full-response behavior over equal-reconstruction controls and replicates across architectures.

### Claims to avoid

- first refusal axis/subspace in an Audio or Speech LLM;
- first multidimensional refusal representation;
- pitch/F0 itself causes the Run 6 result;
- a high-rank activation cloud is a high-rank safety mechanism;
- orthogonality proves causal independence;
- one first-token logit represents the full response;
- all small audio changes jailbreak the model;
- global safety lives in a clean linear subspace separable from general utility.

The last caution is especially important given [Safety Subspaces are Not Linearly Distinct, ICLR 2026](https://openreview.net/forum?id=Fj6LakRHcT).
COAST-R is framed as a **local, operator-reachable transport model**, not a globally distinct safety module.

### Conditional acceptance estimate

- Current one-model/one-operator evidence only: approximately **25–35%** ICLR acceptance likelihood.
- Stage A plus a solid Qwen multi-operator result, but no genuinely new cohort or second architecture:
  approximately **40–50%**.
- The original draft without a non-vacuous reachable basis, interventional coupling, or genuinely new powered
  cohort was independently reviewed at approximately **52–58%** even if its experiments succeeded.
- One independent reviewer estimated **63–68%** after the first identifiability revisions; a second reviewer
  identified the same-pair-induction tautology and initially gave **60–65%** after the out-of-pair, power,
  preservation, and architecture-transfer revisions. A final re-review found one remaining steering loophole:
  `f_train` also had to be separated from all safety labels/gradients and validated on natural delta
  prediction. With that last requirement now included, the final conditional estimate is **63–68%**.

These are judgment estimates, not empirical probabilities. The path above meets the user's ≥60% target only
conditionally on Stage C, a real causal rank increment, and five load-bearing safeguards: non-vacuous
reachability, interventional coupling, label-free natural-delta prediction, out-of-pair induction, and a
genuinely new powered cohort. A clean rank-1 or null outcome can still be a useful scientific conclusion,
but it should not be sold as the multidimensional headline.

## 12. Immediate next action

The next recorded implementation task is **Stage A**, not another F0 experiment:

1. write and freeze a Run 7 exploratory config for the existing n=150 activation set;
2. replace the first-token-only target with the multi-token refusal/compliance continuation-score curve while
   retaining the old target as a baseline;
3. build a behavior-label-free, capped operator basis `B`, report its rank/spectrum/held-out coverage, and
   then implement the rank-1–4 learner inside `B`;
4. estimate `R` and `U` on separate outer-fold roles, then run the `δ_U`/`δ_perp` causal ladder at L18 with
   matched reachable-null ensembles; after freezing `U`, fit the label-free natural-coordinate predictor
   `f_train`, pass its validation gate, and only then run out-of-pair induction and the interventional
   coupling matrix `M_o`;
5. decide whether the multi-operator extraction in Stage B is justified and freeze its preregistration.

No new waveform generation is required for steps 1–4. This is the cheapest way to determine whether the
method adds anything beyond the already implemented SVD and single refusal component.

## 13. Implementation handoff — 2026-07-14

The first executable Stage A core is now present. This records implementation scope, not a result and not a
new preregistration.

Implemented:

1. strict Run 7 config and a resumable `score -> fit -> intervene` CLI;
2. correctly shifted teacher-forced token log-probabilities and refusal-minus-compliance continuation curves,
   while retaining the old first-token endpoint only as an explicit baseline override;
3. source-artifact hashes and official-chat-template P2 alignment checks, so resumed rows cannot silently mix
   another source run or token position;
4. uncentered, behavior-outcome-free reachable `B` with grouped held-out coverage and a declared rank cap;
5. outer item folds with pairwise-disjoint `B`, descriptive `R`, supervised `U`, and label-free `f_train`
   roles; `f_train` predicts natural delta coordinates from neutral state plus declared severity only;
6. nested reduced-rank `U`, exact `delta_U`/`delta_perp`/full-delta decomposition, and a validation gate before
   out-of-pair predicted induction;
7. exact-magnitude L18/P2 prefill-only interventions, greedy decoding, application-count assertions, and one
   long-form row per causal condition for later blinded judging; and
8. CPU tests for token shifting, grouped leakage, basis nesting, natural prediction, source preflight, config,
   and raw-delta hook behavior.

Not yet implemented in this core:

- matched reachable-random causal ensembles and item-level bootstrap/permutation inference;
- replay of the legacy `r_A`, multidimensional RDO cone, and other paper baselines;
- explicit causal aggregation/judging and benign full-response utility interventions;
- a semantically equivalent held-out chat-template condition; and
- Stage B's new multi-transform cohort or any second architecture.

The next execution sequence is therefore: run both harmful and benign continuation scoring on the existing GPU
workspace; run the CPU fit and inspect reachability, available rank, held-out endpoint error, and the frozen
natural-predictor gate; only then generate the neutral-baseline-refuser causal ladder and blind-judge it. Add
matched nulls, baseline replay, and item-level uncertainty before interpreting incremental causal rank. A fit
that finds rank 2 is not yet a multidimensional result.
