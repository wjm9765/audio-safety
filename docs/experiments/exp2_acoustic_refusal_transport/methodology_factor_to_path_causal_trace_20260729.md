# Exp2 Methodology Roadmap — Model-Eye Factor-to-Path Causal Trace

> Date: **2026-07-29**
>
> Status: **methodology roadmap, not a pre-registration and not a run result.**
> This document records the agreed research flow after reviewing the 2026-07-28
> Qwen2-Audio results and an independent Claude Opus 5 cross-check. It does not
> alter the binding yield-pilot rules in [`pilot_protocol.md`](pilot_protocol.md),
> [`pilot_amendment_20260728.md`](pilot_amendment_20260728.md), or
> [`model_selection_rule_20260728.md`](model_selection_rule_20260728.md).
> Any change from the operational-harm switcher estimand to the explicit-refusal
> estimand must be documented as an amendment before a confirmatory run.

## 1. Research question and claim boundary

The broad phenomenon is:

> The linguistic request is held fixed while audio-specific properties are
> changed. Nevertheless, an Audio LLM's explicit refusal behavior changes.

The paper's primary behavioral object is **explicit refusal stability**, not a
claim that every non-refusal is harmful compliance or a successful jailbreak.
The main question is therefore:

> **Which model-eye acoustic input component is changed, where is that change
> read inside the model, through which internal path does it reach the answer
> decision, and how does it alter explicit refusal?**

The target causal chain is:

```text
waveform-level acoustic factor
        ↓
model-eye input difference
        ↓
audio encoder / modality interface
        ↓
early LLM audio-token carrier
        ↓
transfer into non-audio / readout states
        ↓
first-generation decision state t_AB
        ↓
explicit refusal generation
```

The linguistic script and base recording remain unchanged across an acoustic
contrast. This construction-level invariance does **not** guarantee that the
model perceived the same content. Model-side content alignment is therefore a
required construct-validity measurement, not an outcome-based exclusion rule.

## 2. What is meant by a refusal signal

`t_AB` is the hidden state immediately before the first generated token. It is
not itself the first token or a refusal label. Three measurements are separated:

| symbol | measurement | role |
|---|---|---|
| `R_tAB` | refusal-start versus alternative-start logit margin at `t_AB` | internal secondary readout |
| `R_seq` | length-normalized likelihood of short refusal continuations versus direct-answer continuations | secondary, higher-resolution readout |
| `R_gen` | whether the actual generated response contains a frozen, explicit refusal pattern | **primary behavioral endpoint** |

Examples such as `Sorry`, `I cannot`, `I can't`, and `I am unable` belong to a
frozen refusal-prefix bank. A first token such as `I` is ambiguous, so
`R_tAB` alone cannot decide the behavioral result. Conclusions use `R_gen`;
`R_tAB` and `R_seq` diagnose when the generation begins to move.

A "refusal-related signal" is not assumed to be one universal vector. At a
stage `s`, the primary internal quantity is the **causal change in `R_gen` caused
by a controlled intervention at `s`**. Representation distance or linear-probe
decodability can show that information is present, but only an intervention can
show that the information is used.

## 3. Established starting evidence and its limit

The current Qwen2-Audio same-item contrast is `pv_standard` versus `pv_locked`:
the signed semitone shift, vocoder pipeline, length, and linguistic content are
held fixed; phase-handling coherence differs.

- L10 full audio-token-span restore: harmful refusal **+7.3 pp**, 90% CI
  `[+2.7, +12.0]`.
- L18 full audio-token-span restore: **0.0 pp**.
- Earlier L18 effects were full-state patches at the answer-boundary readout
  token, not L18 audio-span patches.
- Harmful-minus-benign interaction for the latest L10 restore remains
  unresolved: `+5.0 pp [-4.1, +14.0]`.
- Coordinate/subspace repairs through rank 64 did not recover full-generation
  refusal, while coarse full-state restoration did.

This supports an **early audio-conditioned causal carrier** and a later readout
decision handle. It does not yet establish a safety-specific coordinate, an
item-specific transport path, or an exact mediation fraction.

The next work must therefore refine the coarse full-state result rather than
search for another fixed refusal axis.

## 4. Model-eye input principle

For model `m`, let `F_m(w)` be the exact representation of waveform `w` at the
model's input boundary. For acoustic factor `f`, define:

```text
delta_(m,f) = F_m(w_f) - F_m(w_reference)
```

The physical waveform manipulation is shared across models, but
`delta_(m,f)` is recomputed separately for every model. A Qwen feature tensor is
never transplanted into a different architecture.

For Qwen2-Audio, `F_Qwen` is the exact Whisper-style log-Mel magnitude consumed
by the model. Waveform phase is not directly visible to the encoder. Phase
handling reaches Qwen only through the magnitude distortion realized after
iSTFT/overlap-add. The Qwen-specific mechanistic cause must therefore be stated
as **structured log-Mel distortion induced by the phase-vocoder contrast**, not
as raw phase entering the model.

The repository already provides the relevant exact-processor functions in
`audio_safety.evaluation.phase_ops`: `model_logmel`, `logmel_deviation`, and
`mel_matched_control`.

## 5. Acoustic-factor scope: phase-vocoder is the first case, not the whole paper

The methodology is designed for a family of content-fixed acoustic factors.
The current and planned scope includes:

| factor family | examples / current assets | principal model-eye question |
|---|---|---|
| phase-vocoder coherence | `pv_locked`, `pv_standard`, signed `±3 st` | which realized spectrotemporal distortion is causal? |
| echo / delay | `echo`; later controlled delay and decay doses | does delayed self-overlap alter temporal evidence or generic intelligibility? |
| tone / pitch | `tone_p8`; later both signs and frozen doses | are pitch/formant or resampling artifacts causal? |
| compositions | `echo_x_tone_p8`; later only after single-factor analysis | additive effect, interaction, or generic degradation? |
| speaker/prosody | planned accent, emotion, speaking-rate, intensity controls | does paralinguistic information change the same internal path? |
| channel distortion | planned noise, filtering, reverberation, codec/compression controls | shared frontend-robustness mechanism or factor-specific route? |

The list records the expansion space; it does not pre-select all factors for one
confirmatory run. Factor names, doses, signs, processing-matched identity arms,
and stopping rules must be frozen in config before target outcomes are observed.

### 5.1 Rules for multi-factor expansion

1. Use the same base item and base waveform across arms.
2. Analyze single factors before compositions.
3. Keep sign and dose assignment deterministic; do not choose the best variant
   per item after observing refusal.
4. Measure the exact model-eye difference for every factor and every model.
5. Match nuisance quantities such as length, gain, and input-distance norm when
   the physical contrast permits it.
6. Never call a factor content-preserving solely because its transcript file is
   unchanged; report model-side content alignment.
7. Strong factors that mainly cause off-topic or malformed generations remain
   robustness controls, not evidence of a refusal-specific mechanism.

### 5.2 Discovery and frozen replication across factors

The full factor-to-path mechanism is first discovered with the narrowest current
contrast, `pv_locked ↔ pv_standard`. Echo, tone, and later factors then test
frozen predictions:

- the model-eye component predicted to matter;
- the relative layer window where the audio carrier is read;
- the direction of the `t_AB` and explicit-refusal effects;
- the internal path selected during discovery.

If a new factor fails the frozen mechanism, it may begin a separately labeled
exploratory analysis. It must not be used to retune the original mechanism and
then be reported as confirmation.

Three cross-factor outcomes are possible:

1. different input distortions, shared internal bottleneck;
2. shared model-eye distortion structure and shared internal bottleneck;
3. factor-specific input components and factor-specific internal routes.

Compositional arms are analyzed only after their component factors. Their role
is to test interaction, not to manufacture a larger refusal effect.

## 6. Core methodology: Factor-to-Path Causal Trace

### Gate 0 — validate the existing L10 causal handle

Before expanding the layer or factor search, repeat the L10 audio-span restore
with the following controls:

- same-item `pv_locked` donor into a `pv_standard` host;
- length- and norm-matched wrong-item donor or wrong-item displacement;
- position-shuffled donor;
- identity self-patch, which must reproduce the unpatched text exactly.

Interpretation:

| observation | licensed conclusion |
|---|---|
| same-item beats wrong-item and shuffled controls | item-specific acoustic-state transport remains viable |
| same-item and wrong-item are similarly effective | generic clean-state / response-mode correction, not item-specific transport |
| real does not beat shuffled control | terminate the L10 transport claim |

A wrong-item null terminates the **item-specific** claim, not automatically the
broader study of a generic acoustic-quality or response-mode mechanism.

### M1 — identify the causal model-eye input component

For the Qwen phase-vocoder contrast:

```text
delta_mel = F_Qwen(pv_standard) - F_Qwen(pv_locked)
F(alpha)  = F_Qwen(pv_locked) + alpha * delta_mel
alpha     ∈ {0, 0.25, 0.50, 0.75, 1.0}
```

The endpoints reproduce the physical waveform arms. Intermediate values are
model-input interventions, not claims about physically realizable waveforms.

The first analysis uses one DSP-motivated decomposition rather than an open-ended
mask search:

- transient / temporal-modulation component;
- its complementary remainder;
- norm-matched structure controls, such as a temporal shift or structured
  permutation that preserves the declared nuisance statistics.

For echo, tone, and later factors, the decomposition is factor-appropriate but
the causal logic is unchanged: partially restore or inject a defined component,
match nuisance magnitude, and test a frozen directional prediction.

Evidence that a named component is causal requires:

1. ordered response to its dose;
2. stronger effect than matched structure controls;
3. replication on held-out items;
4. a corresponding change in `R_gen`, not only `R_tAB`;
5. content alignment that rules out a purely perceptual collapse when making a
   refusal-specific claim.

### M2 — locate the audio-carrier read window

Use same-item full audio-span interchange on a bounded set of layers:

```text
L8, L10, L12, L14, L16, L18
```

L10 and L18 reuse the established positive/null endpoints. The additional
layers fill the predicted early-to-late transition without an all-layer search.

For layer `l`, define the causal transport effect:

```text
T_l = P(R_gen | pv_standard host,
                    audio state at l <- same-item reference state)
      - P(R_gen | pv_standard host)
```

The layer at which audio-span restoration ceases to work is a candidate boundary
where the relevant difference has already left the audio-token carrier and been
written into non-audio or readout states. This is an **escape/read-window
localization**, not by itself a complete path explanation.

Raw activation norms are not compared across layers. Observational representation
distance, if reported, is whitened within each stage and remains secondary to
`T_l`.

### M3 — identify how the signal leaves the audio carrier

Only the one or two transition layers selected by M2 receive deeper analysis.
No all-head fishing is performed.

1. Patch the reference audio span at the candidate layer.
2. Reset the audio span to its host trajectory at a later layer to test whether
   the causal effect has already escaped into other positions.
3. Separate self-attention output from token-wise MLP output.
4. If self-attention is causal, patch the audio-source contribution to
   non-audio/readout queries or the corresponding audio-position K/V path.
5. Use `t_AB` full-state patching only as a downstream positive control.

This stage distinguishes:

- representation difference merely existing in the audio span;
- continued within-span processing;
- transfer through attention into non-audio states;
- final use at the answer-boundary decision state.

The desired path-level claim has the form:

> A defined model-eye acoustic component changes the early audio representation,
> and a bounded attention-mediated transfer window carries that change from
> audio positions into the first-generation decision state, altering explicit
> refusal.

## 7. Minimal endpoint and construct-validity set

The primary paper endpoint remains frozen explicit-refusal string matching.
Mechanism interpretation uses a minimal three-way audit:

| outcome | meaning |
|---|---|
| explicit refusal | frozen refusal pattern present |
| on-topic non-refusal | response remains aligned to the action and target in the request |
| off-topic / perception failure | mishearing, unrelated response, transcription, or malformed output |

Operational harmful assistance may be reported where available, but it is not
required to define the explicit-refusal phenomenon. The three-way audit can be
focused on the mechanistic cohort and intervention-changed outputs; an exhaustive
six-way relabel of every exploratory generation is not a prerequisite for the
first causal gate.

Matched benign items enter the critical M2/M3 cells as a **factor**, not as a
separate defense study:

- harmful-selective effect with stable content supports refusal/safety-specific
  use;
- similar harmful and benign effects support a generic response-mode mechanism;
- content recovery together with refusal recovery supports acoustic/perceptual
  restoration.

## 8. Cross-model generalization

The method is model-agnostic at the causal-framework level and front-end-specific
at M1.

| architecture | model-eye intervention |
|---|---|
| Whisper/log-Mel encoder | recompute that model's exact log-Mel difference and use feature-space partial restoration |
| other spectrogram encoder | use its own sample rate, window, hop, bins, and normalization |
| raw-waveform encoder | intervene on waveform components or the first convolutional latent |
| discrete audio codec/tokenizer | swap defined acoustic-token subsets; do not linearly interpolate token IDs |
| semantic/acoustic dual stream | intervene on each stream separately and test which carries the effect |
| encoder-decoder cross-attention | replace audio-span patching with audio K/V or source-contribution patching |

Cross-model comparisons align semantic stages, not absolute layer numbers:

```text
S0  model-eye input
S1  audio encoder output
S2  projector / modality interface
S3  early-to-middle fusion computation
S4  audio-to-non-audio transfer window
S5  first-generation prelogit / t_AB equivalent
S6  generated explicit refusal
```

Relative depth `l / (L - 1)` is reported where layer counts differ. Raw feature
norms, token counts, and a Qwen-derived `delta_mel` are never compared directly
across models.

### 8.1 Recommended replication order

1. **Qwen discovery:** identify input component, read window, and path.
2. **Same-front-end replication:** test the frozen mechanism on another
   Whisper/log-Mel audio model. This is direct method replication, not necessarily
   independent-backbone evidence.
3. **Different-front-end/backbone replication:** use the same physical waveform
   factors but recompute the model-eye intervention and map architecture-equivalent
   sites.
4. **Cross-factor replication:** echo, tone, and later factors test the frozen
   path before any model-specific retuning.

Claims are graded:

- shared behavioral instability only;
- shared abstract transport path;
- shared causal model-eye component;
- shared mechanism-derived defense.

Agreement between Qwen2-Audio and MiniCPM-o-2.6 alone is a within-Qwen-LLM-family
front-end/interface contrast, not full cross-architecture replication.

## 9. Role of existing defenses

SARSteer and ALMGuard motivate the upstream-versus-downstream hypothesis but do
not define the mechanism-discovery conditions. Their historical `48.3%` and
`19.4%` residual-success estimates are not a paired head-to-head comparison and
must not be used to localize the mechanism.

Existing defenses return only after M1–M3 freeze a causal prediction:

1. derive one defense from the identified input component or transfer path;
2. evaluate it against undefended, SARSteer, and ALMGuard on the same items;
3. match or report benign utility / over-refusal explicitly;
4. test held-out factors and models without refitting on their outcomes.

Examples of mechanism-derived defenses:

- input/frontend canonicalization if a structured spectrotemporal component is
  causal;
- an early-interface robustness adapter if the same component survives the
  encoder but is amplified at the modality interface;
- a bounded fusion-path correction if a specific attention transfer window is
  causal.

Defense-state transplant is optional validation of a frozen mechanism, not part
of mechanism discovery.

## 10. Evidence ladder and stopping rules

| level | required observation | permitted claim |
|---|---|---|
| E0 | explicit-refusal rate differs for a frozen same-item acoustic contrast | behavioral acoustic instability |
| E1 | model-eye component has dose order and beats matched structure controls | input component is causally relevant |
| E2 | same-item state patch beats identity, shuffled, and wrong-item controls | item-specific internal carrier |
| E3 | bounded audio-span restore curve identifies a transition window | candidate audio-carrier read window |
| E4 | source-restricted attention/path patch changes `R_gen` | causal internal transport path |
| E5 | content remains aligned and harmful effect exceeds matched benign | refusal/safety-specific use |
| E6 | mechanism-derived intervention generalizes to held-out factor/model at controlled utility | defense/generalization claim |

Stopping and downgrade rules:

- identity reproduction failure voids the run;
- failure against the shuffled control terminates the claimed carrier;
- failure against wrong-item downgrades item-specific transport to a generic
  state/response-mode hypothesis;
- loss of content alignment prohibits a refusal-specific interpretation;
- a flat M2 curve prohibits a localized audio-span read-window claim;
- a null path patch leaves only stage localization, not a path mechanism;
- failed frozen replication is reported as architecture- or factor-specific and
  is not repaired by outcome-guided retuning.

## 11. Bounded execution order

```text
0. Freeze endpoint, factor definitions, and controls in config
1. L10 same-item vs wrong-item/shuffled kill gate
2. Qwen model-eye input dose and one theory-driven component decomposition
3. Bounded L8/L10/L12/L14/L16/L18 audio-span sweep
4. One- or two-layer path patch at the transition window
5. Matched-benign specificity check in the critical cells
6. Frozen replication on echo and tone, then a second model
7. Mechanism-derived defense and same-cohort baseline comparison
```

The center of the paper is steps 2–4:

> **Which model-eye input component is read at which internal layer, and through
> which path does it reach explicit refusal?**

Any experiment that does not sharpen that input-to-path chain is deferred unless
it is required for construct validity, replication, or the final defense.

## 12. Methodological anchors

- AIA — refusal/compliance margins and bidirectional activation patching:
  <https://arxiv.org/abs/2605.18168>
- VoxParadox — separation of information availability from downstream use:
  <https://arxiv.org/abs/2605.27772>
- GACL — same-example site and path interventions across modality positions:
  <https://arxiv.org/abs/2606.05161>
- Activation patching best practices:
  <https://arxiv.org/abs/2309.16042>
- Path patching:
  <https://arxiv.org/abs/2304.05969>

These works motivate the measurement logic. The contribution sought here is the
factor-resolved, model-eye-input-to-explicit-refusal causal chain across acoustic
factors and model architectures.
