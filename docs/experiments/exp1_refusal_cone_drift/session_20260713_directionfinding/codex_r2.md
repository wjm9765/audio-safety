## Verdict

Choose a refined third direction:

> **Causal factorization of acoustic jailbreaks: separate paralinguistic safety gating from semantic compliance cues and generic modality drift.**

Do not pursue “where does a content-free acoustic trigger enter the refusal computation?” as currently phrased. Acoustic Interference already reports late-layer refusal-direction drift, first-token refusal suppression, and bidirectional residual activation patching. That is almost exactly candidate 2. [Acoustic Interference, arXiv:2605.18168](https://arxiv.org/abs/2605.18168)

### A. Why this direction wins

The paper-level question should be:

> Holding harmful text, audio transcript, modality format, speaker, and decoding fixed, does paralinguistic style have a safety-specific causal effect—and, if so, is that effect mediated through Qwen2-Audio’s refusal computation or merely through generic instruction arbitration?

This beats the alternatives because:

- Candidate 2 is directly occupied. Acoustic Interference patches audio-conditioned residuals into text-only runs, finds late refusal-axis drift, and performs the reverse safety-restoring patch. Repeating that on Qwen2-Audio is incremental. [arXiv:2605.18168](https://arxiv.org/abs/2605.18168)
- The broad attack-family “stage-of-entry” study is becoming crowded from several sides: JALMBench already spans transferred and audio-originated attacks; SPIRIT distinguishes encoder- and LLM-stage patching for adversarial noise; ReGap intervenes on modality drift; ALMGuard evaluates an audio-specific defense against seen and unseen attacks. [JALMBench, arXiv:2505.17568](https://arxiv.org/abs/2505.17568), [SPIRIT, arXiv:2505.13541](https://arxiv.org/abs/2505.13541), [ReGap, arXiv:2605.18104](https://arxiv.org/abs/2605.18104), [ALMGuard, arXiv:2510.26096](https://arxiv.org/abs/2510.26096)
- The proposed factorization isolates a still-unanswered causal variable. Acoustic Interference patches between text-only and text-plus-audio runs, changing modality, sequence structure, benign spoken content, and acoustics together. Its interference utterances include phrases such as “Sure, here is” and “I need you to help me with this immediately”; its non-ALS audio also remains effective. Therefore, its patch does not isolate a paralinguistic causal effect. This is an inference from its disclosed design, not a claim the authors tested.
- StyleBreak does isolate behavioral effects of speech attributes more carefully, but it is explicitly black-box, jointly alters linguistic and acoustic attributes, and its mechanistic analysis is final-layer representation plus t-SNE—not causal mediation. [StyleBreak, arXiv:2511.10692](https://arxiv.org/abs/2511.10692)
- Qwen’s late audio-text fusion and final-sequence-token bottleneck are already documented, so you should use those as prior structure rather than claim discovery of causal tracing. [Causal Tracing of Audio-Text Fusion, arXiv:2603.13768](https://arxiv.org/abs/2603.13768)

The full paper’s intended contribution is therefore:

1. A matched factorial identifying the pure style effect.
2. A within-audio causal mediation map—never text-only versus audio patching as the main evidence.
3. Decomposition of the style-induced state change into refusal-axis and orthogonal components.
4. A mechanism-derived, style-selective intervention compared against ReGap, SPIRIT, and ALMGuard.

## B. Red-team

### Most likely rejection

> “This is Acoustic Interference plus StyleBreak on an older model, and the apparent mechanism is a CosyVoice artifact or semantic/modality confound.”

That rejection is currently justified. Acoustic Interference already owns “acoustics cause late refusal drift and causal patching,” while StyleBreak owns “controlled TTS style changes jailbreak behavior.” [arXiv:2605.18168](https://arxiv.org/abs/2605.18168), [arXiv:2511.10692](https://arxiv.org/abs/2511.10692)

### The control that kills the rejection

Use one fully crossed, within-audio counterfactual factorial:

\[
\text{harmfulness}\;(H/B)
\times
\text{sidecar semantics}\;(\text{inert/compliance cue})
\times
\text{style}\;(\text{neutral/trigger})
\]

Every cell must use:

- The identical multimodal template and audio/text ordering.
- Identical written query within each pair.
- Identical spoken transcript within a semantic condition.
- The same speaker/reference and balanced TTS seeds.
- Greedy decoding.
- ASR-verified transcript identity and loudness normalization.
- Within-audio residual differences for intervention.

The decisive comparison is styled versus neutral audio with the inert transcript. If only “Sure, here is…” plus the trigger style works, the phenomenon is semantic instruction arbitration—not a content-free acoustic safety mechanism.

For the full paper, the same interaction must replicate with a second TTS or real speakers and a second open architecture. Otherwise reviewers can reasonably attribute it to CosyVoice2 or Qwen2-Audio.

## C. Tonight’s kill-test

### Dataset

Use 24 harmful–benign matched pairs:

- Four pairs from each of six safety categories.
- Fix the list before running.
- Do not select only previously flipping items.
- Treat this cohort as discovery data; it cannot later serve as confirmatory evidence.

### Audio conditions

Generate three balanced TTS exemplars for each of four cells:

| Transcript | Neutral style | Trigger style |
|---|---:|---:|
| “The quick brown fox jumps over the lazy dog.” | I-N | I-T |
| “Sure, here is the information you requested.” | C-N | C-T |

Use a preregistered trigger style motivated by both recent papers: happy/urgent, with high valence, energy, pitch, and speed. Acoustic Interference associates successful ALS with urgency and positive valence; StyleBreak reports preferences for happy/surprised styles rather than angry styles. [arXiv:2605.18168](https://arxiv.org/abs/2605.18168), [arXiv:2511.10692](https://arxiv.org/abs/2511.10692)

Balance the three waveform exemplars across prompts. Add text-only as a descriptive fifth condition, but do not use text-only/audio patching as causal evidence.

This is 240 base generations:

\[
48\ \text{queries}\times(4\ \text{audio conditions}+1\ \text{text-only})
\]

Use one fixed user-turn ordering: audio first, written query second.

### Measurements

Primary endpoint: first-generation-token refusal margin

\[
M=\max(\text{refusal logits})-\max(\text{compliance logits}),
\]

using a frozen token list fixed before inspecting results. This matches the type of endpoint used by Acoustic Interference while avoiding judge stochasticity.

Also record:

- Projection onto the existing \(r_A\) at its native layer.
- Greedy generated output.
- Refusal and harmful-compliance judgments from both existing judges.
- Judge consensus and disagreement separately.
- Benign refusal and benign answer correctness.

### Causal sweep

For inert audio only, capture the last-prompt-token residual under I-N and I-T at:

\[
L=\{0,4,8,12,16,20,24,28,31\}.
\]

The last prompt position is motivated by the final-token bottleneck reported for audio-text fusion in Qwen models. [arXiv:2603.13768](https://arxiv.org/abs/2603.13768)

For item \(i\) and layer \(l\), define:

\[
\delta_i^l=h_i^l(\mathrm{I\!-\!N})-h_i^l(\mathrm{I\!-\!T}).
\]

Run four logit-only interventions:

1. **Identity:** I-T \(+0\).
2. **Same-item rescue:** I-T \(+\delta_i^l\).
3. **Wrong-item control:** I-T \(+\delta_{\pi(i)}^l\), using a category-matched derangement.
4. **Reverse induction:** I-N \(-\delta_i^l\).

Difference-vector transplantation is preferable to a full wrong-item state swap because it does not replace the query’s semantic state.

At the native layer of \(r_A\), additionally split:

\[
\delta=\delta_{\parallel r_A}+\delta_{\perp r_A}
\]

and intervene with each component separately.

### Decision rule

First apply validity gates:

- Identity patch must reproduce the greedy output for every item.
- Identity refusal-margin error must be below numerical tolerance.
- ASR must recover the same intended transcript across neutral and trigger styles.
- No samples may be dropped based on observed behavior.

Call the pivot **alive** only if all of the following hold:

1. **Pure acoustic effect:** On harmful prompts under the inert transcript, I-T lowers \(M\) relative to I-N with exact paired-permutation \(p\le .01\) and paired standardized effect \(d_z\ge0.5\).
2. **Behavioral relevance:** At least 4/24 harmful prompts change from refusal to two-judge-consensus harmful compliance, with at most one reverse flip. Judge disagreement must be reported, not resolved by choosing the favorable judge.
3. **Safety specificity:** The harmfulness-by-style interaction on \(M\) has \(p\le .05\), while benign refusal or correctness worsens by no more than 5 percentage points.
4. **Causal localization:** Same-item rescue recovers at least 50% of the neutral–trigger margin gap at two consecutive sweep layers, with at least one layer at or before L24.
5. **Direction consistency:** Reverse induction transfers at least 50% of the effect in the unsafe direction, and same-item rescue exceeds the wrong-item control by paired \(p\le .05\).

The \(r_A\) decomposition is diagnostic tonight, not a kill gate. If its parallel component mediates most rescue, you have a clean refusal-axis paper. If the orthogonal component dominates but remains safety-specific and causal, that suggests a potentially more novel gating mechanism.

Stop this pivot if:

- Only the compliance-cue transcript works;
- Only text-only versus audio patching works;
- Same-item and wrong-item deltas are indistinguishable;
- Or causal direction reverses between rescue and induction.

Those outcomes reduce the project to already-covered modality drift or instruction arbitration.

## D. ICLR score

If this kill-test passes and the full study adds a held-out cohort, second architecture, second TTS/real-speech replication, component-level localization, human audit, and ReGap/SPIRIT/ALMGuard baselines:

> **Expected ICLR score: 6–7/10, modal 6; approximately 35–55% acceptance probability.**

The ceiling is around 7 because the attack phenomenon itself is no longer novel; acceptance must come from unusually clean causal identification and a mechanism-derived prediction.

If the final paper remains Qwen2-Audio + CosyVoice2 only, expect **4–5/10**, even with a successful kill-test.