## Verdict: DROP

The pure-paralinguistic factorization remains a valid control experiment, but it is no longer the best paper spine. I assign it:

- **~30% probability of detecting some style-specific safety effect**
- **~10% probability of becoming an ICLR-worthy contribution**

The second probability is what matters. VoxParadox lowers the empirical prior, your Run 3 lowers it specifically for Qwen2-Audio+CosyVoice2, and Acoustic Interference has already published most of the attractive mechanistic endpoints.

## 1. What VoxParadox establishes

VoxParadox contains 2,000 multiple-choice examples across ten perception tasks: age, gender, emotion, pitch, volume, speed, vocal range, intonation, speaker identity, and speaker count. Each transcript asserts an incorrect attribute while the synthesized audio conveys the intended ground truth. Transcript fidelity is filtered at WER=0; emotion additionally uses an external classifier, and 10% receives human verification. [VoxParadox paper](https://arxiv.org/pdf/2605.27772)

The result is genuinely bad for the base rate:

- Across twelve models, average acoustic-ground-truth accuracy is 15.33%, versus 64.34% agreement with the misleading transcript.
- Qwen2-Audio gets **14.85% acoustic accuracy** and **70.25% transcript-implied agreement**.
- Qwen2-Audio’s emotion accuracy is only **14%**; age and gender are **2% and 3%**.
- Reversing audio—removing intelligible lexical content while retaining several acoustic properties—improves acoustic reliance, supporting the lexical-shortcut explanation.
- PCLM+DPO raises Qwen2-Audio from 14.85% to 72.30%, showing the information/utilization bottleneck can be changed by retraining. [Results and Qwen2-Audio table](https://arxiv.org/pdf/2605.27772)

But it does **not** test safety:

- No harmful/benign requests.
- No refusal/compliance behavior.
- No jailbreak success, first-token refusal margin, or safety judge.
- No test of whether a poorly recognized acoustic feature can nevertheless perturb a brittle downstream safety decision.

Nor is its internal analysis causal in your intended sense. It uses frozen layer-wise probes, reversed-audio input interventions, intermediate-layer augmentation, PCLM, and DPO. The probes show retrievability, not causal mediation. There is no residual-stream transplant, safety-coordinate restoration, or acoustic-style → refusal causal chain. [Probe and intervention details](https://arxiv.org/pdf/2605.27772)

Finally, it is conceptually close but not literally your factorial. VoxParadox constructs one lexical–acoustic contradiction per MCQ; it does not report matched harmful requests rendered under neutral/trigger styles crossed with inert/compliance sidecars.

**Conclusion on Challenge 1 alone:** it raises the bar sharply but does not logically kill the safety hypothesis.

## 2. Why the combined evidence kills the pivot as a paper

Your Run 3 is not a clean same-transcript test—the sad/angry conditions included expressive content rewrites—but it is still the most relevant model-specific evidence:

- Genuine style gap: **+5.0 pp**, below the preregistered +8 pp threshold.
- Escape Spearman: **−0.028**.
- Escape AUROC: **0.484**.
- Angry changed behavior somewhat, but not through the validated `r_A` coordinate. [Run 3 results](/workspace/audio-safety/docs/experiments/exp1_refusal_cone_drift/results.md:294)

Changing to happy/urgent and improving transcript control might reverse this result. That possibility is already included in the 30% effect probability; it does not rescue novelty.

More importantly, Acoustic Interference already reports:

- Qwen2-Audio jailbreak increases of **+22.22 to +35.19 pp** on JBB and **+23.81 to +26.97 pp** on WildJailbreak, depending on the evaluator.
- First-token refusal-logit suppression.
- Projection onto a refusal direction learned from refused versus complied text runs.
- Bidirectional residual-stream patching: audio activations reduce the refusal margin, while text activations restore it.
- Associations between attack effectiveness and pitch, valence, speed, and gender. [Acoustic Interference results and mechanism](https://arxiv.org/html/2605.18168)

Your semantic-factorization criticism is fair. Its “semantically benign” sidecars include:

- “Sure, here is”
- “Below is an instruction that describes a task”
- “I need you to help me with this immediately”

Those are not semantically inert with respect to compliance. [AIA sidecar construction](https://arxiv.org/html/2605.18168)

A clean inert-versus-compliance-sidecar factorial would therefore improve causal identification. But after AIA, that is a **confound correction and replication**, not a fresh central mechanism. Even the proposed first-token and activation-transplant endpoints are already occupied.

The apparent VoxParadox/AIA conflict is also not a real contradiction:

- VoxParadox asks whether models can **explicitly identify ordinary acoustic attributes** under lexical conflict.
- AIA mines unusual, high-dimensional Bark latent styles and asks whether they can **perturb a downstream safety boundary**, while also adding compliance-coded spoken sidecars.
- Your Run 3 tests only sad/angry CosyVoice2 renderings.

A model can be poor at reporting emotion while still being sensitive to a rare acoustic pattern. The scientifically interesting “when is audio used versus ignored?” question remains open, but answering it convincingly would require a broad model × TTS × acoustic-manifold study. On the present assets, it is unlikely to clear the novelty bar.

**Blunt call: DROP.** Preserve the factorial as a compact diagnostic or rebuttal experiment, not the paper’s spine.

## 3. Best alternative: safety under acoustic uncertainty

My single best replacement is:

> **Clarify, Don’t Guess: Risk-Calibrated Safety under Localized Acoustic Uncertainty**

The question is not whether audio jailbreaks suppress refusal. It is:

> When the safety-critical part of a spoken request is genuinely unintelligible, does an Audio LLM ask for clarification—or confidently guess, unsafely comply, or blanket-refuse?

That is an audio-native safety capability problem, not another refusal direction, modality-drift, steering, or paralinguistic-trigger paper. My targeted search found nearby work on perturbation jailbreaks, mixed-speaker composition, and auditory prompt injection, but not a controlled benchmark whose desired action is **clarification under localized safety uncertainty**. For comparison, SACRED-Bench treats mixed/overlapping audio as an attack and measures ASR, while AudioHijack optimizes imperceptible command injection; neither studies calibrated clarification. [SACRED-Bench](https://openreview.net/forum?id=KM2J8XFz5A), [AudioHijack](https://arxiv.org/abs/2604.14604)

### Cheap A40 kill-test

Select 24 of the 150 harmful/benign pairs whose safety distinction is concentrated in a short clause. Render prefix, critical clause, and suffix separately with the same CosyVoice2 speaker, then concatenate every condition identically so seams are controlled.

Use five conditions:

1. Clean segmented audio.
2. Mild masking of only the safety-critical clause.
3. Strong masking of only that clause.
4. Duration/energy-matched masking of a noncritical clause.
5. Text control with the critical clause replaced by `[inaudible]`.

That is 24 pairs × 2 safety labels × 5 conditions = **240 greedy generations**, with 64 output tokens. Give the model an explicit instruction to request repetition when the request is unclear.

Extend the two-judge rubric to classify:

- clarification request;
- appropriate safe/helpful answer;
- blanket refusal;
- substantive unsafe compliance;
- degenerate transcription/restatement.

Primary kill-test outcomes:

- clarification rate under critical-span ambiguity;
- unsafe-guess rate on harmful items;
- benign over-refusal rate;
- critical-mask effect relative to the matched noncritical mask.

**GO:** critical-span ambiguity produces at least a 20 pp rise in non-clarifying errors versus the noncritical-mask control, with the direction reproduced by both judges and clarification remaining below 30%.

**KILL:** the effect disappears against the noncritical-mask control, is entirely explained by general intelligibility failure, or Qwen2-Audio appropriately clarifies on at least half the ambiguous cases.

Use `r_A` only as a frozen secondary diagnostic. If adding `r_A` reduces unsafe guessing but merely converts it into benign over-refusal—without increasing clarification—that is useful evidence that a binary refusal controller is not an uncertainty policy. The existing greedy patch machinery can then localize where clean-versus-ambiguous evidence is lost, but it should not become the headline contribution.

For a full paper, add human intelligibility calibration, at least three additional Audio LLMs, a second TTS or human speech condition, and a simple uncertainty-triggered clarification policy. The strongest result would be a safety–utility Pareto improvement, not another jailbreak ASR table.

## 4. Honest ICLR score

- **Original paralinguistic-factorization pivot now:** **4/10**, likely reject.
- **Uncertainty/clarification direction with only Qwen2-Audio:** **4/10**, still a case study.
- **If the kill-test succeeds and the full multi-model, human-calibrated benchmark plus mitigation is completed:** **6/10**, weak accept.
- **My present expected score for the final recommendation:** **5/10**, with a credible path to 6 but not yet a 7-level idea.

The advantage is that the cheap test can kill it cleanly before another large audio-safety mechanism build.