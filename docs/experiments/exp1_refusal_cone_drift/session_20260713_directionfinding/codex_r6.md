## Verdict

The proposed speaker-conditioned perception-vs-policy project is **not greenlightable**. I found no paper performing the exact speaker-pair interchange-patching experiment, but the remaining novelty is too narrow and the available measurements cannot support its advertised causal claim.

### Collision check

- **VoxSafeBench:** Not merely a behavioral disparity benchmark. Its appendix explicitly compares matched cue-recognition probes against safety behavior and interprets low-probe/low-safety as perception-limited and high-probe/low-safety as downstream alignment failure. It correctly admits that this is not a complete causal decomposition. It has no activation analysis, but it has already posed and behaviorally operationalized the proposed perception-vs-policy question. [VoxSafeBench](https://arxiv.org/abs/2604.14548)

- **Evaluation of Audio Language Models for FSS:** It does use controlled, semantically identical gender/accent/emotion variants and Ordinal Equalized Odds, including African-versus-American/English accent disparities. It is outcome-level and contains no activation analysis. However, I could not verify Claude’s purported quotation that mechanistic blindness is its “most significant unresolved tension”; its stated limitations instead emphasize model coverage, acoustic conditions, threat models, and multi-turn evaluation. [Aloufi et al.](https://arxiv.org/abs/2603.13262)

- **AOR-Bench:** “Behavioral-only” is false. It reports the cited MiMo male/female change, approximately 48.1% to 41.7%, but also extracts refused-versus-complied hidden-state directions in Qwen2-Audio and other open models and performs activation steering. It does not causally explain the speaker effect, but it directly occupies the speaker variation + refusal activation neighborhood. [AOR-Bench](https://arxiv.org/abs/2606.21147)

- **Additional collisions:** VoxParadox already performs layer-wise cue-availability probing and distinguishes information lost before the LLM from information present but ignored by the LLM. The text paper supplies the harmfulness/refusal decomposition almost verbatim. [VoxParadox](https://arxiv.org/abs/2605.27772), [LLMs Encode Harmfulness and Refusal Separately](https://arxiv.org/abs/2507.11878)

- The fairness framing is crowded independently: *The Voice Behind the Words* already holds content fixed through voice cloning and measures accent × gender response disparities. [Satish et al.](https://arxiv.org/abs/2603.16941)

Thus, the exact experiment remains unclaimed, but its components are already published separately:

> VoxSafeBench perception/policy diagnostic + VoxParadox layer-wise localization + Zhao et al. harmfulness/refusal geometry + AOR speaker variation and activation steering.

That is an incremental combination, not a fresh ICLR problem.

## Why the proposed evidence would not work

1. **Projection is not causal decomposition.**  
   Preserved \(c_H\) and shifted \(c_A\) establishes probe geometry, not that harmfulness was causally available to the policy or that a threshold changed. A linear harmfulness probe can remain accurate while the model uses another nonlinear or position-dependent representation.

2. **\(r_A\) cannot be called the native readout.**  
   Escape AUROC 0.48 is direct negative evidence for that interpretation. A direction can steer generation without being the direction naturally read by the model. “Speaker-dependent shift on \(r_A\)” therefore cannot establish readout bias.

3. **The required causal fallback has already failed.**  
   The interchange patch behaved like generic clean-state rescue and lacked item specificity. Reusing it across speakers would invite exactly the reviewer objection you have already validated empirically.

4. **The proposed speaker categories mix protected nuisances with legitimate context.**  
   Accent and gender should ordinarily be invariant. Child status, impairment, emotion, and sometimes authority can legitimately change the appropriate response or wording. Combining them under one “bias” hypothesis makes the estimand normatively incoherent.

5. **Synthetic demographic attribution is fragile.**  
   One `child_female` or `elderly_male` CosyVoice rendering confounds the demographic label with speaker identity, synthesis quality, timbre, and prosody. An ICLR-quality fairness claim needs multiple independent voices per group and crossed factors.

6. **The cited population result may not exist in Qwen2-Audio.**  
   AOR explicitly finds model-dependent directions and weak sensitivity in some LALMs. Before mechanisms, you would need to establish a replicated speaker-conditioned behavioral gap in the exact target model.

Candidate score: **4/10, reject**. Even a positive result likely reads as “activation-space elaboration of VoxSafeBench”; a null result is unsurprising.

## Better direction

The strongest remaining opening is:

> **Hear Only What You Need: Task-conditioned least-privilege gating of speaker information inside Audio LLMs.**

This is not another refusal mechanism. The problem is that semantic tasks such as transcription and content QA expose identity, gender, accent, and other speaker information to every downstream LLM layer even when those features are unnecessary. The method would give the model two explicit modes:

- privacy-default: preserve linguistic/semantic utility while suppressing speaker information;
- authorized: retain speaker information for tasks such as diarization or user verification.

The neighborhood is occupied but the specific problem remains defensible:

- *The Model Hears You* motivates least privilege and demonstrates behavioral identity/attribute inference, but does not build a task-conditioned internal gate. [He et al.](https://arxiv.org/abs/2503.16833)
- HearSay and AudioPrivacy establish large-scale attribute leakage. [HearSay](https://arxiv.org/abs/2601.03783), [AudioPrivacy](https://aclanthology.org/2026.findings-acl.283/)
- Full-duplex work finds speaker information throughout hidden states and uses fixed waveform/feature anonymization, not a task-conditioned Audio-LLM interface. [Kuzmin et al.](https://arxiv.org/abs/2603.08179)
- Conventional selective masking and speaker anonymization are strong baselines, but they primarily alter the waveform or generic speech representation rather than enforce task-specific access inside an instruction-following Audio LLM.

The distinction must remain sharp: **not another speaker-leakage audit, not another fixed anonymizer, and not linear-probe erasure presented as privacy.** The contribution must be a task-conditioned gate evaluated against adaptive post-hoc attackers.

Audio-output/TTS, harmful sound-event, and guardrail directions are worse choices: AudioGuard already claims both input- and output-side safety, TTS/voice-cloning misuse, impersonation, compositional harms, and non-speech harmful events. [AudioGuard](https://arxiv.org/abs/2604.08867) Multi-turn safety is also already occupied by spoken-dialogue judging and VoxSafeBench. [LALM-as-a-Judge](https://arxiv.org/abs/2602.04796)

## Cheap A40 kill-test

Use Qwen2-Audio-7B with the existing hooks and CosyVoice batch path.

**Data**

- 24 independent reference speakers, balanced across gender and at least four accent groups.
- Render the same 48 benign semantic scripts per speaker: 1,152 clips.
- Split by speaker: 16 gate-training speakers and 8 completely held-out speakers.
- Add 48 ordinary content-QA/transcription items for utility; do not use refusal as the primary utility endpoint.

**Conditions**

1. Unmodified Qwen2-Audio.
2. Transcript-only privacy ceiling.
3. Low-rank speaker-subspace removal at the audio-encoder/LLM interface.
4. Rank-matched random projection control.
5. Authorized bypass of the gate.

Start with a closed-form whitened speaker subspace or a tiny adversarial low-rank projector; no model fine-tuning is needed for the kill-test.

**Primary endpoints**

- Privacy: held-out-speaker verification/linkability, gender AUROC, accent macro-accuracy.
- Crucially, train fresh linear and two-layer MLP attackers *after the gate is frozen*. Suppressing the original probe alone does not count.
- Utility: WER, content-QA accuracy, and semantic agreement with unmodified responses.
- Specificity: learned gate must outperform rank-matched random removal.

**Pre-register this decision rule**

GREENLIGHT only if one operating point, on held-out speakers:

- increases speaker-verification EER by at least 10 percentage points;
- reduces adaptive gender/accent attack performance by at least 0.15 AUROC or equivalent normalized advantage;
- loses no more than 3 percentage points of content-QA accuracy and no more than 2 absolute WER points;
- beats the random-projection control with bootstrap 95% confidence intervals;
- and the privacy gain survives the fresh MLP attacker.

KILL if the gain disappears under the adaptive attacker, requires destroying semantic utility, fails on held-out speakers, or exists only at late layers while earlier accessible states still leak identity.

Estimated cost: roughly one A40 day after audio rendering. No expensive causal patching is required.

Honest ICLR score: **6/10 now; plausible 7/10 only with a successful kill-test, two model families, adaptive attacks, and a real task-conditioned privacy–utility frontier.**

**One line:** Yes—finally greenlightable, but **not** the speaker-conditioned refusal decomposition; greenlight **task-conditioned least-privilege speaker-information gating for Audio LLMs**.