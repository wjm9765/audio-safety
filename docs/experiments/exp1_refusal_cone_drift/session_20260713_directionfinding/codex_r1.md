# Verdict

**Recommendation: (c) pivot now—while letting the current cheap run finish strictly as an instrumentation check.**

The locked direction is not dead scientifically, but it is too preempted and too narrowly instantiated for an ICLR paper. Its behavioral headline is already in the literature; its mechanistic ingredients are also no longer new. The remaining daylight is a rigorous, stage-specific causal decomposition across genuinely different audio attack classes—not the current Qwen-only PAP/ICA experiment.

Bluntly:

- A negative result would largely confirm Alignment Curse, JALMBench, ReGap, and the shared-safety-subspace paper.
- A positive result on 19–29 post-selected Qwen flips would be too narrow to establish an audio-specific mechanism.
- That is not “outcome-robust.” It is presently outcome-trapped.

## 1. Novelty collision

### Closest prior work

| Work | What it already establishes | Remaining daylight |
|---|---|---|
| [The Alignment Curse, arXiv:2602.02557](https://arxiv.org/abs/2602.02557) | Compares text attacks, TTS-transferred audio attacks, and audio-originated attacks. Finds text-transferred audio attacks comparable to or stronger than audio-native attacks and connects transfer to modality alignment. | It is behavioral/theoretical, not item-level causal mediation. But it already owns the “maybe this is just a text vulnerability transferred into audio” thesis. |
| [Safety Geometry Collapse / ReGap, arXiv:2605.18104](https://arxiv.org/pdf/2605.18104) | Builds a text refusal direction and a modality-drift direction from semantically paired text/omni inputs; orthogonalizes them; intervenes on modality drift; reports causal safety recovery on audio and image conditions across Qwen2.5-Omni, MiniCPM-o, and Qwen3-Omni. | ReGap uses aggregate direction intervention, not item-matched bidirectional interchange patching. Your controls could support stronger identification, but “first to test whether modality causally contributes at all” is no longer defensible. |
| [SPIRIT, arXiv:2505.13541, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.734/) | On Qwen2-Audio and LLaMA-Omni, identifies noise-sensitive units and replaces attacked activations with clean/denoised reference activations in the encoder or LM. This is activation patching against audio jailbreaks. | SPIRIT is a defense, not a modality-specific causal estimand; it is not bidirectional and focuses on signal perturbations. Still, “activation patching in an audio LLM” is plainly not new. |
| [A Unified Safety Subspace Exists in Speech Language Models, Interspeech 2026](https://www.researchgate.net/publication/405947813_A_Unified_Safety_Subspace_Exists_in_Speech_Language_Models) | Reports a common text–audio refusal/compliance subspace on Qwen2-Audio and GLM-4-Voice, causal bidirectional steering, and transfer across five text/audio attacks. The authors report Interspeech 2026 acceptance; I found no arXiv ID. | It demonstrates shared controllability, not whether natural failures are mediated by that subspace. “Shared actuator versus natively used mechanism” remains open. |
| [JALMBench, arXiv:2505.17568, ICLR 2026](https://arxiv.org/abs/2505.17568) | Twelve ALMs, four text-transferred and four audio-originated attacks, attack representations, and modality comparisons. It already reports Qwen2-Audio as essentially balanced: 6.9% text versus 7.3% audio ASR. | No causal mediation, but it makes a one-model behavioral null unsurprising. |
| [Audio Is the Achilles’ Heel, arXiv:2410.23861, NAACL 2025](https://aclanthology.org/2025.naacl-long.470/) | Explicitly compares matched harmful questions in text and audio, distracting non-speech audio, and speech-specific attacks across five models. | Less carefully factorial and non-mechanistic, but cross-modal behavioral comparison is established prior art. |
| [Audio Jailbreaks in LALMs, arXiv:2605.30031](https://arxiv.org/abs/2605.30031) | Divides attacks into semantic, acoustic, signal, and embedding layers and evaluates attacks/defenses on ten LALMs with benign refusal, latency, and cost. | I found no causal interchange-attribution method. Your “failure mechanism” taxonomy could differ from its “attack surface” taxonomy—but only if validated across those attack strata. |
| [Acoustic Interference, arXiv:2605.18168](https://arxiv.org/abs/2605.18168) | Demonstrates genuinely audio-native, instruction-neutral acoustic interference across ten LALMs and analyzes induced inference-path drift. | It does not provide your matched factorial or rigorous interchange controls, but its attack is far more relevant to audio-specific causality than neutral TTS wrappers. |

### Does Alignment Curse kill the framing?

**It kills the behavioral headline.** “Is this really an audio jailbreak, or a text vulnerability transferred through speech?” is essentially Alignment Curse’s motivating question and primary answer.

It does not completely kill a paper about causal mediation. The precise surviving contribution would be:

> Given a measured modality×attack×harmfulness interaction, which processing stage and which state component causally mediate that interaction?

But that is not what the immediate patch currently identifies. A clean-audio → attacked-audio patch changes the **attack condition**, not the **modality condition**. It can localize a PAP effect but cannot establish that modality mediates the effect.

For modality mediation, donor/recipient pairs must include:

- attacked text ↔ attacked audio, same semantic item;
- clean text ↔ clean audio;
- both repeated for harmful and benign items;
- then difference the patch effects according to the behavioral three-way interaction.

The current “neutral refused → PAP complied” patch is only one leg of that design.

### Exact daylight—and why it is presently insufficient

The genuinely new pieces could be:

1. A pre-registered triple-difference estimand rather than raw audio–text ASR.
2. Item-matched, bidirectional interchange patches.
3. Wrong-item, reverse, identity, random-displacement, alternative-stage, and generic-behavior controls.
4. A separation of the refusal-axis coordinate from its orthogonal complement.

That is stronger causal hygiene than the closest works. But “we use more rigorous controls” is not an ICLR-level central contribution unless it changes the scientific conclusion, reveals a stage-wise double dissociation, or predicts which defense will transfer.

## 2. Reviewer risk

### Most likely rejection reason

**The claimed causal taxonomy is not identified by the evidence.**

A skeptical review would say:

> This is a small, outcome-selected, single-model case study showing that replacing a high-bandwidth hidden state from a refusing prompt can restore refusal. It does not establish modality-specific causation, a general failure taxonomy, or a mechanism distinctive to audio.

Specific problems:

- **Post-selection on flips:** choosing neutral-refused/PAP-complied cases conditions on the outcome. It is valid for studying mechanisms among responders, but says nothing about the population modality effect.
- **Wrong causal contrast:** clean versus PAP differs in attack content, not modality.
- **High-bandwidth transplant:** a full residual-state patch can carry attack semantics, perceived harmfulness, refusal intent, and generic generation state simultaneously. Near-output rescue is not automatically localization.
- **One layer and one anchor:** layer 16 at the first-generation position is a point intervention, not a demonstrated processing pathway.
- **One model and one semantic attack family:** no basis for an audio-jailbreak taxonomy.
- **No human audit:** with \(n\approx19\)–29, a few refusal/compliance judge errors materially alter the result.

### Is the negative acceptable?

Negative results can be ICLR-worthy when they overturn a broad claim using strong coverage, power, and controls. This negative would not: Qwen2-Audio’s small audio–text gap is already reported by JALMBench, Alignment Curse predicts text-transfer dominance, and the Interspeech shared-subspace result predicts modality-agnostic control.

“Outcome-robust” sounds like an ex ante project-management argument, not a contribution reviewers must accept. Reviewers judge the realized result.

### Score prediction

- **If the current patch works cleanly but remains Qwen-only with 19–29 selected semantic-TTS flips:** **3–5/10**, modal score **4/10, weak reject**.
- If the cheap pilot were effectively the paper’s main evidence: **2–4/10**.
- My rough acceptance probability for the locked scope is **below 15%**.

## 3. Neutral-TTS confound

### Is it an audio jailbreak?

It is legitimately a **text-transferred audio jailbreak** under an audio-input threat model. It is not an **audio-native** or **acoustic** jailbreak.

Neutral TTS is actually useful for the narrow question:

> Does delivering the same semantic attack through this synthetic speech channel change safety behavior?

It is fatal only when that result is generalized to “audio jailbreaks” as a whole. A single synthetic voice samples essentially none of the continuous audio attack surface.

The literature now sharply distinguishes these cases:

- Transcript-preserving acoustic edits: [StyleBreak, arXiv:2511.10692, AAAI 2026](https://arxiv.org/abs/2511.10692) and [Tune In, Act Up, arXiv:2501.13772](https://arxiv.org/abs/2501.13772).
- Waveform adversarial attacks: [AdvWave, arXiv:2412.08608, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/45673dbf3f331fbd911b0689872de396-Abstract-Conference.html). The correct ID is **2412.08608**, not 2402.xxxx.
- Universal robust perturbations: [“I am bad,” arXiv:2502.00718](https://arxiv.org/abs/2502.00718).
- Asynchronous, universal, over-the-air perturbations: [AudioJailbreak, arXiv:2505.14103](https://arxiv.org/abs/2505.14103).
- Benign-content acoustic interference: [arXiv:2605.18168](https://arxiv.org/abs/2605.18168).
- Joint text/audio optimization: [JAMA, arXiv:2603.19127](https://arxiv.org/abs/2603.19127).

### Minimum design for a real audio-specificity study

Use the same harmful goals across at least four strata:

1. Text PAP/ICA.
2. Exact PAP/ICA transcript rendered with neutral TTS.
3. Transcript-preserving acoustic changes: speaker, prosody, emotion, emphasis, rate, accent, room response, and noise.
4. Signal-level or audio-native attacks: AdvWave, AudioJailbreak, AIA, or published JALMBench samples.

For every stratum:

- include matched benign controls;
- verify semantic/transcript preservation with both ASR and blinded human audit;
- record perceptual quality and perturbation magnitude;
- patch encoder, projector, early LM, middle LM, and late LM states;
- include at least one second open model.

Without this, the defensible title would concern **text jailbreak transfer through synthetic speech**, not audio jailbreak causality generally.

## 4. GO/PIVOT decision

### Decision: pivot, but let the current test finish

Do not spend additional GPU time expanding the PAP/ICA flip cohort yet. Treat the current run as a test of whether the patch implementation produces nontrivial, controlled effects.

A successful pilot should require all of:

- clean→attack rescue;
- reverse attack→clean induction;
- identity approximately null;
- wrong-item substantially smaller than matched-item;
- random displacement null;
- a localized layer/stage profile rather than generic late-state replacement;
- a meaningful comparison between full-state, \(r_A\)-coordinate-only, and orthogonal-complement patches.

Even passing all of these validates the tool, not the locked paper thesis.

## Strongest alternative direction

### 1. “From Waveform to Refusal: Causal Stage Fingerprints of Audio Jailbreaks”

This is the strongest pivot.

Specific experiment:

- Use 50–100 harmful goals plus matched benign controls.
- Evaluate semantic text attacks, text-transferred TTS, transcript-preserving acoustic attacks, and signal-level attacks.
- Reuse existing rendered audio and obtain Qwen-targeted audio-originated samples from public benchmarks where possible.
- Extract states at audio-encoder blocks, projector output, and an LM layer sweep.
- Run matched bidirectional patches at each stage.
- Decompose rescue into:
  - perception/transcription restoration;
  - generic instruction-following change;
  - refusal-state restoration along \(r_A\);
  - orthogonal residual mediation.
- Test whether the resulting “causal stage fingerprint” predicts defense transfer:
  - denoising for early signal-mediated attacks;
  - modality correction for projector drift;
  - refusal steering for late shared-policy failures.

The paper-worthy result is not merely a taxonomy. It is that **causal stage fingerprints predict which defenses work across attack families**.

### 2. “Shared Actuator or Native Mechanism? Safety-State Use Across Speech and Text”

This directly exploits the existing \(r_A\) work.

Compare at each layer:

- a harm-detection/probe direction;
- a difference-in-means direction;
- the RDO control direction;
- the direction or subspace that actually mediates natural clean→attack flips.

Then patch separately:

- only the \(r_A\) scalar coordinate;
- only its orthogonal complement;
- full matched state.

Measure necessity, sufficiency, cross-modal transfer, and attack-family generalization. This asks whether \(r_A\) is:

1. a shared native refusal mechanism;
2. a generic actuator that can control refusal but is not naturally used;
3. or one coordinate within a higher-dimensional causal mediator.

[Perfect Detection, Failed Control, arXiv:2606.24952](https://arxiv.org/abs/2606.24952) raises exactly the detection-versus-control distinction, but does not answer native causal use for audio refusal. That is the surviving novelty. Angle measurements alone would be incremental; necessity and path-specific mediation are required.

## Bottom line

**Kill the locked title and broad thesis. Preserve the patching machinery.**

Alignment Curse has already answered the behavioral version of “Is it really an audio jailbreak?” largely **no** for text-transferred attacks. ReGap, SPIRIT, and the shared-safety-subspace paper substantially occupy the mechanistic space. The current experiment can at best become a pilot demonstrating that your instrumentation works.

The plausible ICLR paper is instead:

> Different audio jailbreak classes have distinguishable causal entry stages, and those stage fingerprints predict defense transfer.

That question is not answered by the closest papers, requires genuinely audio-native treatments, and gives \(r_A\) a useful role without pretending that discovering another steerable refusal direction is itself novel.