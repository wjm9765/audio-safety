# Exp2 Direction: Acoustic Refusal Transport

> Status: **direction / rationale document.** This is NOT the pre-registration.
> The §0 judgement table is deliberately absent and will be written separately,
> after Phase 0 and Phase 1 fix the estimands. Nothing in this document may be
> used as a confirmatory threshold.
>
> Produced 2026-07-28 from a 12-round Claude↔Codex (`gpt-5.6-sol`, xhigh)
> cross-check. Full record of the reversal:
> [`docs/discussions/2026-07-28_exp2-direction-codex-cross-check.md`](../../discussions/2026-07-28_exp2-direction-codex-cross-check.md)
> (also kept at `outputs/cross_checks/20260727_direction_content_preserving_causal_refusal.md`,
> which is git-ignored per `AGENTS.md`).

## 1. Direction

This project studies audio-native harmful requests whose linguistic content
remains fixed while acoustic properties change. Its central question is: **which
acoustic element causally reaches the model's refusal computation, despite
preserved perception of the harmful request, and through what internal transport
path does it alter operational harmful compliance?** The intended contribution is
a factor-to-mechanism map followed by a small inference-time defense derived from
the identified mechanism and evaluated on held-out acoustic factors at matched
benign utility. This is narrower than AIA, which studies acoustic interference
when harm is supplied in text; narrower than ReGap, which studies modality-level
refusal geometry; and more causal than SARSteer, which assumes that a
text-derived refusal direction transfers to audio. The project does not count
generic non-refusal, transcription, mishearing, or malformed generation as
successful safety bypass.

## 2. What Exp1 Established

Exp1 did not validate its original "manipulable audio-conditioned refusal axis"
hypothesis, but it produced positive and negative assets that constrain Exp2.

### Run 7: initial causal discovery

Independent-bin phase-vocoder incoherence displaced the Qwen2-Audio L18 refusal
representation by `−2.81`, versus `+0.13` for an equal-input-mel-distance
coherent control. Displacement predicted refusal-margin erosion with
item-clustered Spearman `−0.573` (`p=3e-9`). Restoring the frozen L18
refusal-axis component produced `restore − orth ΔM = +1.74 [0.49, 3.07]`, beat 30
orthogonal directions, and flipped 60% of selected behavioral failures back to
refusal. A double dissociation held: the intervention reversed `pv_standard`
flips (`+2.14`) but not mel-matched-control flips (`+0.13`). A dose sweep was
ordered at both representation and margin levels. Pre-registered G5 (causal)
PASSED; G1 behavioural necessity MISSED its `+15 pp` bar; G2 stayed invalid
because the transplant leaked F0.

### Run 10: outcome decomposition and mechanism falsification

Fresh Run 9 renders, recognition and harmful-anchor gates, full-generation
four-way judging. In the gated `n=246` cohort refusal fell `96.3% → 64.6%`, but
harmful compliance rose only `1.5% → 5.7%` (`+4.2 pp`) while decoding failure
rose `1.5% → 22.4%` (`+20.9 pp`). So 66% of the refusal loss became decoding
failure and roughly two outputs were strongly operational jailbreaks — about
**5.0 broken outputs per additional harmful compliance**. Of the total margin
erosion `−1.32`, the isolated `pv_standard − pv_locked` pure-phase component was
only `≈ −0.47`; the rest was shared pitch/vocoder processing. The geometrically
stable audio-span channel (Arm A) was causally **inert** against a
magnitude-matched sham. Readout-token Arm B was strong and bidirectional but used
an unstable, mildly circular direction.

### Run 11: harm/refusal dissociation and a generic causal handle

Coarse harmfulness recognition was **preserved**: paired `dH = +0.00` while
`dM = −2.11`, and a clean-fit harmful-versus-benign direction transferred to
attacked harmful audio with AUROC `0.95–0.997` across layers. Audio-span
restoration beat sham only at L8–L12 and was inert at L16–L28; readout-token
restoration grew with depth. Full-state L18 readout injection flipped 52%
(`46/89`) of non-refusing attack generations back to full refusal. This is a
causal decision-state handle, not a safety-specific coordinate: it could also
repair generic garbling.

### Runs 12–13: precise limits of the low-rank account

Run 12 built a cross-fitted, harmful-specific, harmfulness-orthogonal,
attacked-benign-subtracted L18 coordinate `u_s`; identity validation passed
`300/300`. Restoring `u_s` changed the first-token margin by `+0.096`, beat
matched shams, and was dose-ordered — but changed full-generation refusal by only
`+0.33 pp`, while full-state restoration changed it by `+9.67 pp`. Run 13
extended rank 1 → 64 with no behavioural rescue (max `+0.67 pp`, all intervals
including zero) and the margin lever **weakened** with rank (`+0.096 → +0.053`).
The interaction spectrum was nearly flat (`σ1/σ8 ≈ 1.8`), held-out reconstruction
poor (`0.132/0.182` even at rank 64), and the largest cross-fold principal angle
`≈ 89.4°`. Clean harmful refusal in that cohort was only `49.33%`, which dilutes
the rescue estimand. These runs rule out a stable fixed L18 displacement subspace
as the demonstrated behavioural mechanism; they do **not** distinguish a
distributed mechanism from item-specific garbling or a shared content-dependent
operator.

## 3. Why Exp1 Could Not Answer the Question

### 3.1 Endpoint scale

Binary ASR is floor-censored in aligned models. A null-calibrated reanalysis of
F1 Table 1 (reconstructed denominator `n = 1560` per cell; max−min logit range
corrected for its nonzero extremum expectation via 20k simulations) gives
MiniCPM-o-2.6 the largest excess, `+0.621` logits versus null `E[range] = 0.367`;
Qwen2-Audio a marginal `+0.359`; Qwen2.5-Omni and DeSTA2.5 **unresolvable**
(2–16 events). This is a screening statistic, not an item-clustered effect
estimate, but it shows why percentage-point ASR conceals latent movement.

### 3.2 Operator choice

Independent-bin phase-vocoder incoherence smears transients and destroys
intelligibility. SOX tempo and duration-preserving pitch operations are a
different, gentler regime. Weakening the exp1 operator cannot be assumed to
preserve its safety effect.

### 3.3 Model choice

Three independent sources agree Qwen2-Audio is operationally resistant to single
natural acoustic edits: exp1 Run 8 (emotion slightly *increased* refusal), F1
(neutral speech is Qwen's maximum non-refusal condition), and Jailbreak-AudioBench
Table 2 (most single edits *reduced* Qwen ASR: Volume×10 `−4.0%`, Tone `+8` st
`−3.2%`, machine noise `−6.1%`). This does not prove representational acoustic
invariance, but it makes Qwen a poor discovery model.

**(3.1) and (3.2) mean the intended content-preserved safety effect was never
cleanly tested. (3.3) means the primary model must change.**

## 4. Literature Position: The Unowned Edge

F1 (`2510.16893`) establishes emotion-dependent safety instability under content-
and speaker-controlled speech, then explicitly calls for work on causes and
mitigation. AIA (`2605.18168`) analyses refusal-margin drift and residual patching
but states it provides no decomposition of acoustic properties and no defense;
its malicious content is text-borne. The taxonomy paper (`2605.30031`) names
audio-specific defenses as an open problem and reports severe utility costs:
benign-refusal rate `17.1% → 30.7%` with VoiceShield and `→ 46.1%` with defensive
prompting.

VoxParadox (`2605.27772`, ICML 2026) shows audio LMs "read rather than listen",
localises paralinguistic loss to deep encoder layers and the encoder→LLM
interface, and mitigates it with PCLM+DPO — but never studies safety. GACL
(`2606.05161`) owns same-example patching for modality arbitration, not refusal.
ReGap owns modality-level refusal drift and correction, not factor-resolved
acoustic transport under preserved audio-native harm. SARSteer (`2510.17633`)
assumes a text-derived refusal direction is modality-transferable. SpeechJBB
(`2606.06037`) already performs comprehension-gated speech safety evaluation, so
comprehension gating is **not** a novelty claim.

**Sharpest defensible claim:** *To our knowledge, no prior work causally maps an
isolated, content-preserving acoustic factor into a safety-specific refusal
computation for an audio-native harmful request, and derives a held-out-factor,
utility-matched defense from that transport mechanism.*

## 5. Model Plan

Configs live under `configs/models/`; loading dispatches on `model.loader`.

| Role | Model | Why | Carried caveat |
|---|---|---|---|
| **Primary** | MiniCPM-o-2.6 (8B) | JB-AudioBench `18.2% → ~65.7%`; F1 neutral unsafe rate `3.27%` with the largest null-calibrated excess; multi-accent study calls it "particularly high susceptibility"; SALMONN-Guard `77.95%` | Large published ASRs are adaptive/compositional — **fixed-arm headroom unconfirmed**; low unsafe rate may be over-refusal, so AOR-Bench check required |
| **Secondary** | Kimi-Audio-7B-Instruct | Discrete-semantic + continuous-acoustic dual stream = privileged site to hold semantics fixed and ablate acoustics; MMSU paralinguistic `59.28%` | Stream separation is an architectural *name*; must be empirically validated before use as an instrument |
| Third | Ultravox v0.4.1 (Llama-3.1-8B) | Mature interpretability substrate; attribution patching already demonstrated on it (`2606.18924`) | Thinner safety anchor; no privileged semantic/acoustic split |
| **Resistant control** | Qwen2-Audio-7B-Instruct | Retained, not discarded: supplies the specificity contrast and keeps the SARSteer anchor | Inert to single natural edits — never the discovery model |

## 6. Phase Plan and Exploratory/Confirmatory Firewall

### Phase 0 — mandatory output audit (GPU-free, non-deferrable)

Blindly relabel stored outputs into six classes: operational harmful assistance,
substantive refusal, safe discussion, transcription/description, mishearing,
malformed output. Dataset rebuilding may be deferred; **this may not**, because
without it the analysis strata do not exist and any mechanism found is a
mechanism of an undefined mixture.

### Phase 1 — exploratory internals on existing data

1. All analyses exploratory; log every layer, site, probe, factor and
   intervention attempted. No confirmatory p-values or causal wording.
2. Analyse fixed edit arms and all items. Best-of-N winners cannot identify
   physical factors and are reported only as an adaptive-policy phenotype.
3. Outcome strata are post-treatment diagnostics, not identified causal groups.
4. A candidate intervention must distinguish operational harm from garbling,
   preserve content readouts, and show reciprocal effects where feasible.
5. No exploratory estimate is reused as confirmatory evidence.

### Phase 2 — dataset rebuild

Per-item dose-calibrated factors; processing-matched identity shams; matched
benign controls; a cohort with stable clean-refusal headroom; anchor, semantic
and intelligibility gates evaluated and frozen **before** safety generation.
Failed rows are counted, never deleted after outcomes are observed.

### Phase 3 — sealed confirmation

**May cross from Phase 1:** frozen hypotheses, model/layer/token choices,
intervention code, endpoint definitions, directional predictions, and
byte-identical candidate vectors tested without refitting. Any refit uses a
separate Phase 2 calibration split.

**May not cross:** Phase 1 items, audio renders, outcome-selected edits, factor
winners, exclusion rules, QC thresholds tuned on discovery data,
judge-calibration examples, fitted evaluation thresholds, or samples used for
layer and hyperparameter search.

## 7. Lead-versus-Artifact Rule

| Criterion | Real safety-mechanism lead | Garbling artifact |
|---|---|---|
| Behavioural effect | Restores substantive refusal or suppresses operational harmful assistance | Mainly converts malformed / transcribed / misheard outputs into fluent ones |
| Content | Preserves semantic detail, harmful anchors, harmfulness recognition | Effect covaries with lost anchors, WER, semantic recovery |
| Outcome specificity | Substantially smaller effect on mishearing and decoding-failure strata | Similar repair across harmful, benign, misheard and broken outputs |
| Directionality | clean→edited corruption and edited→clean restoration are reciprocal | Restoration-only, consistent with generic denoising |
| Generic capability | Benign instruction-following and fluency essentially unchanged | Tracks generic fluency, response mode, malformed severity |

A candidate failing the specificity row is terminated **before** dataset rebuild.

## 8. Open Risks

- MiniCPM's strongest published ASRs may vanish under fixed, non-adaptive
  single-factor evaluation.
- A low neutral unsafe rate may reflect over-refusal or weak capability rather
  than strong alignment.
- Existing benchmark labels may still confound operational harm with long
  transcription or fluent mishearing.
- Outcome-stratified discovery conditions on a post-treatment variable and cannot
  identify a mechanism by itself.
- Processing-matched shams may fail to isolate the named factor if the transform
  alters several coupled signal properties at once.
- Kimi's semantic and acoustic streams may remain information-entangled despite
  their architectural names.
- A mechanism may be architecture-specific and fail to transport across MiniCPM,
  Kimi, Ultravox and Qwen.
- An effective repair may merely raise the global refusal threshold and reproduce
  existing defenses' over-refusal costs.
- One A40 limits model/layer/factor/dose/intervention search breadth, raising the
  need for strict staged stopping.
- The eventual effect may remain latent-margin movement with no operational
  harmful-compliance consequence.

## 9. Changelog

- **2026-07-28** — This direction supersedes exp1's "manipulable
  audio-conditioned refusal axis" framing. Codex round 7 concluded "abandon the
  DSP family"; round 8 **retracted** that family-level verdict after the
  stimulus-quality confound was identified, concluding: *"the direction remains
  untested; permit exactly one preregistered quality-calibrated retest."* Round 9
  gave CONDITIONAL GO on starting from Jailbreak-AudioBench; round 11 set the
  model pool and the lead-versus-artifact rule. `design.md` §0 for exp2 is
  deliberately not yet written.
