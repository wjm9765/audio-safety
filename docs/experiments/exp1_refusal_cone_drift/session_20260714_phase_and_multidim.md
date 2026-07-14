# Session 2026-07-14 — Phase-frontend causal result, multidimensional-analysis adjudication, and emotion-attack negative

Comprehensive record of the 2026-07-14 session. Self-contained (outputs/ and /workspace/audio_safety_data/
are git-ignored, so all load-bearing numbers are inlined). Dual-agent throughout: Codex `gpt-5.6-sol`
(high + web_search) ↔ Claude (Opus 4.8). Human PI directive: "the same audio, slightly perturbed
differently, flips the attack; the cause must be internally analyzable; F0/pitch are NOT the point."

---

## 0. Starting point
Run 6 had shown that the librosa "pitch shift" jailbreak of Qwen2-Audio is a **phase-vocoder DSP artifact**,
not pitch (WORLD F0-only reproduced only ~10% of flips). The session goal: run the "multidimensional analysis
verification experiment" and reach ≥7/10 from both a Codex and a Claude ICLR reviewer, else devise another
methodology fitting the PI goal.

## 1. Run 7 — phase-vocoder frontend distortion causally erodes refusal (the 7/10 result)

### Setup & rationale
Codex (2 web-grounded design rounds) reframed a naive "random phase scramble" into a **within-phase-vocoder
counterfactual**: render the SAME harmful sentence at the SAME nominal ±3 st with two vocoders differing ONLY
in phase handling — `pv_standard` (independent-bin phase = librosa) vs `pv_locked` (identity phase-locking;
identical magnitude interpolation / timing / resampling / gain / length; verified STFT magnitude preserved to
~1e-7). Since Whisper mel is magnitude-only, phase reaches the model ONLY via overlap-add realized-mel
distortion. Conditions on 91 neutral-refuser harmful items × {−3,+3}: neutral, pv_standard, pv_locked,
`phase_transplant` (neutral magnitude + measured std−locked phase residual), `mel_matched_ctrl` (pv_locked +
zero-phase EQ, processor log-mel RMS matched to D_pair). Operators in
`src/audio_safety/evaluation/phase_ops.py`; pipeline `scripts/phase_{render_generate,analyze,internal,
causal_patch,dose_response,dose_analyze,power_analyze}.py`. Endpoint = first-token refusal margin + 2-judge
label; frozen L18 refusal direction from run5 (not refit).

### Results (all item-clustered)
- Reproduction valid: pv_standard flips 13/91 at ±3 vs run5 real-librosa 11/91.
- **Behavioral flips modest & NOT flip-specific**: pv_standard 14.3%, pv_locked 6.6%, transplant 7.7%,
  mel_matched_ctrl 11.0%, neutral 2.2% (per-item ever-flip). G1 necessity McNemar p=0.092 (MISS vs +15pp bar);
  G4 flip-specificity MISS (mel_ctrl≈std behaviorally).
- **Margin erosion IS phase-specific**: pv_standard +1.66 [90% 1.30,2.01] ≫ pv_locked +0.97, mel_ctrl +0.90.
- **Representational specificity (robust headline)**: L18 refusal-axis displacement pv_standard −2.81 vs
  mel_matched_ctrl +0.13 (survives decoding-failure exclusion: −2.42 vs +0.02). Per-cell flip median −3.65 vs
  −0.02. corr(displacement, margin erosion) −0.573 (p=3e-9, item-clustered).
- **Causal (G5)**: restoring the frozen L18 refusal-axis component adds ΔM +2.14 vs 30-dir orthogonal-ensemble
  null +0.40; paired bootstrap restore−orth +1.74 [95% CI +0.49,+3.07] (excludes 0); survives leave-two-out;
  60% behavioral flip-back. **Double dissociation**: restoration reverses pv_standard flips (+2.14) but NOT
  mel_ctrl flips (+0.13, n=5) — phase flips are refusal-axis-mediated, equal-mel-distance coherent flips are not.
- **Dose-response (Codex's decisive experiment)**: α∈{0,.25,.5,.75,1} interpolating locked→standard phase on
  the SAME pitched magnitude (pv_lambda; magnitude fixed to ~1e-7). As α rises: L18 refusal-axis displacement
  0→−2.81 (item Spearman −0.579 [90% −0.67,−0.48], 84% items correct sign) and margin erosion +0.97→+1.66
  (Spearman +0.359) rise monotonically; flip 2.7%→7.7%; decoding-failure 18%→28% (two effects: refusal erosion
  + generic disruption, reported honestly).

### Verdict & reviews
PROCEED as a **supporting mechanistic section** (not standalone). **Codex gpt-5.6-sol 7.0/10** ("ship as-is;
do NOT pivot"). **Independent adversarial ICLR reviewer 7/10 as a supporting section** (standalone 5.5;
reproduced every number independently). PI ≥7 gate MET. Binding framing (both reviewers): state the mechanism
at the representation/margin level only; axis is **contributory, not exclusive**; the behavioral flip is NOT
shown refusal-specific (G4 miss + decoding-failure co-moves with α). Title: "Counterfactual Controls for
Acoustic Jailbreaks: Phase Incoherence Causally Erodes Refusal in Qwen2-Audio."

## 2. Multidimensional analysis — the PI's core interest — adjudicated

### What was run (on the frozen L18 activations)
Built a refusal SUBSPACE from 9 per-category difference-in-means vectors → SVD. Participation ratio (effective
dim) = **3.78** (refusal representation IS geometrically multidimensional). Then asked whether the phase attack
is multidimensional and whether multidimensionality predicts the outcome.

### Results (corrected after the blind Codex cross-check caught two errors)
- Rank-k margin-erosion prediction (leave-one-category-out R²): 1D 0.313, 3D 0.302, 5D 0.305, "10D" 0.332
  → **no incremental gain**. (Note: only 9 category vectors exist → max rank 9; "10D" was ≤9D.)
- **CORRECTED**: the phase attack displacement, measured properly as the ENSEMBLE centered-SVD of per-item
  displacements within the ≤8-D refusal subspace, has participation ratio **1.49 (≈1-D)** — Claude's earlier
  "2.2–3.6" was the meaningless PR of a single mean vector; Codex caught it, recomputation confirmed.
- Norm null (Codex-requested): ‖P_S Δ‖/‖Δ‖ = 0.362 vs random 8-D subspace 0.041 → refusal subspace captures
  8.7× more of the displacement than a dimension-matched random subspace (projection is meaningful — but ~1-D
  inside).
- Two content-preserving DSP perturbations (phase-incoherence vs coherent F0/formant shift) have refusal-
  subspace cosine **0.996** (same axis; full-space 0.73).

### Blind cross-check verdict (`outputs/cross_checks/20260714_multidim_methodology.md`)
Claude and Codex AGREED (blind): a naive "SVD shows refusal is multidimensional" claim is **NOT ICLR-caliber**.
The refusal REPRESENTATION is multidimensional (PR≈3.8) but the ATTACK is ~1-D within refusal space (PR 1.49)
and extra dimensions add no predictive value (rank1≈rank9). Preemption is heavy: **SARSteer (ICML 2026)
already does PCA/SVD safe-subspace on Qwen2-Audio**; Wollschläger 2025 (concept cones), Piras 2026 (multi-dir
beats single), and Joad 2026 ("rich geometry, 1-D behavioral knob") already cover the text case. The PI's
field critique ("LALM safety over-relies on single-direction") is **refuted for prediction by the PI's own
data** (rank-1 predicts as well as rank-9); the DEFENSIBLE critique is **"projection onto a refusal direction
is over-interpreted as CAUSAL mediation"** (occupancy ≠ intervention; supported by the earlier Run-4 causal
NO-GO). Defensible multidimensional path exists ONLY under a **causal-bottleneck reframing**: "rich category-
conditioned refusal geometry collapses onto a shared low-dim audio safety bottleneck; matched counterfactuals
identify WHEN it causally mediates; cross-modal & attack-specific experiments identify WHEN the collapse
breaks." Codex-ranked ICLR-caliber options: (1) shared-vs-private causal-bottleneck decomposition via matched
patching on held-out attacks; (2) cross-modal causal refusal geometry (text vs audio pathways).

## 3. Run 8 — emotion attack probe (the PI's "different attacks hit different axes" hypothesis)

### Setup & rationale
PI hypothesis (from Wollschläger concept cones): different audio attacks load DIFFERENT refusal cone axes, so
a combined transform is a stronger jailbreak. Codex's correction: multidimensional geometry ≠ natural attacks
partitioning across it (existence ≠ accessibility); prior ~75% that emotion would funnel. Preregistered probe
(Codex 0.80/0.90 rule): 91 harmful refusers + 91 matched benign × {neutral, sad, fearful, angry}, CosyVoice2
instruct2, fixed base voice; extract L18; project (style−neutral) displacement into the frozen 5-D refusal
subspace; cosine to the phase/pitch DSP axis; split-half stability; benign control.
Scripts `scripts/emotion_{extract,analyze}.py`; renders `outputs/run8_emotion_probe/`.

### Results — emotion does NOT attack Qwen2-Audio
| style | behavioral refusal | mean margin | refusal-axis cos to DSP | margin erosion | split-half stab |
|---|---:|---:|---:|---:|---:|
| neutral | 78.0% | +1.84 | — | — | — |
| sad | 78.0% | +2.08 | −0.76 | −0.24 | 0.65 |
| fearful | 80.2% | +1.71 | −0.45 | +0.13 | 0.39 |
| angry | 79.1% | +2.12 | −0.78 | −0.27 | 0.73 |

Refusal stays ~78–80% across ALL emotion styles (identical to neutral); margin erosion ≈0/negative; displacement
direction unstable (split-half < 0.80). **DECISION = AMBIGUOUS = ineffective manipulation**: emotion is NOT an
effective content-preserving attack on Qwen2-Audio, so it cannot adjudicate funnel-vs-independent-axis. Content
WER identical across styles (~1.2, Qwen mis-transcribes neutral too). Consistent with the LISTEN benchmark
(2026): audio LMs underuse acoustic emotion relative to lexical content — the model refuses on CONTENT, which
emotion preserves. Caveat: emotional intensity of the CosyVoice2 renders was not verified by an emotion
classifier; but the 0-effect + literature strongly indicate a genuine model property, not merely weak renders.

## 4. Synthesis / honest conclusions
1. **The PI's "different attacks → different refusal axes → combined attacks easier" model is NOT supported**:
   the two DSP attacks (phase, pitch) funnel to the SAME refusal axis (cos 0.996) and the attack is ~1-D within
   refusal space; the categorically-different attack (emotion) is not even effective on this model.
2. **The strong, novel, defensible result is the phase matched-counterfactual causal factor-isolation** (Run 7,
   7/10 supporting section). Keep it as the anchor.
3. **The multidimensional angle survives only as the "collapse/funnel" observation** under a causal-bottleneck
   reframing; a plain SVD claim is preempted (SARSteer) and not load-bearing.
4. **Reader-empathy framing** (Codex scope round): broaden the INTRODUCTION to the general phenomenon (pitch/
   pace/style audio jailbreaks) but keep the MECHANISM claim phase-only; present cross-transformation
   generalization as a shared diagnostic READOUT, not an explanatory mechanism.
5. **Landscape finding (honest, and the broad story the PI wanted)**: Qwen2-Audio refusal is content-driven and
   robust to paralinguistic emotion, but vulnerable to low-level DSP frontend distortion (phase incoherence),
   and such acoustic attacks funnel to a single low-margin refusal bottleneck despite multidimensional refusal
   geometry.
6. **Recommended ICLR-caliber extension** (if pursuing beyond the supporting section): cross-modal causal
   refusal geometry (where audio vs text safety pathways diverge/reconverge) — this is the rigorous way to cash
   out the PI's "single-direction is a limitation" intuition, at the CAUSAL rather than the dimension-counting
   level.

## 5. Artifacts
- Direction/pre-reg: `run7_phase_frontend_distortion_direction_20260714.md`.
- Run logs: `results.md` (run7, run8 entries).
- Cross-check: `outputs/cross_checks/20260714_multidim_methodology.md` (git-ignored; conclusions inlined in §2).
- Codex rounds (git-ignored under outputs/): `codex_{phase_design,userfeedback,interpret,rescore,scope,
  multiaxis,crosscheck_multidim}_out.md`.
- Code: `src/audio_safety/evaluation/phase_ops.py`; `scripts/phase_*.py`, `scripts/emotion_*.py`.
