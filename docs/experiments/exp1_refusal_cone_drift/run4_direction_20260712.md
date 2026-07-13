# Run 4 → paper direction (2026-07-12, direction-finding synthesis)

> Synthesis of: the §8 attack-induced-flip results (`outputs/run4_20260712_1931_flip/`), the verified
> literature sweep (`run4_literature_sweep_20260712.md`), and a Codex (gpt-5.6-sol xhigh, web-grounded)
> deep-discussion. **This is a direction decision, not a result claim.** The pre-registered `design.md` §0
> is untouched. The paper-facing study will be a fresh, rigorous re-design.

## The decision

**Lead direction (ICLR): a dissociated audio safety geometry — a harmfulness SENSOR vs a causal refusal
ACTUATOR.**

> An audio-native, causally-verified refusal *actuator* (`r_A`: ADD induces refusal, ABLATE removes it) is
> **distinct** from (a) the model's harmfulness *recognition* (`r_H`) and (b) the *text* refusal geometry.
> Successful attacks can **bypass/attenuate the actuator without erasing the harmfulness signal.**

This is **G1 (native causal audio refusal direction) + G4 (recognition vs natural-readout vs causal-control)**,
with **G3 (text↔audio geometry)** as the cross-modal explanation. The attack-induced flip (G2) is demoted
from an "audio is more vulnerable" claim to a **mechanistic stress test** of the actuator.

### Why this, and not the original "audio-specific flip"
The §8 data does **not** support audio being preferentially vulnerable (see `analysis.md`):
- **jb_pap** produces genuine audio refusal→comply flips (~30%, judge-robust κ≈0.87–0.90) but its
  audio×text interaction ≈ 0 and benign DiD ≈ 0 → currently a **general compliance boost**, not an
  audio- or harmful-specific jailbreak.
- **jb_ica** is 84–89% non-answers in audio → without ASR/intelligibility gates its negative interaction
  cannot separate robustness from input failure.
- So neither attack shows "audio attacks preferentially suppress the refusal axis." The matched-neutral
  null + ICA/PAP boundary are **supporting scope evidence, not the headline.**

The novelty must be the **factorization + cross-modal causal structure**, NOT "we found a direction in
Qwen2-Audio" (that would be dismissed as an audio port of Arditi).

## Positioning (verified prior work)
- Advances **Arditi (2406.11717)** — causal refusal direction, text-only, no sensor/actuator split — into audio.
- Supplies the causal counterpart to **HiddenDetect (2502.14744)** — internal safety signal survives a
  jailbroken output (detection-only) → our `r_H`-preserved-on-flips is the causal, audio version.
- Directly tests **SARSteer (2510.17633)**'s premise that audio-derived steering fails due to an audio/text
  activation gap → a native audio actuator + a text↔audio intervention matrix answers it causally.
- Gives JALMBench (2505.17568) its mechanistic "why"; qualifies the **Alignment Curse (2602.02557)**; adds
  causal audio evidence to **ReGap (2605.18104)**'s text-refusal+drift template.
- Must avoid colliding with occupied claims: "attack drifts off safety" (Acoustic Interference 2605.18168),
  "audio FT suppresses late refusal" (Benign-FT 2604.16659), "multimodal refusal + drift" (ReGap),
  defense-first (SARSteer/ReGap).

## Codex-ranked directions
1. **Dissociated audio safety geometry: harmfulness sensor vs causal refusal actuator.** (lead — above)
2. **Causal transfer matrix of text vs audio refusal geometry** (semantic alignment ≠ safety-controller
   alignment). Lives/dies on a **bidirectional ADD/ABLATE intervention matrix** (audio-native & text-native,
   applied to both modalities, dose-response + utility controls). Directly tests SARSteer's gap premise.
3. **Audio refusal is a conditional cone/routing system, not one axis** (jailbreak vs emotion recruit distinct
   causal channels — consistent with Run 3 emotion NOT `r_A`-mediated). Advances Concept Cones (2502.17420)
   into audio. Needs out-of-sample bidirectional interventions + representational-independence.

## Tonight's test (does the lead survive?) — decision rule
- `r_H` preserved on flips + **selective** `r_A` attenuation/rescue → **direction 1** (headline holds).
- `r_H` preserved + **generic** `r_A` shift → direction 1, but drop PAP from the central causal-attack claim.
- cross-modal causal/readout mismatch but no `r_H` result → **direction 2**.
- multiple stable interventionally-independent controllers → **direction 3**.
- none stable → do **not** force an ICLR narrative from G1 + behavioral nulls (needs multi-model replication).

## Top-3 ICLR rejection risks → preemptions (for the redesign)
1. **"Audio port of Arditi; RDO manufactured the vector (AUROC 0.60)."** → independent discovery/eval split;
   multiple RDO seeds; bidirectional ADD/ABLATE dose-response; norm-matched-random + unrelated-concept
   controls; utility/benign controls; natural attack-linked occupancy change; causal rescue on real flips;
   text↔audio cross-intervention matrix; **replication on ≥2 architecturally distinct open LALMs.** Novelty =
   factorization + cross-modal causal structure.
2. **"Artifact of 1 model, 150 shared FigStep items, 2 authored wrappers."** → multiple architectures;
   established + held-out attack families (e.g. from JALMBench); independent harmful datasets; multiple
   speakers/TTS + acoustic conditions; WER/intelligibility/duration/non-answer gates; matched text/audio
   semantics + benign controls.
3. **"Mediation is circular / measurement-dependent (double-dipping)."** → pre-register layers, token
   positions, metrics, thresholds; grouped cross-fit by item/family; discovery/eval separation for `r_A`,
   `r_H`, attacks; continuous effects + paired categorical outcomes; human audit of a stratified subset
   (incl. non-answers, borderline); multi-judge sensitivity + explicit consensus; **causal rescue/ablation as
   the PRINCIPAL evidence, probe AUROC as observational support only**; precise language ("causal controller",
   not "natural mediation" unless independently established).

**The paper's core intellectual contribution and its own safeguard: teach the distinction between
recognition (`r_H`), natural readout (AUROC), and causal control (`r_A`).**
