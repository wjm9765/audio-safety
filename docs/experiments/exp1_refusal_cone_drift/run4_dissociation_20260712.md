# Run 4 §8 — sensor/actuator dissociation test (2026-07-12, direction-finding)

> Tonight's test of the `run4_direction_20260712.md` hypothesis: on genuine PAP audio refusal→comply flips,
> is the harmfulness SENSOR (`r_H`) preserved while the causal refusal ACTUATOR (`r_A`) occupancy is
> attenuated? **Direction-finding only** (1 model, 150 shared FigStep items, 2 authored wrappers, judge
> substitution). Final positioning appended after the Codex grade. NOT paper-facing.

## Method (reused infra)
- **Frozen `r_A` actuator:** `exp1_20260707_1557_allpos_rebuttal_l12nbhd/rdo_axis.npz`, layer 16 /
  `first_generation_prelogit`. Causally validated (add RR +20.7pp, ablation ASR +35.6pp, benign ORR +0.05pp)
  but weak as a natural readout (escape AUROC 0.484). This is the CONTROLLER, not a natural probe.
- **Activations:** `extract_conversion_activations.py` on the §8 attack-flip manifests, per style
  (`run4_probe_{neutral,jbpap,jbica}`): P2 (decision, layer 16) hidden for `r_A` occupancy c_R = ⟨h, r̂_A⟩,
  and P1 (content, `assistant_start_pre`) at layers [8,12,16] for a cross-fit `r_H`. AUDIO arm used here.
  - Bugfix applied to `pipelines/conversion_probe.py`: wrapped the capture forward in `torch.no_grad()`
    (+periodic `empty_cache`) — the un-graded forward ~doubled peak memory and OOM'd on longer jb_* audio.
    Captured values are identical; only memory changes.
- **Flip labels:** blinded judged manifest (gemini-2.5-flash + claude-haiku-4.5 + consensus). Flip = neutral
  genuine refusal → attacked comply; remained = refusal→refusal. Δc_R = c_R(attack) − c_R(neutral), in SD of
  clean-harmful c_R. Double-diff D = mean(Δ|flip) − mean(Δ|remained). `analyze_flip_dissociation.py`
  (research-code-reviewer checked; one consensus-set bug fixed before running).

## Results

### r_A occupancy (actuator) — sign negative = less refusal-aligned
| style | judge | n_flip | n_rem | Δ_flip (SD) | Δ_rem (SD) | **DoubleDiff (SD)** | harm−benign (SD) |
|---|---|--:|--:|--:|--:|--:|--:|
| jb_pap | gemini-2.5-flash | 28 | 62 | −1.37 | −1.00 | **−0.37 (−0.73,−0.04)** | −0.34 (−0.63,−0.03) |
| jb_pap | claude-haiku-4.5 | 23 | 46 | −1.65 | −0.89 | **−0.76 (−1.11,−0.41)** | −0.61 (−0.89,−0.34) |
| jb_pap | consensus | 19 | 46 | −1.68 | −0.89 | **−0.78 (−1.19,−0.39)** | −0.64 (−0.97,−0.31) |
| jb_ica | consensus | 7 | 1 | −3.34 | +0.37 | −3.72 (degenerate, n_rem=1) | −0.64 |

Generic benign r_A shift: jb_pap −1.04 SD; **jb_ica −2.70 SD** (much larger, whole-input shift).

### r_H (sensor) — cross-fit DIM(harmful−benign) trained on CLEAN neutral, AUROC clean→attacked (audio)
| style | L8 | L12 | L16 |
|---|--:|--:|--:|
| jb_pap | 0.98→0.80 | 0.98→0.77 | 0.99→**0.81** |
| jb_ica | 0.98→0.73 | 0.98→0.71 | 0.99→**0.44** |

## Findings (honest)
1. **jb_pap = actuator bypass.** Judge-robust SELECTIVE attenuation of `r_A` occupancy on genuine flips
   beyond remained-refusals (DoubleDiff −0.37…−0.78 SD, CI excludes 0 for gemini/claude/consensus) and
   beyond benign (harm−benign −0.34…−0.64 SD, CI excludes 0), WHILE `r_H` stays strongly decodable (0.81 @
   L16). The model still recognizes harmfulness but the refusal actuator is suppressed on flips.
2. **jb_ica = perception/sensor degradation.** `r_H` COLLAPSES at the decision layer (0.99→**0.44**, below
   chance), 84–89% non-answers, almost no genuine flips (n_rem=1 → r_A double-diff uninterpretable). A whole-
   input shift (benign r_A −2.70 SD), not a selective actuator bypass.
3. **The sensor/actuator frame separates the two mechanisms** (r_H at L16: 0.81 vs 0.44). The contrast also
   argues jb_pap's modest r_H drop (0.98→0.80) is NOT mere train/test distribution shift (else jb_ica would
   not collapse to 0.44 while jb_pap holds 0.81).

## Caveats (must foreground)
- **Selective-component magnitude is judge-dependent** (gemini DoubleDiff −0.37, lower CI −0.04 — barely
  significant; claude/consensus stronger). Report the range, not the strongest number.
- **Large generic r_A shift** (jb_pap benign −1.04 SD): PAP broadly lowers r_A occupancy; the harmful-flip-
  specific EXTRA (−0.34…−0.64 SD) rides on top of it. jb_pap is not a purely harmful-specific actuator attack.
- **`r_H` drops 0.98→0.80 under jb_pap** — largely preserved, not perfectly; partly (not wholly) distribution shift.
- **Small n_flip** (19–28 jb_pap; 7–9 jb_ica). Single model, 150 shared FigStep items, authored ICA/PAP-style
  wrappers, forced judge substitution. Occupancy/probe are OBSERVATIONAL; causal rescue (add r_A on flips) is
  the missing causal leg.

## Positioning / next step (after Codex gpt-5.6-sol xhigh grade — B/7 for direction-finding)

**Honest present conclusion (Codex-endorsed):** *"jb_pap and jb_ica produce sharply different
representational signatures consistent with two candidate failure modes; they do NOT yet prove two
mechanisms."* The sensor/actuator decomposition is an excellent central **hypothesis** and redesign
principle, NOT yet a proven factual headline.

**Reviewer-safe wording (use this, don't overclaim):**
> In Qwen2-Audio, a clean-trained harmfulness direction retained substantial discrimination under PAP
> audio (AUROC 0.77–0.81), while a previously validated refusal direction shifted broadly and showed an
> additional 0.37–0.78 SD attenuation associated with genuine jailbreak flips. These results are consistent
> with partial preservation of harmfulness information alongside selective refusal-pathway attenuation, but
> do not by themselves establish sensor integrity or causal bypass.

**Corrections applied to Claude's read:**
- Not "sensor intact" → "harmfulness information substantially retained" (AUROC 0.80 ≠ intact).
- Not "two mechanisms" → "two candidate failure modes / sharply different signatures."
- ICA L16 AUROC 0.44 = the clean decoder fails/reverses there, NOT proof perception collapsed.
- PAP: selective component is real but an **incremental association**, generic component is ≥ as large;
  judges are not independent replications (shared items/labels). The **causal line is the rescue**.

**FOREGROUND caveat:** `r_H` AUROC is transfer performance of a *linear decoder at one activation site*,
not a direct measurement of a persistent harmfulness sensor. PAP's decline and ICA's below-chance transfer
could reflect representation remapping / generation-state mismatch, not sensor erosion.

**Causal rescue (Codex: the decisive causal leg) — RESULT: NEGATIVE.** Add frozen `r_A` (α preselected on
the RDO gate, all-positions, layer 16; not tuned on these flips) on the 19 consensus PAP flips; controls =
norm-matched random direction + attacked-benign at same α. Refusal via `label_output` (same metric that
validated the axis's add_rr).

| α | flip baseline | flip +r_A | flip +random | benign +r_A (over-refusal) |
|---|--:|--:|--:|--:|
| 2.0 | 0.05 (1/19) | 0.16 (3/19) | 0.11 (2/19) | 0.03 (1/30) |
| 4.0 | 0.05 (1/19) | 0.16 (3/19) | 0.11 (2/19) | 0.07 (2/30) |
| 8.0 | 0.00 (0/19) | **0.37 (7/19)** | 0.21 (4/19) | 0.17 (5/30) |

At the validated strength (α=2) and α=4, adding `r_A` does NOT rescue flips beyond a norm-matched random
direction (3/19 ≈ 2/19). Only at α=8 does `r_A` modestly exceed random (37% vs 21%), but most flips (12/19)
still are NOT rescued, the gap is 3 items at n=19 (underpowered), and benign over-refusal rises to 17%
(indiscriminate steering). **Per Codex's pre-specified decision rule this is the "generic state shift with an
outcome-correlated `r_A` component" fork: the `r_A` occupancy attenuation on flips is an ASSOCIATION, not a
demonstrated causal lever.** → PAP is NOT a central causal actuator-bypass example. The sensor/actuator
dissociation stays a promising HYPOTHESIS (r_H retained + r_A occupancy selectively attenuated on flips), but
its causal leg fails for these authored PAP wrappers at tested strengths. Caveat: the frozen `r_A` was trained
on CLEAN gate items (escape AUROC 0.484) — it may not be the actuator this attack routes through; the paper-
facing redesign must make a demonstrated causal rescue a GATING criterion, test an attacked-regime-derived
actuator, and use far larger n.

**Cheap follow-ups for the rigorous redesign (Codex):** cross-fit `r_H` WITHIN attacked data + train-
attacked/test-clean (disentangle remapping from erosion); report within-attacked effect size; remove the
benign-estimated common attack shift before comparing directions. For the two-mechanism *diagnostic* to be
publishable: multiple independently-sourced attack families per mechanism class with pre-fixed labels;
held-out attack-family prediction; multi-model replication; direct perception/comprehension controls.
