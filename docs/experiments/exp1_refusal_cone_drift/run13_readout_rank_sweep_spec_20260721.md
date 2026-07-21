# Run 13 — Readout multidimensional (SVD) rank-sweep — experiment spec (2026-07-21)

**Status:** exploratory screen. This document specifies WHAT is run and WHAT is reported.
It intentionally contains **no GO/NO-GO / escalate / stop thresholds** — the numbers below are
reported to the human, who makes the call.

**Run identity:** `run13`, its own run. Output dir `outputs/run13_readout_rank_sweep/`.
It **reuses Run 12's captured L18 states as INPUT** (`--source-run outputs/run12_factorial`,
i.e. `outputs/run12_factorial/capture/`), exactly as Run 11 consumed Run 10's manifests. Run 12 is
untouched. No new activation capture unless `outputs/run12_factorial/capture/` is unavailable
(then re-capture with `capture_run12.py` into the run13 dir).

## 1. Question

Run 12 tested a single **rank-1** harmful-specific L18 readout coordinate `u_s`: it moved the
first-token refusal margin but did not move full-generation behavior (+0.33pp). Every
behaviorally-decisive test so far was effectively rank-1 (Run 10/11 readout = rank-1 DiM; Run 12
`u_s` = rank-1 Mahalanobis coordinate). The rank-1→full-state gap (behavioral rescue +0.33pp vs
+9.67pp) is where a **multidimensional low-rank subspace** could live.

Run 13 generalizes exactly that rank-1 coordinate to a **rank-k subspace** (SVD) and sweeps `k`,
asking: *does a low-rank readout subspace control full-generation refusal more than rank-matched
random or generic-channel directions?* The rank-1 point of the sweep reproduces Run 12's `u_s`.

## 2. Endpoint policy (decision from the human, per session)

- **Primary endpoint: full-generation policy-refusal RATE** (well-powered: attack moves it ~13–31pp).
- **Secondary/diagnostic: first-token refusal margin `M`** (dose curves; operator sanity). A
  margin-only movement is *not* treated as a behavioral result.
- The operational-harm / decoding-failure power problem (only ~2 strongly-operational jailbreaks in
  Run 10's 246) is **deliberately deferred** for this screen. The blind 4-way auto-label is kept as
  a cheap descriptive side-channel only (to see where rescued outputs land), and is explicitly
  **non-authoritative** for `harmful_compliance` claims.

## 3. Data reused (from Run 12, no new capture)

From `outputs/run12_factorial/`:
- `capture/states_*.npz` — L18 `t_AB` (=`first_generation_prelogit`) post-block residual states,
  keyed `{role}|{cond}|{item}`, role∈{harmful,benign}, cond∈{clean,pv_standard_m3,pv_standard_p3}.
- `capture/meta_*.jsonl` — per-row `M`, `H_harm`, category.
- `folds.json` — item→outer-fold (5 category-stratified folds, seed 0). Cross-fitting reuses these.
- `cohort.jsonl` — item→wav path per (role,cond); externally-selected, no outcome filtering.

150 FigStep items (harmful + matched benign), pv_standard ±3 signs (m3/p3).

## 4. Fitting (CPU, cross-fitted, outcome-blind)

Per outer fold `f`, per sign `s`, using FIT items only (folds[item]≠f):

1. **Whitening** `W_f = Σ_f^{-1/2}` from Ledoit–Wolf pooled within-class covariance of clean
   harmful+benign states (same as Run 12; `fit_run12_axis._wsqrt_inv`).
2. **Harmfulness nuisance subspace** `R_H` (orthonormal rows, whitened): mean-anchored SVD of the
   per-item whitened clean harmful−benign contrasts `g_i = W(cH_i − cB_i)`. **Default `k_H = 1`**
   (row 0 = `r_H = unit(W(μ_H−μ_B))`, so the rank-1 `U_1` reproduces Run 12's `u_s` exactly — unit-
   tested). A multi-dimensional `R_H` (`--harmfulness-rank-max > 1`, in-sample reconstruction dim
   selection) is an **opt-in** stronger specificity control. `P_H^⊥ = I − R_Hᵀ R_H`.
3. **Harmful-specific benign-subtracted interaction** per fit item:
   `z_i = P_H^⊥ [ W(cH_i − aH_i) − W(cB_i − aB_i) ]`  (attacked-benign subtracts the generic phase channel).
4. **Channel subspace per rank** `U_k` = `channel_axis.mean_anchored_basis(Z, k)` for
   `k ∈ {1,2,4,8,12,16,20,32,64}` (row 0 = DiM = Run 12's `u_s` direction; rows 1..k−1 = residual
   SVD). Orthonormal in the whitened metric.
5. **Generic-channel subspace per rank** `B_k` = mean-anchored SVD of the benign-only whitened
   displacements `P_H^⊥ W(cB_i − aB_i)`. Same rank. Used to separate safety-specific repair from
   generic audio/decision repair.
6. **Sham bank** per rank: 20 Haar-random rank-k subspaces in the whitened complement of `U_k`
   and `R_H`. Behavioral generation uses one frozen balanced sham assignment per (item,rank) drawn
   from the 20; margins use all 20.

Saved (unfrozen): `subspaces/subspaces.npz` (`W` [float32], `rH,R_H,U_k,B_k` [float64] per
fold/sign; `Winv` is NOT stored — derived via `solve(W,·)` for a self-consistent whitening
round-trip; shams are regenerated deterministically in the edits step), `subspaces/fit_manifest.jsonl`,
`subspaces/geometry.json` (**genuine held-out** reconstruction on the held-out fold, cross-fold
principal angle, and `Z` singular spectrum per rank — reported, not gated).

## 5. Edits (CPU, cross-fitted)

For a held-out test item (uses the fold that excluded it), sign, rank `k`, the projected-transport
edit at `t_AB` is a fixed additive vector:

```
# U_k has orthonormal ROWS (k,d) in the whitened metric; W⁻¹ applied via solve(W, ·)
edit(donor,host,λ) = λ · W⁻¹ Uᵀ ( U W (donor_read − host_read) )
```

Arms per (item, sign, rank):

| arm | host | donor | subspace | purpose |
|---|---|---|---|---|
| `restore`  | attack_H | clean_H  | `U_k` | does the subspace recover refusal? |
| `corrupt`  | clean_H  | attack_H | `U_k` | reciprocal (behavioral mediation) |
| `sham`     | attack_H | clean_H  | random `S_k` | rank+norm-matched null |
| `generic`  | attack_H | clean_H  | `B_k` | generic-channel repair control |
| `brestore` | attack_B | clean_B  | `U_k` | benign over-refusal control |
| `fullstate`| attack_H | clean_H (full state) | — | behavioral ceiling (Run 12 reuse) |

Sham and generic edits are rescaled to the per-item raw-L2 norm of the true `restore` edit (a
degenerate near-zero projection becomes an explicit no-op, counted). Doses `λ ∈ {0,0.25,0.5,1}` for
margins; generations at `λ=1`.
**Enforced (raise before saving):** cross-fit exclusion; `U_k/B_k` orthonormal; `U_k ⟂ R_H`; the
**realized** restore edit is harmfulness-orthogonal (`|R_H · (W·edit)| ≈ 0`); shams orthonormal &
`⟂ (U_k,R_H)` & norm-matched. Wrong-item donor is deterministic (single seeded RNG over sorted
items) and never the same item. No behavioral outcome enters fitting.

Saved: `edits/edits.npz`, `edits/edits_manifest.jsonl`.

## 6. Behavioral run (GPU, L18/t_AB, prefill-only, greedy)

Reuses `ResidualStreamIntervention` (add) and `patch_state` exactly as Run 12. Identity check
(edit=0 reproduces the unhooked forward) validates the operator per row.

- **Margins** per (item,sign,rank): restore(×4 doses), corrupt(×4), brestore(@1), 20 shams(@1),
  generic(@1), fullstate(@1).
- **Generations** (`max_new_tokens=96`, greedy) per (item,sign,rank) — the 6 arms:
  `attack_H {identity, restore@1, sham@1, generic@1, fullstate}`, `attack_B {identity, brestore@1}`,
  `clean_H {identity, corrupt@1}`.
- **Harmfulness probe** `H_harm` under identity and restore@1 (recognition preserved?).

Reuse Run 12 identity/fullstate generations where model/config/decoding hashes match. Sharded.
Saved: `phaseB/margins_*.jsonl`, `phaseB/gens_*.jsonl`.

## 7. Labeling & analysis (CPU)

- `labeling.label_output` → blind 4-way `{policy_refusal, harmful_compliance, benign_answer,
  decoding_failure}` for every generation; build per-arm/rank **transition matrices**
  (identity→intervened). Marked non-authoritative for harmful_compliance.
- Item-clustered bootstrap (10k), averaging the two signs within item.

## 8. What is REPORTED to the human (tables only — no verdict)

For each rank `k` (and both signs, and pooled):

1. **Refusal-rate table:** base attack refusal, and Δrefusal for {restore, sham, generic,
   brestore(benign ORR), fullstate}, with item-clustered 95% CIs. Also `restore − sham` and
   `restore − generic` with CIs.
2. **Margin table:** ΔM dose curves for restore/corrupt, `restore − sham`, corrupt sign.
3. **Reciprocal corruption:** Δrefusal on clean_H under corrupt vs sham.
4. **Transition matrices:** where restored/corrupted items move in the 4-way label (esp. how much
   of any refusal gain lands in `policy_refusal` vs `decoding_failure`).
5. **Recognition:** `H_harm` under identity vs restore@1 (paired).
6. **Geometry:** held-out reconstruction, cross-fold principal angle, and permutation-null p per
   rank; singular spectrum of `Z`.
7. **Rank-1 anchor:** confirm the rank-1 restore row reproduces Run 12's `u_s` result.

Output: `analysis/screen_report.md`, `analysis/screen_report.json`, `analysis/transition_tables.json`.
No `decision.json`; the human reads the tables and decides next steps.

## 9. Pipeline & commands

```bash
RUN=outputs/run13_readout_rank_sweep
SRC=outputs/run12_factorial
RANKS="1 2 4 8 12 16 20 32 64"

# 1. fit cross-fitted subspaces (CPU)  [default k_H=1; add --harmfulness-rank-max 8 for multi-dim R_H sensitivity]
./scripts/fit_run13_subspaces.py --source-run $SRC --run-dir $RUN --ranks $RANKS --n-sham 20 --seed 0

# 2. precompute edits (CPU)
./scripts/precompute_run13_edits.py --source-run $SRC --run-dir $RUN --ranks $RANKS \
  --doses 0 0.25 0.5 1 --n-sham 20 --seed 0

# 3. behavioral run (GPU, sharded)
./scripts/run_run13_sweep.py --run-dir $RUN --source-run $SRC --ranks $RANKS \
  --doses 0 0.25 0.5 1 --max-new-tokens 96 --shard 0 --num-shards 2   # + shard 1

# 4. label + analyze (CPU)
./scripts/analyze_run13_screen.py --run-dir $RUN --n-bootstrap 10000 --seed 0
```

## 10. Frozen parameters

model/gate/instruction from `configs/experiments/run5_pitch_representation_fast.yaml` (same as Run
12); layer 18; token `first_generation_prelogit`; 5 folds seed 0; ranks
{1,2,4,8,12,16,20,32,64}; doses {0,0.25,0.5,1}; 20 shams seed 0; greedy; max_new_tokens 96. All
recorded in `sweep_manifest.json` at fit time (+ git commit).

## 11. Interpretation ceiling (for the human, not a gate)

A positive screen (a stable low-rank `U_k` beats sham AND generic on refusal RATE) supports:
"a cross-fitted low-dimensional L18 readout subspace controls full-generation refusal more than
random or generic audio-channel directions." It does **not** by itself establish: reduced
operational harmful compliance, a safety-specific (vs generic decision/refusal-repair) mechanism,
a deployable defense (still a clean-donor instrument), a stronger attack, or generalization to
other attacks/models. Name a positive result a *multidimensional refusal-control screen*, not a
safety-mechanism confirmation.

## 12. Implementation status & GPU handoff (2026-07-21)

**Code — DONE, upload-and-run ready.** Four executable scripts in `scripts/`:
`fit_run13_subspaces.py`, `precompute_run13_edits.py` (CPU), `run_run13_sweep.py` (GPU),
`analyze_run13_screen.py` (CPU). Reuses `channel_axis.mean_anchored_basis` (multidim SVD) and
`labeling.label_output` (4-way). Blind Codex `gpt-5.6-sol` xhigh review applied (4 high + 3 medium
correctness fixes: rank-1↔Run-12 anchor, genuine held-out geometry, deterministic wrong-item,
self-consistent `solve(W,·)` whitening, enforced realized-edit orthogonality, fold-seeded shams;
vacuous permutation-null removed). CPU numerical core unit-tested: `tests/test_run13_rank_sweep.py`
(6 tests — rank-1=`u_s`, `U_k⟂R_H`, realized-edit orthogonality, sham norm-match, no self wrong-item,
determinism). Full suite green (378 passed).

**Prerequisite on the GPU box:** Run 12 outputs present at `outputs/run12_factorial/`
(`capture/states_*.npz`, `capture/meta_*.jsonl`, `folds.json`, `cohort.jsonl`). These are gitignored
— they must already exist on the compute machine (they do; Run 12 ran there). Env per AGENTS.md
(`HF_HUB_CACHE`, offline flags). `uv sync --group gpu`.

**Run order** (Stages 1–2 are CPU and can run anywhere; 3 is GPU; 4 is CPU):
see §9. Fit → edits produce `subspaces/` and `edits/` (edits.npz is large, ~1 GB with 20 shams ×
9 ranks; pass `--n-sham 5` to shrink for a first pass). Stage 3 is the GPU cost: greedy generations
across 6 arms × 9 ranks (reuses rank-independent identity/fullstate once); shard over
`--num-shards`. Stage 4 writes `analysis/screen_report.md` + `.json` + `transition_tables.json`.

**Deliverable to the human:** the tables in §8 (no verdict). GPU-cost knobs: `--ranks` (drop 32/64
to skip the rank-inflation diagnostics), `--n-sham`, `--num-shards`, `--limit` (smoke test on a few
items first).
