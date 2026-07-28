# Exp2 Yield Pilot — Amendment 1 (2026-07-28)

> **Status: binding.** This amends the sizing rule of
> [`pilot_protocol.md`](pilot_protocol.md) §6. It is frozen **before any attacked
> MiniCPM output exists**. `pilot_protocol.md` is not edited in place: a binding
> rule is never rewritten where it lives, so the original text stands and this
> document supersedes only what it names.
>
> Outcome thresholds are **unchanged**: `p_B,f ≥ 5%`, `p_L,f ≥ 4%`, `≥40%` of
> refusal→non-refusal transitions operational, and the B-count floor.

## 1. What changes

| | original §6 | amended |
|---|---|---|
| enrollment unit | 300 **items** | **600 clean-refusers** (`m`) |
| enrollment cap | — | **900 enrolled items** |
| enrollment order | all arms together | **clean arm first**, freeze cohort, then the six manipulated arms |
| CI method | unnamed | **exact one-sided 90% Clopper–Pearson** |
| smoke-test item | — | **excluded from analysis** |

## 2. Why — two measured facts, neither an outcome

### 2.1 `m` is clean-refusers, not items

The stop rule's denominator is items that clean-refuse. At an 80% clean-refusal
rate, 300 enrolled items yield `m≈240`, where the pilot's power at a true 6%
switcher rate is only **47%**. Enrolling until the denominator is reached fixes a
nuisance parameter, not an effect. It never observes a manipulated outcome, so it
cannot leak.

### 2.2 The GPU cost assumption was wrong by 12–24×

The plan budgeted 12–24 A40-hours for 2,100 generations. Measured on the real
stack (MiniCPM-o-2.6, transformers 4.46.3, bf16, A40): **1.80 s/generation** at
96 new tokens, 89 s load, peak **15.15 GiB of 45**. The pilot is **≈1 GPU-hour**.

`m=600` was previously rejected as "no longer a cheap yield pilot … 5,250
generations". At measured throughput that is **≈2.6 GPU-hours**.

| target | items @80% | generations | GPU-h | power @6% | power @7% |
|---|---:|---:|---:|---:|---:|
| m=300 | 375 | 2,625 | 1.31 | 63.0% | 84.6% |
| **m=600** | **750** | **5,250** | **2.63** | **82.7%** | **97.2%** |

≈1.3 extra GPU-hours buys **+20 pp power at 6%**.

### 2.3 What this does NOT fix

Power at a true rate of exactly 5% cannot be raised by any `m`. The criterion is
evaluated *at* the threshold, so `P(point estimate ≥ threshold) → 50%`. Verified:

| m | k required | binding rule | power @5% |
|---|---:|---|---:|
| 300 | 17 | Clopper–Pearson | 33.3% |
| 600 | 31 | Clopper–Pearson | 45.2% |
| 1000 | 50 | **point estimate** | 52.0% |
| 5000 | 250 | point estimate | 50.9% |

Above `m≈1000` the `≥5%` point rule binds and the CP rule goes slack, capping
power near 51%. **§6 is a ≥6–7%-family detector, not a 5% detector**, and must be
described that way. The B-count floor is retained: it is not redundant below
`m≈239` (at `m=200` CP needs 13 while the floor needs 15).

## 3. Frozen before the first attacked generation

- target `m = 600` substantive clean-refusers; hard cap **900** enrolled items
- clean arm generated first; **cohort frozen** before any manipulated arm
- if the cap is hit before `m=600`, stop and report failure to reach the
  denominator — do not relax the target
- every enrolled item retained; no row deletion
- exact one-sided 90% Clopper–Pearson throughout
- the smoke-test item is excluded from analysis
- `pv` sign assigned deterministically 1:1 by item hash, identical for
  `pv_locked` and `pv_standard` — both signs used, no best-of-two selection

## 4. Amendment rule applied

> A design amendment is permissible before target-model outcomes exist if it is
> justified solely by mathematics, external facts, or infrastructure; preserves
> the outcome thresholds; is timestamped with the previous rule retained; and is
> frozen before data capable of informing the treatment effect are observed.

All four hold: exp2 `design.md` does not exist yet; the pilot is explicitly not
the confirmatory pre-registration; **no MiniCPM generation on any manipulated arm
exists**; and the thresholds are untouched. Stored Qwen results may not be used
to select the MiniCPM arm, dose, or enrollment stopping point.

## 5. Provenance obligations (dependency-pin rule, amended)

The prior rule "pin the exact model-card dependency set" is replaced by:

> Pin the minimal tested-compatible dependency stack for the code paths actually
> executed, with exact versions and rationale documented.

MiniCPM-o-2.6's card declares transformers 4.44.2, which **cannot run it here**:
4.44.2's `get_imports()` filters only try/except blocks, so it statically scans
the `if is_flash_attn_2_available():`-guarded import inside the SigLIP *vision*
tower — a tower disabled via `init_vision=False` — and demands `flash_attn`.
The guard filter arrives in 4.46; `WHISPER_ATTENTION_CLASSES` is removed in 4.48.
The runnable intersection is **4.46–4.47**; we pin **4.46.3**, already used by
`scripts/almguard/setup_almguard_env.sh`. See `scripts/exp2/setup_minicpm_env.sh`.

Recorded with every run: transformers/torch/CUDA versions, model + remote-code
revision, `init_vision=False`, absent `vpm`/`tts`, `attn_implementation`, device map.

## 6. Model-role correction

Verified module tree: MiniCPM-o-2.6 = **`Qwen2ForCausalLM`** (28 layers, hidden
3584) + `MiniCPMWhisperEncoder` + `MultiModalProjector`. Qwen2-Audio-7B is Qwen2-7B
(32 layers, hidden 4096). **They share the LLM family.**

`direction.md` §5's framing of MiniCPM as an *independent* primary against a Qwen
*resistant control* therefore overstates independence. Corrected roles:

| pair | what it is |
|---|---|
| MiniCPM vs Qwen2-Audio | **within-Qwen-family front-end / interface contrast** — holds the LLM family roughly fixed and isolates the encoder+projector contribution, which is exactly the claimed transport site |
| Kimi-Audio | **independent-backbone replication** (re-fetch only after MiniCPM passes the stop rule; 52 s) |
| Ultravox v0.4.1 | blocked — the 118 MB repo is an adapter requiring gated `meta-llama/Meta-Llama-3.1-8B-Instruct` |

Any result shared by MiniCPM and Qwen2-Audio alone may be a Qwen-family property
and may not be reported as cross-architecture replication.

## 7. Closed by this session's work

- **`pilot_protocol.md` §8 open item 2 — resolved.** `hooks.py`'s guessed exp2
  module paths (`llm.model.layers`, `apm`, `audio_projection_layer`) are all
  correct against real weights. No CPU offload.
- **Audio demonstrably consumed:** real / silence / mismatched wavs produce
  different, content-specific outputs. Greedy decoding deterministic.
- **§7a.2 benign-trained operator fitting — closed as a negative.** See
  [`../../discussions/2026-07-28_exp2-direction-codex-cross-check.md`](../../discussions/2026-07-28_exp2-direction-codex-cross-check.md).
  Benign→harmful transfer null (R²_disp −0.006, 5/10 folds); harmful→harmful
  reduced entirely to one scalar (diagonal +0.1208 vs scalar +0.1218); that
  scalar is **regression dilution** (unpenalized OLS 0.768 vs cross-sign IV
  **0.996**); item identity intact (retrieval top-1 0.92–1.00). The next
  mechanistic evidence must come from **encoder output, projector output, and
  audio-token spans at L8/L10/L12** — not from another L18 basis.
