# Exp4 — post-prefill audio/KV routing: pre-registered design

> **Status:** pre-registered and frozen before the first non-smoke GPU run,
> 2026-08-01. Thresholds and primary estimands below must not be edited after a
> non-smoke run begins; any necessary change is a dated amendment.
>
> **Scope:** mechanistic follow-up to Exp3. The endpoint is the frozen literal
> explicit-refusal marker, not harmful compliance or jailbreak success. Human
> verification of audio-content equivalence remains a separate future audit.

## 0. Frozen decision table

All intervals are 95% pair-cluster bootstrap intervals over the **discordant
Exp3 cohort**. The two acoustic directions are averaged within pair before pairs
are resampled. No generated row is excluded because of answer quality.

| quantity | material conditional contributor | negligible at pilot resolution | otherwise |
|---|---|---|---|
| `D_audio = T_II - T_HI` | estimate ≥ 0.20 and CI low > 0 | CI high < 0.10 | ambiguous |
| `D_tAB = T_II - T_IH` | estimate ≥ 0.20 and CI low > 0 | CI high < 0.10 | ambiguous |
| `D_joint = T_II - T_HH` | estimate ≥ 0.20 and CI low > 0 | CI high < 0.10 | ambiguous |

An interval wholly below zero is reported as a reverse effect. These contrasts
are **conditional causal capacities**, not mediation percentages and not shares
that must add to one. The factorial interaction is estimated but has no binary
threshold.

Notation: the first letter is the source of the audio-token cache and the second
is the source of the final prompt position `t_AB`; `I` = injected trajectory and
`H` = physical host trajectory.

## 1. Question left open by Exp3

Exp3 established the following pilot facts for Qwen2-Audio-7B-Instruct:

1. Donor audio-span replacement after decoder block L10 moves the generated
   explicit-refusal marker donorward (`0.781` on discordant cells).
2. Resetting the audio span at L12 removes most of that continuous effect.
3. Closing all 15 post-audio prompt relay positions from L11 through L31 leaves
   donorward transport `0.575`, or retention `0.737 [0.645, 0.836]`.
4. A one-time `t_AB` clamp at L18 also leaves substantial behaviour.

The relay-closure result says that a large route avoids those 15 prompt relay
positions. It does **not** identify that route. During autoregressive decoding,
every generated token can still attend directly to the many audio-token K/V
entries, to `t_AB`, and to earlier generated tokens.

Exp4 asks the narrow next question:

> After the injected prefill has selected the same first generated token `y1`,
> does the remaining refusal trajectory depend on direct rereading of the audio
> K/V reservoir, broadcasting through the cached `t_AB` state, or only on `y1`
> and subsequent generated tokens?

## 2. Causal object and intervention time

The prompt geometry observed in Exp3 is:

```text
P (20 pre-audio)   A (many audio tokens)   R (15 relay tokens)   t_AB
       |                    |                       |                |
       +--------------------+-----------------------+----------------+
                            prefill KV cache
                                  |
                     logits choose one fixed y1
                                  |
             +--------------------+--------------------+
             |                                         |
      swap cached A K/V                         swap cached t_AB K/V
             |                                         |
             +--------------------+--------------------+
                                  |
                         generate y2, y3, ...
```

The cache surgery happens **after** prefill logits have been computed and `y1`
has been chosen, but **before** `y1` is fed back into the model to predict `y2`.
The exact same `y1` token ID is supplied to all four cells. Consequently:

- Exp4 cannot attribute the selection of `y1`; it conditions on that selection.
- Differences from `y2` onward are caused by the prefill-cache intervention.
- Survival in the all-host-cache cell measures an autoregressive `y1` lock-in
  capacity under this intervention, not an unidentified cache route.

## 3. Construction of the injected trajectory

For every frozen Exp3 pair and each direction:

1. The destination arm is the **host**; the opposite arm is the **donor**.
2. Run both physical prefills and capture the donor/host audio span after L10.
3. On the host prompt, replace the complete L10 audio span with the donor span.
4. Clamp all 15 relay positions `R` to their physical-host residual states after
   every block L11–L31.
5. Run this composed prefill with `use_cache=True`. Its last logits select `y1`.

The L10 injection is the Exp3 positive intervention. Relay closure removes the
known prompt-text route and makes the remaining cache difference interpretable.
For a causal decoder, the physical and injected prefill caches can then differ
only at `A` and `t_AB`: pre-audio positions cannot attend forward to audio, and
each relay output is restored before it becomes the next layer's K/V.

## 4. Frozen 2×2 cache surgery

Each cell starts from an independent clone of the injected prefill cache. K and
V are transplanted together at the selected absolute positions in **every
decoder layer**.

| condition | audio-token K/V | `t_AB` K/V | fixed `y1` |
|---|---|---|---|
| `audio_injected__tab_injected` (`II`) | injected | injected | injected |
| `audio_host__tab_injected` (`HI`) | host | injected | injected |
| `audio_injected__tab_host` (`IH`) | injected | host | injected |
| `audio_host__tab_host` (`HH`) | host | host | injected |

Greedy decoding then proceeds manually from `y1` through at most 96 total new
tokens. The generated sequence stored and scored includes `y1`.

## 5. Integrity gates (fail closed)

A non-smoke run is invalid if any gate fails.

1. Source Exp3 config snapshot, cohort, physical behaviour, span-patch records,
   relay-closure records, and provenance are hashed before GPU inference. A
   changed source artifact requires a new run name.
2. The model revision, prompt, matcher, cohort, L10 site, and directions are
   inherited from Exp3; the measured relay count must equal 15 for every pair.
3. Audio positions must be a non-empty contiguous interval and both arms must be
   token/feature aligned under the existing Exp3 check.
4. Every residual intervention hook must fire exactly once during prefill.
5. Cloning the injected cache must be exact.
6. After both `A` and `t_AB` are transplanted from the physical host, the **entire
   mixed cache must equal the physical host cache** (configured tolerance 0).
   This is the crucial proof that no unmanipulated prompt position still carries
   injected information.
7. For two pre-registered cells, manual decoding of `II` must reproduce ordinary
   greedy `model.generate` under the same residual hooks exactly.
8. Whenever injected and physical-host prefills choose the same `y1`, `HH` must
   reproduce the frozen physical-host response exactly.
9. The output checkpoint must contain exactly one row for every
   pair × direction × condition cell.

## 6. Endpoints and estimands

For a discordant direction, let `H` and `D` be the host and donor binary refusal
markers and let `Y_c` be the marker generated in cache condition `c`:

```text
donorward(c) = (Y_c - H) * (D - H)
```

Because `H != D`, this is 1 when the cell matches the donor marker and 0 when it
matches the host marker. Average the two directions within each pair:

```text
T_II, T_HI, T_IH, T_HH
D_audio = T_II - T_HI
D_tAB   = T_II - T_IH
D_joint = T_II - T_HH
I       = T_II - T_HI - T_IH + T_HH
```

The complementary simple effects `T_IH - T_HH` (audio when `t_AB` is host) and
`T_HI - T_HH` (`t_AB` when audio is host) are also reported descriptively. They
help explain redundancy, but only the three contrasts in §0 receive frozen
binary decisions.

Interpretation:

- large `D_audio`: generated tokens causally reread the audio-token cache;
- large `D_tAB`: cached `t_AB` causally broadcasts injected information;
- neither single swap large but `D_joint` large: redundant/conditional parallel
  capacity, captured by the interaction rather than a unique path;
- `T_HH` remains donorward: fixed `y1` and its autoregressive descendants retain
  the effect after complete prompt-cache closure.

`T_HH` is additionally stratified by whether injected prefill changed `y1`
relative to the physical host. Stable Exp3 pairs are reported descriptively but
do not enter the primary donorward estimands because `H == D` gives no direction.

## 7. What this experiment cannot establish

- It does not prove that every audio token contains refusal information; K/V is
  replaced as a whole span. A token/head dose-response experiment would be next.
- It does not identify the direct prefill edge `A -> t_AB`; target-specific
  attention-edge masking is required for that.
- It does not decompose unique mediation shares. The two cache reservoirs may be
  redundant, synergistic, or downstream copies of a common signal.
- It does not test safety specificity, perceptual content invariance, other
  models, or other acoustic transformations.
- It conditions on `y1`, so its claim begins at prediction of `y2`.

## 8. Artifacts and commands

Run names use `exp4_{YYYYMMDD}_{HHMM}_{tag}`. Each run writes:

```text
outputs/<run>/
  config_snapshot.yaml
  provenance.json
  inputs/source_manifest.json
  inputs/cohort.jsonl
  cache_routing/records.jsonl
  metrics.json
  analysis.md
  errors.jsonl                 # only if a stage fails
```

CPU preflight:

```bash
./scripts/exp4/run_audio_kv_routing.py \
  --run-name exp4_20260801_1200_audio_kv \
  --stage preflight
```

Small end-to-end GPU smoke (new run name; not part of the registered estimate):

```bash
./scripts/exp4/run_audio_kv_routing.py \
  --run-name exp4_20260801_1210_smoke \
  --stage all \
  --override exp4.max_pairs=2 \
  --override exp4.n_bootstrap=500 \
  --override exp4.max_new_tokens=24
```

Official run:

```bash
./scripts/exp4/run_audio_kv_routing.py \
  --run-name exp4_20260801_1300_audio_kv \
  --stage all
```

## 9. Amendment log

No amendments.
