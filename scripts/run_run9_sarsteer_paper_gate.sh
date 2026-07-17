#!/usr/bin/env bash
# Run 9 SARSteer paper-faithful gate — held-out defended generation + gate report.
# Assumes: paper vectors built, alpha frozen, held-out undefended baseline cached.
# Usage: run_run9_sarsteer_paper_gate.sh <ALPHA>
set -euo pipefail
ALPHA="${1:?usage: $0 <frozen-alpha>}"
DM=/workspace/audio_safety_data/data/manifests
OUT=/workspace/audio_safety_data/outputs
CAL=$OUT/exp1_20260717_run9_sarsteer_paper_calib59
HELD=$OUT/exp1_20260717_run9_sarsteer_paper_heldout
cd /workspace/audio-safety

./scripts/apply_sarsteer_defense.py \
  --config configs/experiments/run9_sarsteer_paper.yaml \
  --override "sarsteer.alpha=${ALPHA}" \
  --run-name exp1_20260717_run9_sarsteer_paper_heldout \
  --manifest "$DM/run9_sarsteer_paper_heldout.jsonl" \
  --vectors "$CAL/sarsteer_vectors.npz" \
  --undefended-cache "$HELD/heldout_undefended.jsonl" \
  --output "$HELD/heldout_paired_a${ALPHA}.jsonl"
