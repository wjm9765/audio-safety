#!/usr/bin/env bash
# Run 9 SARSteer paper-faithful gate — merge held-out labels + compute the verdict.
# Assumes all 16 judge/labels_*.json exist.
set -euo pipefail
cd /workspace/audio-safety
HELD=/workspace/audio_safety_data/outputs/exp1_20260717_run9_sarsteer_paper_heldout

./scripts/merge_sarsteer_agent_labels.py \
  --paired "$HELD/heldout_paired_a0.1.jsonl" \
  --labels-dir "$HELD/judge" \
  --out "$HELD/heldout.manual_labels.jsonl" \
  --resolution claude_agent_local

./scripts/evaluate_defense_gate.py \
  --defense-name "SARSteer (paper-faithful, alpha=0.1)" \
  --paired "$HELD/heldout_paired_a0.1.jsonl" \
  --paired-labels "$HELD/heldout.manual_labels.jsonl" \
  --clean-condition clean \
  --attack-condition pv_standard \
  --benign-condition clean \
  --positive-control-condition positive_control \
  --output "$HELD/heldout.gate_report.json"
