#!/usr/bin/env -S uv run python
"""Assemble the Run 10 channel-invariance gate input manifest from the Run 9 cohort.

Run 9 froze the faithfulness-gated clean/attack cohort (per-item harmful-anchor +
WER/token-overlap Whisper gate). Run 10 consumes ONLY the pairs that passed:

  clean  = harmful, condition=clean, style=neutral, transcript_control_passed
  attack = harmful, condition in {pv_standard, pv_locked, mel_matched_ctrl},
           transcript_control_passed, and whose item has a passing clean

`style` is set to the CONDITION (pv_standard/pv_locked/mel_matched_ctrl) so the
recognition gate + L18 confirmatory treat one transform family per condition and
POOL the +/-3 semitone context (sign is kept as a separate covariate). pv_standard
is the primary phase attack under test; pv_locked (identity phase-lock) and
mel_matched_ctrl (zero-phase EQ, mel-RMS matched) are the coherent specificity controls.

The item-level 60/20/20 train/dev/test split is frozen ONCE here over the harmful
clean-pass items (same algorithm as channel_patch_l18._split_items) so Step 2 (tau
freeze on clean-dev) and Step 3 (fit U on train+dev, evaluate test) share it. Outputs:

  <out-dir>/manifest.jsonl          clean + attack rows for recognition_gate.py
  <out-dir>/splits.json             {item_id: train|dev|test} frozen split
  <out-dir>/clean_dev_item_ids.txt  dev item_ids -> recognition_gate --dev-items
  <out-dir>/summary.json            counts per condition/sign/fold

CPU-only, no torch. Paths in the manifest are relative to --data-dir (the audio root).
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np

CONDITIONS = ["pv_standard", "pv_locked", "mel_matched_ctrl"]


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _split_items(items: list[str], seed: int) -> dict[str, str]:
    """Item-level 60/20/20, byte-identical to channel_patch_l18._split_items."""
    rng = np.random.RandomState(seed)
    order = sorted(set(items))
    rng.shuffle(order)
    n = len(order)
    n_train, n_dev = int(0.6 * n), int(0.2 * n)
    return {
        item: ("train" if i < n_train else ("dev" if i < n_train + n_dev else "test"))
        for i, item in enumerate(order)
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean", type=Path, required=True, help="Run 9 asr_clean.jsonl")
    ap.add_argument("--attacks", type=Path, required=True, help="Run 9 asr_attacks.jsonl")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data/data"),
                    help="audio root the manifest paths are relative to (recognition_gate --run-dir)")
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS)
    ap.add_argument("--clean-style", default="neutral")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    clean_rows = _load(args.clean)
    attack_rows = _load(args.attacks)

    # Harmful clean that passed faithfulness -> one canonical clean per item.
    clean_by_item: dict[str, dict] = {}
    for r in clean_rows:
        if r.get("safety_label") != "harmful" or not r.get("transcript_control_passed"):
            continue
        item = str(r["item_id"])
        clean_by_item.setdefault(item, r)
    clean_ok = set(clean_by_item)

    # Resolve every referenced audio to an ABSOLUTE path (fail fast, no silent drops).
    # recognition_gate/channel_patch_l18 do `run_dir / row["path"]`; an absolute path makes
    # audio resolution independent of --run-dir (pathlib: abs RHS wins) so --run-dir can be
    # the run OUTPUT dir without polluting the data tree.
    def _abspath(path: str) -> str:
        full = (args.data_dir / path).resolve()
        if not full.is_file():
            raise SystemExit(f"missing audio file: {full}")
        return str(full)

    out_rows: list[dict] = []
    for item in sorted(clean_ok):
        r = clean_by_item[item]
        out_rows.append({
            "item_id": item, "style": args.clean_style, "condition": "clean",
            "sign": 0.0, "path": _abspath(r["path"]), "reference_text": r.get("reference_text"),
            "safety_label": "harmful", "source_style": r.get("style"),
        })

    kept = collections.Counter()
    dropped_no_clean = 0
    for r in attack_rows:
        if r.get("safety_label") != "harmful" or not r.get("transcript_control_passed"):
            continue
        cond = str(r.get("condition"))
        if cond not in args.conditions:
            continue
        item = str(r["item_id"])
        if item not in clean_ok:
            dropped_no_clean += 1
            continue
        out_rows.append({
            "item_id": item, "style": cond, "condition": cond,
            "sign": round(float(r.get("sign", 0.0)), 6), "path": _abspath(r["path"]),
            "reference_text": r.get("reference_text"), "safety_label": "harmful",
            "source_style": r.get("style"), "route": r.get("route"),
            "phase_under_test": r.get("phase_under_test"), "d_pair": r.get("d_pair"),
        })
        kept[(cond, float(r.get("sign", 0.0)))] += 1

    # Freeze the split over harmful clean-pass items (attacks inherit their item's fold).
    split = _split_items(sorted(clean_ok), args.seed)
    for row in out_rows:
        row["split"] = split[row["item_id"]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r) for r in out_rows) + "\n"
    )
    (args.out_dir / "splits.json").write_text(json.dumps(split, indent=2) + "\n")
    dev_items = sorted(i for i, f in split.items() if f == "dev")
    (args.out_dir / "clean_dev_item_ids.txt").write_text("\n".join(dev_items) + "\n")

    fold_counts = collections.Counter(split.values())
    summary = {
        "n_clean_items": len(clean_ok),
        "n_manifest_rows": len(out_rows),
        "conditions": args.conditions,
        "attack_rows_kept": {f"{c}|{s:+g}": n for (c, s), n in sorted(kept.items())},
        "attack_dropped_no_passing_clean": dropped_no_clean,
        "fold_counts_items": dict(fold_counts),
        "data_dir": str(args.data_dir),
        "seed": args.seed,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir}/manifest.jsonl ({len(out_rows)} rows), "
          f"splits.json, clean_dev_item_ids.txt ({len(dev_items)} dev items)")


if __name__ == "__main__":
    main()
