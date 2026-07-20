#!/usr/bin/env -S uv run python
"""Run 12 Phase A (CPU): render the 2x2 (harmfulness x channel) factorial cohort + freeze folds.

Codex-locked design. For each FigStep item that has BOTH a harmful-clean and a matched
benign-clean neutral wav (audio_attack_flip), render pv_standard at signs -3/+3 on BOTH the
harmful and the benign clean audio with the frozen run7 phase operator (pitch_shift_custom
mode='standard'). Selection is on EXTERNAL availability only (no M / refusal / recognition /
harmfulness conditioning). Freeze 5 category-stratified outer folds (seed 0), harmful and its
matched benign in the SAME fold. Writes:
  <run>/audio_run12/{harmful,benign}/pv_standard_{m3,p3}/<item>.wav
  <run>/cohort.jsonl   one row per (item, role in {harmful,benign}, condition in
                       {clean, pv_standard_m3, pv_standard_p3}) with abs path
  <run>/folds.json     {item_id: fold}
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

SIGNS = {-3.0: "m3", 3.0: "p3"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--attack-flip", type=Path,
                    default=Path("/workspace/audio_safety_data/data/audio_attack_flip"))
    ap.add_argument("--pairs", type=Path,
                    default=Path("/workspace/audio_safety_data/data/text/figstep/audio_rdo_pairs.jsonl"))
    ap.add_argument("--category-jsonl", nargs="+", type=Path,
                    default=[Path("/workspace/audio_safety_data/outputs/run9_fresh/asr_attacks.jsonl"),
                             Path("/workspace/audio_safety_data/outputs/run9_fresh/asr_clean.jsonl")],
                    help="source of the canonical SafeBench category_name (NOT the benign-rewrite type)")
    ap.add_argument("--sample-rate", type=int, default=16000)
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import librosa
    import soundfile as sf

    from audio_safety.evaluation.phase_ops import pitch_shift_custom

    # items with BOTH harmful and benign neutral clean audio
    def items_in(sub):
        return {re.search(r"(figstep_safebench_\d+)", f).group(1)
                for f in glob.glob(str(args.attack_flip / sub / "neutral" / "*.wav"))}
    both = sorted(items_in("harmful") & items_in("benign"))
    if not both:
        raise SystemExit("no paired harmful+benign items")

    # category per item (for stratified folds) from the pairs file
    cat = {}
    for line in args.pairs.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            cat[str(r["item_id"])] = str(r.get("category", "unknown"))

    ref = {}
    for line in args.pairs.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            ref[str(r["item_id"])] = {"harmful": r.get("harmful_text"), "benign": r.get("benign_text")}

    out_audio = args.run_dir / "audio_run12"
    rows = []
    for i, item in enumerate(both, 1):
        for role in ("harmful", "benign"):
            clean_path = args.attack_flip / role / "neutral" / f"{item}.wav"
            y, _ = librosa.load(str(clean_path), sr=args.sample_rate)
            rows.append({"item_id": item, "role": role, "condition": "clean", "sign": 0.0,
                         "path": str(clean_path.resolve()),
                         "reference_text": (ref.get(item, {}) or {}).get(role), "category": cat.get(item, "unknown")})
            for sign, tag in SIGNS.items():
                atk = pitch_shift_custom(y, args.sample_rate, sign, mode="standard")
                ap_out = out_audio / role / f"pv_standard_{tag}" / f"{item}.wav"
                ap_out.parent.mkdir(parents=True, exist_ok=True)
                sf.write(str(ap_out), atk, args.sample_rate)
                rows.append({"item_id": item, "role": role, "condition": f"pv_standard_{tag}",
                             "sign": sign, "path": str(ap_out.resolve()),
                             "reference_text": (ref.get(item, {}) or {}).get(role), "category": cat.get(item, "unknown")})
        if i % 25 == 0 or i == len(both):
            print(f"  rendered {i}/{len(both)} items", flush=True)

    # category-stratified 5 folds, seed 0; harmful+benign of an item share the fold (item-level)
    rng = np.random.RandomState(args.seed)
    by_cat: dict[str, list[str]] = {}
    for item in both:
        by_cat.setdefault(cat.get(item, "unknown"), []).append(item)
    folds = {}
    for c, items in sorted(by_cat.items()):
        order = sorted(items)
        rng.shuffle(order)
        for k, it in enumerate(order):
            folds[it] = k % args.n_folds

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "cohort.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (args.run_dir / "folds.json").write_text(json.dumps(folds, indent=2) + "\n")
    import collections
    print(json.dumps({"n_items": len(both), "n_rows": len(rows),
                      "fold_sizes": dict(collections.Counter(folds.values())),
                      "n_categories": len(by_cat)}, indent=2))
    print(f"wrote {args.run_dir}/cohort.jsonl + folds.json + audio_run12/")


if __name__ == "__main__":
    main()
