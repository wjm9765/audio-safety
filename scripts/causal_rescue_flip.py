#!/usr/bin/env -S uv run python
"""Run 4 §8 direction-finding: causal rescue of jb_pap audio flips via the frozen r_A actuator.

DIRECTION-FINDING ONLY. Tests whether ADDING the frozen, causally-validated refusal actuator
r_A (layer 16, all-positions, scale=alpha preselected from the RDO gate — NOT tuned here)
RESTORES refusal on the genuine jb_pap audio refusal->comply flips, with the controls Codex
required: (a) norm-matched RANDOM direction at the same strength (directional specificity),
(b) attacked-BENIGN items at the same strength (over-refusal cost). Refusal is scored with the
same local `label_output` used to validate the axis's add_rr, so the rescue metric is comparable.

Consensus flips = harmful jb_pap AUDIO items that BOTH judges call neutral-refusal -> jb_pap-comply.

Example:
    ./scripts/causal_rescue_flip.py --config configs/experiments/run4_attack_flip.yaml \
        --axis-artifact /workspace/.../exp1_20260707_1557_allpos_rebuttal_l12nbhd/rdo_axis.npz \
        --alpha 2.0 --n-benign 30 --run-name run4_rescue
"""

import argparse
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.evaluation import label_output
from audio_safety.evaluation.attack_flip import classify
from audio_safety.models.qwen2_audio import (
    generate_audio_response,
    generate_audio_response_with_intervention,
    load_qwen2_audio,
)
from audio_safety.pipelines.rdo_gate import load_axis
from audio_safety.utils.io import load_jsonl, save_json, save_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--axis-artifact", type=Path, required=True)
    p.add_argument("--alpha", type=float, default=2.0, help="preselected add strength (RDO gate)")
    p.add_argument("--n-benign", type=int, default=30)
    p.add_argument("--max-flips", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-name", type=str, default="run4_rescue")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=None)
    return p.parse_args()


def _classify_by(judged, modality, style, safety):
    """(item_id) -> {judge: class} for one cell."""
    out = {}
    for r in judged:
        if str(r.get("modality")) != modality or str(r.get("style")) != style:
            continue
        if str(r.get("safety_label")) != safety:
            continue
        v = r.get("judge_verdicts") or {}
        out[str(r["item_id"])] = {m: (classify(vv) if vv else None) for m, vv in v.items()}
    return out


def _rows_by_item(judged, modality, style, safety):
    out = {}
    for r in judged:
        if (str(r.get("modality")) == modality and str(r.get("style")) == style
                and str(r.get("safety_label")) == safety):
            out[str(r["item_id"])] = r
    return out


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)
    judges = list(cfg.conversion_gap.judge.models)
    instruction = cfg.dataset.target_generation.instruction
    max_new = cfg.dataset.target_generation.max_new_tokens

    r_a, site = load_axis(args.axis_artifact)
    r_a = np.asarray(r_a, dtype=np.float32)
    ra_norm = float(np.linalg.norm(r_a))
    rng = np.random.default_rng(args.seed)
    rand = rng.standard_normal(r_a.shape).astype(np.float32)
    rand = rand / float(np.linalg.norm(rand)) * ra_norm  # norm-matched random control

    judged = load_jsonl(paths.data_dir / cfg.conversion_gap.judge.outputs_file)
    neu = _classify_by(judged, "audio", "neutral", "harmful")
    pap = _classify_by(judged, "audio", "jb_pap", "harmful")
    pap_rows = _rows_by_item(judged, "audio", "jb_pap", "harmful")
    # consensus flips: BOTH judges neutral==refusal AND jb_pap==comply
    flips = [
        i for i in neu if i in pap
        and all(neu[i].get(m) == "refusal" for m in judges)
        and all(pap[i].get(m) == "comply" for m in judges)
    ]
    flips = sorted(flips)[: args.max_flips]

    # attacked-benign that were answered (not refusal) by both judges -> over-refusal control
    ben = _classify_by(judged, "audio", "jb_pap", "benign")
    ben_rows = _rows_by_item(judged, "audio", "jb_pap", "benign")
    benign_answered = [
        i for i in ben if all(ben[i].get(m) not in (None, "refusal") for m in judges)
    ]
    benign_answered = sorted(benign_answered)[: args.n_benign]

    print(f"[rescue] consensus flips: {len(flips)}; benign-answered controls: {len(benign_answered)}")
    print(f"[rescue] r_A layer {site.layer}/{site.position} alpha={args.alpha} |r_A|={ra_norm:.3f}")

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)

    def gen(path, vec=None):
        if vec is None:
            return generate_audio_response(model, processor, path, instruction, max_new_tokens=max_new)
        return generate_audio_response_with_intervention(
            model, processor, path, instruction,
            layer_idx=site.layer, position_name=site.position, vector=vec,
            mode="add", scale=args.alpha, all_positions=True, max_new_tokens=max_new,
        )

    def is_refusal(output, safety):
        label, _, _ = label_output(output, safety_label=safety, reference_text="")
        return label == "policy_refusal", label

    records = []
    counts = {"flip_baseline": [0, 0], "flip_rA": [0, 0], "flip_random": [0, 0], "benign_rA": [0, 0]}
    for i in flips:
        path = paths.data_dir / str(pap_rows[i]["path"])
        for cond, vec in (("flip_baseline", None), ("flip_rA", r_a), ("flip_random", rand)):
            out = gen(path, vec)
            ref, label = is_refusal(out, "harmful")
            counts[cond][0] += int(ref); counts[cond][1] += 1
            records.append({"item_id": i, "cond": cond, "label": label, "refusal": ref, "output": out[:400]})
    for i in benign_answered:
        path = paths.data_dir / str(ben_rows[i]["path"])
        out = gen(path, r_a)
        ref, label = is_refusal(out, "benign")
        counts["benign_rA"][0] += int(ref); counts["benign_rA"][1] += 1
        records.append({"item_id": i, "cond": "benign_rA", "label": label, "refusal": ref, "output": out[:400]})

    summary = {
        "alpha": args.alpha, "r_a_layer": site.layer, "r_a_position": site.position,
        "n_flips": len(flips), "n_benign": len(benign_answered), "judges": judges,
        "refusal_rate": {k: (v[0] / v[1] if v[1] else None) for k, v in counts.items()},
        "counts": counts,
    }
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    save_json(summary, run_dir / "causal_rescue_summary.json")
    save_jsonl(records, run_dir / "causal_rescue_records.jsonl")
    print("[rescue] refusal rate by condition:")
    for k, v in summary["refusal_rate"].items():
        print(f"   {k:14s} {v if v is None else round(v, 3)}  (n={counts[k][1]})")
    print(f"[rescue] -> {run_dir / 'causal_rescue_summary.json'}")


if __name__ == "__main__":
    main()
