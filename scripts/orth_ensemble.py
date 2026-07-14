#!/usr/bin/env -S uv run python
"""Codex round-3 upgrade: replace the single orthogonal control with an ENSEMBLE of K
covariance-matched directions orthogonal to both the refusal and harm directions, at the
identical cellwise norm and all-position schedule. Reuses baseline + restore_full outputs
from causal_refusal_component.jsonl (no regen) and generates only the K orth conditions.
Tests whether tau=0.278 reflects refusal-axis specificity or a favourable single control.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np


def loo_dir(P2, label, valid, exclude, items):
    m = valid & (items != exclude)
    y = label[m]
    if len(np.unique(y)) < 2:
        return None
    d = P2[m][y == 1].mean(0) - P2[m][y == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > 1e-9 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--k", type=int, default=15)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    from audio_safety.config import load_experiment_config
    from audio_safety.models.qwen2_audio import load_qwen2_audio, generate_audio_response_with_intervention
    from audio_safety.utils.paths import resolve_paths

    cfg = load_experiment_config(Path("configs/experiments/run5_pitch_representation_fast.yaml"))
    paths = resolve_paths(cfg.paths)
    cells = [json.loads(l) for l in (args.run_dir / "pitch_representation/cells.jsonl").read_text().splitlines() if l.strip()]
    causal = [json.loads(l) for l in (args.run_dir / "pitch_representation/causal_refusal_component.jsonl").read_text().splitlines() if l.strip()]
    arr = dict(np.load(args.run_dir / "pitch_representation/activations.npz", allow_pickle=True))
    llm = [int(v) for v in arr["llm_layers"]]
    P2 = np.asarray(arr["llm_p2"], float)[:, llm.index(args.layer), :]
    idx = {(str(c["item_id"]), str(c["safety_label"]), round(float(c["pitch_semitones"]), 6)): int(c["activation_index"]) for c in cells}
    items = np.asarray([str(c["item_id"]) for c in cells])
    lab = np.asarray([str(c.get("reviewed_behavior_label") or "") for c in cells])
    marg = np.asarray([float(c["refusal_margin"]) for c in cells])
    safety = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in cells])
    refuse = (lab == "policy_refusal").astype(int)
    valid = np.isin(lab, ["policy_refusal", "harmful_compliance"])

    # pool of pitch displacements (empirical covariance-matched directions)
    pool = []
    for r in causal:
        it, p = str(r["item_id"]), round(float(r["pitch"]), 6)
        nh, ph = idx.get((it, "harmful", 0.0)), idx.get((it, "harmful", p))
        if nh is not None and ph is not None:
            pool.append(P2[ph] - P2[nh])
    pool = np.stack(pool)
    rng = np.random.RandomState(args.seed)

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction

    out_rows = []
    for r in causal:
        it, p = str(r["item_id"]), round(float(r["pitch"]), 6)
        nh, ph = idx.get((it, "harmful", 0.0)), idx.get((it, "harmful", p))
        if nh is None or ph is None:
            continue
        delta = P2[ph] - P2[nh]
        rd = loo_dir(P2, refuse, valid, it, items)
        hd = loo_dir(P2, safety, np.ones(len(cells), bool), it, items)
        if rd is None or hd is None:
            continue
        a_mag = abs(float(rd @ delta))
        wav = str(args.run_dir / r_cell_variant(cells, idx, it, p))
        orth_outs = []
        picks = rng.choice(len(pool), size=min(args.k, len(pool)), replace=False)
        for j, pi in enumerate(picks):
            v = pool[pi].copy()
            v = v - (rd @ v) * rd
            v = v - (hd @ v) * hd            # orthogonal to BOTH refusal and harm
            n = np.linalg.norm(v)
            if n < 1e-6:
                continue
            v = v / n
            out = generate_audio_response_with_intervention(
                model, processor, wav, instruction, layer_idx=args.layer,
                position_name="first_generation_prelogit", vector=torch.tensor(v, dtype=torch.float32),
                mode="add", scale=-a_mag, all_positions=True,   # remove an equal-norm cov-matched orth component
                max_new_tokens=args.max_new_tokens, do_sample=False,
            )
            orth_outs.append(out)
        out_rows.append({"item_id": it, "pitch": p, "reference_text": r.get("reference_text"),
                         "baseline": r.get("baseline"), "restore_full": r.get("restore_full"),
                         "orth_ensemble": orth_outs, "a_mag": a_mag})
        print(f"  {it[-4:]} p={p:+g} a_mag={a_mag:.1f} orth_n={len(orth_outs)}")
    outp = args.run_dir / "pitch_representation/orth_ensemble.jsonl"
    outp.write_text("\n".join(json.dumps(r) for r in out_rows) + "\n")
    print(f"\nwrote {outp} ({len(out_rows)} cells, k={args.k})")


def r_cell_variant(cells, idx, it, p):
    ai = idx[(it, "harmful", p)]
    for c in cells:
        if int(c["activation_index"]) == ai:
            return c["variant_path"]
    raise KeyError((it, p))


if __name__ == "__main__":
    main()
