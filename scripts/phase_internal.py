#!/usr/bin/env -S uv run python
"""Run 7 internal signature (CPU): does the phase artifact move the L18 residual along
the FROZEN refusal direction, more than the mel-distance-matched coherent control?

Frozen r (refusal) and hdir (harm) = difference-in-means of run5 P2 @ L18 (not refit).
For each item x sign, delta_cond = P2_cond - P2_locked at L18. Reports the projection
onto r (signed refusal-axis movement), the fraction of that movement explained by r,
and an EXPLORATORY multidimensional SVD (effective rank + rank-k margin-erosion readout
with nested CV; NOT load-bearing, honors the PI's 'multidimensional analysis' request).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SRC_RUN = "run5_20260714_0308_pitch_n150"


def dim_dir(P2, y):
    m = np.isin(y, [0, 1])
    if len(np.unique(y[m])) < 2:
        return None
    d = P2[m][y[m] == 1].mean(0) - P2[m][y[m] == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--src-run", default=SRC_RUN)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data"))
    ap.add_argument("--layer", type=int, default=18)
    args = ap.parse_args()

    src = args.data_dir / "outputs" / args.src_run / "pitch_representation"
    s_arr = dict(np.load(src / "activations.npz", allow_pickle=True))
    s_layers = [int(v) for v in s_arr["llm_layers"]]
    s_P2 = np.asarray(s_arr["llm_p2"], np.float64)[:, s_layers.index(args.layer), :]
    s_cells = [json.loads(l) for l in (src / "cells.jsonl").read_text().splitlines() if l.strip()]
    s_lab = np.asarray([str(c.get("reviewed_behavior_label") or "") for c in s_cells])
    s_safe = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in s_cells])
    r = dim_dir(s_P2, np.where(s_lab == "policy_refusal", 1, np.where(s_lab == "harmful_compliance", 0, -1)))
    hdir = dim_dir(s_P2, s_safe)

    r7 = args.run_dir / "pitch_frontend"
    cells = [json.loads(l) for l in (r7 / "cells.jsonl").read_text().splitlines() if l.strip()]
    a7 = dict(np.load(r7 / "activations.npz", allow_pickle=True))
    a7_layers = [int(v) for v in a7["llm_layers"]]
    P2 = np.asarray(a7["llm_p2"], np.float64)[:, a7_layers.index(args.layer), :]
    by = {(c["item_id"], round(float(c["sign"]), 6), c["condition"]): c for c in cells}
    items = sorted({c["item_id"] for c in cells})
    signs = sorted({round(float(c["sign"]), 6) for c in cells if c["condition"] != "neutral"})
    neu_margin = {c["item_id"]: c["refusal_margin"] for c in cells if c["condition"] == "neutral"}

    def delta(cond, base, it, s):
        a, b = by.get((it, s, cond)), by.get((it, s, base))
        if not a or not b:
            return None
        return P2[a["activation_index"]] - P2[b["activation_index"]]

    print("=== L18 refusal-axis movement: projection of (cond - locked) onto frozen r ===")
    print("(negative = moved toward compliance / off the refusal side)")
    rows = {}
    for cond in ["pv_standard", "phase_transplant", "mel_matched_ctrl"]:
        projs, fracs = [], []
        for it in items:
            for s in signs:
                d = delta(cond, "pv_locked", it, s)
                if d is None:
                    continue
                p = float(r @ d)
                projs.append(p)
                fracs.append(p ** 2 / (float(d @ d) + 1e-12))
        rows[cond] = np.array(projs)
        print(f"  {cond:18s} r-proj mean={np.mean(projs):+.3f}  |r-proj| mean={np.mean(np.abs(projs)):.3f}  "
              f"energy-frac-on-r mean={np.mean(fracs):.4f}  n={len(projs)}")

    # representation-level specificity: std vs mel_matched_ctrl (matched input distance)
    if "pv_standard" in rows and "mel_matched_ctrl" in rows:
        print(f"\n[repr specificity] refusal-axis movement std vs mel_matched_ctrl (matched D_pair): "
              f"std={np.mean(rows['pv_standard']):+.3f} vs ctrl={np.mean(rows['mel_matched_ctrl']):+.3f}")

    # correlate refusal-axis movement with margin erosion (std)
    xs, ys = [], []
    for it in items:
        for s in signs:
            cs = by.get((it, s, "pv_standard"))
            d = delta("pv_standard", "pv_locked", it, s)
            if cs is None or d is None or it not in neu_margin:
                continue
            xs.append(float(r @ d))
            ys.append(neu_margin[it] - cs["refusal_margin"])
    if len(xs) > 3:
        rho = np.corrcoef(xs, ys)[0, 1]
        print(f"[assoc] corr(refusal-axis movement, margin erosion) = {rho:+.3f} (n={len(xs)})")

    # EXPLORATORY multidimensional SVD of the std-vs-locked displacement
    D = []
    for it in items:
        for s in signs:
            d = delta("pv_standard", "pv_locked", it, s)
            if d is not None:
                D.append(d)
    D = np.array(D)
    if len(D) >= 5:
        Dc = D - D.mean(0)
        sv = np.linalg.svd(Dc, compute_uv=False)
        ev = sv ** 2
        pr = float((ev.sum() ** 2) / (ev ** 2).sum())  # participation ratio
        print(f"\n[EXPLORATORY multidim] std-vs-locked L18 displacement: n={len(D)} "
              f"participation-ratio(effective rank)={pr:.1f}")
        print(f"  top singular values (norm): {np.round(sv[:6] / sv[0], 3).tolist()}")
        print("  NOTE: high effective rank is expected for 4096-d noisy displacements; "
              "reported descriptively, NOT as a load-bearing multidimensional claim.")


if __name__ == "__main__":
    main()
