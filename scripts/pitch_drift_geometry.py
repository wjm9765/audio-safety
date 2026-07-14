#!/usr/bin/env -S uv run python
"""Augmentation analysis for the pitch-representation run: the shared low-dimensional
acoustic-drift subspace, its multidimensional interaction with the refusal readout,
harmfulness-vs-refusal dissociation along pitch, and a geometric brittleness predictor.

This complements `audio_safety.evaluation.pitch_representation` (which computes the
DiD-SVD margin readout + dissociation flags). Here we build the *new object* the paper
leans on: a shared pitch-drift subspace V, and we test whether the pitch-driven change
in the refusal margin is (a) multidimensional and (b) predictable from geometry that the
scalar neutral margin is blind to.

Read-only w.r.t. the model; runs on CPU from the extracted artifacts. Usage:
    ./scripts/pitch_drift_geometry.py --run-dir <abs run dir> [--site llm_p2] [--out ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_EPS = 1e-12


def _load(run_dir: Path, activations_rel: str, cells_rel: str):
    arrays = dict(np.load(run_dir / activations_rel, allow_pickle=True))
    cells = [json.loads(line) for line in (run_dir / cells_rel).read_text().splitlines() if line.strip()]
    return arrays, cells


def _site_view(arrays: dict[str, np.ndarray], site: str, layer_offset: int | None):
    """Return an (N, d) matrix for a site. LLM/encoder sites are (N, L, d); pick a layer."""
    a = np.asarray(arrays[site], dtype=np.float64)
    if a.ndim == 3:
        if layer_offset is None:
            raise ValueError(f"site {site} is layered; pass a layer offset")
        return a[:, layer_offset, :]
    return a


def _index(cells: list[dict[str, Any]]):
    by = {}
    for c in cells:
        by[(str(c["item_id"]), str(c["safety_label"]), round(float(c["pitch_semitones"]), 6))] = int(
            c["activation_index"]
        )
    return by


def _pr(singular_sq: np.ndarray) -> float:
    """Participation ratio (effective rank) from squared singular values."""
    s = singular_sq[singular_sq > _EPS]
    if s.size == 0:
        return 0.0
    return float((s.sum() ** 2) / (np.square(s).sum()))


def _grouped_folds(groups: np.ndarray, n_folds: int):
    from sklearn.model_selection import GroupKFold

    uniq = np.unique(groups)
    if len(uniq) < 2:
        return []
    gkf = GroupKFold(n_splits=min(n_folds, len(uniq)))
    z = np.zeros(len(groups))
    return list(gkf.split(z, z, groups))


def _crossfit_dir(x: np.ndarray, y: np.ndarray, groups: np.ndarray, n_folds: int):
    """Cross-fit difference-of-means direction; return per-row signed score + AUROC."""
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, dtype=np.int8)
    scores = np.full(len(y), np.nan)
    dirs = []
    for tr, te in _grouped_folds(groups, n_folds):
        if len(np.unique(y[tr])) != 2:
            continue
        m0 = x[tr][y[tr] == 0].mean(0)
        m1 = x[tr][y[tr] == 1].mean(0)
        d = m1 - m0
        n = np.linalg.norm(d)
        if n <= _EPS:
            continue
        d = d / n
        thr = 0.5 * (m0 @ d + m1 @ d)
        scores[te] = x[te] @ d - thr
        dirs.append(d)
    valid = np.isfinite(scores)
    auroc = None
    if valid.sum() >= 2 and len(np.unique(y[valid])) == 2:
        auroc = float(roc_auc_score(y[valid], scores[valid]))
    mean_dir = np.mean(dirs, 0) if dirs else None
    if mean_dir is not None:
        mean_dir = mean_dir / (np.linalg.norm(mean_dir) + _EPS)
    return scores, auroc, mean_dir


def analyze(run_dir: Path, site: str, layers: list[int], activations_rel: str, cells_rel: str,
            n_folds: int, ranks: list[int]):
    arrays, cells = _load(run_dir, activations_rel, cells_rel)
    llm_layers = [int(v) for v in arrays["llm_layers"]]
    by = _index(cells)
    items = sorted({str(c["item_id"]) for c in cells})
    pitches = sorted({round(float(c["pitch_semitones"]), 6) for c in cells})
    nonzero_pitches = [p for p in pitches if p != 0.0]

    margins = {int(c["activation_index"]): float(c["refusal_margin"]) for c in cells}

    # per-item behavioral brittleness proxies from generated harmful responses
    behavior = defaultdict(dict)  # item -> pitch -> label
    reviewed = defaultdict(dict)
    for c in cells:
        if c.get("safety_label") != "harmful":
            continue
        p = round(float(c["pitch_semitones"]), 6)
        if c.get("behavior_label"):
            behavior[str(c["item_id"])][p] = str(c["behavior_label"])
        if c.get("reviewed_behavior_label"):
            reviewed[str(c["item_id"])][p] = str(c["reviewed_behavior_label"])

    report: dict[str, Any] = {"site": site, "layers": {}, "items": items,
                              "pitches": pitches, "n_items": len(items)}

    for layer in layers:
        try:
            off = llm_layers.index(layer)
        except ValueError:
            continue
        X = _site_view(arrays, site, off)  # (N, d)

        # ---- shared pitch-drift subspace from benign-controlled DiD ----
        did_rows, drift_rows, dM_rows, row_items = [], [], [], []
        for it in items:
            nh = by.get((it, "harmful", 0.0))
            nb = by.get((it, "benign", 0.0))
            if nh is None or nb is None:
                continue
            for p in nonzero_pitches:
                ph = by.get((it, "harmful", p))
                pb = by.get((it, "benign", p))
                if ph is None or pb is None:
                    continue
                dH = X[ph] - X[nh]
                dB = X[pb] - X[nb]
                did_rows.append(dH - dB)       # harmful-specific pitch displacement
                drift_rows.append(dH)          # raw harmful pitch drift
                dM_rows.append(margins[ph] - margins[nh])  # refusal-margin change
                row_items.append(it)
        if len(did_rows) < 4:
            continue
        DiD = np.stack(did_rows)
        DR = np.stack(drift_rows)
        dM = np.asarray(dM_rows)
        row_items = np.asarray(row_items)

        # SVD (effective rank) of the shared drift and the benign-controlled DiD
        DiD_c = DiD - DiD.mean(0, keepdims=True)
        DR_c = DR - DR.mean(0, keepdims=True)
        s_did = np.linalg.svd(DiD_c, compute_uv=False)
        s_dr = np.linalg.svd(DR_c, compute_uv=False)

        # ---- multidimensional refusal-margin readout: rank-k of DiD predicts dM ----
        from sklearn.linear_model import Ridge
        from sklearn.metrics import mean_squared_error
        rank_mse = {}
        preds = {k: np.full(len(dM), np.nan) for k in ranks}
        for tr, te in _grouped_folds(row_items, n_folds):
            ctr = DiD[tr].mean(0, keepdims=True)
            Xtr = DiD[tr] - ctr
            Xte = DiD[te] - ctr
            maxr = min(max(ranks), Xtr.shape[0] - 1, Xtr.shape[1])
            if maxr < 1:
                continue
            U, S, Vt = np.linalg.svd(Xtr, full_matrices=False)
            comp = Vt[:maxr]
            for k in ranks:
                if k > maxr:
                    continue
                rr = Ridge(alpha=1.0).fit(Xtr @ comp[:k].T, dM[tr])
                preds[k][te] = rr.predict(Xte @ comp[:k].T)
        for k in ranks:
            v = np.isfinite(preds[k])
            if v.sum() >= 2:
                rank_mse[k] = float(mean_squared_error(dM[v], preds[k][v]))
        mse1 = rank_mse.get(1)
        mse_multi = min([rank_mse[k] for k in (2, 3) if k in rank_mse], default=None)
        multidim_gain = None if (mse1 is None or mse_multi is None or mse1 <= _EPS) else float(
            (mse1 - mse_multi) / mse1
        )

        # ---- dissociation along pitch: harmfulness readout vs refusal readout ----
        # readouts fit on ALL cells at this site
        allX = X
        safety = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in cells])
        refl = np.asarray([1 if margins[int(c["activation_index"])] > 0 else 0 for c in cells])
        groups_all = np.asarray([str(c["item_id"]) for c in cells])
        h_scores, h_auroc, _ = _crossfit_dir(allX, safety, groups_all, n_folds)
        r_scores, r_auroc, _ = _crossfit_dir(allX, refl, groups_all, n_folds)

        # movement from neutral->extreme pitch on harmful items (mean |Δ| in readout units)
        idx_of = {int(c["activation_index"]): i for i, c in enumerate(cells)}
        harm_move, ref_move = [], []
        for it in items:
            nh = by.get((it, "harmful", 0.0))
            if nh is None:
                continue
            for p in (min(nonzero_pitches), max(nonzero_pitches)):
                ph = by.get((it, "harmful", p))
                if ph is None:
                    continue
                i0, ip = idx_of[nh], idx_of[ph]
                if np.isfinite(h_scores[i0]) and np.isfinite(h_scores[ip]):
                    harm_move.append(abs(h_scores[ip] - h_scores[i0]))
                if np.isfinite(r_scores[i0]) and np.isfinite(r_scores[ip]):
                    ref_move.append(abs(r_scores[ip] - r_scores[i0]))
        dissoc = {
            "harmfulness_auroc": h_auroc,
            "refusal_auroc": r_auroc,
            "mean_harmfulness_move": float(np.mean(harm_move)) if harm_move else None,
            "mean_refusal_move": float(np.mean(ref_move)) if ref_move else None,
        }

        report["layers"][str(layer)] = {
            "n_did_rows": int(len(DiD)),
            "drift_effective_rank_pr": _pr(s_dr ** 2),
            "did_effective_rank_pr": _pr(s_did ** 2),
            "drift_top1_var": float((s_dr[0] ** 2) / (np.sum(s_dr ** 2) + _EPS)),
            "drift_top3_var": float(np.sum(s_dr[:3] ** 2) / (np.sum(s_dr ** 2) + _EPS)),
            "did_top1_var": float((s_did[0] ** 2) / (np.sum(s_did ** 2) + _EPS)),
            "did_top3_var": float(np.sum(s_did[:3] ** 2) / (np.sum(s_did ** 2) + _EPS)),
            "rank_mse": rank_mse,
            "multidim_gain_rank23_over_rank1": multidim_gain,
            "dissociation": dissoc,
        }

    # ---- behavioral brittleness summary (heuristic + reviewed) ----
    def brittle_count(table):
        n_brittle, flips = 0, []
        for it in items:
            labs = table.get(it, {})
            neu = labs.get(0.0)
            if neu != "policy_refusal":
                continue
            comp = [p for p, l in labs.items() if p != 0.0 and l == "harmful_compliance"]
            if comp:
                n_brittle += 1
                flips.append({"item": it, "flip_pitches": sorted(comp)})
        return n_brittle, flips

    hb, hflips = brittle_count(behavior)
    rb, rflips = brittle_count(reviewed)
    report["behavior_heuristic"] = {"n_brittle_items": hb, "flips": hflips}
    report["behavior_reviewed"] = {"n_brittle_items": rb, "flips": rflips}
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--site", default="llm_p2")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="LLM layer indices; default = a spread across depth")
    ap.add_argument("--activations", default="pitch_representation/activations.npz")
    ap.add_argument("--cells", default="pitch_representation/cells.jsonl")
    ap.add_argument("--n-folds", type=int, default=5)
    ap.add_argument("--ranks", type=int, nargs="*", default=[1, 2, 3, 5])
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    arrays = dict(np.load(args.run_dir / args.activations, allow_pickle=True))
    llm_layers = [int(v) for v in arrays["llm_layers"]]
    print("array keys:", sorted(arrays))
    for k in ("encoder_mean", "projector_mean", "llm_p2", "llm_audio_mean"):
        if k in arrays:
            print(f"  {k}: {np.asarray(arrays[k]).shape}")
    print("llm_layers:", llm_layers)

    layers = args.layers or [l for l in (8, 12, 14, 16, 18, 20, 24, 28) if l in llm_layers]
    rep = analyze(args.run_dir, args.site, layers, args.activations, args.cells,
                  args.n_folds, args.ranks)
    out = args.out or (args.run_dir / "pitch_representation" / "drift_geometry.json")
    out.write_text(json.dumps(rep, indent=2))
    print(f"\nwrote {out}")
    # concise console summary
    print(f"\nsite={rep['site']}  n_items={rep['n_items']}")
    print(f"behavior heuristic brittle items: {rep['behavior_heuristic']['n_brittle_items']}")
    print(f"behavior reviewed  brittle items: {rep['behavior_reviewed']['n_brittle_items']}")
    for layer, m in rep["layers"].items():
        d = m["dissociation"]
        print(f"L{layer}: drift_PR={m['drift_effective_rank_pr']:.1f} "
              f"did_PR={m['did_effective_rank_pr']:.1f} "
              f"multidim_gain={m['multidim_gain_rank23_over_rank1']} "
              f"harm_AUROC={d['harmfulness_auroc']} ref_AUROC={d['refusal_auroc']} "
              f"harm_move={d['mean_harmfulness_move']} ref_move={d['mean_refusal_move']}")


if __name__ == "__main__":
    main()
