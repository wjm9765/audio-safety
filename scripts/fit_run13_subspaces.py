#!/usr/bin/env -S uv run python
"""Run 13 fit (CPU): cross-fitted, outcome-blind readout SUBSPACES for the rank sweep.

Generalizes Run 12's rank-1 ``u_s`` to a rank-k subspace ``U_k`` and sweeps k. Reuses Run 12's
captured L18 t_AB states (``--source-run``). Per outer fold f, sign s (FIT items only):

  W       = Sigma_f^-1/2   (Ledoit-Wolf pooled within-class covariance of clean H+B states)
  rH      = unit(W (muH - muB))
  R_H     = mean_anchored_basis({W(cH_i - cB_i)}, k_H)  # harmfulness NUISANCE subspace
  P_H^perp x = x - project(x, R_H)
  z_i     = P_H^perp [ W(cH_i - aH_i) - W(cB_i - aB_i) ] # benign-subtracted attack interaction
  U_k     = mean_anchored_basis({z_i}, k)               # channel subspace per rank k
  B_k     = mean_anchored_basis({ P_H^perp W(cB_i - aB_i) }, k)  # generic benign-channel control

DEFAULT k_H = 1 (--harmfulness-rank-max 1) => R_H row 0 = rH, so U_1 reproduces Run 12's u_s
direction exactly (up to a 1e-12 normalization epsilon). Raising the cap turns R_H into a
multi-dimensional harmfulness nuisance subspace (stronger specificity control, opt-in sensitivity;
NOTE: k_H selection is then in-sample-reconstruction based, an approximation).

Only ``W`` (float32) is stored; ``Winv`` is NOT stored — the edit is derived with solve(W, .) so
the whitening round-trip is self-consistent. Orthonormal bases R_H/U_k/B_k are stored float64.
Shams are regenerated deterministically in precompute_run13_edits.py.

Writes <run>/subspaces/{subspaces.npz, fit_manifest.jsonl, geometry.json, sweep_manifest.json}.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from itertools import combinations
from pathlib import Path

import numpy as np

from audio_safety.pipelines.channel_axis import (
    largest_principal_angle,
    mean_anchored_basis,
    project,
    reconstruction_ratio,
    unit,
)

SIGNS = {-3.0: "m3", 3.0: "p3"}


def _wsqrt_inv(sigma: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.clip(vals, eps * float(vals.max()), None)
    return (vecs * (1.0 / np.sqrt(vals))) @ vecs.T


def _select_kH(G: np.ndarray, cap: int, recon_min: float) -> int:
    """Smallest dim in 1..cap whose (in-sample) reconstruction of G reaches recon_min (else cap).

    Only exercised when cap>1 (multi-dim R_H is an opt-in sensitivity). Approximate; the primary
    run uses cap=1 (k_H=1).
    """
    cap = min(cap, G.shape[0], G.shape[1])
    for k in range(1, cap + 1):
        if reconstruction_ratio(G, mean_anchored_basis(G, k)) >= recon_min:
            return k
    return cap


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=Path, required=True, help="Run 12 dir with capture/, folds.json")
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20, 32, 64])
    ap.add_argument("--harmfulness-rank-max", type=int, default=1)
    ap.add_argument("--harmfulness-recon-min", type=float, default=0.90)
    ap.add_argument("--n-sham", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from sklearn.covariance import LedoitWolf

    cap = args.source_run / "capture"
    meta = []
    for mf in sorted(cap.glob("meta_*.jsonl")):
        meta += [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
    st: dict[str, np.ndarray] = {}
    for sf in sorted(cap.glob("states_*.npz")):
        z = np.load(sf)
        st.update({k: z[k].astype(np.float64) for k in z.files})
    folds = json.loads((args.source_run / "folds.json").read_text())

    def S(role, cond, it):
        return st.get(f"{role}|{cond}|{it}")

    items = sorted({m["item_id"] for m in meta})

    def complete(it):
        need = [("harmful", "clean"), ("benign", "clean")]
        need += [(r, f"pv_standard_{t}") for r in ("harmful", "benign") for t in SIGNS.values()]
        return all(S(r, c, it) is not None for r, c in need)

    items = [it for it in items if complete(it) and it in folds]
    n_folds = max(folds[it] for it in items) + 1
    ranks = sorted(set(args.ranks))

    def interaction(W, R_H, fit_items):
        dH = np.stack([W @ (S("harmful", "clean", it) - S("harmful", f"pv_standard_{TAG}", it)) for it in fit_items])
        dB = np.stack([W @ (S("benign", "clean", it) - S("benign", f"pv_standard_{TAG}", it)) for it in fit_items])
        return (dH - dB) - project(dH - dB, R_H)

    store: dict[str, np.ndarray] = {}
    fit_manifest = []
    ztrain: dict[tuple[str, int], np.ndarray] = {}
    ztest: dict[tuple[str, int], np.ndarray] = {}

    for sign, TAG in SIGNS.items():
        for f in range(n_folds):
            fit = [it for it in items if folds[it] != f]
            heldout = [it for it in items if folds[it] == f]
            muH = np.mean([S("harmful", "clean", it) for it in fit], axis=0)
            muB = np.mean([S("benign", "clean", it) for it in fit], axis=0)
            Xc = np.concatenate([
                np.stack([S("harmful", "clean", it) - muH for it in fit]),
                np.stack([S("benign", "clean", it) - muB for it in fit]),
            ])
            W = _wsqrt_inv(LedoitWolf().fit(Xc).covariance_)
            rH = unit(W @ (muH - muB))
            G = np.stack([W @ (S("harmful", "clean", it) - S("benign", "clean", it)) for it in fit])
            k_H = _select_kH(G, args.harmfulness_rank_max, args.harmfulness_recon_min)
            R_H = mean_anchored_basis(G, k_H)  # (k_H,d) orthonormal rows; row0 == rH direction
            Z = interaction(W, R_H, fit)
            DBp = np.stack([W @ (S("benign", "clean", it) - S("benign", f"pv_standard_{TAG}", it)) for it in fit])
            DBp = DBp - project(DBp, R_H)
            ztrain[(TAG, f)] = Z
            # genuine held-out interaction using the SAME (fold-f-fit) W and R_H
            ztest[(TAG, f)] = interaction(W, R_H, heldout) if heldout else np.empty((0, W.shape[0]))
            store[f"W|{TAG}|f{f}"] = W.astype(np.float32)      # big; float32 like Run 12
            store[f"rH|{TAG}|f{f}"] = rH.astype(np.float64)
            store[f"RH|{TAG}|f{f}"] = R_H.astype(np.float64)
            for k in ranks:
                store[f"U|{TAG}|f{f}|r{k}"] = mean_anchored_basis(Z, k).astype(np.float64)
                store[f"B|{TAG}|f{f}|r{k}"] = mean_anchored_basis(DBp, k).astype(np.float64)
            fit_manifest.append({"tag": TAG, "sign": sign, "fold": int(f), "n_fit": len(fit),
                                 "n_heldout": len(heldout), "k_H": int(k_H)})

    # geometry (reported, NOT gated): genuine held-out reconstruction + cross-fold angle + spectrum
    geometry = {"ranks": ranks, "signs": {}}
    for TAG in SIGNS.values():
        per_rank = {}
        spectrum = np.linalg.svd(ztrain[(TAG, 0)], compute_uv=False)[:20]
        for k in ranks:
            recon_ho = [reconstruction_ratio(ztest[(TAG, f)], store[f"U|{TAG}|f{f}|r{k}"])
                        for f in range(n_folds) if ztest[(TAG, f)].shape[0] > 0]
            recon_train = [reconstruction_ratio(ztrain[(TAG, f)], store[f"U|{TAG}|f{f}|r{k}"])
                           for f in range(n_folds)]
            bases = [store[f"U|{TAG}|f{f}|r{k}"] for f in range(n_folds)]
            angles = [largest_principal_angle(a, b) for a, b in combinations(bases, 2)]
            per_rank[str(k)] = {
                "heldout_recon_mean": round(float(np.mean(recon_ho)), 4) if recon_ho else None,
                "train_recon_mean": round(float(np.mean(recon_train)), 4),
                "cross_fold_angle_rad_median": round(float(np.median(angles)), 4) if angles else None,
                "cross_fold_angle_rad_max": round(float(np.max(angles)), 4) if angles else None,
            }
        geometry["signs"][TAG] = {"per_rank": per_rank,
                                  "z_singular_spectrum_fold0": [round(float(x), 4) for x in spectrum]}

    out = args.run_dir / "subspaces"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "subspaces.npz", **store)
    (out / "fit_manifest.jsonl").write_text("\n".join(json.dumps(m) for m in fit_manifest) + "\n")
    (out / "geometry.json").write_text(json.dumps(geometry, indent=2) + "\n")
    (args.run_dir / "sweep_manifest.json").write_text(json.dumps({
        "run": "run13_readout_rank_sweep", "source_run": str(args.source_run),
        "layer": 18, "token": "first_generation_prelogit",
        "ranks": ranks, "n_folds": int(n_folds), "n_items": len(items),
        "harmfulness_rank_max": args.harmfulness_rank_max,
        "harmfulness_recon_min": args.harmfulness_recon_min,
        "n_sham": args.n_sham, "seed": args.seed, "git_commit": _git_commit(),
    }, indent=2) + "\n")
    kHs = sorted({m["k_H"] for m in fit_manifest})
    print(f"wrote {out}/ ({len(items)} items, {n_folds} folds, ranks {ranks}); k_H in {kHs}")
    print(json.dumps(geometry, indent=2))


if __name__ == "__main__":
    main()
