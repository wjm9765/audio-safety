#!/usr/bin/env -S uv run python
"""Run 12 Phase A (CPU): whitened cross-fit of r_H and the harmful-specific channel axis u_s,
plus the pre-registered instrument-validity gate — evaluated BEFORE any behavioral outcome.

Codex-locked §2. Per outer fold f, per sign s (FIT items only):
  Sigma_f = LedoitWolf pooled within-class covariance of clean states centered within safety class
  W_f = Sigma_f^-1/2
  r_H,f = unit(W_f (mu_H - mu_B))
  d^H_i = W_f (h^C_H,i - h^A_H,i,s),  d^B_i = W_f (h^C_B,i - h^A_B,i,s)
  v = E_fit[d^H - d^B];  q = (I - r_H r_H^T) v;  u_f,s = unit(q)
Validity per sign: every fold ||q||/B >= 0.10 (B = max(||mean d^B||, sqrt(tr Cov(d^B)/n)));
cross-fold signed median pairwise cosine >= 0.80 and min >= 0.50 of the RAW edit directions
e = W_f^-1 u_f,s compared in a common clean-only whitening metric. Fail(either sign) => AMBIGUOUS.

Writes <run>/axis/{axis_report.json, axes.npz (r_H_f,s, u_f,s, e_f,s, W_f, Winv_f per fold/sign)}.
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np

SIGNS = {-3.0: "m3", 3.0: "p3"}


def _wsqrt_inv(sigma: np.ndarray, eps: float = 1e-8):
    vals, vecs = np.linalg.eigh(sigma)
    vals = np.clip(vals, eps * float(vals.max()), None)
    W = (vecs * (1.0 / np.sqrt(vals))) @ vecs.T
    Winv = (vecs * np.sqrt(vals)) @ vecs.T
    return W, Winv


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--signal-min", type=float, default=0.10)
    ap.add_argument("--cos-median-min", type=float, default=0.80)
    ap.add_argument("--cos-min", type=float, default=0.50)
    args = ap.parse_args()

    from sklearn.covariance import LedoitWolf

    cap = args.run_dir / "capture"
    meta = []
    for mf in sorted(cap.glob("meta_*.jsonl")):
        meta += [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
    npz = {}
    for sf in sorted(cap.glob("states_*.npz")):
        z = np.load(sf)
        npz.update({k: z[k].astype(np.float64) for k in z.files})
    folds = json.loads((args.run_dir / "folds.json").read_text())

    def state(role, cond, item):
        return npz.get(f"{role}|{cond}|{item}")

    items = sorted({m["item_id"] for m in meta})
    # keep items with all needed conditions present (technical completeness)
    def complete(it):
        need = [("harmful", "clean"), ("benign", "clean")]
        need += [(r, f"pv_standard_{t}") for r in ("harmful", "benign") for t in SIGNS.values()]
        return all(state(r, c, it) is not None for r, c in need)
    items = [it for it in items if complete(it)]
    n_folds = max(folds.values()) + 1

    # common clean-only whitening metric on ALL clean states (for cross-fold cosine only)
    muH_all = np.mean([state("harmful", "clean", it) for it in items], axis=0)
    muB_all = np.mean([state("benign", "clean", it) for it in items], axis=0)
    Xall = np.concatenate([
        np.stack([state("harmful", "clean", it) - muH_all for it in items]),
        np.stack([state("benign", "clean", it) - muB_all for it in items]),
    ])
    Wc, _ = _wsqrt_inv(LedoitWolf().fit(Xall).covariance_)

    report = {"signs": {}, "n_items": len(items), "n_folds": int(n_folds)}
    axes = {}
    verdict_valid = True
    for sign, tag in SIGNS.items():
        per_fold = []  # (u, e_raw_whitened_common, signal_ratio)
        for f in range(n_folds):
            fit = [it for it in items if folds[it] != f]
            muH = np.mean([state("harmful", "clean", it) for it in fit], axis=0)
            muB = np.mean([state("benign", "clean", it) for it in fit], axis=0)
            Xc = np.concatenate([
                np.stack([state("harmful", "clean", it) - muH for it in fit]),
                np.stack([state("benign", "clean", it) - muB for it in fit]),
            ])
            W, Winv = _wsqrt_inv(LedoitWolf().fit(Xc).covariance_)
            rH = _unit(W @ (muH - muB))
            dH = np.stack([W @ (state("harmful", "clean", it) - state("harmful", f"pv_standard_{tag}", it)) for it in fit])
            dB = np.stack([W @ (state("benign", "clean", it) - state("benign", f"pv_standard_{tag}", it)) for it in fit])
            v = (dH - dB).mean(axis=0)
            q = v - (rH @ v) * rH
            u = _unit(q)
            e_raw = Winv @ u
            B = max(float(np.linalg.norm(dB.mean(axis=0))),
                    float(np.sqrt(np.sum(dB.var(axis=0)) / max(len(fit), 1))))
            ratio = float(np.linalg.norm(q) / (B + 1e-12))
            per_fold.append((u, _unit(Wc @ e_raw), ratio))
            axes[f"rH|{tag}|f{f}"] = rH.astype(np.float32)
            axes[f"u|{tag}|f{f}"] = u.astype(np.float32)
            axes[f"eraw|{tag}|f{f}"] = e_raw.astype(np.float32)
            axes[f"W|{tag}|f{f}"] = W.astype(np.float32)
            axes[f"Winv|{tag}|f{f}"] = Winv.astype(np.float32)
        ratios = [p[2] for p in per_fold]
        ecommon = [p[1] for p in per_fold]
        cosines = [float(a @ b) for a, b in combinations(ecommon, 2)]
        signal_ok = all(r >= args.signal_min for r in ratios)
        stab_ok = (float(np.median(cosines)) >= args.cos_median_min) and (min(cosines) >= args.cos_min)
        ok = bool(signal_ok and stab_ok)
        verdict_valid = verdict_valid and ok
        report["signs"][tag] = {
            "sign": sign, "signal_ratios": [round(r, 3) for r in ratios],
            "signal_min": round(min(ratios), 3), "signal_ok": bool(signal_ok),
            "cross_fold_cos_median": round(float(np.median(cosines)), 3),
            "cross_fold_cos_min": round(float(min(cosines)), 3), "stability_ok": bool(stab_ok),
            "valid": ok,
        }
    report["verdict"] = "INSTRUMENT_VALID" if verdict_valid else "AMBIGUOUS_instrument_collapse"

    out = args.run_dir / "axis"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "axes.npz", **axes)
    (out / "axis_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    print(f"\nVERDICT: {report['verdict']} -> {'proceed to Phase B behavioral arms' if verdict_valid else 'STOP, do not read behavioral outcomes'}")


if __name__ == "__main__":
    main()
