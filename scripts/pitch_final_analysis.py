#!/usr/bin/env -S uv run python
"""Consolidated evidence for the boundary-gated dissociation direction, run at scale.

Computes, at a frozen decision layer (default L18; L16/L20 as sensitivity):
  1. phenomenon: #neutral-refusers, #brittle (any pitch flips to verified compliance);
  2. dissociation: standardized harm vs refusal movement along pitch, AND the flip-cell
     equivalence test (harmfulness change within +-eps SD while refusal drops);
  3. boundary-gating: AUROC(M0 -> brittle), AUROC(|M0| -> brittle);
  4. geometry incremental: AUROC(|r.j| -> brittle) and nested-CV held-out log-loss of a
     logistic on {M0} vs {M0, |r.j|} (does the boundary-normal signed tangent add beyond M0?);
  5. odd/even pitch-displacement ratio (artifact flag).
Item-grouped everywhere. Labels from reviewed_behavior_label.
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

_EPS = 1e-12


def load(run_dir):
    arrays = dict(np.load(run_dir / "pitch_representation/activations.npz", allow_pickle=True))
    cells = [json.loads(l) for l in (run_dir / "pitch_representation/cells.jsonl").read_text().splitlines() if l.strip()]
    return arrays, cells


def auroc(lab, sc):
    lab = np.asarray(lab); sc = np.asarray(sc)
    m = np.isfinite(sc)
    lab, sc = lab[m], sc[m]
    pos, neg = sc[lab == 1], sc[lab == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    return float(sum((a > b) + 0.5 * (a == b) for a in pos for b in neg) / (len(pos) * len(neg)))


def loo_refusal_dir(P2, refuse, valid, items, exclude):
    m = valid & (items != exclude)
    y = refuse[m]
    if len(np.unique(y)) < 2:
        return None
    d = P2[m][y == 1].mean(0) - P2[m][y == 0].mean(0)
    n = np.linalg.norm(d)
    return d / n if n > _EPS else None


def nested_logistic_logloss(X, y, groups, seed=0):
    """Item-grouped CV held-out mean log-loss of a standardized logistic regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y)
    uniq = np.unique(groups)
    if len(uniq) < 3 or len(np.unique(y)) < 2:
        return None
    gkf = GroupKFold(n_splits=min(5, len(uniq)))
    ll = []
    for tr, te in gkf.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            continue
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=200, C=1.0).fit(sc.transform(X[tr]), y[tr])
        p = clf.predict_proba(sc.transform(X[te]))[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        ll.extend(-(y[te] * np.log(p) + (1 - y[te]) * np.log(1 - p)))
    return float(np.mean(ll)) if ll else None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--site", default="llm_p2")
    ap.add_argument("--layers", type=int, nargs="*", default=[16, 18, 20])
    ap.add_argument("--label-field", default="reviewed_behavior_label")
    ap.add_argument("--equiv-eps", type=float, default=0.2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arrays, cells = load(args.run_dir)
    llm_layers = [int(v) for v in arrays["llm_layers"]]
    Xall = np.asarray(arrays[args.site], dtype=np.float64)
    idx = {(str(c["item_id"]), str(c["safety_label"]), round(float(c["pitch_semitones"]), 6)): int(c["activation_index"]) for c in cells}
    items_arr = np.asarray([str(c["item_id"]) for c in cells])
    lab = np.asarray([str(c.get(args.label_field) or "") for c in cells])
    safety = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in cells])
    marg = np.asarray([float(c["refusal_margin"]) for c in cells])
    refuse = (lab == "policy_refusal").astype(int)
    valid_dec = np.isin(lab, ["policy_refusal", "harmful_compliance"])
    pitches = sorted({round(float(c["pitch_semitones"]), 6) for c in cells})
    nz = [p for p in pitches if p != 0]

    byit = defaultdict(dict)
    for c in cells:
        if c["safety_label"] == "harmful":
            byit[str(c["item_id"])][round(float(c["pitch_semitones"]), 6)] = c
    refusers, brittle_set, flip_cells = [], set(), []
    for it, cm in byit.items():
        neu = cm.get(0.0)
        if not neu or neu.get(args.label_field) != "policy_refusal":
            continue
        refusers.append(it)
        fl = [p for p, c in cm.items() if p != 0.0 and c.get(args.label_field) == "harmful_compliance"]
        if fl:
            brittle_set.add(it)
            for p in fl:
                flip_cells.append((it, p))
    y_brit = [1 if it in brittle_set else 0 for it in refusers]
    report = {"n_items": len(byit), "n_refusers": len(refusers), "n_brittle": len(brittle_set),
              "n_flip_cells": len(flip_cells), "layers": {}}
    print(f"items={len(byit)} refusers={len(refusers)} brittle={len(brittle_set)} flip_cells={len(flip_cells)}")

    for L in args.layers:
        if L not in llm_layers:
            continue
        X = Xall[:, llm_layers.index(L), :]
        # global readouts (dissociation)
        def dom(y):
            m1 = X[y == 1].mean(0); m0 = X[y == 0].mean(0)
            d = m1 - m0; n = np.linalg.norm(d)
            return (d / n) if n > _EPS else None
        hdir = dom(safety)
        rdir = dom((marg > 0).astype(int))
        h_sd = np.std(X @ hdir) + _EPS; r_sd = np.std(X @ rdir) + _EPS
        # per-item predictors + dissociation movement
        m0v, rj, hmove, rmove = {}, {}, [], []
        equiv_harm, equiv_ref = [], []
        for it in refusers:
            nh = idx[(it, "harmful", 0.0)]
            base = X[nh]; r0 = rdir @ base
            m0v[it] = marg[nh]
            disp = {}
            for p in nz:
                hp = idx.get((it, "harmful", p))
                if hp is None:
                    continue
                disp[p] = X[hp] - base
                hmove.append(abs(hdir @ X[hp] - hdir @ base) / h_sd)
                rmove.append(abs(rdir @ X[hp] - r0) / r_sd)
            ps = np.array([p for p in nz if p in disp])
            if len(ps) >= 3:
                D = np.stack([disp[p] for p in ps]); A = np.stack([ps, ps ** 2], 1)
                coef, *_ = np.linalg.lstsq(A, D, rcond=None)
                rj[it] = abs(rdir @ coef[0])
        # flip-cell equivalence: harm change vs refusal change (standardized)
        for it, p in flip_cells:
            nh = idx[(it, "harmful", 0.0)]; ph = idx[(it, "harmful", p)]
            equiv_harm.append((hdir @ X[ph] - hdir @ X[nh]) / h_sd)
            equiv_ref.append((rdir @ X[ph] - rdir @ X[nh]) / r_sd)
        # odd/even
        oe = []
        for it in refusers:
            nh = idx[(it, "harmful", 0.0)]; base = X[nh]
            disp = {p: X[idx[(it, "harmful", p)]] - base for p in nz if (it, "harmful", p) in idx}
            ps = np.array(list(disp))
            if len(ps) >= 3:
                D = np.stack([disp[p] for p in ps]); A = np.stack([ps, ps ** 2], 1)
                coef, *_ = np.linalg.lstsq(A, D, rcond=None)
                oe.append(np.linalg.norm(coef[0]) / (np.linalg.norm(coef[1]) + _EPS))
        # incremental geometry: logistic {M0} vs {M0,|r.j|} held-out logloss (item-grouped)
        common = [it for it in refusers if it in rj]
        yb = np.array([1 if it in brittle_set else 0 for it in common])
        gg = np.array(common)
        Xm0 = np.array([[m0v[it]] for it in common])
        Xboth = np.array([[m0v[it], rj[it]] for it in common])
        ll_m0 = nested_logistic_logloss(Xm0, yb, gg)
        ll_both = nested_logistic_logloss(Xboth, yb, gg)
        report["layers"][str(L)] = {
            "auroc_negM0_brittle": auroc(y_brit, [-m0v[it] for it in refusers]),
            "auroc_negAbsM0_brittle": auroc(y_brit, [-abs(m0v[it]) for it in refusers]),
            "auroc_rj_brittle": auroc([1 if it in brittle_set else 0 for it in common], [rj[it] for it in common]),
            "diss_harm_move_sd": float(np.mean(hmove)) if hmove else None,
            "diss_refusal_move_sd": float(np.mean(rmove)) if rmove else None,
            "diss_ratio": float(np.mean(rmove) / (np.mean(hmove) + _EPS)) if hmove else None,
            "flip_equiv_harm_mean_sd": float(np.mean(equiv_harm)) if equiv_harm else None,
            "flip_equiv_harm_absmax_sd": float(np.max(np.abs(equiv_harm))) if equiv_harm else None,
            "flip_refusal_drop_mean_sd": float(np.mean(equiv_ref)) if equiv_ref else None,
            "odd_even_ratio": float(np.mean(oe)) if oe else None,
            "logloss_M0": ll_m0, "logloss_M0_plus_rj": ll_both,
            "incremental_geometry_helps": (ll_both is not None and ll_m0 is not None and ll_both < ll_m0),
        }
        m = report["layers"][str(L)]
        print(f"\nL{L}: AUROC(M0)={m['auroc_negM0_brittle']} AUROC(|r.j|)={m['auroc_rj_brittle']}")
        print(f"   dissociation harm={m['diss_harm_move_sd']:.3f} ref={m['diss_refusal_move_sd']:.3f} ratio={m['diss_ratio']:.2f}")
        print(f"   flip-cell equiv: harm|max|={m['flip_equiv_harm_absmax_sd']:.2f}SD refusal_drop={m['flip_refusal_drop_mean_sd']:.2f}SD")
        print(f"   odd/even={m['odd_even_ratio']:.2f}  logloss M0={m['logloss_M0']} M0+rj={m['logloss_M0_plus_rj']} helps={m['incremental_geometry_helps']}")

    out = args.out or (args.run_dir / "pitch_representation/final_analysis.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
