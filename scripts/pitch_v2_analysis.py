#!/usr/bin/env -S uv run python
"""Reviewer-hardened analysis for the pitch refusal-transport direction.

Adds, over pitch_final_analysis:
  - item-bootstrap 90% CIs on every headline AUROC and on the dissociation ratio;
  - odd vs even split of the boundary-normal tangent predictor (|r.j_odd| vs |r.j_even|)
    so the signal is attributed to signed transport, not a symmetric artifact;
  - a boundary-referenced (d'-normalized) dissociation metric so "harm barely moves"
    is not just a saturation artifact of a near-ceiling probe.
Item-grouped; labels from reviewed_behavior_label. Frozen layer default L18.
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
    from scipy.stats import rankdata
    lab = np.asarray(lab, float); sc = np.asarray(sc, float)
    m = np.isfinite(sc); lab, sc = lab[m], sc[m]
    n1 = int((lab == 1).sum()); n0 = int((lab == 0).sum())
    if n1 == 0 or n0 == 0:
        return None
    r = rankdata(sc)
    return float((r[lab == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def boot_ci(items, lab_of, score_of, stat="auroc", n=2000, seed=0):
    """Item-level bootstrap CI. items: list of ids; lab_of/score_of: dict id->value."""
    rng = np.random.RandomState(seed)
    ids = list(items)
    vals = []
    for _ in range(n):
        samp = rng.choice(len(ids), len(ids), replace=True)
        sids = [ids[i] for i in samp]
        lab = [lab_of[i] for i in sids]; sc = [score_of[i] for i in sids]
        if stat == "auroc":
            v = auroc(lab, sc)
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return [float(np.percentile(vals, 5)), float(np.percentile(vals, 95))]


def dprime(scores, y):
    y = np.asarray(y); s = np.asarray(scores)
    p, n = s[y == 1], s[y == 0]
    if len(p) < 2 or len(n) < 2:
        return _EPS
    pooled = np.sqrt(0.5 * (p.var() + n.var())) + _EPS
    return abs(p.mean() - n.mean()) / pooled


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="*", default=[18])
    ap.add_argument("--label-field", default="reviewed_behavior_label")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arrays, cells = load(args.run_dir)
    llm_layers = [int(v) for v in arrays["llm_layers"]]
    Xall = np.asarray(arrays["llm_p2"], dtype=np.float64)
    idx = {(str(c["item_id"]), str(c["safety_label"]), round(float(c["pitch_semitones"]), 6)): int(c["activation_index"]) for c in cells}
    lab = np.asarray([str(c.get(args.label_field) or "") for c in cells])
    safety = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in cells])
    marg = np.asarray([float(c["refusal_margin"]) for c in cells])
    pitches = sorted({round(float(c["pitch_semitones"]), 6) for c in cells})
    nz = [p for p in pitches if p != 0.0]

    byit = defaultdict(dict)
    for c in cells:
        if c["safety_label"] == "harmful":
            byit[str(c["item_id"])][round(float(c["pitch_semitones"]), 6)] = c
    refusers, brittle = [], set()
    for it, cm in byit.items():
        neu = cm.get(0.0)
        if not neu or neu.get(args.label_field) != "policy_refusal":
            continue
        refusers.append(it)
        if any(p != 0.0 and c.get(args.label_field) == "harmful_compliance" for p, c in cm.items()):
            brittle.add(it)
    labof = {it: (1 if it in brittle else 0) for it in refusers}
    report = {"n_items": len(byit), "n_refusers": len(refusers), "n_brittle": len(brittle), "layers": {}}
    print(f"items={len(byit)} refusers={len(refusers)} brittle={len(brittle)}")

    for L in args.layers:
        if L not in llm_layers:
            continue
        X = Xall[:, llm_layers.index(L), :]
        def dom(y):
            d = X[y == 1].mean(0) - X[y == 0].mean(0); n = np.linalg.norm(d)
            return d / n if n > _EPS else None
        hdir = dom(safety); rdir = dom((marg > 0).astype(int))
        # class separations (d') for boundary-referencing
        hd = dprime(X @ hdir, safety); rd = dprime(X @ rdir, (marg > 0).astype(int))

        m0, rj_odd, rj_even, rj_full, hmove, rmove = {}, {}, {}, {}, {}, {}
        for it in refusers:
            nh = idx[(it, "harmful", 0.0)]; base = X[nh]
            m0[it] = marg[nh]
            disp = {}
            hm, rm = [], []
            for p in nz:
                k = (it, "harmful", p)
                if k in idx:
                    disp[p] = X[idx[k]] - base
                    hm.append(abs(hdir @ X[idx[k]] - hdir @ base))
                    rm.append(abs(rdir @ X[idx[k]] - rdir @ base))
            hmove[it] = float(np.mean(hm)) if hm else np.nan
            rmove[it] = float(np.mean(rm)) if rm else np.nan
            ps = np.array([p for p in nz if p in disp])
            if len(ps) >= 3:
                D = np.stack([disp[p] for p in ps]); A = np.stack([ps, ps ** 2], 1)
                coef, *_ = np.linalg.lstsq(A, D, rcond=None)
                j, cc = coef[0], coef[1]
                rj_odd[it] = abs(rdir @ j); rj_even[it] = abs(rdir @ cc)
                rj_full[it] = abs(rdir @ (D.mean(0)))  # crude full displacement proxy
        common = [it for it in refusers if it in rj_odd]
        # boundary-referenced dissociation: movement per unit class-separation (d')
        harm_ref = {it: hmove[it] / hd for it in refusers if np.isfinite(hmove[it])}
        ref_ref = {it: rmove[it] / rd for it in refusers if np.isfinite(rmove[it])}
        hbar = np.mean([harm_ref[it] for it in harm_ref]); rbar = np.mean([ref_ref[it] for it in ref_ref])

        def auc_ci(scoremap):
            cids = [it for it in refusers if it in scoremap]
            return auroc([labof[it] for it in cids], [scoremap[it] for it in cids]), boot_ci(cids, labof, scoremap)

        a_m0, ci_m0 = auc_ci({it: -m0[it] for it in refusers})
        a_odd, ci_odd = auc_ci(rj_odd)
        a_even, ci_even = auc_ci(rj_even)

        # dissociation ratio bootstrap (boundary-referenced)
        rng = np.random.RandomState(0); ratios = []
        cids = [it for it in refusers if it in harm_ref and it in ref_ref]
        for _ in range(2000):
            s = rng.choice(len(cids), len(cids), replace=True)
            hh = np.mean([harm_ref[cids[i]] for i in s]); rr = np.mean([ref_ref[cids[i]] for i in s])
            ratios.append(rr / (hh + _EPS))
        report["layers"][str(L)] = {
            "harm_dprime": hd, "refusal_dprime": rd,
            "auroc_M0": a_m0, "auroc_M0_ci90": ci_m0,
            "auroc_rj_odd": a_odd, "auroc_rj_odd_ci90": ci_odd,
            "auroc_rj_even": a_even, "auroc_rj_even_ci90": ci_even,
            "diss_harm_perdprime": float(hbar), "diss_refusal_perdprime": float(rbar),
            "diss_ratio_boundaryref": float(rbar / (hbar + _EPS)),
            "diss_ratio_ci90": [float(np.percentile(ratios, 5)), float(np.percentile(ratios, 95))],
        }
        m = report["layers"][str(L)]
        print(f"\nL{L}: harm d'={hd:.2f} refusal d'={rd:.2f}")
        print(f"  AUROC(M0)={a_m0:.2f} CI{ci_m0}")
        print(f"  AUROC(|r.j_odd|)={a_odd:.2f} CI{ci_odd}   AUROC(|r.j_even|)={a_even:.2f} CI{ci_even}")
        print(f"  boundary-ref dissociation: harm/d'={hbar:.3f} refusal/d'={rbar:.3f} ratio={m['diss_ratio_boundaryref']:.2f} CI{m['diss_ratio_ci90']}")

    out = args.out or (args.run_dir / "pitch_representation/v2_analysis.json")
    out.write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
