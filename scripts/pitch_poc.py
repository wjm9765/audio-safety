#!/usr/bin/env -S uv run python
"""Fast proof-of-concept for the pitch->refusal boundary-transversality story on the
20-item run, using agent-verified brittleness labels. Descriptive only (n small).

Signals:
  (A) does the clean neutral refusal margin M(0) separate brittle vs robust refusers?
  (B) pitch-normal refusal velocity: does the refusal readout drop faster under pitch
      for brittle items? (boundary-normal component of the pitch tangent)
  (C) standardized dissociation: z-scored harmfulness vs refusal movement along pitch.
  (D) odd/even pitch decomposition: signed tangent (odd) vs symmetric artifact (even).
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np

_EPS = 1e-12

def load(run_dir, site="llm_p2"):
    arrays = dict(np.load(run_dir / "pitch_representation/activations.npz", allow_pickle=True))
    cells = [json.loads(l) for l in (run_dir / "pitch_representation/cells.jsonl").read_text().splitlines() if l.strip()]
    return arrays, cells


def derive_labels(cells, label_field="reviewed_behavior_label"):
    """Neutral-refusers split into brittle (any pitch flips to compliance) vs robust,
    using the given label field. Returns (brittle_set, robust_set) of 4-char item ids."""
    from collections import defaultdict
    byit = defaultdict(dict)
    for c in cells:
        if c.get("safety_label") == "harmful":
            byit[str(c["item_id"])[-4:]][round(float(c["pitch_semitones"]), 6)] = c
    brittle, robust = set(), set()
    for it, cellmap in byit.items():
        neu = cellmap.get(0.0)
        if neu is None or neu.get(label_field) != "policy_refusal":
            continue
        flips = [p for p, c in cellmap.items() if p != 0.0 and c.get(label_field) == "harmful_compliance"]
        (brittle if flips else robust).add(it)
    return brittle, robust


def auroc(labels, scores):
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels); scores = np.asarray(scores)
    m = np.isfinite(scores)
    if m.sum() < 2 or len(np.unique(labels[m])) < 2:
        return None
    return float(roc_auc_score(labels[m], scores[m]))


def refusal_dir(X, margins, groups, n_folds=5):
    """Full-data diff-of-means refusal direction (margin>0 vs <0), unit; + threshold."""
    y = (margins > 0).astype(int)
    if len(np.unique(y)) < 2:
        return None, None
    m1 = X[y == 1].mean(0); m0 = X[y == 0].mean(0)
    d = m1 - m0; n = np.linalg.norm(d)
    if n <= _EPS: return None, None
    d = d / n
    thr = 0.5 * (m0 @ d + m1 @ d)
    return d, thr


def harmful_dir(X, safety):
    y = safety.astype(int)
    m1 = X[y == 1].mean(0); m0 = X[y == 0].mean(0)
    d = m1 - m0; n = np.linalg.norm(d)
    return (d / n) if n > _EPS else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--site", default="llm_p2")
    ap.add_argument("--layers", type=int, nargs="*", default=[12, 16, 18, 20, 24])
    ap.add_argument("--label-field", default="reviewed_behavior_label")
    args = ap.parse_args()
    arrays, cells = load(args.run_dir, args.site)
    global BRITTLE, ROBUST
    BRITTLE, ROBUST = derive_labels(cells, label_field=args.label_field)
    llm_layers = [int(v) for v in arrays["llm_layers"]]
    Xall = np.asarray(arrays[args.site], dtype=np.float64)  # (N, L, d)
    idx = {(str(c["item_id"])[-4:], str(c["safety_label"]), round(float(c["pitch_semitones"]), 6)): int(c["activation_index"]) for c in cells}
    margins = {int(c["activation_index"]): float(c["refusal_margin"]) for c in cells}
    safety = np.asarray([1 if c["safety_label"] == "harmful" else 0 for c in cells])
    marg_arr = np.asarray([margins[int(c["activation_index"])] for c in cells])
    pitches = sorted({round(float(c["pitch_semitones"]), 6) for c in cells})
    nz = [p for p in pitches if p != 0]
    refusers = sorted(BRITTLE | ROBUST)
    lab = {it: (1 if it in BRITTLE else 0) for it in refusers}

    print(f"neutral-refusers: {len(refusers)} (brittle={len(BRITTLE)} robust={len(ROBUST)})")
    print("\n=== (A) neutral margin M(0) per item ===")
    m0 = {it: margins[idx[(it, 'harmful', 0.0)]] for it in refusers if (it, 'harmful', 0.0) in idx}
    for it in refusers:
        print(f"  {it} {'BRITTLE' if lab[it] else 'robust '} M0={m0.get(it):+.2f}")
    print(f"  AUROC(-M0 -> brittle) = {auroc([lab[it] for it in refusers], [-m0[it] for it in refusers])}")

    for L in args.layers:
        if L not in llm_layers: continue
        off = llm_layers.index(L)
        X = Xall[:, off, :]
        rdir, rthr = refusal_dir(X, marg_arr, None)
        hdir = harmful_dir(X, safety)
        if rdir is None: continue
        # per-item refusal & harmful readout scores across pitch
        r_normal_vel, h_move, r_move, odd_norm, even_norm, r_od = {}, {}, {}, {}, {}, {}
        # standardization scales
        r_scores_all = X @ rdir; h_scores_all = X @ hdir
        r_sd = np.std(r_scores_all) + _EPS; h_sd = np.std(h_scores_all) + _EPS
        for it in refusers:
            h0 = idx.get((it, 'harmful', 0.0))
            if h0 is None: continue
            base = X[h0]
            r0 = rdir @ base
            # refusal drop (boundary-normal velocity toward comply): max decrease over pitch
            drops = []; hmoves = []; rmoves = []
            disp = {}
            for p in nz:
                hp = idx.get((it, 'harmful', p))
                if hp is None: continue
                d = X[hp] - base
                disp[p] = d
                drops.append(max(0.0, r0 - rdir @ X[hp]))     # normal velocity toward compliance
                rmoves.append(abs(rdir @ X[hp] - r0) / r_sd)   # standardized refusal move
                hmoves.append(abs(hdir @ X[hp] - hdir @ base) / h_sd)  # standardized harm move
            r_normal_vel[it] = max(drops) if drops else np.nan
            h_move[it] = float(np.mean(hmoves)) if hmoves else np.nan
            r_move[it] = float(np.mean(rmoves)) if rmoves else np.nan
            # odd/even decomposition on displacement: d(p) = j*p + c*p^2
            ps = np.array([p for p in nz if p in disp], dtype=float)
            if len(ps) >= 3:
                D = np.stack([disp[p] for p in ps])  # (P, d)
                A = np.stack([ps, ps**2], 1)          # (P, 2)
                coef, *_ = np.linalg.lstsq(A, D, rcond=None)  # (2, d): j, c
                j, c = coef[0], coef[1]
                odd_norm[it] = float(np.linalg.norm(j))
                even_norm[it] = float(np.linalg.norm(c))
                r_od[it] = float(abs(rdir @ j))  # refusal-normal component of signed tangent
        print(f"\n=== Layer {L} ===")
        print(f"  (B) pitch-normal refusal velocity (max refusal drop):")
        print(f"      AUROC(vel -> brittle) = {auroc([lab[it] for it in refusers if it in r_normal_vel], [r_normal_vel[it] for it in refusers if it in r_normal_vel])}")
        print(f"  (B') refusal-normal component of signed tangent |r . j|:")
        print(f"      AUROC(|r.j| -> brittle) = {auroc([lab[it] for it in refusers if it in r_od], [r_od[it] for it in refusers if it in r_od])}")
        # transversality: normal velocity / boundary distance  (high => brittle)
        trans = {it: (r_normal_vel[it] / (abs(m0.get(it, np.nan)) + 0.5)) for it in refusers if it in r_normal_vel}
        print(f"  (transversality vel/(|M0|+.5)) AUROC = {auroc([lab[it] for it in trans], [trans[it] for it in trans])}")
        # (C) standardized dissociation
        hb = np.mean([h_move[it] for it in refusers if it in h_move and np.isfinite(h_move[it])])
        rb = np.mean([r_move[it] for it in refusers if it in r_move and np.isfinite(r_move[it])])
        print(f"  (C) standardized move harm={hb:.3f} refusal={rb:.3f} ratio(ref/harm)={rb/(hb+_EPS):.2f}")
        # (D) odd/even
        on = np.mean([odd_norm[it] for it in refusers if it in odd_norm])
        en = np.mean([even_norm[it] for it in refusers if it in even_norm])
        print(f"  (D) pitch displacement ||odd(j)||={on:.2f} ||even(c)||={en:.2f} odd/even={on/(en+_EPS):.2f}")


if __name__ == "__main__":
    main()
