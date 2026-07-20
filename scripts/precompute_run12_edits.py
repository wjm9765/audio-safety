#!/usr/bin/env -S uv run python
"""Run 12 Phase B (CPU): precompute the Mahalanobis rank-1 coordinate EDIT VECTORS per test item.

The intervention h' = h + lambda W^-1 u [u^T W (donor - host)] is, because host = the captured
t_AB state (deterministic, prefill-only), a FIXED additive vector at t_AB:
    edit(lambda) = lambda * c * (Winv @ u),   c = u . (W @ (donor_read - host_read)).
So Phase B applies it with the existing ResidualStreamIntervention(mode='add', token_index=t_AB).
Each item uses the axis fitted on its HELD-OUT fold (folds[item]) — genuine cross-fitting.

Arms precomputed per (item, sign):
  restore   : host=attack_H, donor=clean_H   (does u_s recover refusal?)
  corrupt   : host=clean_H,  donor=attack_H   (reciprocal)
  brestore  : host=attack_B, donor=clean_B    (benign over-refusal control, G4)
  sham x5   : random whitened dirs perp to u and r_H, raw-L2 matched to the restore edit
  full_state donor = clean_H read; wrong_item donor = deranged same-category clean_H read
Also verifies the edit is harmfulness-invariant (|r_H . u| ~ 0 => whitened g_H change ~ 0).

Reads <run>/{axis/axes.npz, capture/states_*.npz, capture/meta_*.jsonl, folds.json};
writes <run>/edits/{edits.npz, edits_manifest.jsonl}.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SIGNS = {-3.0: "m3", 3.0: "p3"}


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--doses", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--n-sham", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cap = args.run_dir / "capture"
    meta = []
    for mf in sorted(cap.glob("meta_*.jsonl")):
        meta += [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
    st = {}
    for sf in sorted(cap.glob("states_*.npz")):
        z = np.load(sf)
        st.update({k: z[k].astype(np.float64) for k in z.files})
    axes = {k: np.load(args.run_dir / "axis" / "axes.npz")[k].astype(np.float64)
            for k in np.load(args.run_dir / "axis" / "axes.npz").files}
    folds = json.loads((args.run_dir / "folds.json").read_text())
    cat = {m["item_id"]: m.get("category") for m in meta}
    Mcap = {m["key"]: m["M"] for m in meta}

    def S(role, cond, it):
        return st.get(f"{role}|{cond}|{it}")

    items = sorted({m["item_id"] for m in meta})
    items = [it for it in items if all(S(r, c, it) is not None for r in ("harmful", "benign")
                                       for c in ("clean", "pv_standard_m3", "pv_standard_p3"))]
    rng = np.random.RandomState(args.seed)
    d = axes[f"u|m3|f0"].shape[0]

    edits, manifest = {}, []
    for it in items:
        f = folds[it]
        # a deranged same-category donor item (frozen) for the wrong-item full-state control
        same = [j for j in items if cat.get(j) == cat.get(it) and j != it]
        wrong = same[rng.randint(len(same))] if same else it
        for sign, tag in SIGNS.items():
            u = axes[f"u|{tag}|f{f}"]; W = axes[f"W|{tag}|f{f}"]; Winv = axes[f"Winv|{tag}|f{f}"]
            rH = axes[f"rH|{tag}|f{f}"]
            aH, cH = S("harmful", f"pv_standard_{tag}", it), S("harmful", "clean", it)
            aB, cB = S("benign", f"pv_standard_{tag}", it), S("benign", "clean", it)
            raw_u = Winv @ u  # raw-space edit direction
            c_restore = float(u @ (W @ (cH - aH)))
            c_corrupt = float(u @ (W @ (aH - cH)))
            c_brestore = float(u @ (W @ (cB - aB)))
            base = f"{it}|{tag}"
            for lam in args.doses:
                edits[f"{base}|restore|{lam}"] = (lam * c_restore * raw_u).astype(np.float32)
                edits[f"{base}|corrupt|{lam}"] = (lam * c_corrupt * raw_u).astype(np.float32)
                edits[f"{base}|brestore|{lam}"] = (lam * c_brestore * raw_u).astype(np.float32)
            restore_norm = float(np.linalg.norm(c_restore * raw_u))  # match sham to this
            # sham: whitened random dirs perp to u and rH, raw-matched norm
            for j in range(args.n_sham):
                g = rng.standard_normal(d)
                g = g - (g @ u) * u - (g @ rH) * rH
                sraw = _unit(Winv @ _unit(g))
                edits[f"{base}|sham{j}|1.0"] = (restore_norm * sraw).astype(np.float32)
            edits[f"{base}|fullstate_donor"] = cH.astype(np.float32)
            edits[f"{base}|wrongitem_donor"] = S("harmful", "clean", wrong).astype(np.float32)
            manifest.append({
                "item_id": it, "sign": sign, "tag": tag, "fold": f, "category": cat.get(it),
                "wrong_item": wrong,
                "M_attack_H": Mcap[f"harmful|pv_standard_{tag}|{it}"],
                "M_clean_H": Mcap[f"harmful|clean|{it}"],
                "M_attack_B": Mcap[f"benign|pv_standard_{tag}|{it}"],
                "M_clean_B": Mcap[f"benign|clean|{it}"],
                "c_restore": c_restore, "restore_edit_norm": restore_norm,
                "rH_dot_u": float(abs(rH @ u)),  # harmfulness-invariance check (~0)
                "attack_H_path": None,  # filled by phaseB from cohort
            })
        if (len(manifest)) % 100 == 0:
            print(f"  precomputed {len(manifest)} (item,sign) arms", flush=True)

    out = args.run_dir / "edits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "edits.npz", **edits)
    (out / "edits_manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    inv = max(m["rH_dot_u"] for m in manifest)
    print(f"wrote {out}/ ({len(manifest)} arms, {len(items)} items); max |rH.u| = {inv:.2e} "
          f"(harmfulness-invariance {'OK' if inv < 1e-4 else 'VIOLATED'})")


if __name__ == "__main__":
    main()
