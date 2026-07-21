#!/usr/bin/env -S uv run python
"""Run 13 edits (CPU): precompute rank-k projected-transport EDIT VECTORS per test item.

For orthonormal whitened subspace M (k,d), the rank-k projected transport of a clean/attack pair is
a FIXED additive vector at t_AB (host = captured state, prefill-only):

    edit(M, donor, host, lambda) = lambda * W^{-1} @ M.T @ ( M @ ( W @ (donor_read - host_read) ) )

W^{-1} is applied via solve(W, .) using the SINGLE stored W (self-consistent whitening round-trip).
Rank-1 with M=U_1 reproduces Run 12's u_s edit when k_H=1. Each item uses subspaces fitted on its
HELD-OUT fold (folds[item]).

Arms per (item, sign, rank k):
  restore  : M=U_k, host=attack_H, donor=clean_H   (dose curve)
  corrupt  : M=U_k, host=clean_H,  donor=attack_H   (reciprocal, dose curve)
  brestore : M=U_k, host=attack_B, donor=clean_B    (benign over-refusal control, dose curve)
  generic  : M=B_k, host=attack_H, donor=clean_H    (generic-channel control, @1, norm-matched)
  sham0..N : M=random S_k perp U_k,R_H, host=attack_H, donor=clean_H (@1, norm-matched)
Per (item, sign) rank-independent: fullstate donor = clean_H read; wrongitem donor = a DIFFERENT
same-category clean_H read (deterministically drawn; never the same item).

Invariants are ENFORCED (raise before saving): U_k/B_k orthonormal, U_k perp R_H, realized restore
edit is harmfulness-orthogonal (|R_H @ W @ edit| ~ 0), shams orthonormal & perp to (U_k,R_H).

Reads <src>/{capture, folds.json}, <run>/subspaces/subspaces.npz; writes <run>/edits/*.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from audio_safety.pipelines.channel_axis import unit

SIGNS = {-3.0: "m3", 3.0: "p3"}
ORTHO_TOL = 1e-5   # float32-W-aware realized-edit tolerance
DEGEN_TOL = 1e-9   # a projection this small (relative to restore norm) is treated as a no-op


def _random_orthocomplement_basis(k: int, d: int, avoid: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    """Orthonormal (k,d) basis, orthogonal to the rows of orthonormal ``avoid`` (m,d)."""
    if avoid.size and k > d - avoid.shape[0]:
        raise ValueError(f"cannot draw rank-{k} subspace in a {d - avoid.shape[0]}-dim complement")
    G = rng.standard_normal((k, d))
    for _ in range(2):  # project-out then re-project for numerical safety
        if avoid.size:
            G = G - (G @ avoid.T) @ avoid
        Q, _ = np.linalg.qr(G.T)
        G = Q[:, :k].T
    return G


def _transport(M: np.ndarray, W: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Raw-space rank-k projected transport of delta = donor_read - host_read (W^{-1} via solve)."""
    proj = M.T @ (M @ (W @ delta))
    return np.linalg.solve(W, proj)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-run", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 12, 16, 20, 32, 64])
    ap.add_argument("--doses", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    ap.add_argument("--n-sham", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    cap = args.source_run / "capture"
    meta = []
    for mf in sorted(cap.glob("meta_*.jsonl")):
        meta += [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
    st: dict[str, np.ndarray] = {}
    for sf in sorted(cap.glob("states_*.npz")):
        z = np.load(sf)
        st.update({k: z[k].astype(np.float64) for k in z.files})
    folds = json.loads((args.source_run / "folds.json").read_text())
    cat = {m["item_id"]: m.get("category") for m in meta}
    Mcap = {m["key"]: m["M"] for m in meta}
    sub = np.load(args.run_dir / "subspaces" / "subspaces.npz")
    subs = {k: sub[k].astype(np.float64) for k in sub.files}

    def S(role, cond, it):
        return st.get(f"{role}|{cond}|{it}")

    items = sorted({m["item_id"] for m in meta})
    items = [it for it in items
             if it in folds and all(S(r, c, it) is not None for r in ("harmful", "benign")
                                    for c in ("clean", "pv_standard_m3", "pv_standard_p3"))]
    ranks = sorted(set(args.ranks))

    # deterministic wrong-item donor: single seeded RNG over sorted items, never same item
    wrong_rng = np.random.RandomState(args.seed)
    wrong_of = {}
    for it in items:
        same = [j for j in items if cat.get(j) == cat.get(it) and j != it]
        pool = same or [j for j in items if j != it]
        wrong_of[it] = pool[wrong_rng.randint(len(pool))] if pool else it

    def assert_ortho(name, B, tol=1e-6):
        if not np.allclose(B @ B.T, np.eye(B.shape[0]), atol=tol):
            raise ValueError(f"{name} not orthonormal (max dev {np.abs(B @ B.T - np.eye(B.shape[0])).max():.2e})")

    edits: dict[str, np.ndarray] = {}
    manifest = []
    n_degen = 0
    max_realized = 0.0
    for it in items:
        f = folds[it]
        wrong = wrong_of[it]
        for sign, tag in SIGNS.items():
            W = subs[f"W|{tag}|f{f}"]; R_H = subs[f"RH|{tag}|f{f}"]
            cH, aH = S("harmful", "clean", it), S("harmful", f"pv_standard_{tag}", it)
            cB, aB = S("benign", "clean", it), S("benign", f"pv_standard_{tag}", it)
            base = f"{it}|{tag}"
            for k in ranks:
                U = subs[f"U|{tag}|f{f}|r{k}"]; B = subs[f"B|{tag}|f{f}|r{k}"]
                assert_ortho(f"U r{k}", U); assert_ortho(f"B r{k}", B)
                if np.abs(U @ R_H.T).max() > 1e-4:
                    raise ValueError(f"U_k not perp R_H (r{k}, {tag}, f{f})")
                restore1 = _transport(U, W, cH - aH)
                # realized harmfulness-orthogonality of the ACTUAL edit (whitened metric)
                realized = float(np.abs(R_H @ (W @ restore1)).max())
                max_realized = max(max_realized, realized)
                if realized > ORTHO_TOL * (np.linalg.norm(W @ restore1) + 1e-12):
                    raise ValueError(f"realized restore edit not harmfulness-orthogonal: {realized:.2e}")
                corrupt1 = _transport(U, W, aH - cH)
                brestore1 = _transport(U, W, cB - aB)
                restore_norm = float(np.linalg.norm(restore1))
                for lam in args.doses:
                    edits[f"{base}|r{k}|restore|{lam}"] = (lam * restore1).astype(np.float32)
                    edits[f"{base}|r{k}|corrupt|{lam}"] = (lam * corrupt1).astype(np.float32)
                    edits[f"{base}|r{k}|brestore|{lam}"] = (lam * brestore1).astype(np.float32)

                def norm_matched(raw):
                    nonlocal n_degen
                    nr = float(np.linalg.norm(raw))
                    if nr < DEGEN_TOL * (restore_norm + 1e-12):
                        n_degen += 1
                        return np.zeros_like(raw)  # degenerate projection -> explicit no-op
                    return restore_norm * (raw / nr)

                gen_raw = _transport(B, W, cH - aH)
                edits[f"{base}|r{k}|generic|1.0"] = norm_matched(gen_raw).astype(np.float32)
                avoid = np.concatenate([U, R_H], axis=0)
                srng = np.random.RandomState(args.seed + 1000003 * k + 31 * f + (0 if tag == "m3" else 1))
                for j in range(args.n_sham):
                    Sk = _random_orthocomplement_basis(k, U.shape[1], avoid, srng)
                    if np.abs(Sk @ avoid.T).max() > 1e-5:
                        raise ValueError(f"sham{j} not perp to (U,R_H) (r{k})")
                    edits[f"{base}|r{k}|sham{j}|1.0"] = norm_matched(_transport(Sk, W, cH - aH)).astype(np.float32)
            edits[f"{base}|fullstate_donor"] = cH.astype(np.float32)
            edits[f"{base}|wrongitem_donor"] = S("harmful", "clean", wrong).astype(np.float32)
            manifest.append({
                "item_id": it, "sign": sign, "tag": tag, "fold": int(f), "category": cat.get(it),
                "wrong_item": wrong, "wrong_item_same_category": bool(cat.get(wrong) == cat.get(it)),
                "M_attack_H": Mcap[f"harmful|pv_standard_{tag}|{it}"],
                "M_clean_H": Mcap[f"harmful|clean|{it}"],
                "M_attack_B": Mcap[f"benign|pv_standard_{tag}|{it}"],
                "M_clean_B": Mcap[f"benign|clean|{it}"],
            })
        if len(manifest) % 100 == 0:
            print(f"  precomputed {len(manifest)} (item,sign)", flush=True)

    out = args.run_dir / "edits"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "edits.npz", **edits)
    (out / "edits_manifest.jsonl").write_text("\n".join(json.dumps(m) for m in manifest) + "\n")
    print(f"wrote {out}/ ({len(manifest)} (item,sign), ranks {ranks}, {args.n_sham} shams); "
          f"max realized |R_H.W.edit| = {max_realized:.2e}; degenerate norm-matches = {n_degen}")


if __name__ == "__main__":
    main()
