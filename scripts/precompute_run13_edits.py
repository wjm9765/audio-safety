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
from scipy.linalg import lu_factor, lu_solve

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


def _proj_col(M: np.ndarray, Wd: np.ndarray) -> np.ndarray:
    """Whitened-space projection column ``M^T M (W delta)`` for subspace M, with ``W @ delta`` given.

    The raw-space transport is ``W^{-1} @ _proj_col(M, W @ delta)``. Two batchings make this cheap
    WITHOUT changing the arithmetic (same operands, same order, just fewer redundant passes):
      * ``W @ delta`` depends on neither the rank nor M, so it is hoisted per (item,sign).
      * ``lu_solve`` takes a MATRIX rhs, so all 216 arm/rank/sham columns of one (item,sign) are
        solved in a single call.
    Together these collapse ~64.8k reads of the 134MB W / LU factor to ~600 (the run was memory-
    bandwidth bound at ~5 GB/s, i.e. ~61 min; per-column solving was the wall, not the flops).
    """
    return M.T @ (M @ Wd)


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
    wlu_cache: dict[tuple[str, int], object] = {}  # lu_factor(W) per (tag,fold); W has only 2*n_folds distinct values
    sham_cache: dict[tuple[str, int, int], list[np.ndarray]] = {}  # sham bank per (tag,fold,rank) -- item-independent by seed
    for it in items:
        f = folds[it]
        wrong = wrong_of[it]
        for sign, tag in SIGNS.items():
            W = subs[f"W|{tag}|f{f}"]; R_H = subs[f"RH|{tag}|f{f}"]
            Wlu = wlu_cache.get((tag, f))
            if Wlu is None:
                Wlu = wlu_cache[(tag, f)] = lu_factor(W)
            cH, aH = S("harmful", "clean", it), S("harmful", f"pv_standard_{tag}", it)
            cB, aB = S("benign", "clean", it), S("benign", f"pv_standard_{tag}", it)
            base = f"{it}|{tag}"
            # W @ delta depends on neither rank nor subspace -> hoist (was recomputed ~200x per row).
            # W(aH-cH) is the exact IEEE negation of W(cH-aH) (every op is sign-symmetric), so the
            # corrupt column is exactly -(restore column) rather than a second matmul.
            Wd_H = W @ (cH - aH)
            Wd_B = W @ (cB - aB)
            cols: list[np.ndarray] = []
            slots: list[tuple[str, int, int]] = []
            for k in ranks:
                U = subs[f"U|{tag}|f{f}|r{k}"]; B = subs[f"B|{tag}|f{f}|r{k}"]
                assert_ortho(f"U r{k}", U); assert_ortho(f"B r{k}", B)
                if np.abs(U @ R_H.T).max() > 1e-4:
                    raise ValueError(f"U_k not perp R_H (r{k}, {tag}, f{f})")
                p_res = _proj_col(U, Wd_H)
                cols += [p_res, -p_res, _proj_col(U, Wd_B), _proj_col(B, Wd_H)]
                slots += [("restore", k, 0), ("corrupt", k, 0), ("brestore", k, 0), ("generic", k, 0)]
                # The sham seed has NO item term -- bases depend only on (tag,fold,rank), so every
                # item in a fold already gets the SAME bank. Build it once per group with the exact
                # same draw order and reuse (54k basis constructions -> 1.8k; bases bit-identical).
                shams = sham_cache.get((tag, f, k))
                if shams is None:
                    avoid = np.concatenate([U, R_H], axis=0)
                    srng = np.random.RandomState(args.seed + 1000003 * k + 31 * f + (0 if tag == "m3" else 1))
                    shams = []
                    for j in range(args.n_sham):
                        Sk = _random_orthocomplement_basis(k, U.shape[1], avoid, srng)
                        if np.abs(Sk @ avoid.T).max() > 1e-5:
                            raise ValueError(f"sham{j} not perp to (U,R_H) (r{k})")
                        shams.append(Sk)
                    sham_cache[(tag, f, k)] = shams
                for j, Sk in enumerate(shams):
                    cols.append(_proj_col(Sk, Wd_H)); slots.append(("sham", k, j))
            # ONE W^{-1} solve for every arm/rank/sham of this (item,sign) (lu_solve takes a matrix rhs)
            raw_all = lu_solve(Wlu, np.column_stack(cols))
            raw = {s: raw_all[:, i] for i, s in enumerate(slots)}
            # realized harmfulness-orthogonality of the ACTUAL edits, batched over ranks (one W matmul)
            WR = W @ np.column_stack([raw[("restore", k, 0)] for k in ranks])
            for ki, k in enumerate(ranks):
                wr = WR[:, ki]
                realized = float(np.abs(R_H @ wr).max())
                max_realized = max(max_realized, realized)
                if realized > ORTHO_TOL * (np.linalg.norm(wr) + 1e-12):
                    raise ValueError(f"realized restore edit not harmfulness-orthogonal: {realized:.2e}")
                restore1 = raw[("restore", k, 0)]
                restore_norm = float(np.linalg.norm(restore1))
                for lam in args.doses:
                    edits[f"{base}|r{k}|restore|{lam}"] = (lam * restore1).astype(np.float32)
                    edits[f"{base}|r{k}|corrupt|{lam}"] = (lam * raw[("corrupt", k, 0)]).astype(np.float32)
                    edits[f"{base}|r{k}|brestore|{lam}"] = (lam * raw[("brestore", k, 0)]).astype(np.float32)

                def norm_matched(v, _rn=restore_norm):
                    nonlocal n_degen
                    nr = float(np.linalg.norm(v))
                    if nr < DEGEN_TOL * (_rn + 1e-12):
                        n_degen += 1
                        return np.zeros_like(v)  # degenerate projection -> explicit no-op
                    return _rn * (v / nr)

                edits[f"{base}|r{k}|generic|1.0"] = norm_matched(raw[("generic", k, 0)]).astype(np.float32)
                for j in range(args.n_sham):
                    edits[f"{base}|r{k}|sham{j}|1.0"] = norm_matched(raw[("sham", k, j)]).astype(np.float32)
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
