"""Attack-family residual-subspace geometry (Candidate-4 kill-test).

Pure numpy/scipy; CPU-only and unit-testable. Implements the representational
core of the pre-registered kill-test:

1. Build per-item family displacement vectors
       d_{f,i} = [h(A_f H_i) - h(A_0 H_i)] - [h(A_f B_i) - h(A_0 B_i)]
   (the benign term removes the attack's generic, non-harmful effect).
2. Remove the shared refusal component (unit r_A, or the top-k shared PC across
   families) and whiten by the benign nuisance covariance.
3. Fit rank-k family subspaces (SVD), item-grouped cross-validated.
4. Family-structure gate: nearest-subspace family identifiability vs a
   label-permuted null (necessary, not sufficient).
5. Principal-angle overlap tr(P_f P_g)/min(k_f,k_g) between family subspaces.

The decisive behavioral defense-transfer test lives elsewhere (it needs the
model); this module supplies the geometry those predictions are scored against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EPS = 1e-12


# --------------------------------------------------------------------------
# small linear-algebra helpers
# --------------------------------------------------------------------------
def unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + _EPS)


def orthonormal_basis(matrix: np.ndarray, rank: int) -> np.ndarray:
    """Top-``rank`` left-singular directions of ``matrix`` (rows = samples).

    Returns an (rank, d) orthonormal basis of the dominant row-space directions
    (i.e. the principal directions the sample vectors span). Uncentered SVD:
    the mean displacement direction is itself signal, so we do NOT subtract it.
    """
    m = np.asarray(matrix, dtype=np.float64)
    if m.ndim != 2 or m.shape[0] == 0:
        raise ValueError("matrix must be (n_samples, d) with n_samples > 0")
    # right singular vectors V are the directions in d-space.
    _, _, vt = np.linalg.svd(m, full_matrices=False)
    r = int(min(rank, vt.shape[0]))
    return vt[:r]


def projector(basis: np.ndarray) -> np.ndarray:
    """Orthogonal projection matrix P = B^T B for an orthonormal (k,d) basis."""
    b = np.asarray(basis, dtype=np.float64)
    return b.T @ b


def principal_angle_overlap(basis_f: np.ndarray, basis_g: np.ndarray) -> float:
    """Normalised subspace overlap tr(P_f P_g)/min(k_f,k_g) in [0, 1].

    Equals the mean of squared cosines of the principal angles between the two
    subspaces; 1 iff the smaller subspace is contained in the larger, 0 iff
    orthogonal. Bases must be orthonormal (rows).
    """
    bf = np.asarray(basis_f, dtype=np.float64)
    bg = np.asarray(basis_g, dtype=np.float64)
    k = min(bf.shape[0], bg.shape[0])
    if k == 0:
        return 0.0
    # tr(P_f P_g) = ||B_f B_g^T||_F^2 for orthonormal bases.
    cross = bf @ bg.T
    return float(np.sum(cross * cross) / k)


def shrink_covariance(x: np.ndarray, shrinkage: float = 0.1) -> np.ndarray:
    """Diagonally-loaded covariance for n < d regimes.

    Cov = (1-s) * empirical + s * (mean-variance) * I. ``x`` is (n, d).
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    xc = x - x.mean(axis=0, keepdims=True)
    emp = (xc.T @ xc) / max(n - 1, 1)
    mu = float(np.trace(emp) / emp.shape[0])
    s = float(np.clip(shrinkage, 0.0, 1.0))
    return (1.0 - s) * emp + s * mu * np.eye(emp.shape[0])


def whitening_transform(cov: np.ndarray) -> np.ndarray:
    """Inverse matrix square-root W s.t. W cov W^T = I (symmetric)."""
    cov = np.asarray(cov, dtype=np.float64)
    w, v = np.linalg.eigh(cov)
    w = np.clip(w, _EPS, None)
    return (v / np.sqrt(w)) @ v.T


def item_grouped_folds(item_ids: list[str], k: int, seed: int = 0) -> list[np.ndarray]:
    """Contiguous item-grouped folds (variants of one base item never split)."""
    uniq = sorted(set(item_ids))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    parts = np.array_split(np.array(uniq, dtype=object), k)
    id_to_pos = {iid: i for i, iid in enumerate(item_ids)}
    folds = []
    for part in parts:
        idx = [id_to_pos[i] for i in item_ids if i in set(part.tolist())]
        # keep every row whose item is in this part
        idx = [j for j, iid in enumerate(item_ids) if iid in set(part.tolist())]
        folds.append(np.array(sorted(set(idx)), dtype=int))
    return folds


# --------------------------------------------------------------------------
# displacement construction
# --------------------------------------------------------------------------
@dataclass
class FamilyDisplacements:
    """Per-family displacement matrices at one layer, aligned by item.

    ``harmful[f]`` and ``benign[f]`` are (n_items_f, d) arrays; ``items[f]`` the
    aligned item ids. ``harmful`` is the safety-relevant displacement d_{f,i};
    ``benign`` is the matched benign-side attack effect (the nuisance control).
    """

    harmful: dict[str, np.ndarray]
    benign: dict[str, np.ndarray]
    items: dict[str, list[str]]
    families: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.families:
            self.families = sorted(self.harmful)


def build_displacements(
    cell: dict[tuple[str, str, str], np.ndarray],
    families: list[str],
    items: list[str],
    *,
    clean_family: str = "clean",
) -> FamilyDisplacements:
    """Assemble family displacements from a per-cell activation lookup.

    ``cell[(family, safety_label, item_id)] = h`` (d,). For each attack family f
    and item i present in both harmful+benign & clean+attack cells:
        harmful d_{f,i} = (h[f,H,i]-h[clean,H,i]) - (h[f,B,i]-h[clean,B,i])
        benign  b_{f,i} =  h[f,B,i]-h[clean,B,i]
    """
    harmful: dict[str, np.ndarray] = {}
    benign: dict[str, np.ndarray] = {}
    keep_items: dict[str, list[str]] = {}
    for f in families:
        if f == clean_family:
            continue
        dh, db, iid = [], [], []
        for i in items:
            need = [
                (clean_family, "harmful", i),
                (f, "harmful", i),
                (clean_family, "benign", i),
                (f, "benign", i),
            ]
            if any(k not in cell for k in need):
                continue
            hH = cell[(f, "harmful", i)] - cell[(clean_family, "harmful", i)]
            hB = cell[(f, "benign", i)] - cell[(clean_family, "benign", i)]
            dh.append(hH - hB)
            db.append(hB)
            iid.append(i)
        if dh:
            harmful[f] = np.asarray(dh, dtype=np.float64)
            benign[f] = np.asarray(db, dtype=np.float64)
            keep_items[f] = iid
    return FamilyDisplacements(harmful=harmful, benign=benign, items=keep_items)


# --------------------------------------------------------------------------
# shared-refusal removal + benign whitening
# --------------------------------------------------------------------------
def shared_refusal_basis(
    disp: FamilyDisplacements, mode: str, *, r_a: np.ndarray | None = None, k: int = 1
) -> np.ndarray:
    """Return an orthonormal (m, d) basis of the shared refusal component.

    - ``mode='rA'``: the single frozen refusal axis (requires ``r_a``).
    - ``mode='shared_pc'``: top-k principal components of the *stack of family
      mean displacements* (directions common across families).
    - ``mode='none'``: empty basis.
    """
    if mode == "none":
        d = next(iter(disp.harmful.values())).shape[1]
        return np.zeros((0, d), dtype=np.float64)
    if mode == "rA":
        if r_a is None:
            raise ValueError("mode='rA' requires r_a")
        return unit(r_a)[None, :]
    if mode == "shared_pc":
        means = np.stack([disp.harmful[f].mean(axis=0) for f in disp.families])
        return orthonormal_basis(means, k)
    raise ValueError(f"unknown mode {mode!r}")


def remove_subspace(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Project rows of ``x`` off the row-space of orthonormal ``basis``."""
    x = np.asarray(x, dtype=np.float64)
    if basis.shape[0] == 0:
        return x
    return x - (x @ basis.T) @ basis


@dataclass
class ProcessedDisplacements:
    harmful: dict[str, np.ndarray]
    items: dict[str, list[str]]
    families: list[str]
    whitener: np.ndarray
    shared_basis: np.ndarray


def process_displacements(
    disp: FamilyDisplacements,
    *,
    shared_mode: str = "shared_pc",
    r_a: np.ndarray | None = None,
    shared_k: int = 1,
    whiten: bool = True,
    shrinkage: float = 0.1,
) -> ProcessedDisplacements:
    """Remove shared refusal + whiten by benign nuisance covariance.

    Whitener is estimated on the pooled *benign* attack displacements (the
    nuisance manifold) so that whitening cannot use harmful-specific variance.
    """
    shared = shared_refusal_basis(disp, shared_mode, r_a=r_a, k=shared_k)
    benign_pool = np.concatenate([disp.benign[f] for f in disp.families], axis=0)
    benign_pool = remove_subspace(benign_pool, shared)
    if whiten:
        cov = shrink_covariance(benign_pool, shrinkage)
        w = whitening_transform(cov)
    else:
        d = benign_pool.shape[1]
        w = np.eye(d)
    out: dict[str, np.ndarray] = {}
    for f in disp.families:
        h = remove_subspace(disp.harmful[f], shared)
        out[f] = h @ w.T
    return ProcessedDisplacements(
        harmful=out, items=disp.items, families=disp.families, whitener=w, shared_basis=shared
    )


# --------------------------------------------------------------------------
# family-structure gate: nearest-subspace identifiability vs permuted null
# --------------------------------------------------------------------------
def _fit_subspaces(
    train: dict[str, np.ndarray], rank: int
) -> dict[str, np.ndarray]:
    return {f: orthonormal_basis(train[f], rank) for f in train if train[f].shape[0] >= 1}


def _projection_ratio(v: np.ndarray, basis: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    nv = np.linalg.norm(v) + _EPS
    p = v @ basis.T
    return float(np.linalg.norm(p) / nv)


def nearest_subspace_cv(
    proc: ProcessedDisplacements, *, rank: int = 3, n_folds: int = 5, seed: int = 0
) -> dict:
    """Item-grouped CV nearest-subspace family classification accuracy.

    For each held-out row (true family f), fit rank-k subspaces on the other
    folds' rows per family, classify to the family whose subspace captures the
    largest fraction of the row's norm. Returns accuracy + confusion + per-row.
    """
    families = [f for f in proc.families if proc.harmful[f].shape[0] > 0]
    # global row table
    rows, labels, items = [], [], []
    for fi, f in enumerate(families):
        for j in range(proc.harmful[f].shape[0]):
            rows.append(proc.harmful[f][j])
            labels.append(fi)
            items.append(proc.items[f][j])
    rows = np.asarray(rows)
    labels = np.asarray(labels)
    folds = item_grouped_folds(items, n_folds, seed)
    conf = np.zeros((len(families), len(families)), dtype=int)
    per_row = []
    for fold in folds:
        test_mask = np.zeros(len(rows), dtype=bool)
        test_mask[fold] = True
        train = {}
        for fi, f in enumerate(families):
            sel = (labels == fi) & (~test_mask)
            if sel.sum() >= 1:
                train[f] = rows[sel]
        bases = _fit_subspaces(train, rank)
        order = [f for f in families if f in bases]
        for r in np.where(test_mask)[0]:
            ratios = np.array([_projection_ratio(rows[r], bases[f]) for f in order])
            pred = order[int(np.argmax(ratios))]
            pred_i = families.index(pred)
            conf[labels[r], pred_i] += 1
            per_row.append((items[r], families[labels[r]], pred, float(ratios.max())))
    acc = float(np.trace(conf) / max(conf.sum(), 1))
    return {
        "accuracy": acc,
        "chance": 1.0 / len(families),
        "confusion": conf.tolist(),
        "families": families,
        "n": int(conf.sum()),
        "per_row": per_row,
    }


def family_structure_permutation(
    proc: ProcessedDisplacements,
    *,
    rank: int = 3,
    n_folds: int = 5,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict:
    """Compare observed nearest-subspace accuracy to a family-label-permuted null.

    Labels are permuted ACROSS families within the same item block (so item
    identity and the nuisance structure are preserved; only the family tag is
    scrambled). Returns observed acc, null mean/95pct, and a permutation p-value.
    """
    obs = nearest_subspace_cv(proc, rank=rank, n_folds=n_folds, seed=seed)
    # build the flat table once
    families = obs["families"]
    rows, labels, items = [], [], []
    for fi, f in enumerate(families):
        for j in range(proc.harmful[f].shape[0]):
            rows.append(proc.harmful[f][j])
            labels.append(fi)
            items.append(proc.items[f][j])
    rows = np.asarray(rows)
    labels = np.asarray(labels)
    items_arr = np.asarray(items, dtype=object)
    rng = np.random.default_rng(seed + 1)
    null_acc = []
    # group row indices by item, permute the family labels among rows of an item
    from collections import defaultdict

    by_item = defaultdict(list)
    for idx, it in enumerate(items):
        by_item[it].append(idx)
    for _ in range(n_permutations):
        perm = labels.copy()
        for _it, idxs in by_item.items():
            lab = perm[idxs].copy()
            rng.shuffle(lab)
            perm[idxs] = lab
        # rebuild a ProcessedDisplacements-like structure and score
        permuted = {f: rows[perm == fi] for fi, f in enumerate(families)}
        permuted_items = {f: items_arr[perm == fi].tolist() for fi, f in enumerate(families)}
        pobj = ProcessedDisplacements(
            harmful=permuted, items=permuted_items, families=families,
            whitener=proc.whitener, shared_basis=proc.shared_basis,
        )
        null_acc.append(nearest_subspace_cv(pobj, rank=rank, n_folds=n_folds, seed=seed)["accuracy"])
    null_acc = np.array(null_acc)
    p = float((1 + np.sum(null_acc >= obs["accuracy"])) / (1 + len(null_acc)))
    return {
        "observed_accuracy": obs["accuracy"],
        "chance": obs["chance"],
        "null_mean": float(null_acc.mean()),
        "null_p95": float(np.percentile(null_acc, 95)),
        "p_value": p,
        "confusion": obs["confusion"],
        "families": families,
        "n": obs["n"],
    }


# --------------------------------------------------------------------------
# subspace overlap matrix
# --------------------------------------------------------------------------
def family_overlap_matrix(
    proc: ProcessedDisplacements, *, rank: int = 3
) -> dict:
    """Full-data rank-k subspace + pairwise principal-angle overlap matrix."""
    families = [f for f in proc.families if proc.harmful[f].shape[0] >= 1]
    bases = {f: orthonormal_basis(proc.harmful[f], rank) for f in families}
    m = len(families)
    ov = np.eye(m)
    for a in range(m):
        for b in range(a + 1, m):
            o = principal_angle_overlap(bases[families[a]], bases[families[b]])
            ov[a, b] = ov[b, a] = o
    return {"families": families, "overlap": ov.tolist(), "bases": bases}
