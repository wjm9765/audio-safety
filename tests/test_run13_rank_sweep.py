"""CPU tests for the Run 13 readout rank-sweep numerical core (fit + edits).

Builds a small synthetic Run-12-style capture dir, runs the two CPU scripts, and asserts the
invariants Codex flagged: cross-fit, U_k orthonormal & perp R_H, realized edit harmfulness-
orthogonal, rank-1 reproduces the Run 12 u_s direction, sham norm-matching, and determinism.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SIGNS = {"m3": -3.0, "p3": 3.0}


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv):
    old = sys.argv
    sys.argv = ["prog", *argv]
    try:
        mod.main()
    finally:
        sys.argv = old


def _make_capture(src: Path, *, n_items=24, d=48, seed=0):
    rng = np.random.RandomState(seed)
    cap = src / "capture"
    cap.mkdir(parents=True, exist_ok=True)
    mu_h, mu_b = rng.randn(d) * 0.5, rng.randn(d) * 0.5
    chan = rng.randn(d)  # shared "phase channel" direction
    states, meta, cohort, folds = {}, [], [], {}
    items = [f"item{i:02d}" for i in range(n_items)]
    for i, it in enumerate(items):
        cat = "A" if i % 2 == 0 else "B"
        folds[it] = i % 5
        cH = mu_h + rng.randn(d) * 0.3
        cB = mu_b + rng.randn(d) * 0.3
        for role, clean in (("harmful", cH), ("benign", cB)):
            states[f"{role}|clean|{it}"] = clean.astype(np.float32)
            meta.append({"item_id": it, "role": role, "condition": "clean", "sign": 0,
                         "category": cat, "key": f"{role}|clean|{it}",
                         "M": float(rng.randn()), "H_harm": float(rng.randn())})
            cohort.append({"role": role, "condition": "clean", "item_id": it, "path": f"/x/{role}_clean_{it}.wav"})
            for tag, sgn in SIGNS.items():
                cond = f"pv_standard_{tag}"
                atk = clean + chan * (0.4 * sgn / 3.0) + rng.randn(d) * 0.15
                states[f"{role}|{cond}|{it}"] = atk.astype(np.float32)
                meta.append({"item_id": it, "role": role, "condition": cond, "sign": sgn,
                             "category": cat, "key": f"{role}|{cond}|{it}",
                             "M": float(rng.randn()), "H_harm": float(rng.randn())})
                cohort.append({"role": role, "condition": cond, "item_id": it, "path": f"/x/{role}_{cond}_{it}.wav"})
    np.savez_compressed(cap / "states_all.npz", **states)
    (cap / "meta_all.jsonl").write_text("\n".join(json.dumps(m) for m in meta) + "\n")
    (src / "folds.json").write_text(json.dumps(folds))
    (src / "cohort.jsonl").write_text("\n".join(json.dumps(c) for c in cohort) + "\n")
    return items, folds, states, d


@pytest.fixture()
def built(tmp_path):
    src, run = tmp_path / "run12", tmp_path / "run13"
    items, folds, states, d = _make_capture(src)
    fit = _load("fit_run13_subspaces")
    edits = _load("precompute_run13_edits")
    ranks = ["1", "2", "4"]
    _run(fit, ["--source-run", str(src), "--run-dir", str(run), "--ranks", *ranks, "--seed", "0"])
    _run(edits, ["--source-run", str(src), "--run-dir", str(run), "--ranks", *ranks,
                 "--doses", "0", "1", "--n-sham", "4", "--seed", "0"])
    return src, run, items, folds, states, d


def _S(states, role, cond, it):
    return states[f"{role}|{cond}|{it}"].astype(np.float64)


def test_fit_outputs_and_kH(built):
    _src, run, _items, _folds, _states, _d = built
    assert (run / "subspaces" / "subspaces.npz").exists()
    assert (run / "subspaces" / "geometry.json").exists()
    man = [json.loads(x) for x in (run / "subspaces" / "fit_manifest.jsonl").read_text().splitlines() if x.strip()]
    assert all(m["k_H"] == 1 for m in man)  # default cap => rank-1 harmfulness


def test_U_orthonormal_and_perp_RH(built):
    _src, run, _items, _folds, _states, _d = built
    sub = np.load(run / "subspaces" / "subspaces.npz")
    for k in (1, 2, 4):
        U = sub[f"U|m3|f0|r{k}"]
        R_H = sub["RH|m3|f0"]
        assert np.allclose(U @ U.T, np.eye(k), atol=1e-8)
        assert np.abs(U @ R_H.T).max() < 1e-6


def test_rank1_reproduces_run12_us(built):
    """U_1 must equal unit((I - rH rH^T) mean(dH-dB)) computed the Run 12 way (k_H=1)."""
    src, run, items, folds, states, _d = built
    from sklearn.covariance import LedoitWolf
    fitmod = _load("fit_run13_subspaces")
    sub = np.load(run / "subspaces" / "subspaces.npz")
    f, tag = 0, "m3"
    fit = [it for it in items if folds[it] != f]
    muH = np.mean([_S(states, "harmful", "clean", it) for it in fit], axis=0)
    muB = np.mean([_S(states, "benign", "clean", it) for it in fit], axis=0)
    Xc = np.concatenate([
        np.stack([_S(states, "harmful", "clean", it) - muH for it in fit]),
        np.stack([_S(states, "benign", "clean", it) - muB for it in fit]),
    ])
    W = fitmod._wsqrt_inv(LedoitWolf().fit(Xc).covariance_)
    rH = (W @ (muH - muB)); rH = rH / np.linalg.norm(rH)
    dH = np.stack([W @ (_S(states, "harmful", "clean", it) - _S(states, "harmful", f"pv_standard_{tag}", it)) for it in fit])
    dB = np.stack([W @ (_S(states, "benign", "clean", it) - _S(states, "benign", f"pv_standard_{tag}", it)) for it in fit])
    v = (dH - dB).mean(axis=0)
    u_ref = v - (rH @ v) * rH
    u_ref = u_ref / np.linalg.norm(u_ref)
    U1 = sub["U|m3|f0|r1"][0]
    assert abs(abs(float(U1 @ u_ref)) - 1.0) < 1e-6  # same direction (sign-free)


def test_realized_edit_harmfulness_orthogonal_and_sham_norm_matched(built):
    src, run, items, folds, states, _d = built
    edmod = _load("precompute_run13_edits")
    ed = np.load(run / "edits" / "edits.npz")
    sub = np.load(run / "edits" / "../subspaces/subspaces.npz")
    it = items[0]; f = folds[it]; tag = "m3"; k = 4
    W = sub[f"W|{tag}|f{f}"].astype(np.float64)
    R_H = sub[f"RH|{tag}|f{f}"].astype(np.float64)
    restore = ed[f"{it}|{tag}|r{k}|restore|1.0"].astype(np.float64)
    # realized edit must be harmfulness-orthogonal in the whitened metric
    assert np.abs(R_H @ (W @ restore)).max() < 1e-4 * (np.linalg.norm(W @ restore) + 1e-12)
    # sham edits norm-matched to restore
    rn = np.linalg.norm(restore)
    for j in range(4):
        sh = ed[f"{it}|{tag}|r{k}|sham{j}|1.0"].astype(np.float64)
        assert abs(np.linalg.norm(sh) - rn) < 1e-4 * rn


def test_wrong_item_never_self(built):
    _src, run, _items, _folds, _states, _d = built
    man = [json.loads(x) for x in (run / "edits" / "edits_manifest.jsonl").read_text().splitlines() if x.strip()]
    assert all(m["wrong_item"] != m["item_id"] for m in man)


def test_determinism(built, tmp_path):
    src, run, items, folds, states, _d = built
    run2 = tmp_path / "run13b"
    edits = _load("precompute_run13_edits")
    # reuse the already-fitted subspaces by copying them
    (run2 / "subspaces").mkdir(parents=True)
    for p in (run / "subspaces").glob("*"):
        (run2 / "subspaces" / p.name).write_bytes(p.read_bytes())
    _run(edits, ["--source-run", str(src), "--run-dir", str(run2), "--ranks", "1", "2", "4",
                 "--doses", "0", "1", "--n-sham", "4", "--seed", "0"])
    a = np.load(run / "edits" / "edits.npz")
    b = np.load(run2 / "edits" / "edits.npz")
    assert set(a.files) == set(b.files)
    for key in list(a.files)[:200]:
        assert np.array_equal(a[key], b[key]), f"non-deterministic edit {key}"
    ma = [json.loads(x) for x in (run / "edits" / "edits_manifest.jsonl").read_text().splitlines() if x.strip()]
    mb = [json.loads(x) for x in (run2 / "edits" / "edits_manifest.jsonl").read_text().splitlines() if x.strip()]
    assert [m["wrong_item"] for m in ma] == [m["wrong_item"] for m in mb]
