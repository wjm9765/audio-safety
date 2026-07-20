#!/usr/bin/env -S uv run python
"""Run 11a analysis (CPU): is HARMFULNESS preserved under the phase attack, or degraded?

Decides the bypass-vs-degradation fork from the captured states/margins:
  - Fit a harmfulness direction r_H = difference-in-means(clean_harmful - clean_benign) per
    (layer, site), CLEAN only, item cross-fit.
  - IN-condition AUROC: r_H separates clean_harmful vs clean_benign (sanity).
  - TRANSFER AUROC: does the SAME clean-fit r_H still separate ATTACK_harmful vs clean_benign?
    (is harmful intent still linearly available after the attack?)
  - Occupancy: paired <h, r_H> clean_harmful vs attack_harmful (per item). Preserved => bypass;
    dropped => harmfulness degraded.
  - Behavioral harmfulness recognition: forced-choice H_harm (Yes-No) clean vs attack, and the
    refusal margin M clean vs attack.
  - Readout orthogonal decomposition: split the paired attack displacement into its r_H
    (harmfulness) component vs the residual, at the readout token.

VERDICT: BYPASS-consistent if harmfulness (transfer AUROC, occupancy, H_harm) stays ~intact while
M erodes; DEGRADATION if harmfulness drops with M. Reads <run>/dissociation/{states_*.npz,
meta_*.jsonl}; writes <run>/dissociation/dissociation_report.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    ranks = allv.argsort().argsort().astype(float) + 1.0
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[8, 12, 16, 18, 20, 24, 28])
    ap.add_argument("--sites", nargs="+", default=["readout", "audiomean"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d = args.run_dir / "dissociation"
    meta = []
    for mf in sorted(d.glob("meta_*.jsonl")):
        meta += [json.loads(x) for x in mf.read_text().splitlines() if x.strip()]
    npz = {}
    for sf in sorted(d.glob("states_*.npz")):
        z = np.load(sf)
        npz.update({k: z[k] for k in z.files})
    by_key = {m["key"]: m for m in meta}

    def stack(cond_prefix, layer, site):
        keys = [m["key"] for m in meta if m["condition"] == cond_prefix]
        if not keys:
            return np.zeros((0, 0)), [], []
        X = np.stack([npz[f"{k}|L{layer}|{site}"] for k in keys]).astype(np.float64)
        items = [by_key[k]["item_id"] for k in keys]
        return X, items, keys

    from audio_safety.evaluation.conversion_probe import cross_fit_dim, item_grouped_folds  # noqa: F401

    report = {"layers": {}, "behavioral": {}}
    # ---- behavioral margins (condition-level) ----
    for cond in ["clean_harmful", "attack_harmful", "clean_benign"]:
        ms = [m["M"] for m in meta if m["condition"] == cond]
        hh = [m["H_harm"] for m in meta if m["condition"] == cond]
        report["behavioral"][cond] = {
            "n": len(ms), "M_mean": float(np.mean(ms)), "M_refuse_rate": float(np.mean(np.array(ms) > 0)),
            "H_harm_mean": float(np.mean(hh)), "H_harm_judged_harmful_rate": float(np.mean(np.array(hh) > 0)),
        }
    # paired M / H_harm clean_harmful vs attack_harmful (by item)
    ch = {m["item_id"]: m for m in meta if m["condition"] == "clean_harmful"}
    pairsM, pairsH = [], []
    for m in meta:
        if m["condition"] == "attack_harmful" and m["item_id"] in ch:
            c = ch[m["item_id"]]
            pairsM.append((c["M"], m["M"])); pairsH.append((c["H_harm"], m["H_harm"]))
    pairsM, pairsH = np.array(pairsM, dtype=float).reshape(-1, 2), np.array(pairsH, dtype=float).reshape(-1, 2)
    if len(pairsM):
        report["behavioral"]["paired_attack_minus_clean"] = {
            "n": int(len(pairsM)),
            "dM_mean": float(np.mean(pairsM[:, 1] - pairsM[:, 0])),
            "dH_harm_mean": float(np.mean(pairsH[:, 1] - pairsH[:, 0])),
            "H_harm_clean_judged": float(np.mean(pairsH[:, 0] > 0)),
            "H_harm_attack_judged": float(np.mean(pairsH[:, 1] > 0)),
        }
    else:
        report["behavioral"]["paired_attack_minus_clean"] = {"n": 0}

    for L in args.layers:
        report["layers"][str(L)] = {}
        for site in args.sites:
            Xh, ih, _ = stack("clean_harmful", L, site)
            Xb, ib, _ = stack("clean_benign", L, site)
            Xa, ia, _ = stack("attack_harmful", L, site)
            if not (len(Xh) and len(Xb) and len(Xa)):
                report["layers"][str(L)][site] = {"skipped": "a condition is empty"}
                continue
            # r_H direction (clean harmful - clean benign), full-data unit
            rH = _unit(Xh.mean(0) - Xb.mean(0))
            # in-condition cross-fit AUROC (harmful vs benign, clean)
            X = np.concatenate([Xh, Xb]); mask = np.array([True] * len(Xh) + [False] * len(Xb))
            items = ih + ib
            proj = cross_fit_dim(X, mask, items, k=5, seed=args.seed)
            auroc_in = _auroc(proj[mask], proj[~mask])
            # transfer: clean-fit rH separates ATTACK harmful vs clean benign?
            pa, pb = Xa @ rH, Xb @ rH
            auroc_transfer = _auroc(pa, pb)
            # occupancy paired clean_harmful vs attack_harmful
            ch_occ = {it: (Xh[k] @ rH) for k, it in enumerate(ih)}
            occ_pairs = [(ch_occ[it], Xa[k] @ rH) for k, it in enumerate(ia) if it in ch_occ]
            occ_pairs = np.array(occ_pairs)
            occ = {
                "clean_harmful_mean": float(np.mean([ch_occ[i] for i in ch_occ])),
                "attack_harmful_mean": float(np.mean(occ_pairs[:, 1])) if len(occ_pairs) else float("nan"),
                "benign_mean": float(np.mean(pb)),
                "paired_attack_minus_clean": float(np.mean(occ_pairs[:, 1] - occ_pairs[:, 0])) if len(occ_pairs) else float("nan"),
                "preserved_frac": float(np.mean(occ_pairs[:, 1] > (np.mean(pb) + occ_pairs[:, 0]) / 2)) if len(occ_pairs) else float("nan"),
            }
            entry = {"auroc_harmful_vs_benign_clean": auroc_in,
                     "auroc_transfer_attack_vs_benign": auroc_transfer,
                     "harmfulness_occupancy": occ}
            # readout orthogonal decomposition of the attack displacement
            if site == "readout":
                disp = np.stack([Xh[ih.index(it)] - Xa[k] for k, it in enumerate(ia) if it in ih])
                comp_rH = disp @ rH  # signed harmfulness-axis component of (clean-attack)
                energy_rH = float(np.mean(comp_rH**2))
                energy_tot = float(np.mean(np.sum(disp**2, axis=1)))
                entry["displacement_decomp"] = {
                    "n": int(len(disp)),
                    "frac_energy_along_rH": energy_rH / (energy_tot + 1e-12),
                    "mean_rH_component": float(np.mean(comp_rH)),
                }
            report["layers"][str(L)][site] = entry

    (d / "dissociation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    # ---- console summary + verdict heuristic ----
    b = report["behavioral"]["paired_attack_minus_clean"]
    if b.get("n"):
        print(f"behavioral: dM(attack-clean)={b['dM_mean']:+.2f}  dH_harm={b['dH_harm_mean']:+.2f}  "
              f"judged-harmful clean={b['H_harm_clean_judged']:.2f} attack={b['H_harm_attack_judged']:.2f}")
    for L in args.layers:
        r = report["layers"][str(L)].get("readout", {})
        if r and "auroc_harmful_vs_benign_clean" in r:
            occ = r["harmfulness_occupancy"]
            print(f"L{L} readout: AUROC(harm|clean)={r['auroc_harmful_vs_benign_clean']:.3f} "
                  f"AUROC_transfer(attack)={r['auroc_transfer_attack_vs_benign']:.3f} "
                  f"occ dAttack={occ['paired_attack_minus_clean']:+.2f} "
                  f"frac_disp_along_rH={r.get('displacement_decomp',{}).get('frac_energy_along_rH',float('nan')):.3f}")
    print(f"\nwrote {d}/dissociation_report.json")
    print("READ: harmfulness PRESERVED under attack (transfer AUROC ~ clean AUROC, occ ~unchanged, "
          "H_harm judged-harmful stays high) while M erodes => BYPASS-consistent; harmfulness DROPS "
          "with M => DEGRADATION (perception/decoding).")


if __name__ == "__main__":
    main()
