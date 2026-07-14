#!/usr/bin/env -S uv run python
"""Dose-response monotonicity (CPU, item-clustered per Codex): as STFT phase goes from
locked (alpha=0) to incoherent (alpha=1) on the SAME pitched magnitude, do margin
erosion, L18 refusal-axis displacement, flip rate, and decoding-failure rate rise
monotonically? Item-level Spearman(alpha, .) + item-bootstrap CIs on per-alpha means.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

SRC_RUN = "run5_20260714_0308_pitch_n150"
MAIN_RUN = "run7_20260714_phase_frontend"


def dim_dir(P2, y):
    m = np.isin(y, [0, 1])
    d = P2[m][y[m] == 1].mean(0) - P2[m][y[m] == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-12)


def item_boot(items, per_item, n=2000, seed=0):
    rng = np.random.RandomState(seed)
    ks = list(items)
    vals = [np.mean([per_item[k] for k in np.array(ks)[rng.choice(len(ks), len(ks), True)]]) for _ in range(n)]
    return float(np.percentile(vals, 5)), float(np.percentile(vals, 95))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data"))
    ap.add_argument("--src-run", default=SRC_RUN)
    ap.add_argument("--main-run", default=MAIN_RUN)
    ap.add_argument("--layer", type=int, default=18)
    args = ap.parse_args()
    from scipy.stats import spearmanr

    # frozen refusal dir from run5 L18
    sd = args.data_dir / "outputs" / args.src_run / "pitch_representation"
    s = dict(np.load(sd / "activations.npz", allow_pickle=True))
    sl = [int(v) for v in s["llm_layers"]]
    sP2 = np.asarray(s["llm_p2"], np.float64)[:, sl.index(args.layer), :]
    sc = [json.loads(l) for l in (sd / "cells.jsonl").read_text().splitlines() if l.strip()]
    slab = np.asarray([str(c.get("reviewed_behavior_label") or "") for c in sc])
    r = dim_dir(sP2, np.where(slab == "policy_refusal", 1, np.where(slab == "harmful_compliance", 0, -1)))

    # neutral margins per item from main run
    mc = [json.loads(l) for l in (args.data_dir / "outputs" / args.main_run / "pitch_frontend/cells.jsonl").read_text().splitlines() if l.strip()]
    neu_margin = {c["item_id"]: c["refusal_margin"] for c in mc if c["condition"] == "neutral"}

    dd = args.run_dir / "pitch_dose"
    cells = [json.loads(l) for l in (dd / "cells.jsonl").read_text().splitlines() if l.strip()]
    a = dict(np.load(dd / "activations.npz", allow_pickle=True))
    al = [int(v) for v in a["llm_layers"]]
    P2 = np.asarray(a["llm_p2"], np.float64)[:, al.index(args.layer), :]
    by = {(c["item_id"], round(float(c["sign"]), 6), round(float(c["alpha"]), 4)): c for c in cells}
    items = sorted({c["item_id"] for c in cells})
    signs = sorted({round(float(c["sign"]), 6) for c in cells})
    alphas = sorted({round(float(c["alpha"]), 4) for c in cells})
    is_flip = lambda c: c.get("reviewed_behavior_label") == "harmful_compliance"
    is_dec = lambda c: c.get("reviewed_behavior_label") == "decoding_failure"

    print(f"items={len(items)} signs={signs} alphas={alphas}")
    print(f"{'alpha':>6} {'flip':>7} {'decfail':>8} {'margin_eros':>18} {'refusal_disp(vs a0)':>22}")
    per_alpha = {}
    for a_ in alphas:
        flip_pi, dec_pi, eros_pi, disp_pi = {}, {}, {}, {}
        for it in items:
            fs, ds, es, dp = [], [], [], []
            for sg in signs:
                c = by.get((it, sg, a_))
                if not c:
                    continue
                fs.append(is_flip(c)); ds.append(is_dec(c))
                if it in neu_margin:
                    es.append(neu_margin[it] - c["refusal_margin"])
                c0 = by.get((it, sg, 0.0))
                if c0:
                    dp.append(float(r @ (P2[c["activation_index"]] - P2[c0["activation_index"]])))
            if fs: flip_pi[it] = float(np.mean(fs))
            if ds: dec_pi[it] = float(np.mean(ds))
            if es: eros_pi[it] = float(np.mean(es))
            if dp: disp_pi[it] = float(np.mean(dp))
        er_lo, er_hi = item_boot(list(eros_pi), eros_pi)
        dp_lo, dp_hi = item_boot(list(disp_pi), disp_pi)
        per_alpha[a_] = {"flip": np.mean(list(flip_pi.values())), "dec": np.mean(list(dec_pi.values())),
                         "eros": np.mean(list(eros_pi.values())), "disp": np.mean(list(disp_pi.values())),
                         "eros_pi": eros_pi, "disp_pi": disp_pi, "flip_pi": flip_pi}
        print(f"{a_:>6.2f} {per_alpha[a_]['flip']:>7.3f} {per_alpha[a_]['dec']:>8.3f} "
              f"{per_alpha[a_]['eros']:>+8.3f}[{er_lo:+.2f},{er_hi:+.2f}] {per_alpha[a_]['disp']:>+10.3f}[{dp_lo:+.2f},{dp_hi:+.2f}]")

    # item-level Spearman(alpha, metric): does each item trend monotonically?
    def item_spearman(metric):
        rhos = []
        for it in items:
            xs = [a_ for a_ in alphas if it in per_alpha[a_][metric + "_pi"]]
            ys = [per_alpha[a_][metric + "_pi"][it] for a_ in xs]
            if len(set(ys)) > 1 and len(xs) >= 3:
                rhos.append(spearmanr(xs, ys).statistic)
        return np.array(rhos)
    for metric, name in [("eros", "margin erosion"), ("disp", "refusal-axis displacement (more negative=off refusal)")]:
        rh = item_spearman(metric)
        lo, hi = item_boot(list(range(len(rh))), {i: rh[i] for i in range(len(rh))}) if len(rh) else (float("nan"), float("nan"))
        sign_note = "(expect >0)" if metric == "eros" else "(expect <0)"
        print(f"\n[monotonicity] item-level Spearman(alpha, {name}) {sign_note}: "
              f"mean={np.nanmean(rh):+.3f} [90% {lo:+.3f},{hi:+.3f}] median={np.nanmedian(rh):+.3f} n_items={len(rh)} "
              f"frac_correct_sign={np.mean(rh>0 if metric=='eros' else rh<0):.2f}")


if __name__ == "__main__":
    main()
