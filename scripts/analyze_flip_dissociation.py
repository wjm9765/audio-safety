#!/usr/bin/env -S uv run python
"""Run 4 §8 direction-finding: sensor(r_H) vs actuator(r_A) dissociation on flips.

DIRECTION-FINDING ONLY (not a §0 gate, not paper-facing). Tests the tonight-hypothesis
from `run4_direction_20260712.md`: on genuine PAP audio refusal->comply flips, is the
harmfulness signal r_H PRESERVED while the causal refusal actuator r_A's occupancy is
ATTENUATED? Uses activations captured by extract_conversion_activations.py for the
neutral (clean) and attack (jb_pap / jb_ica) style cells, plus the frozen r_A axis and
the blinded judged manifest (for flip labels).

Reads, per style run-dir: conversion_probe.activations_file (npz: ch_stack (N,Lh,d) at
content P1 layers, cr_hidden (N,d) at decision P2/layer c_r_layer) + metadata jsonl.

Emits a JSON report + a markdown summary + scatter data (r_H vs r_A) to the attack run-dir.
Judge labels: uses each judge separately AND their consensus (never cherry-picks a judge).

Example:
    ./scripts/analyze_flip_dissociation.py \
        --config configs/experiments/run4_attack_flip.yaml \
        --neutral-run run4_probe_neutral --attack-run run4_probe_jbpap \
        --control-run run4_probe_jbica \
        --axis-artifact /workspace/.../exp1_20260707_1557_allpos_rebuttal_l12nbhd/rdo_axis.npz
"""

import argparse
import json
from pathlib import Path

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.attack_flip import classify
from audio_safety.evaluation.conversion_probe import readout_auroc
from audio_safety.pipelines.rdo_gate import load_axis
from audio_safety.utils.io import load_jsonl, save_json
from audio_safety.utils.paths import resolve_paths, run_output_dir

HARMFUL, BENIGN = "harmful", "benign"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--neutral-run", type=str, required=True)
    p.add_argument("--attack-run", type=str, required=True)
    p.add_argument("--control-run", type=str, default=None, help="e.g. jb_ica negative control")
    p.add_argument("--axis-artifact", type=Path, required=True)
    p.add_argument("--modality", type=str, default="audio")
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--data-dir", type=Path, default=None)
    return p.parse_args()


def _unit(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v)), 1e-12)


def _load_style(run_dir: Path, probe) -> tuple[dict, np.ndarray, np.ndarray, list]:
    data = np.load(run_dir / probe.activations_file, allow_pickle=True)
    meta = load_jsonl(run_dir / probe.metadata_file)
    return data, data["ch_stack"], data["cr_hidden"], meta


def _key(m: dict) -> tuple[str, str]:
    return (str(m["item_id"]), str(m["safety_label"]))


def _judge_classify(judged_rows, model, modality):
    """(item_id, safety_label, style) -> coarse class for one judge, one modality."""
    out = {}
    for r in judged_rows:
        if str(r.get("modality")) != modality:
            continue
        v = (r.get("judge_verdicts") or {}).get(model)
        if v is None:
            continue
        out[(str(r["item_id"]), str(r["safety_label"]), str(r["style"]))] = classify(v)
    return out


def _boot_ci(x: np.ndarray, n_boot: int, seed: int, alpha=0.05):
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(n_boot)])
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def _boot_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, seed: int, alpha=0.05):
    """CI for mean(a) - mean(b) via independent resampling of each group."""
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    d = np.array(
        [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
         for _ in range(n_boot)]
    )
    return float(a.mean() - b.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def _item_folds(item_ids, k, seed):
    uniq = sorted(set(item_ids))
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(uniq))
    fold_of = {uniq[order[i]]: i % k for i in range(len(uniq))}
    return np.array([fold_of[i] for i in item_ids])


def train_clean_score_attacked(feat_neu, harm_neu, items_neu, feat_att, items_att, k, seed):
    """Item-grouped: train DIM(harmful-benign) on CLEAN train items, project held-out
    CLEAN and ATTACKED rows of the same items. Returns (score_neu, score_att) out-of-fold.
    AUROC is shift/scale invariant, so raw projections suffice."""
    fold_neu = _item_folds(items_neu, k, seed)
    item2fold = {it: f for it, f in zip(items_neu, fold_neu)}
    fold_att = np.array([item2fold.get(it, -1) for it in items_att])
    s_neu = np.full(len(feat_neu), np.nan)
    s_att = np.full(len(feat_att), np.nan)
    for f in range(k):
        tr = fold_neu != f
        if not (harm_neu[tr].any() and (~harm_neu[tr]).any()):
            continue
        d = _unit(feat_neu[tr & harm_neu].mean(0) - feat_neu[tr & ~harm_neu].mean(0))
        s_neu[fold_neu == f] = feat_neu[fold_neu == f] @ d
        s_att[fold_att == f] = feat_att[fold_att == f] @ d
    return s_neu, s_att


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    probe = cfg.conversion_probe
    if probe is None:
        raise SystemExit("config has no conversion_probe block")
    paths = resolve_paths(cfg.paths, output_dir=args.output_dir, data_dir=args.data_dir)
    r_a, site = load_axis(args.axis_artifact)
    ra_u = _unit(r_a)

    judged = load_jsonl(paths.data_dir / cfg.conversion_gap.judge.outputs_file)
    judges = list(cfg.conversion_gap.judge.models)
    c_h_layers = [int(x) for x in np.load(
        run_output_dir(paths.output_dir, args.neutral_run) / probe.activations_file,
        allow_pickle=True)["c_h_layers"]]

    styles = {"neutral": args.neutral_run, "attack": args.attack_run}
    if args.control_run:
        styles["control"] = args.control_run

    # Load activations, index by (item_id, safety_label) for the chosen modality.
    store = {}  # style -> {(item,safety): {"cr": float, "ch": (Lh,d)}}
    raw = {}
    for sname, run in styles.items():
        rd = run_output_dir(paths.output_dir, run)
        data, ch, cr, meta = _load_style(rd, probe)
        raw[sname] = (ch, cr, meta)
        idx = {}
        for i, m in enumerate(meta):
            if str(m["modality"]) != args.modality:
                continue
            idx[_key(m)] = {"cr": float(cr[i] @ ra_u), "ch": ch[i], "row": i}
        store[sname] = idx

    report = {
        "run": args.attack_run, "modality": args.modality,
        "r_a_layer": site.layer, "r_a_position": site.position,
        "c_h_layers": c_h_layers, "judges": judges, "styles": list(styles),
        "results": {},
    }

    for attack_style in [s for s in ("attack", "control") if s in store]:
        # judge classes per judge + consensus, for neutral & this attack style (audio)
        style_name = {"attack": "jb_pap-ish", "control": "jb_ica-ish"}[attack_style]
        neu_run_style = None  # style strings live in judged manifest
        # discover actual style strings from metadata
        neu_style = str(raw["neutral"][2][0].get("style"))
        att_style = str(raw[attack_style][2][0].get("style"))

        per_judge = {}
        # r_A occupancy Δ = c_R(attack) - c_R(neutral), per item, harmful & benign
        # baseline sd from clean harmful c_R (to express Δ in SD units)
        harm_items = [k for k in store["neutral"] if k[1] == HARMFUL and k in store[attack_style]]
        ben_items = [k for k in store["neutral"] if k[1] == BENIGN and k in store[attack_style]]
        cr_clean_harm = np.array([store["neutral"][k]["cr"] for k in harm_items])
        sd = max(float(cr_clean_harm.std()), 1e-9)
        dCR_harm = {k: store[attack_style][k]["cr"] - store["neutral"][k]["cr"] for k in harm_items}
        dCR_ben = np.array([store[attack_style][k]["cr"] - store["neutral"][k]["cr"] for k in ben_items])

        for judge in list(judges) + ["consensus"]:
            if judge == "consensus":
                cls_neu = {}
                cls_att = {}
                jm = [_judge_classify(judged, jj, args.modality) for jj in judges]
                for k in harm_items:
                    kn = (k[0], k[1], neu_style); ka = (k[0], k[1], att_style)
                    n_all = [d.get(kn) for d in jm]; a_all = [d.get(ka) for d in jm]
                    cls_neu[k] = "refusal" if all(c == "refusal" for c in n_all) else "mixed"
                    # Consensus attacked class must be able to be "refusal" too, else the
                    # remained (refusal->refusal) set is always empty (reviewer finding 1).
                    if all(c == "comply" for c in a_all):
                        cls_att[k] = "comply"
                    elif all(c == "refusal" for c in a_all):
                        cls_att[k] = "refusal"
                    else:
                        cls_att[k] = "mixed"
            else:
                cj = _judge_classify(judged, judge, args.modality)
                cls_neu = {k: cj.get((k[0], k[1], neu_style)) for k in harm_items}
                cls_att = {k: cj.get((k[0], k[1], att_style)) for k in harm_items}

            flip = [k for k in harm_items if cls_neu.get(k) == "refusal" and cls_att.get(k) == "comply"]
            remained = [k for k in harm_items if cls_neu.get(k) == "refusal" and cls_att.get(k) == "refusal"]

            dflip = np.array([dCR_harm[k] for k in flip]) / sd
            drem = np.array([dCR_harm[k] for k in remained]) / sd
            dd, dd_lo, dd_hi = _boot_diff_ci(dflip, drem, args.n_bootstrap, args.seed)
            f_m, f_lo, f_hi = _boot_ci(dflip, args.n_bootstrap, args.seed)
            r_m, r_lo, r_hi = _boot_ci(drem, args.n_bootstrap, args.seed)
            # harmful-specificity of the r_A shift: harmful-flip Δ vs benign Δ (SD units)
            hs, hs_lo, hs_hi = _boot_diff_ci(dflip, dCR_ben / sd, args.n_bootstrap, args.seed)

            per_judge[judge] = {
                "n_flip": len(flip), "n_remained": len(remained),
                "dCR_flip_sd": [f_m, f_lo, f_hi],
                "dCR_remained_sd": [r_m, r_lo, r_hi],
                "rA_occupancy_double_diff_sd": [dd, dd_lo, dd_hi],
                "rA_shift_harmful_vs_benign_sd": [hs, hs_lo, hs_hi],
            }

        # --- cross-fit r_H (train on CLEAN neutral, eval on attacked), best content layer ---
        # Build aligned arrays over items present in BOTH neutral & attack (harmful+benign)
        both = [k for k in store["neutral"] if k in store[attack_style]]
        harm_mask = np.array([k[1] == HARMFUL for k in both])
        items = [k[0] for k in both]
        rH = {}
        for j, layer in enumerate(c_h_layers):
            feat_neu = np.array([store["neutral"][k]["ch"][j] for k in both])
            feat_att = np.array([store[attack_style][k]["ch"][j] for k in both])
            s_neu, s_att = train_clean_score_attacked(
                feat_neu, harm_mask, items, feat_att, items, probe.n_cross_fit_folds, args.seed)
            ok = ~np.isnan(s_neu)
            auroc_clean = readout_auroc(s_neu[ok], harm_mask[ok].astype(int))
            oka = ~np.isnan(s_att)
            auroc_att = readout_auroc(s_att[oka], harm_mask[oka].astype(int))
            # r_H on the FLIP-harmful vs attacked-benign (does harmfulness survive the flip?)
            # use consensus flip set from the primary judge list intersection
            rH[int(layer)] = {"auroc_clean": auroc_clean, "auroc_attacked": auroc_att}
        report["results"][f"{attack_style}({att_style})"] = {
            "neutral_style": neu_style, "attack_style": att_style,
            "n_harm_matched": len(harm_items), "n_benign_matched": len(ben_items),
            "clean_cR_harm_sd": sd,
            "rA_occupancy": per_judge,
            "rH_cross_fit_auroc": rH,
            "benign_rA_shift_sd_mean": float((dCR_ben / sd).mean()) if len(dCR_ben) else float("nan"),
        }

    out_dir = run_output_dir(paths.output_dir, args.attack_run)
    save_json(report, out_dir / "flip_dissociation_report.json")
    print(json.dumps(report["results"], indent=2, default=str))
    print(f"[dissoc] report -> {out_dir / 'flip_dissociation_report.json'}")


if __name__ == "__main__":
    main()
