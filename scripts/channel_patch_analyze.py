#!/usr/bin/env -S uv run python
"""Run 10 STEP 3 analysis (CPU): G1/G4 paired stats + restore-vs-matched-sham + GO/STOP.

Reads <run>/channel_patch/l18_patch.json (written by channel_patch_l18.py) and, per arm,
at the maximum dose, computes:
  - restoration Delta_M = M(attack + restore clean U-coord) - M(attack)   [fixability]
  - corruption  Delta_M = M(clean + inject attack U-coord)  - M(clean)    [necessity]
  - restoration vs the MATCHED SHAM null (same projected-transport operator/support/max-dose
    along random rank-matched subspaces perp to U) -> is the recovery U-SPECIFIC?
with by-item bootstrap CIs (evaluation.conversion_probe.paired_mean_diff_ci). Per-sign runs
have one test row per item, so the paired CI is already item-clustered; a pooled-sign run
must instead resample items (both signs together) -- do NOT pool signs into this script.

VERDICT (registered G1 in the margin form, since flips are few -- Run 7 lineage): GO if some
arm's restoration Delta_M has item-LB > 0 AND beats the MATCHED SHAM null (item-LB > 0). A run
with no matched sham CANNOT be a GO (fail-closed; Codex 2026-07-19). The behavioral 15pp /
flip-specificity (G4), recognition-equivalence-under-patch, and the pv_locked<->pv_standard
specificity contrast are applied on top of this by the analyst -- this script reports the
continuous-margin evidence and the matched-sham specificity only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _paired(a: list[float], b: list[float], *, n_boot: int, alpha: float, seed: int) -> dict:
    from audio_safety.evaluation.conversion_probe import paired_mean_diff_ci

    if not a:
        return {"n": 0}
    return paired_mean_diff_ci(np.asarray(a, float), np.asarray(b, float), n_boot=n_boot, alpha=alpha, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--in-name", default="l18_patch.json")
    ap.add_argument("--out-name", default="l18_analysis.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-no-sham", action="store_true",
                    help="permit reporting without the matched sham (SMOKE ONLY: cannot be a GO)")
    args = ap.parse_args()

    patch_dir = args.run_dir / "channel_patch"
    data = json.loads((patch_dir / args.in_name).read_text())
    results = data["results"]
    if not results:
        raise SystemExit("no test-pair results to analyze")
    dose = max(float(d) for d in data["dose"])
    dose_key = f"{dose}"

    arms = sorted({arm for r in results for arm in r.get("arms", {})})
    ambiguous_by_arm = data.get("rank_ambiguous", {})
    report: dict[str, dict] = {"dose": dose, "arms": {}}
    go = False
    for arm in arms:
        ambiguous = bool(ambiguous_by_arm.get(arm, False))
        rows = [r for r in results if arm in r.get("arms", {})]
        restore = [r["arms"][arm]["restore"][dose_key] for r in rows]
        corrupt = [r["arms"][arm]["corrupt"][dose_key] for r in rows]
        base_att = [r["base_margin_attack"] for r in rows]
        base_cln = [r["base_margin_clean"] for r in rows]
        restore_ci = _paired(restore, base_att, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
        corrupt_ci = _paired(corrupt, base_cln, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

        # PRIMARY specificity: restoration vs the matched projected-transport sham (per pair,
        # mean over the sham ensemble). Both are Delta_M vs the same attack baseline.
        with_sham = [
            (rw, rs) for rw, rs in zip(rows, restore, strict=True)
            if rw["arms"][arm].get("sham_restore_max_dose")
        ]
        vs_sham = {"n": 0}
        if with_sham:
            r_dm = [rs - rw["base_margin_attack"] for rw, rs in with_sham]
            s_dm = [float(np.mean(rw["arms"][arm]["sham_restore_max_dose"])) - rw["base_margin_attack"]
                    for rw, _ in with_sham]
            vs_sham = _paired(r_dm, s_dm, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
            vs_sham["restore_gt_sham_pct"] = float(np.mean([
                np.mean(np.asarray(rw["arms"][arm]["sham_restore_max_dose"]) < rs)
                for rw, rs in with_sham
            ]))

        # SECONDARY (legacy, rig sanity only): restoration vs the global-additive orth null.
        with_null = [(rw, rs) for rw, rs in zip(rows, restore, strict=True) if rw.get("orth_null_margins")]
        vs_orth = {"n": 0}
        if with_null:
            r_dm = [rs - rw["base_margin_attack"] for rw, rs in with_null]
            o_dm = [float(np.mean(rw["orth_null_margins"])) - rw["base_margin_attack"] for rw, _ in with_null]
            vs_orth = _paired(r_dm, o_dm, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

        # GO requires the matched sham (fail-closed) AND a NON-ambiguous (stable multi-rank)
        # channel axis: restoration LB>0 AND beats matched sham LB>0. An ambiguous arm ran on
        # the rank-1 DiM fallback and is reported but never a confirmatory GO.
        has_sham = vs_sham.get("n", 0) > 0
        arm_go = bool(
            restore_ci.get("ci_low", float("-inf")) > 0
            and has_sham
            and vs_sham.get("ci_low", float("-inf")) > 0
            and not ambiguous
        )
        go = go or arm_go
        report["arms"][arm] = {
            "n": len(rows),
            "rank_ambiguous": ambiguous,
            "restore_dmargin": restore_ci,
            "corrupt_dmargin": corrupt_ci,
            "restore_vs_matched_sham": vs_sham,
            "restore_vs_orth_null_legacy": vs_orth,
            "has_matched_sham": has_sham,
            "arm_go": arm_go,
        }
        print(
            f"[arm {arm}{' AMBIGUOUS/rank-1-DiM' if ambiguous else ''}] n={len(rows)} "
            f"restore ΔM={restore_ci.get('mean', float('nan')):+.3f} "
            f"CI[{restore_ci.get('ci_low', float('nan')):+.3f},{restore_ci.get('ci_high', float('nan')):+.3f}] | "
            f"corrupt ΔM={corrupt_ci.get('mean', float('nan')):+.3f} | "
            f"vs-sham LB={vs_sham.get('ci_low', float('nan'))} "
            f"(restore>sham {vs_sham.get('restore_gt_sham_pct', float('nan'))}) -> {'GO' if arm_go else 'stop'}"
        )

    if not any(r.get("has_matched_sham") for r in report["arms"].values()) and not args.allow_no_sham:
        raise SystemExit(
            "no matched-sham margins in l18_patch.json (re-run channel_patch_l18 with --k-sham>0). "
            "A confirmatory GO requires the matched projected-transport sham; pass --allow-no-sham "
            "to emit a non-GO report anyway."
        )

    report["verdict"] = "GO" if go else "STOP"
    (patch_dir / args.out_name).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nVERDICT (margin evidence, matched-sham-specific): {report['verdict']}  "
          "(apply behavioral G4 + recognition-equivalence + pv_locked/mel specificity on top)")
    print(f"wrote {patch_dir / args.out_name}")


if __name__ == "__main__":
    main()
