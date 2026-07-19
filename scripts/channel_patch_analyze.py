#!/usr/bin/env -S uv run python
"""Run 10 STEP 3 analysis (CPU): G1/G4 paired stats + restore-vs-orth-null + GO/STOP.

Reads <run>/channel_patch/l18_patch.json (written by channel_patch_l18.py) and, per arm,
at the maximum dose, computes:
  - restoration Delta_M = M(attack + restore clean U-coord) - M(attack)   [fixability]
  - corruption  Delta_M = M(clean + inject attack U-coord)  - M(clean)    [necessity]
  - restoration vs the sham/orthogonal null (matched magnitude, perp to refusal-DiM)
with by-item bootstrap CIs (evaluation.conversion_probe.paired_mean_diff_ci).

VERDICT (registered G1 in the margin form, since flips are few — Run 7 lineage): GO if some
arm's restoration Delta_M has item-LB > 0 AND beats the orthogonal null; else STOP. The
behavioral 15pp / flip-specificity (G4) and recognition-equivalence gates are applied on
top of this by the analyst — this script reports the continuous-margin evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _paired(a: list[float], b: list[float], *, n_boot: int, alpha: float, seed: int) -> dict:
    from audio_safety.evaluation.conversion_probe import paired_mean_diff_ci

    return paired_mean_diff_ci(np.asarray(a, float), np.asarray(b, float), n_boot=n_boot, alpha=alpha, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--in-name", default="l18_patch.json")
    ap.add_argument("--out-name", default="l18_analysis.json")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    patch_dir = args.run_dir / "channel_patch"
    data = json.loads((patch_dir / args.in_name).read_text())
    results = data["results"]
    if not results:
        raise SystemExit("no test-pair results to analyze")
    dose = max(float(d) for d in data["dose"])
    dose_key = f"{dose}"

    arms = sorted({arm for r in results for arm in r.get("arms", {})})
    report: dict[str, dict] = {"dose": dose, "arms": {}}
    go = False
    for arm in arms:
        rows = [r for r in results if arm in r.get("arms", {})]
        restore = [r["arms"][arm]["restore"][dose_key] for r in rows]
        corrupt = [r["arms"][arm]["corrupt"][dose_key] for r in rows]
        base_att = [r["base_margin_attack"] for r in rows]
        base_cln = [r["base_margin_clean"] for r in rows]
        restore_ci = _paired(restore, base_att, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
        corrupt_ci = _paired(corrupt, base_cln, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)

        # restoration vs orthogonal null (only pairs that have a null recorded)
        with_null = [(rw, rs) for rw, rs in zip(rows, restore, strict=True) if rw.get("orth_null_margins")]
        vs_null = {"n": 0}
        if with_null:
            r_dm = [rs - rw["base_margin_attack"] for rw, rs in with_null]
            o_dm = [float(np.mean(rw["orth_null_margins"])) - rw["base_margin_attack"] for rw, _ in with_null]
            vs_null = _paired(r_dm, o_dm, n_boot=args.n_boot, alpha=args.alpha, seed=args.seed)
            vs_null["restore_gt_null_pct"] = float(
                np.mean([
                    np.mean(np.asarray(rw["orth_null_margins"]) < rs) for rw, rs in with_null
                ])
            )
        arm_go = restore_ci["ci_low"] > 0 and (vs_null.get("n", 0) == 0 or vs_null["ci_low"] > 0)
        go = go or arm_go
        report["arms"][arm] = {
            "n": len(rows),
            "restore_dmargin": restore_ci,
            "corrupt_dmargin": corrupt_ci,
            "restore_vs_orth_null": vs_null,
            "arm_go": bool(arm_go),
        }
        print(
            f"[arm {arm}] n={len(rows)} restore ΔM={restore_ci['mean']:+.3f} "
            f"CI[{restore_ci['ci_low']:+.3f},{restore_ci['ci_high']:+.3f}] | "
            f"corrupt ΔM={corrupt_ci['mean']:+.3f} | "
            f"vs-null LB={vs_null.get('ci_low', float('nan'))} -> {'GO' if arm_go else 'stop'}"
        )

    report["verdict"] = "GO" if go else "STOP"
    (patch_dir / args.out_name).write_text(json.dumps(report, indent=2) + "\n")
    print(f"\nVERDICT (margin evidence): {report['verdict']}  "
          "(apply G4 flip-specificity + recognition-equivalence gates on top)")
    print(f"wrote {patch_dir / args.out_name}")


if __name__ == "__main__":
    main()
