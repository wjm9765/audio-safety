#!/usr/bin/env -S uv run python
"""Harmful x benign interaction for the L10 phase patch. CPU only.

THE POINT. The harmful-only run showed restoration at L10 (+7.3 pp, 90% CI
[+2.7,+12.0]) and nothing at L18. That is a restorative handle, not a safety
mechanism: the identical patch might repair benign audio just as well, in which
case what was repaired is general audio processing, not a safety computation.

THE TEST IS THE INTERACTION ITSELF. Codex was explicit that concluding
"specificity" because harmful is significant and benign is not would be invalid -
a difference in significance is not a significant difference. So this estimates

    interaction = delta_patch(harmful) - delta_patch(benign)

directly, with one JOINT bootstrap over items so the two arms' sampling error is
combined rather than compared post hoc.

Also reported:
  * reciprocity S = delta_restore + delta_corrupt, which a symmetric bidirectional
    mechanism predicts to be 0 (observed +5.3 pp on harmful, i.e. asymmetric)
  * L18 as the downstream control, which must stay null in both roles

CAVEAT carried into the output: the benign items are `soft_overrefusal` controls,
not content-matched counterparts of the harmful items. So a difference here is a
cohort interaction, not proof of safety specificity, and is labelled that way.

Example:
    ./scripts/exp2/analyze_causal_interaction.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

BASE = Path("/workspace/audio_safety_data/outputs/exp2_causal_phase")


def load(path: Path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.open() if x.strip() and "error" not in json.loads(x)]


def deltas(rows, direction, site, condition="real"):
    """Per-item (patched - baseline) refusal, paired within item."""
    sub = [r for r in rows if r["direction"] == direction and r["site"] == site
           and r["condition"] == condition]
    return np.array([float(r["refusal"]) - float(r["baseline_refusal"]) for r in sub])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--harmful", type=Path, default=BASE / "patches.jsonl")
    ap.add_argument("--benign", type=Path, default=BASE / "patches_benign.jsonl")
    ap.add_argument("--boot", type=int, default=10000)
    args = ap.parse_args()

    H, B = load(args.harmful), load(args.benign)
    if not H:
        print("no harmful rows")
        return
    rng = np.random.default_rng(0)
    print(f"harmful rows={len(H)} items={len({r['item_id'] for r in H})}   "
          f"benign rows={len(B)} items={len({r['item_id'] for r in B})}")

    # integrity: identity must reproduce baseline exactly, in both roles
    for tag, rows in (("harmful", H), ("benign", B)):
        idn = [r for r in rows if r["condition"] == "identity"]
        if idn:
            same = sum(r["response"] == r["baseline_response"] for r in idn)
            flag = "OK" if same == len(idn) else "*** MISALIGNED - VOID ***"
            print(f"  identity {tag}: {same}/{len(idn)} exact  {flag}")

    def ci(d, q=(5, 95)):
        if len(d) == 0:
            return (float("nan"),) * 3
        b = [np.mean(rng.choice(d, len(d), replace=True)) for _ in range(args.boot)]
        return d.mean(), np.percentile(b, q[0]), np.percentile(b, q[1])

    sites = sorted({r["site"] for r in H})
    print(f"\n{'site':6s}{'direction':10s}{'harmful':>22s}{'benign':>22s}"
          f"{'INTERACTION (H-B)':>26s}")
    print("-" * 88)
    for s in sites:
        for d in ("restore", "corrupt"):
            dh, db = deltas(H, d, s), deltas(B, d, s)
            mh, hl, hh = ci(dh)
            mb, bl, bh = ci(db) if len(db) else (float("nan"),) * 3
            if len(db):
                # JOINT bootstrap: resample both item sets each draw, so the
                # interaction interval carries both arms' sampling error.
                bo = [np.mean(rng.choice(dh, len(dh), replace=True))
                      - np.mean(rng.choice(db, len(db), replace=True))
                      for _ in range(args.boot)]
                mi, il, ih = mh - mb, np.percentile(bo, 5), np.percentile(bo, 95)
                itxt = f"{100 * mi:+.1f}pp [{100 * il:+.1f},{100 * ih:+.1f}]"
                verdict = "excl 0" if (il > 0 or ih < 0) else "spans 0"
            else:
                itxt, verdict = "(benign pending)", ""
            htxt = f"{100 * mh:+.1f}pp [{100 * hl:+.1f},{100 * hh:+.1f}]"
            btxt = (f"{100 * mb:+.1f}pp [{100 * bl:+.1f},{100 * bh:+.1f}]"
                    if len(db) else "-")
            print(f"L{s:<5d}{d:10s}{htxt:>22s}{btxt:>22s}{itxt:>20s} {verdict}")

    print("\nRECIPROCITY  S = delta_restore + delta_corrupt   (symmetric mechanism -> 0)")
    for s in sites:
        for tag, rows in (("harmful", H), ("benign", B)):
            dr, dc = deltas(rows, "restore", s), deltas(rows, "corrupt", s)
            if not len(dr) or not len(dc):
                continue
            bo = [np.mean(rng.choice(dr, len(dr), replace=True))
                  + np.mean(rng.choice(dc, len(dc), replace=True))
                  for _ in range(args.boot)]
            print(f"  L{s} {tag:8s} S = {100 * (dr.mean() + dc.mean()):+.1f}pp "
                  f"[{100 * np.percentile(bo, 5):+.1f},{100 * np.percentile(bo, 95):+.1f}]")

    print("\nREADING")
    print("  interaction spans 0  -> the patch repairs benign audio too: this is")
    print("     general representation repair, NOT a safety mechanism.")
    print("  interaction excludes 0, harmful larger -> safety-related candidate;")
    print("     still a COHORT interaction, since benign items are soft_overrefusal")
    print("     controls rather than content-matched counterparts.")
    print("  L18 must stay null in both roles for the localization to hold.")


if __name__ == "__main__":
    main()
