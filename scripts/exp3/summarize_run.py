#!/usr/bin/env -S uv run python
"""Read-only reporting over an Exp3 run directory.

Usage:
    ./scripts/exp3/summarize_run.py <run_dir>

Complements the pipeline's own metrics.json by separating the *raw* R_tAB drift
a condition causes from the *donor-aligned* transport score, which is the
distinction that tells apart genuine bidirectional transport from a fixed
refusal-ward push (e.g. random_direction).
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

LAYERS = (8, 10, 12, 14, 16, 18)
CONDS = ("identity", "real", "wrong_item", "random_direction", "position_sham")


def load(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def mean(values: list[float]) -> float:
    return statistics.mean(values) if values else float("nan")


def behavior_table(run: Path) -> None:
    rows = load(run / "behavior" / "generations.jsonl")
    if not rows:
        return
    pairs = load(run / "inputs" / "pairs.jsonl")
    by = {(r["item_id"], r["arm"]): r for r in rows}
    print(f"\n=== BEHAVIOR (n_generations={len(rows)}) ===")
    overall: Counter = Counter()
    for role in ("harmful", "benign"):
        tab: Counter = Counter()
        for pair in pairs:
            src = by.get((pair["item_id"], pair["source_arm"]))
            tgt = by.get((pair["item_id"], pair["target_arm"]))
            if src is None or tgt is None or pair["role"] != role:
                continue
            tab[(bool(src["explicit_refusal"]), bool(tgt["explicit_refusal"]))] += 1
        n = sum(tab.values())
        if not n:
            continue
        overall.update(tab)
        disc = tab[(True, False)] + tab[(False, True)]
        print(
            f"  {role:8s} n={n:4d}  instability={disc / n:.3f}  "
            f"R->NR={tab[(True, False)]:3d}  NR->R={tab[(False, True)]:3d}  "
            f"stableR={tab[(True, True)]:3d}  stableNR={tab[(False, False)]:3d}"
        )
    n = sum(overall.values())
    if n:
        disc = overall[(True, False)] + overall[(False, True)]
        src_rate = (overall[(True, True)] + overall[(True, False)]) / n
        tgt_rate = (overall[(True, True)] + overall[(False, True)]) / n
        print(
            f"  {'ALL':8s} n={n:4d}  instability={disc / n:.3f}  "
            f"R->NR={overall[(True, False)]:3d}  NR->R={overall[(False, True)]:3d}"
        )
        print(
            f"           net refusal rate: source={src_rate:.3f} -> target={tgt_rate:.3f} "
            f"(net {tgt_rate - src_rate:+.3f}, hidden by cancellation)"
        )
    fails = [r for r in rows if r.get("decoding_failure")]
    print(f"  decoding failures: {len(fails)}/{len(rows)}")


def dose_table(run: Path) -> None:
    rows = load(run / "input_dose" / "records.jsonl")
    if not rows:
        return
    print(f"\n=== M1 INPUT DOSE (n_cells={len(rows)}) ===")
    ep = [r for r in rows if r.get("endpoint_expected")]
    print(
        f"  endpoint integrity: {sum(bool(r['endpoint_exact']) for r in ep)}/{len(ep)} exact, "
        f"max margin err={max((r['endpoint_margin_abs_error'] for r in ep), default=0):.2e}"
    )
    pads = {
        r["pair_id"]: r["padding_delta_frac_of_total_fro"]
        for r in rows
        if "padding_delta_frac_of_total_fro" in r
    }
    if pads:
        vals = list(pads.values())
        print(
            f"  whisper pad-floor share of ||dF||: mean={mean(vals):.3f} "
            f"min={min(vals):.3f} max={max(vals):.3f}"
        )
    print(
        f"  {'component':16s} {'dose':>5s} {'n':>4s} {'raw dR_tAB':>11s} "
        f"{'donor-aligned':>14s} {'flip%':>7s}"
    )
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["component"], r["dose"])].append(r)
    for key in sorted(groups, key=lambda k: (k[0] != "all", k[0], k[1])):
        vals = groups[key]
        raw = [r["r_tab_margin"] - r["baseline_r_tab_margin"] for r in vals]
        aligned = [
            (r["r_tab_margin"] - r["baseline_r_tab_margin"])
            * (r["donor_r_tab_margin"] - r["baseline_r_tab_margin"])
            for r in vals
        ]
        flips = [r for r in vals if r.get("explicit_refusal") is not None]
        flip = (
            mean(
                [
                    float(bool(r["explicit_refusal"]) != bool(r["baseline_explicit_refusal"]))
                    for r in flips
                ]
            )
            * 100
        )
        print(
            f"  {key[0]:16s} {key[1]:5.2f} {len(vals):4d} {mean(raw):+11.3f} "
            f"{mean(aligned):+14.3f} {flip:6.1f}%"
        )


def patch_table(run: Path, name: str, rel: str, site: str) -> None:
    rows = load(run / rel)
    if not rows:
        return
    layers = sorted({r[site] for r in rows})
    conds = [c for c in CONDS if any(r["condition"] == c for r in rows)]
    conds += sorted({r["condition"] for r in rows} - set(conds))
    print(f"\n=== {name} (n_cells={len(rows)}, cohort={len({r['pair_id'] for r in rows})}) ===")
    for label, fn in (
        ("RAW shift (patched - host)", lambda r: r["r_tab_margin"] - r["baseline_r_tab_margin"]),
        (
            "DONOR-ALIGNED (raw x desired)",
            lambda r: (
                (r["r_tab_margin"] - r["baseline_r_tab_margin"])
                * (r["donor_r_tab_margin"] - r["baseline_r_tab_margin"])
            ),
        ),
    ):
        print(f"  -- {label} --")
        print(f"  {'condition':18s}" + "".join(f"{lay:>9}" for lay in layers))
        for cond in conds:
            line = f"  {cond:18s}"
            for lay in layers:
                vals = [fn(r) for r in rows if r["condition"] == cond and r[site] == lay]
                line += f"{mean(vals):+9.3f}" if vals else f"{'-':>9}"
            print(line)
    ident = [r for r in rows if r["condition"] == "identity"]
    if ident:
        errs = [
            r["identity_margin_abs_error"]
            for r in ident
            if r.get("identity_margin_abs_error") is not None
        ]
        texts = [r for r in ident if r.get("identity_response_exact") is not None]
        print(
            f"  identity integrity: max margin err={max(errs, default=0):.2e}, "
            f"exact text {sum(bool(r['identity_response_exact']) for r in texts)}/{len(texts)}"
        )
    counts = {r.get("applied_count") or (r.get("applied_counts") or [None])[0] for r in rows}
    print(f"  applied_count values: {sorted(c for c in counts if c is not None)}")


def main() -> None:
    run = Path(sys.argv[1])
    print(f"RUN: {run}")
    behavior_table(run)
    dose_table(run)
    patch_table(run, "M2 SPAN PATCH", "span_patch/records.jsonl", "layer")
    patch_table(run, "M3 ESCAPE/RESET", "escape_reset/records.jsonl", "inject_layer")
    patch_table(run, "STAGE 5 READOUT PATCH", "readout_patch/records.jsonl", "layer")


if __name__ == "__main__":
    main()
