#!/usr/bin/env -S uv run python
"""Freeze SARSteer alpha from adjudicated development-control labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.evaluation.agent_judge_io import merge_label_batches
from audio_safety.evaluation.alpha_selection import select_alpha
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep", type=Path, required=True, help="alpha_sweep_dev.jsonl")
    parser.add_argument(
        "--judge-root",
        type=Path,
        required=True,
        help="dir holding judge_a<ALPHA>/labels_*.json per alpha",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    sweep = load_jsonl(args.sweep)
    alphas = sorted({float(r["sweep_alpha"]) for r in sweep})

    rows_by_alpha: dict[float, list[dict[str, object]]] = {}
    for alpha in alphas:
        subset = [r for r in sweep if float(r["sweep_alpha"]) == alpha]
        label_dir = args.judge_root / f"judge_a{alpha:g}"
        label_rows: list[dict[str, object]] = []
        for path in sorted(label_dir.glob("labels_*.json")):
            payload = json.loads(path.read_text("utf-8"))
            if not isinstance(payload, list):
                raise SystemExit(f"{path} must be a JSON array")
            label_rows.extend(payload)
        merged = merge_label_batches(subset, label_rows, resolution="claude_agent_local")
        # Carry gate_role/item_id from the sweep rows onto the merged labels.
        by_id = {r["record_id"]: r for r in subset}
        for m in merged:
            src = by_id[m["record_id"]]
            m["gate_role"] = src.get("gate_role")
            m["item_id"] = src.get("item_id")
        rows_by_alpha[alpha] = merged

    report = select_alpha(
        rows_by_alpha,
        n_bootstrap=args.n_bootstrap,
        ci_alpha=args.ci_alpha,
        seed=args.seed,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", "utf-8")

    print(f"[alpha] selected: {report['selected_alpha']}")
    print(f"[alpha] rule: {report['selection_rule']}")
    print(
        f"{'alpha':>6} {'ASR_red':>8} {'CI_low':>8} {'CI_high':>8} {'newDF':>6} "
        f"{'benign+':>8} {'pass':>5}"
    )
    for g in report["grid"]:
        ci = g["asr_ci"]
        print(
            f"{g['alpha']:>6} {g['asr_reduction']:>8.3f} {ci[0]:>8.3f} {ci[1]:>8.3f} "
            f"{g['new_decoding_failures']:>6} {g['benign_refusal_increase']:>8.3f} "
            f"{'YES' if g['passes'] else 'no':>5}"
            + ("" if g["passes"] else f"   <- {'; '.join(g['reject_reasons'])}")
        )
    print(f"[alpha] report -> {args.out}")


if __name__ == "__main__":
    main()
