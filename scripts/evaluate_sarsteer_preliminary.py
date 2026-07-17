#!/usr/bin/env -S uv run python
"""Run the fully local, non-authoritative SARSteer Run 9 direction check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl
from audio_safety.evaluation.sarsteer_preliminary import evaluate_preliminary_sarsteer
from audio_safety.utils.io import load_jsonl, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True, help="merged paired generations")
    parser.add_argument(
        "--expected-manifest",
        type=Path,
        required=True,
        help="frozen canonical manifest used to generate the pairs",
    )
    parser.add_argument("--audit-rows-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paired = args.paired.resolve()
    expected = args.expected_manifest.resolve()
    audit_out = args.audit_rows_out.resolve()
    report_out = args.report_out.resolve()
    inputs = {paired, expected}
    outputs = {audit_out, report_out}
    if len(outputs) != 2 or inputs & outputs:
        raise SystemExit("output paths must be distinct from each other and from inputs")
    for path, role in ((paired, "paired output"), (expected, "expected manifest")):
        if not path.is_file():
            raise SystemExit(f"{role} not found: {path}")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(
            "refusing to replace existing outputs without --overwrite: "
            + ", ".join(str(path) for path in sorted(existing))
        )

    try:
        audit_rows, report = evaluate_preliminary_sarsteer(
            load_jsonl(paired),
            load_jsonl(expected),
            n_bootstrap=args.n_bootstrap,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    report.update(
        {
            "paired": str(paired),
            "expected_manifest": str(expected),
            "audit_rows": str(audit_out),
        }
    )
    atomic_save_jsonl(audit_rows, audit_out)
    save_json(report, report_out)

    survival = report["gate"]["vulnerable_set"]["survival"]
    benign = report["gate"]["benign"]["over_refusal_cost"]
    utility = report["utility"]["refusal_delta"]
    positive = report["gate"]["positive_control"]["asr_reduction"]
    summary = {
        "status": report["status"],
        "rows": report["validation"]["paired_rows"],
        "vulnerable_items": report["gate"]["vulnerable_set"]["n_vulnerable_items"],
        "attack_survival": None if survival is None else survival["estimate"],
        "soft_overrefusal_delta": None if benign is None else benign["estimate"],
        "utility_refusal_delta": None if utility is None else utility["estimate"],
        "positive_control_reduction": None if positive is None else positive["estimate"],
        "heuristic_threshold_verdict": report["gate"]["decision"]["threshold_verdict"],
        "report": str(report_out),
    }
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
