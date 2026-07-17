#!/usr/bin/env -S uv run python
"""Prepare leak-free, phase-excluded ALMGuard SAP inputs for the Run 9 gate."""

from __future__ import annotations

import argparse
from pathlib import Path

from audio_safety.pipelines.almguard_training_data import (
    FORBIDDEN_TOKENS,
    DataContractError,
    build_plan,
    prepare_data,
    validate_prepared,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--almguard-repo", type=Path, default=Path("/workspace/almguard/ALMGuard"))
    parser.add_argument("--data-dir", type=Path, default=Path("/workspace/audio_safety_data/data"))
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--metadata", type=Path, default=None)
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        action="append",
        default=None,
        help="shared Run9 eval manifest; repeat for every eval view",
    )
    parser.add_argument("--train-per-family", type=int, default=10)
    parser.add_argument("--holdout-per-family", type=int, default=9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--forbid-token",
        action="append",
        default=None,
        help="case-insensitive token forbidden in lexical and resolved candidate paths",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="validate plan without writing")
    mode.add_argument(
        "--validate-only", action="store_true", help="validate an existing output root"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    repo = args.almguard_repo.resolve()
    output_root = (
        args.output_root.resolve(strict=False)
        if args.output_root
        else data_dir / f"almguard_run9_sap_official_seed{args.seed}"
    )
    metadata = (
        args.metadata.resolve() if args.metadata else repo / "datasets" / "AdvBench_Audio.json"
    )
    eval_manifests = args.eval_manifest or [data_dir / "manifests" / "run9_fresh_clean.jsonl"]
    forbidden = tuple(args.forbid_token or FORBIDDEN_TOKENS)

    try:
        if args.validate_only:
            contract = validate_prepared(output_root, data_dir=data_dir)
            label = "validated"
        elif args.dry_run:
            plan = build_plan(
                almguard_repo=repo,
                data_dir=data_dir,
                output_root=output_root,
                metadata_path=metadata,
                eval_manifests=[path.resolve() for path in eval_manifests],
                train_per_family=args.train_per_family,
                holdout_per_family=args.holdout_per_family,
                seed=args.seed,
                forbidden_tokens=forbidden,
            )
            print(
                f"[almguard-data] dry-run passed: common={len(plan['common_indices'])}, "
                f"train={len(plan['train_rows'])}, positive_control={len(plan['holdout_rows'])}"
            )
            return
        else:
            contract = prepare_data(
                almguard_repo=repo,
                data_dir=data_dir,
                output_root=output_root,
                metadata_path=metadata,
                eval_manifests=[path.resolve() for path in eval_manifests],
                train_per_family=args.train_per_family,
                holdout_per_family=args.holdout_per_family,
                seed=args.seed,
                forbidden_tokens=forbidden,
            )
            label = "prepared"
    except DataContractError as exc:
        raise SystemExit(f"[almguard-data] ERROR: {exc}") from exc

    print(
        f"[almguard-data] {label}: train={contract['train_total']} "
        f"({contract['train_per_family']}/family), "
        f"positive_control={contract['positive_control_total']} "
        f"({contract['positive_control_per_family']}/family)"
    )
    print("[almguard-data] train dirs: " + " ".join(contract["train_wav_dirs"]))
    print(f"[almguard-data] contract: {output_root / 'contract.json'}")


if __name__ == "__main__":
    main()
