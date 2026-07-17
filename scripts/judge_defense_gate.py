#!/usr/bin/env -S uv run python
"""Adjudicate Run 9 defense generations with the existing blinded judges.

The script accepts either SARSteer's paired output or ALMGuard's two aligned arm
files.  It checkpoints content-addressed judge verdicts atomically, resumes
without paying for completed records, and writes only fully resolved four-way
label sidecars for ``evaluate_defense_gate.py``.  Judge disagreement fails closed
unless an explicit reviewed override is supplied.

Request and response bodies are never printed and are omitted from checkpoints,
unresolved reports, and evaluator label sidecars.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from audio_safety.config import load_experiment_config
from audio_safety.evaluation.defense_judge import (
    atomic_save_jsonl,
    bind_judge_identities,
    build_aligned_label_sidecars,
    build_paired_label_sidecar,
    expand_aligned_rows,
    expand_paired_rows,
    resolve_checkpoint_labels,
    run_judge_checkpoint,
)
from audio_safety.utils.io import load_jsonl


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--judge-config",
        type=Path,
        required=True,
        help="experiment YAML whose conversion_gap.judge block defines the judges",
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--paired", type=Path, help="SARSteer paired generation JSONL")
    mode.add_argument("--undefended", type=Path, help="ALMGuard undefended generation JSONL")
    parser.add_argument("--defended", type=Path, help="ALMGuard defended generation JSONL")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="atomic, resumable two-judge verdict checkpoint JSONL",
    )
    parser.add_argument(
        "--paired-labels-out",
        type=Path,
        help="paired label sidecar consumed by evaluate_defense_gate.py --paired-labels",
    )
    parser.add_argument(
        "--undefended-labels-out",
        type=Path,
        help="aligned undefended label sidecar",
    )
    parser.add_argument(
        "--defended-labels-out",
        type=Path,
        help="aligned defended label sidecar",
    )
    parser.add_argument(
        "--unresolved-out",
        type=Path,
        default=None,
        help="judge-disagreement report (default: <checkpoint>.unresolved.jsonl)",
    )
    parser.add_argument(
        "--reviewed-overrides",
        type=Path,
        default=None,
        help=(
            "optional JSONL resolving disagreements by judge_record_id, "
            "reviewed_behavior_label, and reviewed_by"
        ),
    )
    parser.add_argument("--save-every", type=int, default=16)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser.parse_args(argv)


def _validate_mode_args(args: argparse.Namespace) -> str:
    if args.paired is not None:
        forbidden = {
            "--defended": args.defended,
            "--undefended-labels-out": args.undefended_labels_out,
            "--defended-labels-out": args.defended_labels_out,
        }
        used = [name for name, value in forbidden.items() if value is not None]
        if used:
            raise SystemExit(f"paired mode cannot use {', '.join(used)}")
        if args.paired_labels_out is None:
            raise SystemExit("--paired-labels-out is required with --paired")
        return "paired"

    if args.defended is None:
        raise SystemExit("--defended is required with --undefended")
    if args.paired_labels_out is not None:
        raise SystemExit("--paired-labels-out is only valid with --paired")
    if args.undefended_labels_out is None or args.defended_labels_out is None:
        raise SystemExit(
            "--undefended-labels-out and --defended-labels-out are required with --undefended"
        )
    return "aligned"


def _unresolved_path(args: argparse.Namespace) -> Path:
    if args.unresolved_out is not None:
        return args.unresolved_out
    return args.checkpoint.with_name(f"{args.checkpoint.name}.unresolved.jsonl")


def _invalidate_label_outputs(args: argparse.Namespace, mode: str) -> None:
    """Replace stale sidecars with empty files so disagreement cannot be evaluated."""

    outputs = (
        [args.paired_labels_out]
        if mode == "paired"
        else [args.undefended_labels_out, args.defended_labels_out]
    )
    for output in outputs:
        atomic_save_jsonl([], output)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = _validate_mode_args(args)
    if args.save_every < 1:
        raise SystemExit("--save-every must be >= 1")

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
    cfg = load_experiment_config(args.judge_config, overrides=args.override)
    if cfg.conversion_gap is None:
        raise SystemExit("--judge-config needs a conversion_gap.judge block")
    judge_cfg = cfg.conversion_gap.judge
    if not judge_cfg.enabled:
        raise SystemExit("conversion_gap.judge is disabled")
    if len(judge_cfg.models) < 2:
        raise SystemExit("Run 9 defense adjudication requires at least two judge models")

    if mode == "paired":
        paired_rows = load_jsonl(args.paired)
        undefended_rows: list[dict] = []
        defended_rows: list[dict] = []
        expanded = expand_paired_rows(paired_rows)
        input_count = len(paired_rows)
    else:
        paired_rows = []
        undefended_rows = load_jsonl(args.undefended)
        defended_rows = load_jsonl(args.defended)
        expanded = expand_aligned_rows(undefended_rows, defended_rows)
        input_count = len(undefended_rows)

    bound = bind_judge_identities(expanded, judge_cfg)
    print(
        f"[defense-judge] mode={mode} inputs={input_count} expanded={len(bound)} "
        f"judges={len(judge_cfg.models)} checkpoint={args.checkpoint}",
        flush=True,
    )
    checkpoint_rows = run_judge_checkpoint(
        bound,
        judge_cfg,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
        overwrite=args.overwrite,
        show_progress=not args.no_progress,
    )

    override_rows = (
        load_jsonl(args.reviewed_overrides) if args.reviewed_overrides is not None else []
    )
    resolutions, unresolved = resolve_checkpoint_labels(
        checkpoint_rows,
        reviewed_overrides=override_rows,
    )
    unresolved_path = _unresolved_path(args)
    atomic_save_jsonl(unresolved, unresolved_path)
    if unresolved:
        _invalidate_label_outputs(args, mode)
        raise SystemExit(
            f"{len(unresolved)} judge disagreements remain unresolved; evaluator label "
            f"sidecars were invalidated. Review categorical audit rows in {unresolved_path}"
        )

    if mode == "paired":
        labels = build_paired_label_sidecar(paired_rows, bound, resolutions)
        atomic_save_jsonl(labels, args.paired_labels_out)
        output_summary = str(args.paired_labels_out)
    else:
        undefended_labels, defended_labels = build_aligned_label_sidecars(
            undefended_rows,
            defended_rows,
            bound,
            resolutions,
        )
        atomic_save_jsonl(undefended_labels, args.undefended_labels_out)
        atomic_save_jsonl(defended_labels, args.defended_labels_out)
        output_summary = f"{args.undefended_labels_out}, {args.defended_labels_out}"
    print(
        f"[defense-judge] resolved={len(resolutions)} unresolved=0 labels={output_summary}",
        flush=True,
    )


if __name__ == "__main__":
    main()
