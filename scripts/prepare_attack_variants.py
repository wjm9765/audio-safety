#!/usr/bin/env -S uv run python
"""Freeze the Run 4 §8 text-jailbreak attack variants (deterministic, offline).

Wraps every base harmful/benign request with the frozen jailbreak templates and
writes a style-override manifest that the render pipeline consumes as new neutral-
acoustic ``style`` conditions (design §8.2). This runs on CPU with no model, so it
is the freeze step: it records each template's hash and the rendered-text hash so
the exact spoken attack cohort is auditable before any r_A alignment is observed
(design §8.6).

Example (local):
    ./scripts/prepare_attack_variants.py \
        --config configs/experiments/run4_attack_flip.yaml
"""

import argparse
import json
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.data.attacks import (
    build_attack_variant_records,
    freeze_summary,
    load_jailbreak_templates,
)
from audio_safety.utils.io import save_jsonl
from audio_safety.utils.paths import resolve_paths

DEFAULT_TEMPLATES = Path("configs/attacks/jailbreak_templates.yaml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument(
        "--templates",
        type=Path,
        default=DEFAULT_TEMPLATES,
        help="frozen jailbreak template YAML",
    )
    parser.add_argument("--limit", type=int, default=None, help="max base pairs to wrap")
    parser.add_argument(
        "--safety-label",
        choices=("harmful", "benign", "both"),
        default="both",
        help="which pair side(s) to wrap",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="override output file (defaults to dataset.attack_variant_file)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)

    if args.output is not None:
        output_path = (
            args.output if args.output.is_absolute() else paths.data_dir / args.output
        )
    elif cfg.dataset.attack_variant_file is not None:
        output_path = paths.data_dir / cfg.dataset.attack_variant_file
    else:
        raise SystemExit(
            "no output path: set dataset.attack_variant_file or pass --output"
        )

    templates = load_jailbreak_templates(args.templates)
    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    pairs = pairs[: args.limit] if args.limit is not None else pairs[: cfg.dataset.n_pairs]

    labels = ("harmful", "benign") if args.safety_label == "both" else (args.safety_label,)
    records = build_attack_variant_records(pairs, templates, safety_labels=labels)
    save_jsonl(records, output_path)

    summary = freeze_summary(records)
    print(f"[attack] wrote {len(records)} attack-variant records -> {output_path}")
    print(f"[attack] frozen template set ({args.templates}):")
    print(json.dumps(summary["by_style"], indent=2, ensure_ascii=False))
    print(
        "[attack] add these style keys to dataset.styles + dataset.neutral_acoustic_styles, "
        "then run render_audio -> score_transcripts -> generate_behavior."
    )


if __name__ == "__main__":
    main()
