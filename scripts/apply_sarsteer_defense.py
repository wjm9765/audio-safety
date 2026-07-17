#!/usr/bin/env -S uv run python
"""Apply the SARSteer defense on the held-out eval set (Run 9 defense gate).

For every held-out row (harmful clean + harmful attacked + benign), generate an
UNDEFENDED and a SARSteer-DEFENDED response with the SAME greedy decoding, and
write both to a JSONL manifest. Downstream: judge_behavior.py labels both, then a
survival step computes, on the vulnerable set S (clean-refuse and attack-comply),
the fraction still flipping under the defense (STRONG >=50% / WEAK <=20%) plus the
benign over-refusal cost. This script only produces the raw generations; it does
NOT judge or decide.
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.pipelines.rdo_gate import rows_for_split, split_ids
from audio_safety.pipelines.sarsteer import (
    generate_audio_response_with_sarsteer,
    load_sarsteer_vectors,
)
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="heldout", help="eval split name")
    parser.add_argument(
        "--vectors",
        type=Path,
        default=None,
        help="SARSteer vectors npz (default: <run_dir>/<sarsteer.vectors_file>)",
    )
    parser.add_argument("--limit", type=int, default=None, help="debug: cap eval rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.sarsteer is None or not cfg.sarsteer.enabled:
        raise SystemExit("cfg.sarsteer is disabled; enable it in the run9 config")
    sar = cfg.sarsteer
    paths = resolve_paths(
        cfg.paths, data_dir=args.data_dir, output_dir=args.output_dir, cache_dir=args.cache_dir
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    vectors_path = args.vectors if args.vectors is not None else run_dir / sar.vectors_file
    vectors = load_sarsteer_vectors(vectors_path)

    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    split_map = split_ids(pairs, cfg)
    rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    eval_rows = rows_for_split(rows, split_map, args.split)
    if args.limit is not None:
        eval_rows = eval_rows[: args.limit]
    print(
        f"[sarsteer] apply on {len(eval_rows)} '{args.split}' rows "
        f"({len(vectors)} layer vectors, alpha={sar.alpha}); run dir {run_dir}",
        flush=True,
    )

    from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction

    records = []
    for row in eval_rows:
        audio_path = paths.data_dir / str(row["path"])
        undefended = generate_audio_response(
            model,
            processor,
            audio_path,
            instruction,
            max_new_tokens=sar.max_new_tokens,
            do_sample=False,
        )
        defended = generate_audio_response_with_sarsteer(
            model,
            processor,
            audio_path,
            instruction,
            vectors=vectors,
            alpha=sar.alpha,
            max_new_tokens=sar.max_new_tokens,
            do_sample=False,
        )
        records.append(
            {
                "item_id": row["item_id"],
                "safety_label": row["safety_label"],
                "style": row["style"],
                "path": row["path"],
                "reference_text": row.get("reference_text"),
                "defense": "sarsteer",
                "undefended_output": undefended,
                "defended_output": defended,
            }
        )

    out_path = run_dir / "sarsteer_defended_outputs.jsonl"
    save_jsonl(records, out_path)
    print(f"[sarsteer] wrote {len(records)} undefended/defended pairs -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
