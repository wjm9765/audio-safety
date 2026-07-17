#!/usr/bin/env -S uv run python
"""Build the faithful SARSteer defense vectors (Run 9 defense gate).

Calibrates on the TRAIN split only (disjoint from the held-out eval set used by
apply_sarsteer_defense.py), so the defense never sees the items it is graded on:

1. per-layer text-derived refusal vector from harmful TEXT (train, clean style);
2. per-layer benign-SPEECH PCA safe-space from benign audio (train, clean style);
3. v_perp = (I - U U^T) v, saved to <run_dir>/<sarsteer.vectors_file>.

SARSteer has no public code; see src/audio_safety/pipelines/sarsteer.py for the
paper reconstruction and the fidelity NOTES.
"""

import argparse
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.pipelines.rdo_gate import rows_for_split, split_ids
from audio_safety.pipelines.sarsteer import (
    build_sarsteer_vectors,
    extract_benign_speech_activations,
    extract_text_refusal_vectors,
    save_sarsteer_vectors,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--clean-style",
        type=str,
        default="neutral",
        help="style whose rows are the clean/undefended calibration source",
    )
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

    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    split_map = split_ids(pairs, cfg)
    rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
    train_rows = rows_for_split(rows, split_map, "train")

    # Refusal calibration: unique harmful TEXT (clean style), disjoint from eval.
    harmful_texts: list[str] = []
    seen: set[str] = set()
    for row in train_rows:
        if row.get("safety_label") != "harmful" or row.get("style") != args.clean_style:
            continue
        item_id = str(row["item_id"])
        text = str(row.get("reference_text") or "").strip()
        if not text or item_id in seen:
            continue
        seen.add(item_id)
        harmful_texts.append(text)
        if len(harmful_texts) >= sar.n_refusal_calib:
            break

    # Safe-space PCA: benign SPEECH (clean style), disjoint from eval.
    benign_paths: list[Path] = []
    for row in train_rows:
        if row.get("safety_label") != "benign" or row.get("style") != args.clean_style:
            continue
        benign_paths.append(paths.data_dir / str(row["path"]))
        if len(benign_paths) >= sar.n_benign_pca:
            break

    if not harmful_texts:
        raise SystemExit("no harmful clean-style train rows for refusal calibration")
    if len(benign_paths) < 2:
        raise SystemExit("need >= 2 benign clean-style train rows for safe-space PCA")
    print(
        f"[sarsteer] calibrate: {len(harmful_texts)} harmful texts, "
        f"{len(benign_paths)} benign clips; run dir {run_dir}",
        flush=True,
    )

    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)

    refusal_vectors = extract_text_refusal_vectors(
        model,
        processor,
        harmful_texts,
        refusal_text=sar.refusal_prompt,
        extraction_position=sar.extraction_position,
    )
    benign_acts = extract_benign_speech_activations(
        model,
        processor,
        benign_paths,
        cfg.dataset.target_generation.instruction,
        position_name=sar.benign_position,
    )
    vectors = build_sarsteer_vectors(refusal_vectors, benign_acts, n_pcs=sar.n_pcs)
    if sar.layers is not None:
        keep = set(sar.layers)
        vectors = {ell: vec for ell, vec in vectors.items() if ell in keep}
        if not vectors:
            raise SystemExit(f"sarsteer.layers {sorted(keep)} selected no available layer")

    # Report per-layer norms (Codex 2026-07-17): alpha=0.1 multiplies the RAW
    # v_perp, so its magnitude — and how much of v survives safe-space ablation —
    # is load-bearing and must be inspectable, not hidden.
    import numpy as np

    norm_report = {}
    for ell in sorted(vectors):
        v_norm = float(np.linalg.norm(refusal_vectors[ell]))
        vperp_norm = float(np.linalg.norm(vectors[ell]))
        norm_report[str(ell)] = {
            "v_norm": v_norm,
            "v_perp_norm": vperp_norm,
            "retained_ratio": vperp_norm / v_norm if v_norm > 0 else float("nan"),
        }
    ratios = [r["retained_ratio"] for r in norm_report.values()]
    print(
        f"[sarsteer] v_perp/v retained ratio: "
        f"min={min(ratios):.3f} median={float(np.median(ratios)):.3f} max={max(ratios):.3f}",
        flush=True,
    )

    import subprocess

    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
    except Exception:
        commit = "unknown"

    out_path = run_dir / sar.vectors_file
    save_sarsteer_vectors(
        out_path,
        vectors,
        {
            "method": "sarsteer_faithful",
            "seed": cfg.seed,  # fixes the train/heldout split -> the exact calibration set
            "git_commit": commit,
            "alpha": sar.alpha,
            "n_pcs": sar.n_pcs,
            "refusal_prompt": sar.refusal_prompt,
            "extraction_position": sar.extraction_position,
            "benign_position": sar.benign_position,
            "n_refusal_calib": len(harmful_texts),
            "n_benign_pca": len(benign_paths),
            "layers": sorted(vectors),
            "retained_norm_ratio": norm_report,
        },
    )
    print(f"[sarsteer] saved {len(vectors)} per-layer defense vectors -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
