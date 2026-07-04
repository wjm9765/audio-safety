#!/usr/bin/env -S uv run python
"""Generate a harmful-benign pair draft manifest with OpenRouter.

Example:
    ./scripts/prepare_audio_rdo_pairs.py \
        --config configs/experiments/exp1_refusal_cone_drift.yaml \
        --limit 150
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from audio_safety.config import load_experiment_config
from audio_safety.data import generate_pair_manifest, load_harmful_seed_rows
from audio_safety.utils.paths import resolve_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="experiment YAML")
    parser.add_argument("--data-dir", type=Path, default=None, help="override data root")
    parser.add_argument("--limit", type=int, default=None, help="max seed rows to generate")
    parser.add_argument(
        "--no-auto-seed",
        action="store_true",
        help="fail instead of downloading the configured harmful seed file",
    )
    return parser.parse_args()


def _run(cmd: list[str]) -> None:
    print("[pairs] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ensure_figstep_seed(seed_path: Path, cache_dir: Path, source_url: str) -> None:
    if seed_path.exists():
        return
    source_root = cache_dir / "sources" / "FigStep"
    source_csv = source_root / "data" / "question" / "safebench.csv"
    if not source_csv.exists():
        if source_root.exists() and any(source_root.iterdir()):
            raise RuntimeError(f"FigStep cache exists but safebench.csv is missing: {source_root}")
        source_root.parent.mkdir(parents=True, exist_ok=True)
        _run(["git", "clone", source_url, str(source_root)])
    if not source_csv.exists():
        raise FileNotFoundError(f"FigStep safebench.csv not found after clone: {source_csv}")
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_csv, seed_path)
    print(f"[pairs] copied seed CSV -> {seed_path}", flush=True)


def ensure_seed_file(seed_path: Path, cache_dir: Path, *, source: str, source_url: str) -> None:
    if seed_path.exists():
        return
    if source == "figstep_safebench":
        ensure_figstep_seed(seed_path, cache_dir, source_url)
        return
    raise FileNotFoundError(f"seed file not found and no downloader is configured: {seed_path}")


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)
    seed_path = paths.data_dir / cfg.dataset.seed_file
    output_path = paths.data_dir / cfg.dataset.source_file
    if not args.no_auto_seed:
        ensure_seed_file(
            seed_path,
            paths.cache_dir,
            source=cfg.dataset.harmful_source,
            source_url=cfg.dataset.source_url,
        )
    seed_rows = load_harmful_seed_rows(seed_path, source=cfg.dataset.harmful_source)
    pairs = generate_pair_manifest(
        seed_rows,
        cfg.dataset.pair_generation,
        output_path,
        limit=args.limit or cfg.dataset.n_pairs,
    )
    print(f"[pairs] wrote {len(pairs)} draft pairs -> {output_path}")
    if cfg.dataset.pair_generation.review_required:
        print("[pairs] review_required=true; inspect/edit needs_review rows before geometry.")


if __name__ == "__main__":
    main()
