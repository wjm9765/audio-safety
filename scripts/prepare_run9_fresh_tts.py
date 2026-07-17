#!/usr/bin/env -S uv run python
"""Prepare resumable CosyVoice2 jobs for the Run 9 full SafeBench cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from audio_safety.data.run9_fresh import (
    build_run9_fresh_plan,
    load_safebench_rows,
    write_job_shards,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--seed-csv", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--output-audio-root",
        type=Path,
        default=None,
        help="default: <data-dir>/audio_run9/clean",
    )
    parser.add_argument(
        "--reuse-harmful-root",
        type=Path,
        default=None,
        help="default: <data-dir>/audio_attack_flip/harmful/neutral",
    )
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--harmful-only", action="store_true")
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="default: <data-dir>/manifests/run9_fresh_clean.jsonl",
    )
    parser.add_argument(
        "--jobs-out",
        type=Path,
        default=None,
        help="default: <data-dir>/manifests/run9_fresh_tts_jobs.jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    default_data = Path(
        os.environ.get("AUDIO_SAFETY_DATA_DIR", "/workspace/audio_safety_data/data")
    )
    data_dir = (args.data_dir or default_data).resolve()
    seed_csv = args.seed_csv or data_dir / "text/figstep/safebench.csv"
    output_root = args.output_audio_root or data_dir / "audio_run9/clean"
    reuse_root = args.reuse_harmful_root or data_dir / "audio_attack_flip/harmful/neutral"
    manifest_out = args.manifest_out or data_dir / "manifests/run9_fresh_clean.jsonl"
    jobs_out = args.jobs_out or data_dir / "manifests/run9_fresh_tts_jobs.jsonl"

    plan = build_run9_fresh_plan(
        load_safebench_rows(seed_csv),
        data_dir=data_dir,
        output_audio_root=output_root,
        reuse_harmful_root=None if args.no_reuse else reuse_root,
        include_soft_controls=not args.harmful_only,
    )
    write_jsonl(manifest_out, plan.manifest_rows)
    write_jsonl(jobs_out, plan.tts_jobs)
    shard_paths = write_job_shards(plan.tts_jobs, jobs_out, args.workers)
    summary = {
        "seed_csv": str(seed_csv),
        "manifest": str(manifest_out),
        "jobs": str(jobs_out),
        "shards": [str(path) for path in shard_paths],
        "cohort_rows": len(plan.manifest_rows),
        "tts_jobs": len(plan.tts_jobs),
        "reused_hash_verified": plan.reused_count,
        "pending_tts": plan.pending_count,
        "workers": args.workers,
        "asr_required": True,
    }
    summary_path = jobs_out.with_name(f"{jobs_out.stem}_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print("[run9] render shards with scripts/cosyvoice2_tts.py --batch-jsonl <shard>")


if __name__ == "__main__":
    main()
