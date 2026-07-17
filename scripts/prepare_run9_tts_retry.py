#!/usr/bin/env -S uv run python
"""Prepare versioned CosyVoice jobs for failed Run 9 clean ASR controls."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from audio_safety.data.run9_tts_retry import build_retry_plan
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.utils.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--clean-manifest", type=Path, default=None)
    parser.add_argument("--asr", type=Path, default=None)
    parser.add_argument("--original-tts-jobs", type=Path, default=None)
    parser.add_argument("--retry-root", type=Path, default=None)
    parser.add_argument("--retry-id", required=True)
    parser.add_argument("--base-seed", type=int, default=1709)
    parser.add_argument("--manifest-out", type=Path, default=None)
    parser.add_argument("--jobs-out", type=Path, default=None)
    parser.add_argument("--summary-out", type=Path, default=None)
    return parser.parse_args()


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(value, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)


def _resolve(path: Path | None, default: Path) -> Path:
    return (path or default).resolve()


def main() -> None:
    args = parse_args()
    data_dir = _resolve(
        args.data_dir,
        Path(os.environ.get("AUDIO_SAFETY_DATA_DIR", "/workspace/audio_safety_data/data")),
    )
    workspace = data_dir.parent
    clean_manifest = _resolve(args.clean_manifest, data_dir / "manifests/run9_fresh_clean.jsonl")
    asr_path = _resolve(args.asr, workspace / "outputs/run9_fresh/asr_clean.jsonl")
    original_jobs = _resolve(
        args.original_tts_jobs,
        data_dir / "manifests/run9_fresh_tts_jobs.jsonl",
    )
    retry_root = _resolve(args.retry_root, data_dir / "audio_run9/retries")
    stem = f"run9_fresh_tts_retry_{args.retry_id}"
    manifest_out = _resolve(args.manifest_out, data_dir / f"manifests/{stem}.jsonl")
    jobs_out = _resolve(args.jobs_out, data_dir / f"manifests/{stem}_jobs.jsonl")
    summary_out = _resolve(args.summary_out, data_dir / f"manifests/{stem}_summary.json")
    for path, label in (
        (clean_manifest, "clean manifest"),
        (asr_path, "clean ASR output"),
        (original_jobs, "original TTS jobs"),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} not found: {path}")

    plan = build_retry_plan(
        load_jsonl(clean_manifest),
        load_jsonl(asr_path),
        load_jsonl(original_jobs),
        data_dir=data_dir,
        retry_root=retry_root,
        retry_id=args.retry_id,
        base_seed=args.base_seed,
    )
    atomic_save_jsonl(plan.candidate_rows, manifest_out)
    atomic_save_jsonl(plan.tts_jobs, jobs_out)
    summary = {
        **plan.summary,
        "clean_manifest": str(clean_manifest),
        "asr": str(asr_path),
        "candidate_manifest": str(manifest_out),
        "tts_jobs": str(jobs_out),
        "retry_root": str(retry_root),
    }
    _atomic_json(summary_out, summary)
    print(json.dumps({**summary, "summary": str(summary_out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
