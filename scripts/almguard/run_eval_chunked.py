#!/usr/bin/env -S uv run python
"""Run ALMGuard evaluation in fail-closed, resumable canonical chunks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_safety.pipelines.almguard_io import staged_wav_name
from audio_safety.pipelines.almguard_run9 import (
    ALMGuardRun9Error,
    align_arm_to_manifest,
    atomic_save_jsonl,
    index_eval_rows,
    load_object_jsonl,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("undefended", "defended"), required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=200)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--almguard-root", type=Path, default=Path("/workspace/almguard"))
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen2-Audio-7B-Instruct",
    )
    parser.add_argument("--perturb-path", type=Path)
    parser.add_argument("--zero-like", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--resume", action="store_true")
    action.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _expected_defense(mode: str) -> str:
    return "none" if mode == "undefended" else "almguard"


def _local_index(row: Mapping[str, Any], *, role: str, position: int) -> int:
    value = row.get("invocation_index")
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ALMGuardRun9Error(
            f"{role} row {position} has invalid invocation_index"
        )
    if row.get("staging_index") != value:
        raise ALMGuardRun9Error(
            f"{role} row {position} staging_index does not match invocation_index"
        )
    if row.get("staged_wav_name") != staged_wav_name(value):
        raise ALMGuardRun9Error(
            f"{role} row {position} has invalid staged_wav_name"
        )
    return value


def validate_resume_prefix(
    canonical: Sequence[Mapping[str, Any]],
    completed: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    expected_defense: str,
) -> list[dict[str, Any]]:
    if len(completed) > len(canonical):
        raise ALMGuardRun9Error("resume output is longer than the canonical manifest")
    expected_ids = [str(row["record_id"]) for row in canonical[: len(completed)]]
    actual_ids = [row.get("record_id") for row in completed]
    if actual_ids != expected_ids:
        raise ALMGuardRun9Error("resume output is not an exact canonical record_id prefix")
    aligned = align_arm_to_manifest(
        canonical[: len(completed)],
        completed,
        data_dir=data_dir,
        role="chunked resume",
        expected_defense=expected_defense,
    )
    for index, row in enumerate(aligned):
        if row.get("index") != index or row.get("canonical_index") != index:
            raise ALMGuardRun9Error(
                f"resume output row {index} has invalid global index/canonical_index"
            )
        _local_index(row, role="resume output", position=index)
    return aligned


def normalize_child_chunk(
    canonical_chunk: Sequence[Mapping[str, Any]],
    child_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    expected_defense: str,
    global_start: int,
) -> list[dict[str, Any]]:
    expected_ids = [str(row["record_id"]) for row in canonical_chunk]
    actual_ids = [row.get("record_id") for row in child_rows]
    if actual_ids != expected_ids:
        raise ALMGuardRun9Error("child output order differs from the canonical chunk")
    aligned = align_arm_to_manifest(
        canonical_chunk,
        child_rows,
        data_dir=data_dir,
        role=f"chunk starting at {global_start}",
        expected_defense=expected_defense,
    )
    normalized: list[dict[str, Any]] = []
    for local_index, row in enumerate(aligned):
        if row.get("index") != local_index:
            raise ALMGuardRun9Error(
                f"child row {local_index} has invalid local index"
            )
        if row.get("staging_index") != local_index:
            raise ALMGuardRun9Error(
                f"child row {local_index} has invalid staging_index"
            )
        if row.get("staged_wav_name") != staged_wav_name(local_index):
            raise ALMGuardRun9Error(
                f"child row {local_index} has invalid staged_wav_name"
            )
        if row.get("defense") != expected_defense:
            raise ALMGuardRun9Error(
                f"child row {local_index} has invalid defense tag"
            )
        if not isinstance(row.get("output"), str):
            raise ALMGuardRun9Error(
                f"child row {local_index} lacks a string output"
            )
        global_index = global_start + local_index
        normalized.append(
            {
                **row,
                "invocation_index": local_index,
                "index": global_index,
                "canonical_index": global_index,
            }
        )
    return normalized


def _child_command(
    args: argparse.Namespace,
    *,
    manifest: Path,
    out: Path,
    work_dir: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).with_name("run_almguard.py")),
        "--mode",
        args.mode,
        "--manifest",
        str(manifest),
        "--data-dir",
        str(args.data_dir.resolve()),
        "--out",
        str(out),
        "--work-dir",
        str(work_dir),
        "--almguard-root",
        str(args.almguard_root.resolve()),
        "--model-path",
        args.model_path,
    ]
    if args.mode == "defended":
        if args.perturb_path is None:
            raise ALMGuardRun9Error("--perturb-path is required for defended mode")
        command.extend(["--perturb-path", str(args.perturb_path.resolve())])
    elif args.zero_like is not None:
        command.extend(["--zero-like", str(args.zero_like.resolve())])
    return command


def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.chunk_size <= 0:
        raise ALMGuardRun9Error("--chunk-size must be positive")
    manifest = args.manifest.resolve()
    out = args.out.resolve()
    if manifest == out:
        raise ALMGuardRun9Error("--out must differ from --manifest")
    canonical = load_object_jsonl(manifest, role="canonical manifest")
    data_dir = args.data_dir.resolve()
    index_eval_rows(canonical, data_dir=data_dir, role="canonical manifest")
    defense = _expected_defense(args.mode)

    if args.resume:
        if not out.is_file():
            raise ALMGuardRun9Error("--resume requires an existing output JSONL")
        completed = validate_resume_prefix(
            canonical,
            load_object_jsonl(out, role="resume output"),
            data_dir=data_dir,
            expected_defense=defense,
        )
    elif out.exists() and not args.overwrite:
        raise ALMGuardRun9Error("output exists; pass --resume or --overwrite")
    else:
        completed = []

    cache_root = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else out.parent / "_almguard_chunk_cache"
    )
    run_cache = cache_root / uuid.uuid4().hex
    run_cache.mkdir(parents=True, exist_ok=False)
    chunks_run = 0
    for start in range(len(completed), len(canonical), args.chunk_size):
        chunk = canonical[start : start + args.chunk_size]
        chunk_dir = run_cache / f"chunk-{start:06d}"
        chunk_dir.mkdir(parents=True)
        chunk_manifest = chunk_dir / "manifest.jsonl"
        child_out = chunk_dir / "output.jsonl"
        atomic_save_jsonl(chunk, chunk_manifest)
        subprocess.run(
            _child_command(
                args,
                manifest=chunk_manifest,
                out=child_out,
                work_dir=chunk_dir / "work",
            ),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        child_rows = load_object_jsonl(child_out, role=f"child output at {start}")
        completed.extend(
            normalize_child_chunk(
                chunk,
                child_rows,
                data_dir=data_dir,
                expected_defense=defense,
                global_start=start,
            )
        )
        atomic_save_jsonl(completed, out)
        chunks_run += 1
    return {
        "status": "complete",
        "mode": args.mode,
        "rows": len(completed),
        "chunks_run": chunks_run,
        "chunk_size": args.chunk_size,
        "output": str(out),
        "run_cache": str(run_cache),
        "child_streams_suppressed": True,
    }


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        summary = _run(args)
    except ALMGuardRun9Error as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
