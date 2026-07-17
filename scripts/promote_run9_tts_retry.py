#!/usr/bin/env -S uv run python
"""Promote ASR-passing Run 9 TTS retries and invalidate dependent attacks."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from audio_safety.data.run9_tts_retry import (
    build_promotion_plan,
    filter_checkpoint_items,
    invalidate_attack_artifacts,
)
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.utils.io import load_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--retry-id", required=True)
    parser.add_argument("--clean-manifest", type=Path, default=None)
    parser.add_argument("--clean-asr", type=Path, default=None)
    parser.add_argument("--retry-manifest", type=Path, required=True)
    parser.add_argument("--retry-asr", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, default=None)
    parser.add_argument("--report-out", type=Path, default=None)
    parser.add_argument("--invalidate-attacks", action="store_true")
    parser.add_argument("--attack-manifest", type=Path, default=None)
    parser.add_argument("--attack-asr", type=Path, default=None)
    return parser.parse_args()


def _resolve(path: Path | None, default: Path) -> Path:
    return (path or default).resolve()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _backup(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"backup already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _require_complete(rows: list[dict[str, Any]], expected: int, label: str) -> None:
    ok = sum(row.get("asr_status") == "ok" for row in rows)
    if len(rows) != expected or ok != expected:
        raise RuntimeError(
            f"{label} must be complete before promotion: rows={len(rows)}, "
            f"status_ok={ok}, expected={expected}"
        )


def main() -> None:
    args = parse_args()
    data_dir = _resolve(
        args.data_dir,
        Path(os.environ.get("AUDIO_SAFETY_DATA_DIR", "/workspace/audio_safety_data/data")),
    )
    workspace = data_dir.parent
    clean_manifest = _resolve(args.clean_manifest, data_dir / "manifests/run9_fresh_clean.jsonl")
    clean_asr = _resolve(args.clean_asr, workspace / "outputs/run9_fresh/asr_clean.jsonl")
    retry_manifest = args.retry_manifest.resolve()
    retry_asr = args.retry_asr.resolve()
    backup_root = _resolve(args.backup_root, workspace / f"backups/run9_tts_retry/{args.retry_id}")
    report_out = _resolve(
        args.report_out,
        workspace / f"outputs/run9_fresh/tts_retry_{args.retry_id}_promotion.json",
    )
    required = [clean_manifest, clean_asr, retry_manifest, retry_asr]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    clean_rows = load_jsonl(clean_manifest)
    clean_asr_rows = load_jsonl(clean_asr)
    retry_rows = load_jsonl(retry_manifest)
    retry_asr_rows = load_jsonl(retry_asr)
    _require_complete(clean_asr_rows, len(clean_rows), "main clean ASR")
    _require_complete(retry_asr_rows, len(retry_rows), "retry ASR")
    plan = build_promotion_plan(
        clean_rows,
        retry_rows,
        retry_asr_rows,
        data_dir=data_dir,
        retry_id=args.retry_id,
    )
    if not plan.promoted_item_ids:
        report = {**plan.summary, "clean_manifest_changed": False}
        _atomic_json(report_out, report)
        print(json.dumps({**report, "report": str(report_out)}, indent=2), flush=True)
        return

    attack_manifest: Path | None = None
    attack_asr: Path | None = None
    attack_rows: list[dict[str, Any]] = []
    attack_asr_rows: list[dict[str, Any]] = []
    if args.invalidate_attacks and plan.promoted_harmful_item_ids:
        attack_manifest = _resolve(
            args.attack_manifest, data_dir / "manifests/run9_fresh_attacks.jsonl"
        )
        attack_asr = _resolve(args.attack_asr, workspace / "outputs/run9_fresh/asr_attacks.jsonl")
        if not attack_manifest.is_file() or not attack_asr.is_file():
            raise FileNotFoundError(
                "--invalidate-attacks requires complete attack manifest and ASR output"
            )
        attack_rows = load_jsonl(attack_manifest)
        attack_asr_rows = load_jsonl(attack_asr)
        _require_complete(attack_asr_rows, len(attack_rows), "attack ASR")

    # Immutable copies are completed before any authoritative file changes.
    _backup(clean_manifest, backup_root / "manifests" / clean_manifest.name)
    _backup(clean_asr, backup_root / "asr" / clean_asr.name)
    if attack_manifest is not None and attack_asr is not None:
        _backup(attack_manifest, backup_root / "manifests" / attack_manifest.name)
        _backup(attack_asr, backup_root / "asr" / attack_asr.name)

    # The manifest itself is promoted with one atomic replace. Original WAVs are
    # not copied over or deleted: the path now selects the versioned candidate.
    atomic_save_jsonl(plan.clean_rows, clean_manifest)
    filtered_clean_asr, removed_clean_asr = filter_checkpoint_items(
        clean_asr_rows, plan.promoted_item_ids
    )
    if removed_clean_asr != len(plan.promoted_item_ids):
        raise RuntimeError(
            "clean ASR invalidation count mismatch: "
            f"removed={removed_clean_asr}, promoted={len(plan.promoted_item_ids)}"
        )
    atomic_save_jsonl(filtered_clean_asr, clean_asr)

    invalidation: dict[str, Any] = {
        "invalidated_attack_rows": 0,
        "moved_attack_artifacts": 0,
        "missing_attack_artifacts": 0,
        "invalidated_attack_asr_rows": 0,
    }
    if attack_manifest is not None and attack_asr is not None:
        kept_attacks, artifact_counts = invalidate_attack_artifacts(
            attack_rows=attack_rows,
            harmful_item_ids=plan.promoted_harmful_item_ids,
            data_dir=data_dir,
            backup_root=backup_root,
        )
        atomic_save_jsonl(kept_attacks, attack_manifest)
        kept_attack_asr, removed_attack_asr = filter_checkpoint_items(
            attack_asr_rows, plan.promoted_harmful_item_ids
        )
        atomic_save_jsonl(kept_attack_asr, attack_asr)
        invalidation = {
            **artifact_counts,
            "invalidated_attack_asr_rows": removed_attack_asr,
        }

    report = {
        **plan.summary,
        **invalidation,
        "clean_manifest_changed": True,
        "invalidated_clean_asr_rows": removed_clean_asr,
        "backup_root": str(backup_root),
        "original_wavs_overwritten": False,
        "rerender_required": bool(invalidation["invalidated_attack_rows"]),
    }
    _atomic_json(report_out, report)
    print(json.dumps({**report, "report": str(report_out)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
