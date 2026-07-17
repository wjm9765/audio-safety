#!/usr/bin/env -S uv run python
"""Render frozen Run 7 DSP attacks from existing Run 9 fresh clean WAVs.

This CPU-only stage reads a ``run9_fresh_clean.jsonl`` snapshot and processes only
rows whose ``gate_role`` is ``harmful_eval`` and whose clean WAV currently exists.
It never loads Qwen model weights: each worker loads only ``AutoProcessor`` from the
configured cache and keeps its Whisper feature extractor for mel matching.

For every clean item and sign, the frozen Run 7 operators produce:

* ``pv_standard``: primary phase-vocoder/channel attack under test;
* ``pv_locked``: phase-repaired negative twin;
* ``mel_matched_ctrl``: coherent smooth-EQ other-route control, matched to the
  processor-space ``pv_standard`` vs ``pv_locked`` distance ``d_pair``.

Per-item/sign sidecars make the stage resumable even if the combined manifest is
not reached. Missing clean WAVs are reported and left for a later rerun while TTS
continues. Existing inconsistent artifacts are never replaced without
``--overwrite``. Use ``--dry-run`` to inspect the current snapshot without loading
the processor or writing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import uuid
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from audio_safety.config import load_experiment_config
from audio_safety.utils.io import load_jsonl, save_json, save_jsonl
from audio_safety.utils.paths import resolve_paths

OPERATOR_VERSION = "run7_phase_ops_20260714_v1"
CONDITIONS = ("pv_standard", "pv_locked", "mel_matched_ctrl")
_WORKER_FEATURE_EXTRACTOR: Any | None = None


@dataclass(frozen=True)
class RenderTask:
    item_id: str
    category_id: int | None
    category_name: str | None
    reference_text: str
    reference_sha256: str | None
    clean_manifest_path: str
    clean_path: Path
    clean_sha256: str
    clean_text_hash_verified: bool
    sign: float
    data_dir: Path
    output_root: Path
    model_id: str
    cache_dir: Path
    sample_rate: int
    local_files_only: bool


@dataclass(frozen=True)
class RenderPlan:
    tasks: list[RenderTask]
    harmful_rows: int
    available_items: int
    missing_clean_items: list[str]
    ignored_rows: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, required=True, help="experiment YAML for model/cache"
    )
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="default: <data-dir>/audio_run9/attacks",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=None,
        help="default: <data-dir>/manifests/run9_fresh_attacks.jsonl",
    )
    parser.add_argument("--model-id", default=None, help="default: config model.model_id")
    parser.add_argument("--signs", type=float, nargs="+", default=[-3.0, 3.0])
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="cap currently available items")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow processor download; default is cache-only/local_files_only",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace inconsistent per-task artifacts; valid completed tasks are still reused",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sign_key(value: float) -> float:
    return round(float(value), 6)


def _sign_tag(value: float) -> str:
    value = _sign_key(value)
    prefix = "m" if value < 0 else "p"
    return prefix + f"{abs(value):g}".replace(".", "p")


def _resolve_data_path(path: Path, data_dir: Path) -> Path:
    resolved = path if path.is_absolute() else data_dir / path
    return resolved.resolve()


def _relative_to_data(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Run 9 rendered audio must stay under data_dir: {path}") from exc


def _text_hash_status(clean_path: Path, expected: str | None) -> bool:
    if not expected:
        return False
    sidecar = clean_path.with_suffix(f"{clean_path.suffix}.sha256")
    if not sidecar.is_file():
        return False
    actual = sidecar.read_text(encoding="utf-8").strip()
    if actual != expected:
        raise ValueError(f"clean text-hash sidecar mismatch: {clean_path}")
    return True


def _validate_signs(signs: Sequence[float]) -> tuple[float, ...]:
    values = tuple(dict.fromkeys(_sign_key(sign) for sign in signs))
    if not values or any(sign == 0.0 for sign in values):
        raise ValueError("--signs must contain one or more non-zero values")
    return values


def build_render_plan(
    clean_rows: Sequence[dict[str, Any]],
    *,
    data_dir: Path,
    output_root: Path,
    model_id: str,
    cache_dir: Path,
    signs: Sequence[float] = (-3.0, 3.0),
    sample_rate: int = 16000,
    local_files_only: bool = True,
    limit: int | None = None,
) -> RenderPlan:
    if sample_rate < 1:
        raise ValueError("sample_rate must be positive")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    sign_values = _validate_signs(signs)
    data_dir = data_dir.resolve()
    output_root = output_root.resolve()
    if not output_root.is_relative_to(data_dir):
        raise ValueError("output_root must be under data_dir")

    seen: set[str] = set()
    candidates: list[tuple[dict[str, Any], Path, bool]] = []
    missing: list[str] = []
    ignored = 0
    harmful_rows = 0
    for index, row in enumerate(clean_rows):
        is_harmful_clean = (
            row.get("gate_role") == "harmful_eval"
            and row.get("safety_label") == "harmful"
            and row.get("condition") == "clean"
        )
        if not is_harmful_clean:
            ignored += 1
            continue
        harmful_rows += 1
        required = {"item_id", "path", "reference_text"}
        absent = sorted(required - row.keys())
        if absent:
            raise ValueError(f"clean manifest row {index} missing fields: {absent}")
        item_id = str(row["item_id"])
        if item_id in seen:
            raise ValueError(f"duplicate harmful clean item_id: {item_id}")
        seen.add(item_id)
        clean_path = _resolve_data_path(Path(str(row["path"])), data_dir)
        if not clean_path.is_relative_to(data_dir):
            raise ValueError(f"clean WAV escapes data_dir: {clean_path}")
        if not clean_path.is_file():
            missing.append(item_id)
            continue
        text_verified = _text_hash_status(clean_path, row.get("reference_sha256"))
        candidates.append((row, clean_path, text_verified))

    candidates.sort(key=lambda entry: str(entry[0]["item_id"]))
    if limit is not None:
        candidates = candidates[:limit]
    tasks: list[RenderTask] = []
    for row, clean_path, text_verified in candidates:
        clean_digest = _sha256(clean_path)
        for sign in sign_values:
            tasks.append(
                RenderTask(
                    item_id=str(row["item_id"]),
                    category_id=(
                        int(row["category_id"]) if row.get("category_id") is not None else None
                    ),
                    category_name=(
                        str(row["category_name"]) if row.get("category_name") is not None else None
                    ),
                    reference_text=str(row["reference_text"]),
                    reference_sha256=(
                        str(row["reference_sha256"])
                        if row.get("reference_sha256") is not None
                        else None
                    ),
                    clean_manifest_path=str(row["path"]),
                    clean_path=clean_path,
                    clean_sha256=clean_digest,
                    clean_text_hash_verified=text_verified,
                    sign=sign,
                    data_dir=data_dir,
                    output_root=output_root,
                    model_id=model_id,
                    cache_dir=cache_dir.resolve(),
                    sample_rate=sample_rate,
                    local_files_only=local_files_only,
                )
            )
    return RenderPlan(
        tasks=tasks,
        harmful_rows=harmful_rows,
        available_items=len(candidates),
        missing_clean_items=missing,
        ignored_rows=ignored,
    )


def _output_paths(task: RenderTask) -> dict[str, Path]:
    tag = _sign_tag(task.sign)
    return {
        condition: task.output_root / condition / tag / f"{task.item_id}.wav"
        for condition in CONDITIONS
    }


def _sidecar_path(task: RenderTask) -> Path:
    return task.output_root / "_metadata" / f"{task.item_id}_{_sign_tag(task.sign)}.json"


def _task_identity(task: RenderTask) -> dict[str, Any]:
    return {
        "operator_version": OPERATOR_VERSION,
        "item_id": task.item_id,
        "sign": task.sign,
        "source_clean_sha256": task.clean_sha256,
        "processor_model_id": task.model_id,
        "sample_rate": task.sample_rate,
    }


def _atomic_save_json(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_json(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_jsonl(rows: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_jsonl(rows, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _sidecar_rows(task: RenderTask) -> list[dict[str, Any]] | None:
    sidecar = _sidecar_path(task)
    if not sidecar.is_file():
        return None
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    if payload.get("identity") != _task_identity(task):
        return None
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != len(CONDITIONS):
        return None
    if {row.get("condition") for row in rows} != set(CONDITIONS):
        return None
    for row in rows:
        output = _resolve_data_path(Path(str(row.get("path", ""))), task.data_dir)
        if not output.is_file() or row.get("output_sha256") != _sha256(output):
            return None
    return rows


def classify_tasks(
    tasks: Sequence[RenderTask], *, overwrite: bool
) -> tuple[list[RenderTask], list[dict[str, Any]]]:
    pending: list[RenderTask] = []
    resumed: list[dict[str, Any]] = []
    for task in tasks:
        rows = _sidecar_rows(task)
        if rows is not None:
            resumed.extend(rows)
            continue
        artifacts = [*_output_paths(task).values(), _sidecar_path(task)]
        conflicts = [path for path in artifacts if path.exists() or path.is_symlink()]
        if conflicts and not overwrite:
            raise FileExistsError(
                "incomplete or incompatible render artifacts exist; rerun with --overwrite: "
                f"{[str(path) for path in conflicts[:3]]}"
            )
        pending.append(task)
    return pending, resumed


def _load_audio(path: Path, sample_rate: int) -> np.ndarray:
    import librosa

    audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    return np.asarray(audio, dtype=np.float32)


def _compute_variants(
    clean: np.ndarray, sample_rate: int, sign: float, feature_extractor: Any
) -> dict[str, Any]:
    from audio_safety.evaluation.phase_ops import (
        mel_matched_control,
        model_logmel,
        pitch_shift_custom,
    )

    standard = pitch_shift_custom(clean, sample_rate, sign, mode="standard")
    locked = pitch_shift_custom(clean, sample_rate, sign, mode="locked")
    standard_mel = model_logmel(standard, sample_rate, feature_extractor)
    locked_mel = model_logmel(locked, sample_rate, feature_extractor)
    frames = min(standard_mel.shape[1], locked_mel.shape[1])
    d_pair = float(np.sqrt(((standard_mel[:, :frames] - locked_mel[:, :frames]) ** 2).mean()))
    eq_control, realized_rms, eq_strength = mel_matched_control(
        locked, sample_rate, d_pair, feature_extractor
    )
    return {
        "waveforms": {
            "pv_standard": standard,
            "pv_locked": locked,
            "mel_matched_ctrl": eq_control,
        },
        "d_pair": d_pair,
        "mel_ctrl_realized_rms": float(realized_rms),
        "mel_ctrl_strength": float(eq_strength),
    }


def _atomic_write_wav(path: Path, waveform: np.ndarray, sample_rate: int) -> None:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp.wav")
    try:
        sf.write(temporary, waveform, sample_rate)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _route_metadata(condition: str) -> tuple[str, bool, str]:
    if condition == "pv_standard":
        return "phase_channel", True, "exclude_primary_phase_attack_under_test"
    if condition == "pv_locked":
        return "phase_repaired_negative_twin", False, "eligible_non_target_control"
    if condition == "mel_matched_ctrl":
        return "eq_other_route", False, "eligible_non_target_acoustic_route"
    raise ValueError(f"unknown condition: {condition}")


def _render_task_with_extractor(
    task: RenderTask, feature_extractor: Any, *, overwrite: bool
) -> list[dict[str, Any]]:
    if _sha256(task.clean_path) != task.clean_sha256:
        raise RuntimeError(f"clean WAV changed after planning: {task.clean_path}")
    clean = _load_audio(task.clean_path, task.sample_rate)
    rendered = _compute_variants(clean, task.sample_rate, task.sign, feature_extractor)
    paths = _output_paths(task)
    for condition, waveform in rendered["waveforms"].items():
        path = paths[condition]
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to replace render without --overwrite: {path}")
        _atomic_write_wav(path, np.asarray(waveform), task.sample_rate)
    if _sha256(task.clean_path) != task.clean_sha256:
        raise RuntimeError(f"clean WAV changed during rendering: {task.clean_path}")

    rows: list[dict[str, Any]] = []
    for condition in CONDITIONS:
        route, under_test, training_policy = _route_metadata(condition)
        path = paths[condition]
        rows.append(
            {
                "item_id": task.item_id,
                "source": "figstep_safebench",
                "category_id": task.category_id,
                "category_name": task.category_name,
                "safety_label": "harmful",
                "gate_role": "harmful_eval",
                "style": f"{condition}_{_sign_tag(task.sign)}",
                "condition": condition,
                "sign": task.sign,
                "route": route,
                "phase_under_test": under_test,
                "phase_under_test_tag": ("run7_frozen_pv_standard" if under_test else None),
                "almguard_training_policy": training_policy,
                "path": _relative_to_data(path, task.data_dir),
                "output_sha256": _sha256(path),
                "reference_text": task.reference_text,
                "reference_sha256": task.reference_sha256,
                "source_clean_path": task.clean_manifest_path,
                "source_clean_sha256": task.clean_sha256,
                "source_text_hash_verified": task.clean_text_hash_verified,
                "operator_version": OPERATOR_VERSION,
                "processor_model_id": task.model_id,
                "sample_rate": task.sample_rate,
                "d_pair": rendered["d_pair"],
                "mel_ctrl_realized_rms": (
                    rendered["mel_ctrl_realized_rms"] if condition == "mel_matched_ctrl" else None
                ),
                "mel_ctrl_strength": (
                    rendered["mel_ctrl_strength"] if condition == "mel_matched_ctrl" else None
                ),
                "render_status": "rendered",
                "asr_required": True,
            }
        )
    _atomic_save_json(
        {"schema_version": 1, "identity": _task_identity(task), "rows": rows},
        _sidecar_path(task),
    )
    return rows


def _load_feature_extractor(model_id: str, cache_dir: Path, local_files_only: bool) -> Any:
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=str(cache_dir),
        local_files_only=local_files_only,
    )
    return processor.feature_extractor


def _init_worker(model_id: str, cache_dir: Path, local_files_only: bool) -> None:
    global _WORKER_FEATURE_EXTRACTOR
    _WORKER_FEATURE_EXTRACTOR = _load_feature_extractor(model_id, cache_dir, local_files_only)


def _render_worker(payload: tuple[RenderTask, bool]) -> list[dict[str, Any]]:
    task, overwrite = payload
    if _WORKER_FEATURE_EXTRACTOR is None:
        raise RuntimeError("worker feature extractor was not initialized")
    return _render_task_with_extractor(task, _WORKER_FEATURE_EXTRACTOR, overwrite=overwrite)


def _existing_manifest_rows(path: Path) -> list[dict[str, Any]]:
    return load_jsonl(path) if path.is_file() else []


def _row_key(row: dict[str, Any]) -> tuple[str, str, float]:
    return str(row["item_id"]), str(row["condition"]), _sign_key(row["sign"])


def _merge_rows(
    existing: Sequence[dict[str, Any]], updates: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    indexed: dict[tuple[str, str, float], dict[str, Any]] = {}
    for row in existing:
        key = _row_key(row)
        if key in indexed:
            raise ValueError(f"duplicate existing attack-manifest key: {key}")
        indexed[key] = row
    for row in updates:
        indexed[_row_key(row)] = row
    condition_order = {condition: index for index, condition in enumerate(CONDITIONS)}
    return sorted(
        indexed.values(),
        key=lambda row: (
            str(row["item_id"]),
            _sign_key(row["sign"]),
            condition_order.get(str(row["condition"]), len(CONDITIONS)),
        ),
    )


def execute_tasks(
    tasks: Sequence[RenderTask], *, workers: int, overwrite: bool
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if not tasks:
        return []
    first = tasks[0]
    rows: list[dict[str, Any]] = []
    if workers == 1:
        extractor = _load_feature_extractor(first.model_id, first.cache_dir, first.local_files_only)
        for index, task in enumerate(tasks, start=1):
            rows.extend(_render_task_with_extractor(task, extractor, overwrite=overwrite))
            print(f"[run9-render] {index}/{len(tasks)} item-sign tasks", flush=True)
        return rows

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(first.model_id, first.cache_dir, first.local_files_only),
    ) as pool:
        futures = [pool.submit(_render_worker, (task, overwrite)) for task in tasks]
        for index, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            print(f"[run9-render] {index}/{len(tasks)} item-sign tasks", flush=True)
    return rows


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    cfg = load_experiment_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir, cache_dir=args.cache_dir)
    clean_manifest = _resolve_data_path(args.clean_manifest, paths.data_dir)
    if not clean_manifest.is_file():
        raise FileNotFoundError(f"fresh clean manifest not found: {clean_manifest}")
    output_root = (
        _resolve_data_path(args.output_root, paths.data_dir)
        if args.output_root is not None
        else (paths.data_dir / "audio_run9" / "attacks").resolve()
    )
    manifest_out = (
        _resolve_data_path(args.manifest_out, paths.data_dir)
        if args.manifest_out is not None
        else (paths.data_dir / "manifests" / "run9_fresh_attacks.jsonl").resolve()
    )
    model_id = args.model_id or cfg.model.model_id
    plan = build_render_plan(
        load_jsonl(clean_manifest),
        data_dir=paths.data_dir,
        output_root=output_root,
        model_id=model_id,
        cache_dir=paths.cache_dir,
        signs=args.signs,
        sample_rate=args.sample_rate,
        local_files_only=not args.allow_download,
        limit=args.limit,
    )
    pending, resumed = classify_tasks(plan.tasks, overwrite=args.overwrite)
    summary = {
        "clean_manifest": str(clean_manifest),
        "output_root": str(output_root),
        "manifest_out": str(manifest_out),
        "operator_version": OPERATOR_VERSION,
        "phase_under_test": "pv_standard",
        "no_qwen_model_loaded": True,
        "processor_model_id": model_id,
        "processor_cache_only": not args.allow_download,
        "harmful_clean_rows": plan.harmful_rows,
        "currently_available_items": plan.available_items,
        "missing_clean_items": len(plan.missing_clean_items),
        "missing_clean_item_ids": plan.missing_clean_items,
        "item_sign_tasks": len(plan.tasks),
        "pending_tasks": len(pending),
        "resumed_tasks": len(resumed) // len(CONDITIONS),
        "workers": args.workers,
        "dry_run": args.dry_run,
    }
    if args.dry_run:
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    rendered = execute_tasks(pending, workers=args.workers, overwrite=args.overwrite)
    merged = _merge_rows(
        _existing_manifest_rows(manifest_out),
        [*resumed, *rendered],
    )
    _atomic_save_jsonl(merged, manifest_out)
    summary.update(
        {
            "newly_rendered_tasks": len(rendered) // len(CONDITIONS),
            "manifest_rows": len(merged),
            "manifest_condition_counts": dict(
                sorted(Counter(str(row["condition"]) for row in merged).items())
            ),
        }
    )
    summary_path = manifest_out.with_name(f"{manifest_out.stem}_summary.json")
    _atomic_save_json(summary, summary_path)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
