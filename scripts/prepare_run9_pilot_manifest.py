#!/usr/bin/env -S uv run python
"""Adapt the held-out Run 7 phase cohort into a Run 9 SARSteer pilot manifest.

The Run 7 ``cells.jsonl`` stores clean audio as absolute paths and rendered variants
relative to the Run 7 directory.  SARSteer's build/apply CLIs instead consume one
dataset ``target_generation.outputs_file`` and resolve its ``path`` fields from the
configured data directory.  This adapter bridges those contracts without changing
the SARSteer implementation:

* TRAIN: neutral harmful + benign rows from an existing calibration manifest;
* HELDOUT: Run 7 neutral, ``pv_standard`` (phase-vocoder/channel route), and
  ``mel_matched_ctrl`` (coherent EQ/other route) for the requested pitch signs;
* VALIDATION: omitted from this pilot manifest.

The deterministic split comes from the supplied experiment config.  Calibration
and evaluation item IDs are required to be disjoint.  By default, held-out audio is
symlink-staged below ``dataset.tts.audio_subdir`` so every output ``path`` is relative
to ``data_dir``.  ``--stage-mode absolute`` is available for a non-portable manifest.

This is a direction-finding pilot adapter, not a fresh confirmatory data build.  Its
metadata sidecar fails closed on ASR validity and records the Run 7 cohort's known
limitations.  Use ``--dry-run`` or ``--preflight-only`` before materializing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import uuid
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from audio_safety.config import load_experiment_config
from audio_safety.config.schema import ExperimentConfig
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.data.datasets import AudioRdoPair
from audio_safety.pipelines.rdo_gate import split_ids
from audio_safety.utils.io import get_git_commit, load_jsonl, save_json, save_jsonl
from audio_safety.utils.paths import resolve_paths


@dataclass(frozen=True)
class StageAction:
    source: Path
    destination: Path


@dataclass(frozen=True)
class PilotPlan:
    records: list[dict[str, Any]]
    stage_actions: list[StageAction]
    metadata: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Run 9 pilot experiment YAML")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", type=Path, default=None, help="override configured data root")
    parser.add_argument("--cells", type=Path, required=True, help="Run 7 cells.jsonl")
    parser.add_argument(
        "--variant-root",
        type=Path,
        default=None,
        help="base for relative variant_path values (default: parent of cells subdir)",
    )
    parser.add_argument(
        "--calibration-manifest",
        type=Path,
        required=True,
        help="existing behavior/render manifest containing neutral harmful+benign audio",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSONL (default: data_dir/dataset.target_generation.outputs_file)",
    )
    parser.add_argument(
        "--metadata-output",
        type=Path,
        default=None,
        help="audit JSON sidecar (default: <output-stem>.metadata.json)",
    )
    parser.add_argument(
        "--stage-subdir",
        type=Path,
        default=None,
        help="data-dir-relative staging root (default: dataset.tts.audio_subdir)",
    )
    parser.add_argument(
        "--stage-mode",
        choices=("symlink", "copy", "absolute"),
        default="symlink",
        help="stage held-out WAVs or reference their absolute source paths",
    )
    parser.add_argument("--clean-condition", default="neutral")
    parser.add_argument("--phase-condition", default="pv_standard")
    parser.add_argument("--eq-condition", default="mel_matched_ctrl")
    parser.add_argument("--clean-style", default="neutral")
    parser.add_argument("--phase-style-prefix", default="phase_pv_standard")
    parser.add_argument("--eq-style-prefix", default="eq_mel_matched_ctrl")
    parser.add_argument("--signs", type=float, nargs="+", default=[-3.0, 3.0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate and print the complete plan without writing or staging",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="alias for a no-write validation pass, intended for run orchestration",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing output/metadata files (never replaces conflicting staged WAVs)",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_cli_path(path: Path, data_dir: Path) -> Path:
    return path if path.is_absolute() else data_dir / path


def _variant_source(variant_path: str, variant_root: Path) -> Path:
    raw = Path(variant_path)
    source = raw if raw.is_absolute() else variant_root / raw
    source = source.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Run 7 variant audio not found: {source}")
    if source.suffix.lower() != ".wav":
        raise ValueError(f"Run 7 variant_path is not a WAV: {source}")
    return source


def _manifest_path(path: Path, data_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _sign_key(value: float) -> float:
    return round(float(value), 6)


def _sign_tag(value: float) -> str:
    value = _sign_key(value)
    prefix = "m" if value < 0 else "p"
    magnitude = f"{abs(value):g}".replace(".", "p")
    return f"{prefix}{magnitude}"


def _condition_style(
    condition: str,
    sign: float,
    *,
    clean_condition: str,
    phase_condition: str,
    eq_condition: str,
    clean_style: str,
    phase_style_prefix: str,
    eq_style_prefix: str,
) -> str:
    if condition == clean_condition:
        return clean_style
    if condition == phase_condition:
        return f"{phase_style_prefix}_{_sign_tag(sign)}"
    if condition == eq_condition:
        return f"{eq_style_prefix}_{_sign_tag(sign)}"
    raise ValueError(f"unsupported selected condition: {condition!r}")


def _index_cells(cells: Sequence[dict[str, Any]]) -> dict[tuple[str, str, float], dict[str, Any]]:
    indexed: dict[tuple[str, str, float], dict[str, Any]] = {}
    required = {"item_id", "condition", "sign", "variant_path", "reference_text"}
    for index, row in enumerate(cells):
        missing = sorted(required - row.keys())
        if missing:
            raise ValueError(f"Run 7 cell {index} missing fields: {missing}")
        key = (str(row["item_id"]), str(row["condition"]), _sign_key(row["sign"]))
        if key in indexed:
            raise ValueError(f"duplicate Run 7 cell key: {key}")
        indexed[key] = row
    return indexed


def _index_calibration_rows(
    rows: Sequence[dict[str, Any]], clean_style: str
) -> dict[tuple[str, str], dict[str, Any]]:
    indexed: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if str(row.get("style")) != clean_style:
            continue
        label = str(row.get("safety_label"))
        if label not in {"harmful", "benign"}:
            continue
        if "item_id" not in row or "path" not in row or "reference_text" not in row:
            raise ValueError(f"calibration row {index} lacks item_id/path/reference_text")
        key = (str(row["item_id"]), label)
        if key in indexed:
            raise ValueError(f"duplicate neutral calibration row: {key}")
        indexed[key] = row
    return indexed


def _stage_root(data_dir: Path, stage_subdir: Path) -> Path:
    if stage_subdir.is_absolute() or ".." in stage_subdir.parts:
        raise ValueError("stage subdir must be a data-dir-relative path without '..'")
    root = (data_dir / stage_subdir).resolve()
    if not root.is_relative_to(data_dir.resolve()):
        raise ValueError(f"staging root escapes data_dir: {root}")
    return root


def _calibration_record(
    row: dict[str, Any], pair: AudioRdoPair, data_dir: Path, clean_style: str
) -> dict[str, Any]:
    label = str(row["safety_label"])
    expected_text = pair.harmful_text if label == "harmful" else pair.benign_text
    if str(row["reference_text"]).strip() != expected_text.strip():
        raise ValueError(f"calibration reference text mismatch for {pair.item_id}/{label}")
    source = _resolve_cli_path(Path(str(row["path"])), data_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"calibration audio not found: {source}")
    return {
        "item_id": pair.item_id,
        "category": pair.category,
        "safety_label": label,
        "style": clean_style,
        "path": _manifest_path(source, data_dir),
        "reference_text": expected_text,
        "source_role": "sarsteer_calibration",
        "pilot_split": "train",
        "source_manifest_status": row.get("status"),
        "transcript_control_passed": bool(row.get("transcript_control_passed", True)),
        "transcript_control_skipped": bool(row.get("transcript_control_skipped", False)),
    }


def _eval_record(
    *,
    cell: dict[str, Any],
    pair: AudioRdoPair,
    source: Path,
    manifest_audio_path: str,
    style: str,
) -> dict[str, Any]:
    if str(cell.get("safety_label", "harmful")) != "harmful":
        raise ValueError(f"Run 7 pilot cell is not harmful: {pair.item_id}/{style}")
    if str(cell["reference_text"]).strip() != pair.harmful_text.strip():
        raise ValueError(f"Run 7 reference text mismatch for {pair.item_id}/{style}")
    wer = cell.get("wer")
    measured_wer = isinstance(wer, int | float) and math.isfinite(float(wer))
    return {
        "item_id": pair.item_id,
        "category": pair.category,
        "safety_label": "harmful",
        "style": style,
        "path": manifest_audio_path,
        "reference_text": pair.harmful_text,
        "source_role": "run7_phase_frontend_pilot_eval",
        "pilot_split": "heldout",
        "run7_condition": str(cell["condition"]),
        "run7_sign": float(cell["sign"]),
        "run7_variant_path": str(cell["variant_path"]),
        "run7_source_behavior_label": cell.get("reviewed_behavior_label"),
        "run7_source_refusal_margin": cell.get("refusal_margin"),
        "run7_source_wer": float(wer) if measured_wer else None,
        "source_audio_sha256": _sha256(source),
        # Run 7 transcribed only a calibration subset with the target model itself;
        # the report says those WERs are unreliable.  Fail closed so this adapter
        # cannot be mistaken for the fresh ASR faithfulness gate required by Run 9.
        "transcript_control_passed": False,
        "transcript_control_skipped": True,
        "pilot_asr_status": (
            "run7_qwen_wer_measured_but_not_gate_qualified" if measured_wer else "not_measured"
        ),
    }


def build_pilot_plan(
    *,
    cfg: ExperimentConfig,
    pairs: Sequence[AudioRdoPair],
    cells: Sequence[dict[str, Any]],
    calibration_rows: Sequence[dict[str, Any]],
    cells_path: Path,
    calibration_manifest_path: Path,
    data_dir: Path,
    variant_root: Path,
    stage_subdir: Path,
    stage_mode: str,
    clean_condition: str = "neutral",
    phase_condition: str = "pv_standard",
    eq_condition: str = "mel_matched_ctrl",
    clean_style: str = "neutral",
    phase_style_prefix: str = "phase_pv_standard",
    eq_style_prefix: str = "eq_mel_matched_ctrl",
    signs: Sequence[float] = (-3.0, 3.0),
) -> PilotPlan:
    if len({clean_condition, phase_condition, eq_condition}) != 3:
        raise ValueError("clean, phase, and EQ conditions must be distinct")
    sign_values = tuple(dict.fromkeys(_sign_key(sign) for sign in signs))
    if not sign_values or any(sign == 0 for sign in sign_values):
        raise ValueError("--signs must contain one or more non-zero pitch values")
    if stage_mode not in {"symlink", "copy", "absolute"}:
        raise ValueError(f"unsupported stage mode: {stage_mode}")

    pair_by_id = {pair.item_id: pair for pair in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("pair manifest contains duplicate item IDs")
    split_map = split_ids(list(pairs), cfg)
    if any(
        split_map[a] & split_map[b]
        for a, b in (("train", "validation"), ("train", "heldout"), ("validation", "heldout"))
    ):
        raise ValueError("configured item splits overlap")

    cells_by_key = _index_cells(cells)
    cell_item_ids = {key[0] for key in cells_by_key}
    unknown_cell_ids = sorted(cell_item_ids - pair_by_id.keys())
    if unknown_cell_ids:
        raise ValueError(f"Run 7 cells contain unknown pair IDs: {unknown_cell_ids[:3]}")
    eval_ids = sorted(cell_item_ids & split_map["heldout"])
    if not eval_ids:
        raise ValueError("Run 7 cells have no items in the configured held-out split")

    calibration_by_key = _index_calibration_rows(calibration_rows, clean_style)
    records: list[dict[str, Any]] = []
    calibration_ids: set[str] = set()
    for item_id in sorted(split_map["train"]):
        pair = pair_by_id[item_id]
        for label in ("harmful", "benign"):
            key = (item_id, label)
            if key not in calibration_by_key:
                raise ValueError(f"missing neutral calibration row: {key}")
            records.append(
                _calibration_record(calibration_by_key[key], pair, data_dir, clean_style)
            )
        calibration_ids.add(item_id)

    overlap = calibration_ids & set(eval_ids)
    if overlap:
        raise ValueError(f"calibration/eval item leakage: {sorted(overlap)[:3]}")

    root = _stage_root(data_dir, stage_subdir)
    actions: list[StageAction] = []
    eval_records: list[dict[str, Any]] = []
    selected_cells: list[dict[str, Any]] = []
    for item_id in eval_ids:
        pair = pair_by_id[item_id]
        requested = [(clean_condition, 0.0)]
        requested.extend((phase_condition, sign) for sign in sign_values)
        requested.extend((eq_condition, sign) for sign in sign_values)
        for condition, sign in requested:
            key = (item_id, condition, _sign_key(sign))
            if key not in cells_by_key:
                raise ValueError(f"missing required Run 7 cell: {key}")
            cell = cells_by_key[key]
            source = _variant_source(str(cell["variant_path"]), variant_root)
            style = _condition_style(
                condition,
                sign,
                clean_condition=clean_condition,
                phase_condition=phase_condition,
                eq_condition=eq_condition,
                clean_style=clean_style,
                phase_style_prefix=phase_style_prefix,
                eq_style_prefix=eq_style_prefix,
            )
            if stage_mode == "absolute":
                manifest_audio_path = str(source)
            else:
                destination = root / "harmful" / style / f"{item_id}.wav"
                actions.append(StageAction(source=source, destination=destination))
                manifest_audio_path = _manifest_path(destination, data_dir)
            eval_records.append(
                _eval_record(
                    cell=cell,
                    pair=pair,
                    source=source,
                    manifest_audio_path=manifest_audio_path,
                    style=style,
                )
            )
            selected_cells.append(cell)
    records.extend(eval_records)

    destination_counts = Counter(action.destination for action in actions)
    collisions = [str(path) for path, count in destination_counts.items() if count > 1]
    if collisions:
        raise ValueError(f"staging destination collisions: {collisions[:3]}")

    style_counts = Counter(str(record["style"]) for record in eval_records)
    configured_styles = set(cfg.dataset.styles)
    missing_styles = sorted(set(style_counts) - configured_styles)
    if missing_styles:
        raise ValueError(
            "adapter styles are absent from dataset.styles: "
            f"{missing_styles}; update the pilot config"
        )
    measured_wer = [
        float(cell["wer"])
        for cell in selected_cells
        if isinstance(cell.get("wer"), int | float) and math.isfinite(float(cell["wer"]))
    ]
    available_harmful = sum(
        record["source_role"] == "sarsteer_calibration" and record["safety_label"] == "harmful"
        for record in records
    )
    available_benign = sum(
        record["source_role"] == "sarsteer_calibration" and record["safety_label"] == "benign"
        for record in records
    )
    warnings = [
        (
            "Direction-finding pilot only: the 91-item Run 7 cohort was selected "
            "as prior neutral refusers."
        ),
        f"Only {len(eval_ids)} Run 7 items fall in this config's held-out split.",
        (
            "The pilot reuses SafeBench/CosyVoice2 audio; it is not the fresh "
            "~350-harmful Run 9 cohort."
        ),
        (
            "Run 7 WER was measured on a subset with Qwen2-Audio and is explicitly "
            "reported as unreliable."
        ),
        (
            "All pilot eval rows fail closed on transcript_control_passed; run a "
            "fresh ASR gate before confirmatory use."
        ),
        (
            "Only the existing requested pitch signs are included; this adapter "
            "does not add acoustic coverage."
        ),
        (
            "Run 7 pv_standard has a high decoding-failure rate, so report "
            "refusal/compliance/failure separately."
        ),
        (
            "Source behavior labels are provenance only; Run 9 undefended/defended "
            "responses must be regenerated and rejudged."
        ),
    ]
    if stage_mode == "symlink":
        warnings.append("Staged symlinks require the Run 7 source directory to remain available.")
    if stage_mode == "absolute":
        warnings.append(
            "Absolute-path mode is non-portable and ties the manifest to this workspace layout."
        )
    if cfg.sarsteer is not None:
        if cfg.sarsteer.n_refusal_calib > available_harmful:
            warnings.append(
                f"Config requests {cfg.sarsteer.n_refusal_calib} harmful calibration "
                "items but only "
                f"{available_harmful} train items are available."
            )
        if cfg.sarsteer.n_benign_pca > available_benign:
            warnings.append(
                f"Config requests {cfg.sarsteer.n_benign_pca} benign PCA items but only "
                f"{available_benign} train items are available."
            )

    metadata = {
        "schema_version": 1,
        "created_utc": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit(),
        "experiment_name": cfg.name,
        "inputs": {
            "cells": str(cells_path.resolve()),
            "cells_sha256": _sha256(cells_path),
            "calibration_manifest": str(calibration_manifest_path.resolve()),
            "calibration_manifest_sha256": _sha256(calibration_manifest_path),
            "variant_root": str(variant_root.resolve()),
            "data_dir": str(data_dir.resolve()),
        },
        "split": {
            "seed": cfg.seed,
            "configured_counts": {name: len(ids) for name, ids in split_map.items()},
            "calibration_split": "train",
            "evaluation_split": "heldout",
            "calibration_item_count": len(calibration_ids),
            "evaluation_item_count": len(eval_ids),
            "calibration_eval_overlap": sorted(overlap),
            "leakage_check_passed": not overlap,
            "evaluation_item_ids": eval_ids,
        },
        "conditions": {
            "clean": {"run7_condition": clean_condition, "signs": [0.0], "style": clean_style},
            "phase_channel_route": {
                "run7_condition": phase_condition,
                "signs": list(sign_values),
                "style_prefix": phase_style_prefix,
                "role": "phase-vocoder independent-bin incoherence / primary channel attack",
            },
            "eq_other_route": {
                "run7_condition": eq_condition,
                "signs": list(sign_values),
                "style_prefix": eq_style_prefix,
                "role": "phase-coherent mel-distance-matched smooth linear-phase EQ control",
            },
        },
        "counts": {
            "manifest_rows": len(records),
            "calibration_rows": len(records) - len(eval_records),
            "evaluation_rows": len(eval_records),
            "evaluation_rows_by_style": dict(sorted(style_counts.items())),
            "stage_actions": len(actions),
        },
        "asr": {
            "gate_qualified": False,
            "selected_rows_with_run7_wer": len(measured_wer),
            "selected_rows_without_run7_wer": len(selected_cells) - len(measured_wer),
            "run7_wer_at_or_below_config_threshold": sum(
                value <= cfg.dataset.transcript_control.wer_max for value in measured_wer
            ),
            "threshold_for_descriptive_count_only": cfg.dataset.transcript_control.wer_max,
            "required_followup": (
                "fresh transcript-faithfulness ASR screen on the pilot/fresh cohort"
            ),
        },
        "staging": {"mode": stage_mode, "root": str(root)},
        "known_limitations": warnings,
    }
    return PilotPlan(records=records, stage_actions=actions, metadata=metadata)


def _files_identical(left: Path, right: Path) -> bool:
    return left.stat().st_size == right.stat().st_size and _sha256(left) == _sha256(right)


def preflight_stage(actions: Sequence[StageAction], stage_mode: str) -> dict[str, int]:
    if stage_mode == "absolute":
        return {"new": 0, "reused": 0}
    new = 0
    reused = 0
    for action in actions:
        if not action.source.is_file():
            raise FileNotFoundError(f"staging source disappeared: {action.source}")
        destination = action.destination
        if destination.exists() or destination.is_symlink():
            same = False
            if destination.is_symlink():
                same = destination.resolve() == action.source.resolve()
            elif destination.is_file():
                same = _files_identical(destination, action.source)
            if not same:
                raise FileExistsError(
                    f"refusing to replace conflicting staged audio: {destination}"
                )
            reused += 1
        else:
            new += 1
    return {"new": new, "reused": reused}


def _atomic_save_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_jsonl(records, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_save_json(value: dict[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        save_json(value, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def materialize_plan(
    plan: PilotPlan,
    *,
    output_path: Path,
    metadata_path: Path,
    stage_mode: str,
    dry_run: bool,
    overwrite: bool,
) -> dict[str, Any]:
    stage_status = preflight_stage(plan.stage_actions, stage_mode)
    existing = [path for path in (output_path, metadata_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to replace existing adapter output(s): {[str(path) for path in existing]}"
        )
    summary = {
        "dry_run": dry_run,
        "output": str(output_path),
        "metadata_output": str(metadata_path),
        "manifest_rows": len(plan.records),
        "calibration_rows": plan.metadata["counts"]["calibration_rows"],
        "evaluation_rows": plan.metadata["counts"]["evaluation_rows"],
        "calibration_item_count": plan.metadata["split"]["calibration_item_count"],
        "evaluation_item_count": plan.metadata["split"]["evaluation_item_count"],
        "leakage_check_passed": plan.metadata["split"]["leakage_check_passed"],
        "stage_mode": stage_mode,
        "stage_new": stage_status["new"],
        "stage_reused": stage_status["reused"],
        "asr_gate_qualified": plan.metadata["asr"]["gate_qualified"],
        "warnings": plan.metadata["known_limitations"],
    }
    if dry_run:
        return summary

    for action in plan.stage_actions:
        destination = action.destination
        if destination.exists() or destination.is_symlink():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        if stage_mode == "symlink":
            destination.symlink_to(action.source.resolve())
        elif stage_mode == "copy":
            shutil.copy2(action.source, destination)
        else:
            raise ValueError(f"unexpected materialized stage mode: {stage_mode}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_save_jsonl(plan.records, output_path)
    metadata = dict(plan.metadata)
    metadata["outputs"] = {
        "manifest": str(output_path.resolve()),
        "metadata": str(metadata_path.resolve()),
    }
    _atomic_save_json(metadata, metadata_path)
    return summary


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment_config(args.config, overrides=args.override)
    paths = resolve_paths(cfg.paths, data_dir=args.data_dir)
    cells_path = args.cells.resolve()
    calibration_path = _resolve_cli_path(args.calibration_manifest, paths.data_dir).resolve()
    if not cells_path.is_file():
        raise FileNotFoundError(f"Run 7 cells file not found: {cells_path}")
    if not calibration_path.is_file():
        raise FileNotFoundError(f"calibration manifest not found: {calibration_path}")
    variant_root = (
        args.variant_root.resolve()
        if args.variant_root is not None
        else cells_path.parent.parent.resolve()
    )
    stage_subdir = args.stage_subdir or cfg.dataset.tts.audio_subdir
    output_path = (
        _resolve_cli_path(args.output, paths.data_dir)
        if args.output is not None
        else paths.data_dir / cfg.dataset.target_generation.outputs_file
    ).resolve()
    metadata_path = (
        _resolve_cli_path(args.metadata_output, paths.data_dir)
        if args.metadata_output is not None
        else output_path.with_name(f"{output_path.stem}.metadata.json")
    ).resolve()

    pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
    plan = build_pilot_plan(
        cfg=cfg,
        pairs=pairs,
        cells=load_jsonl(cells_path),
        calibration_rows=load_jsonl(calibration_path),
        cells_path=cells_path,
        calibration_manifest_path=calibration_path,
        data_dir=paths.data_dir,
        variant_root=variant_root,
        stage_subdir=stage_subdir,
        stage_mode=args.stage_mode,
        clean_condition=args.clean_condition,
        phase_condition=args.phase_condition,
        eq_condition=args.eq_condition,
        clean_style=args.clean_style,
        phase_style_prefix=args.phase_style_prefix,
        eq_style_prefix=args.eq_style_prefix,
        signs=args.signs,
    )
    summary = materialize_plan(
        plan,
        output_path=output_path,
        metadata_path=metadata_path,
        stage_mode=args.stage_mode,
        dry_run=args.dry_run or args.preflight_only,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
