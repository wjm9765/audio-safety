#!/usr/bin/env -S uv run python
"""Fail-closed integrity audit for the frozen Run 9 phase/EQ attack cohort.

The report deliberately excludes prompt text.  It validates the manifest grid,
audio payloads and provenance sidecars, then byte-compares every overlapping
Run 7 render to prove that the frozen operator was reproduced.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_CONDITIONS = ("pv_standard", "pv_locked", "mel_matched_ctrl")
EXPECTED_SIGNS = (-3.0, 3.0)
EXPECTED_OPERATOR_VERSION = "run7_phase_ops_20260714_v1"
EXPECTED_PROCESSOR = "Qwen/Qwen2-Audio-7B-Instruct"
EXPECTED_TAGS = {
    "pv_standard": {
        "route": "phase_channel",
        "phase_under_test": True,
        "phase_under_test_tag": "run7_frozen_pv_standard",
        "almguard_training_policy": "exclude_primary_phase_attack_under_test",
    },
    "pv_locked": {
        "route": "phase_repaired_negative_twin",
        "phase_under_test": False,
        "phase_under_test_tag": None,
        "almguard_training_policy": "eligible_non_target_control",
    },
    "mel_matched_ctrl": {
        "route": "eq_other_route",
        "phase_under_test": False,
        "phase_under_test_tag": None,
        "almguard_training_policy": "eligible_non_target_acoustic_route",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL row {line_number} is not an object: {path}")
            rows.append(_without_prompt_fields(value))
    return rows


def _without_prompt_fields(row: dict[str, Any]) -> dict[str, Any]:
    """Drop text-bearing fields before any validation or reporting."""
    return {
        key: value
        for key, value in row.items()
        if key not in {"reference_text", "judge_request_text", "prompt", "target"}
    }


def _sign(value: Any) -> float | None:
    try:
        result = round(float(value), 6)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _sign_tag(value: float) -> str:
    return ("m" if value < 0 else "p") + f"{abs(value):g}".replace(".", "p")


def _key(row: dict[str, Any]) -> tuple[str, float | None, str]:
    return str(row.get("item_id", "")), _sign(row.get("sign")), str(row.get("condition", ""))


def _resolve_under(path_value: Any, root: Path) -> Path | None:
    if not isinstance(path_value, str) or not path_value:
        return None
    candidate = Path(path_value)
    resolved = (candidate if candidate.is_absolute() else root / candidate).resolve()
    return resolved if resolved.is_relative_to(root.resolve()) else None


@dataclass
class Issues:
    error_counts: Counter[str] = field(default_factory=Counter)
    warning_counts: Counter[str] = field(default_factory=Counter)
    error_examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    warning_examples: dict[str, list[str]] = field(default_factory=lambda: defaultdict(list))
    max_examples: int = 12

    def error(self, code: str, context: str) -> None:
        self.error_counts[code] += 1
        if len(self.error_examples[code]) < self.max_examples:
            self.error_examples[code].append(context)

    def warning(self, code: str, context: str) -> None:
        self.warning_counts[code] += 1
        if len(self.warning_examples[code]) < self.max_examples:
            self.warning_examples[code].append(context)

    def report(self) -> dict[str, Any]:
        return {
            "error_total": sum(self.error_counts.values()),
            "warning_total": sum(self.warning_counts.values()),
            "error_counts": dict(sorted(self.error_counts.items())),
            "warning_counts": dict(sorted(self.warning_counts.items())),
            "error_examples": dict(sorted(self.error_examples.items())),
            "warning_examples": dict(sorted(self.warning_examples.items())),
        }


def _inspect_audio(path: Path) -> dict[str, Any]:
    import soundfile as sf

    digest = _sha256(path)
    finite = True
    with sf.SoundFile(path) as handle:
        sample_rate = int(handle.samplerate)
        channels = int(handle.channels)
        frames = int(handle.frames)
        audio_format = str(handle.format)
        for block in handle.blocks(blocksize=65536, dtype="float32", always_2d=True):
            if not np.isfinite(block).all():
                finite = False
                break
    return {
        "sha256": digest,
        "sample_rate": sample_rate,
        "channels": channels,
        "frames": frames,
        "format": audio_format,
        "finite": finite,
    }


def _inspect_paths(
    paths: Iterable[Path],
    *,
    workers: int,
    issues: Issues,
    prefix: str,
    expected_sample_rate: int,
) -> dict[Path, dict[str, Any]]:
    unique = sorted(set(paths))
    results: dict[Path, dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_inspect_audio, path): path for path in unique if path.is_file()}
        for path in unique:
            if not path.is_file():
                issues.error(f"{prefix}_missing", str(path))
        for future in as_completed(futures):
            path = futures[future]
            try:
                results[path] = future.result()
            except Exception as exc:  # corrupt decoders must become reportable failures
                issues.error(f"{prefix}_unreadable", f"{path}: {type(exc).__name__}")
    for path, meta in results.items():
        if meta["sample_rate"] != expected_sample_rate:
            issues.error(f"{prefix}_sample_rate", f"{path}: {meta['sample_rate']}")
        if meta["channels"] != 1:
            issues.error(f"{prefix}_channels", f"{path}: {meta['channels']}")
        if meta["frames"] < 1:
            issues.error(f"{prefix}_empty", str(path))
        if not meta["finite"]:
            issues.error(f"{prefix}_nonfinite", str(path))
        if meta["format"] != "WAV":
            issues.error(f"{prefix}_format", f"{path}: {meta['format']}")
    return results


def _validate_grid(
    rows: Sequence[dict[str, Any]], *, expected_items: int, issues: Issues
) -> tuple[dict[tuple[str, float | None, str], dict[str, Any]], set[str]]:
    indexed: dict[tuple[str, float | None, str], dict[str, Any]] = {}
    items: set[str] = set()
    for row_number, row in enumerate(rows, start=1):
        key = _key(row)
        context = f"row={row_number},key={key}"
        if not key[0] or key[1] not in EXPECTED_SIGNS or key[2] not in EXPECTED_CONDITIONS:
            issues.error("invalid_stable_key", context)
        if key in indexed:
            issues.error("duplicate_stable_key", context)
        else:
            indexed[key] = row
        items.add(key[0])
        if row.get("operator_version") != EXPECTED_OPERATOR_VERSION:
            issues.error("operator_version_mismatch", context)
        if row.get("processor_model_id") != EXPECTED_PROCESSOR:
            issues.error("processor_model_mismatch", context)
        if row.get("sample_rate") != 16000:
            issues.error("manifest_sample_rate_mismatch", context)
        if row.get("render_status") != "rendered":
            issues.error("render_status_mismatch", context)
        if row.get("safety_label") != "harmful" or row.get("gate_role") != "harmful_eval":
            issues.error("gate_role_mismatch", context)
        if row.get("asr_required") is not True:
            issues.error("asr_required_missing", context)
        expected_tag = EXPECTED_TAGS.get(key[2])
        if expected_tag is not None:
            for field_name, expected_value in expected_tag.items():
                if row.get(field_name) != expected_value:
                    issues.error("condition_tag_mismatch", f"{context},field={field_name}")
        reference_digest = row.get("reference_sha256")
        if not (
            isinstance(reference_digest, str)
            and len(reference_digest) == 64
            and all(character in "0123456789abcdef" for character in reference_digest)
        ):
            issues.error("reference_digest_invalid", context)

    if len(rows) != expected_items * len(EXPECTED_SIGNS) * len(EXPECTED_CONDITIONS):
        issues.error("row_count_mismatch", f"actual={len(rows)},expected={expected_items * 6}")
    if len(items) != expected_items:
        issues.error("item_count_mismatch", f"actual={len(items)},expected={expected_items}")
    expected_cells = {
        (sign, condition) for sign in EXPECTED_SIGNS for condition in EXPECTED_CONDITIONS
    }
    for item_id in sorted(items):
        actual_cells = {(sign, condition) for item, sign, condition in indexed if item == item_id}
        if actual_cells != expected_cells:
            issues.error("incomplete_item_grid", item_id)
    return indexed, items


def _validate_clean_provenance(
    rows: Sequence[dict[str, Any]],
    clean_rows: Sequence[dict[str, Any]],
    *,
    data_dir: Path,
    items: set[str],
    workers: int,
    issues: Issues,
) -> tuple[dict[Path, dict[str, Any]], dict[str, int]]:
    clean_index: dict[str, dict[str, Any]] = {}
    for row_number, row in enumerate(clean_rows, start=1):
        if not (
            row.get("gate_role") == "harmful_eval"
            and row.get("safety_label") == "harmful"
            and row.get("condition") == "clean"
        ):
            continue
        item_id = str(row.get("item_id", ""))
        if item_id in clean_index:
            issues.error("duplicate_clean_item", f"row={row_number},item={item_id}")
        clean_index[item_id] = row
        reference_digest = row.get("reference_sha256")
        if not (
            isinstance(reference_digest, str)
            and len(reference_digest) == 64
            and all(character in "0123456789abcdef" for character in reference_digest)
        ):
            issues.error("clean_reference_digest_invalid", item_id)
    if set(clean_index) != items:
        clean_only = len(set(clean_index) - items)
        attack_only = len(items - set(clean_index))
        issues.error(
            "clean_attack_item_set_mismatch",
            f"clean_only={clean_only},attack_only={attack_only}",
        )

    clean_paths: dict[str, Path] = {}
    for item_id, clean_row in clean_index.items():
        path = _resolve_under(clean_row.get("path"), data_dir)
        if path is None:
            issues.error("clean_path_invalid", item_id)
        else:
            clean_paths[item_id] = path
    clean_audio = _inspect_paths(
        clean_paths.values(),
        workers=workers,
        issues=issues,
        prefix="clean_audio",
        expected_sample_rate=24000,
    )

    sidecars_present = 0
    sidecars_verified = 0
    sidecars_missing = 0
    for item_id, clean_row in clean_index.items():
        path = clean_paths.get(item_id)
        if path is None or path not in clean_audio:
            continue
        digest = clean_audio[path]["sha256"]
        text_sidecar = path.with_suffix(f"{path.suffix}.sha256")
        has_valid_text_sidecar = False
        if text_sidecar.is_file():
            sidecars_present += 1
            sidecar_value = text_sidecar.read_text(encoding="utf-8").strip()
            has_valid_text_sidecar = sidecar_value == clean_row.get("reference_sha256")
            if has_valid_text_sidecar:
                sidecars_verified += 1
            else:
                issues.error("clean_text_sidecar_mismatch", item_id)
        else:
            sidecars_missing += 1
        item_rows = [row for row in rows if str(row.get("item_id")) == item_id]
        for row in item_rows:
            context = f"item={item_id},condition={row.get('condition')},sign={row.get('sign')}"
            if row.get("source_clean_path") != clean_row.get("path"):
                issues.error("source_clean_path_mismatch", context)
            if row.get("source_clean_sha256") != digest:
                issues.error("source_clean_hash_mismatch", context)
            if row.get("reference_sha256") != clean_row.get("reference_sha256"):
                issues.error("source_reference_hash_mismatch", context)
            for field_name in (
                "source",
                "category_id",
                "category_name",
                "safety_label",
                "gate_role",
            ):
                if row.get(field_name) != clean_row.get(field_name):
                    issues.error("source_metadata_mismatch", f"{context},field={field_name}")
            if row.get("source_text_hash_verified") is not has_valid_text_sidecar:
                issues.error("source_text_sidecar_flag_mismatch", context)

    return clean_audio, {
        "unique_clean_audio_checked": len(clean_audio),
        "text_hash_sidecars_present": sidecars_present,
        "text_hash_sidecars_verified": sidecars_verified,
        "text_hash_sidecars_missing": sidecars_missing,
        "text_sidecar_flags_consistent": len(clean_index),
    }


def _validate_attack_audio(
    rows: Sequence[dict[str, Any]],
    *,
    data_dir: Path,
    clean_audio: dict[Path, dict[str, Any]],
    workers: int,
    issues: Issues,
) -> tuple[dict[Path, dict[str, Any]], dict[str, Any]]:
    row_paths: dict[int, Path] = {}
    for row_number, row in enumerate(rows, start=1):
        path = _resolve_under(row.get("path"), data_dir)
        if path is None:
            issues.error("attack_path_invalid", f"row={row_number}")
        else:
            row_paths[row_number - 1] = path
    if len(set(row_paths.values())) != len(row_paths):
        issues.error(
            "duplicate_attack_path", f"unique={len(set(row_paths.values()))},rows={len(row_paths)}"
        )
    attack_audio = _inspect_paths(
        row_paths.values(),
        workers=workers,
        issues=issues,
        prefix="attack_audio",
        expected_sample_rate=16000,
    )
    hashes: dict[str, list[int]] = defaultdict(list)
    for row_index, path in row_paths.items():
        meta = attack_audio.get(path)
        if meta is None:
            continue
        hashes[meta["sha256"]].append(row_index)
        if rows[row_index].get("output_sha256") != meta["sha256"]:
            issues.error("output_hash_mismatch", f"row={row_index + 1},path={path}")
    duplicate_groups = [indices for indices in hashes.values() if len(indices) > 1]
    if duplicate_groups:
        issues.error("exact_duplicate_attack_audio", f"groups={len(duplicate_groups)}")
    clean_hashes = {meta["sha256"] for meta in clean_audio.values()}
    attack_clean_hash_overlap = sum(digest in clean_hashes for digest in hashes)
    if attack_clean_hash_overlap:
        issues.error("attack_equals_clean_audio", f"hashes={attack_clean_hash_overlap}")
    clean_hash_counts = Counter(meta["sha256"] for meta in clean_audio.values())
    duplicate_clean_groups = sum(count > 1 for count in clean_hash_counts.values())
    if duplicate_clean_groups:
        issues.warning("exact_duplicate_clean_audio", f"groups={duplicate_clean_groups}")
    return attack_audio, {
        "rows_checked": len(rows),
        "unique_paths_checked": len(attack_audio),
        "unique_output_hashes": len(hashes),
        "duplicate_output_hash_groups": len(duplicate_groups),
        "attack_hashes_equal_to_clean": attack_clean_hash_overlap,
        "duplicate_clean_hash_groups": duplicate_clean_groups,
    }


def _validate_pair_metadata(
    indexed: dict[tuple[str, float | None, str], dict[str, Any]], *, issues: Issues
) -> dict[str, Any]:
    matched = 0
    relative_errors: list[float] = []
    for item_id, sign, condition in sorted(
        indexed, key=lambda value: (value[0], value[1] or 0, value[2])
    ):
        if condition != "pv_standard" or sign not in EXPECTED_SIGNS:
            continue
        group = [indexed.get((item_id, sign, candidate)) for candidate in EXPECTED_CONDITIONS]
        if any(row is None for row in group):
            continue
        values = [row.get("d_pair") for row in group if row is not None]
        if not all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in values
        ):
            issues.error("invalid_d_pair", f"item={item_id},sign={sign}")
            continue
        if max(values) != min(values):
            issues.error("inconsistent_d_pair", f"item={item_id},sign={sign}")
            continue
        control = group[2]
        assert control is not None
        realized = control.get("mel_ctrl_realized_rms")
        strength = control.get("mel_ctrl_strength")
        if not isinstance(realized, (int, float)) or not math.isfinite(realized) or realized <= 0:
            issues.error("invalid_mel_control_rms", f"item={item_id},sign={sign}")
            continue
        if not isinstance(strength, (int, float)) or not math.isfinite(strength) or strength <= 0:
            issues.error("invalid_mel_control_strength", f"item={item_id},sign={sign}")
            continue
        relative_error = abs(float(realized) - float(values[0])) / float(values[0])
        relative_errors.append(relative_error)
        if relative_error > 0.05 + 1e-12:
            issues.error("mel_control_outside_5pct", f"item={item_id},sign={sign}")
        for other in group[:2]:
            assert other is not None
            if (
                other.get("mel_ctrl_realized_rms") is not None
                or other.get("mel_ctrl_strength") is not None
            ):
                issues.error("mel_control_metadata_leak", f"item={item_id},sign={sign}")
        matched += 1
    return {
        "item_sign_pairs_checked": matched,
        "mel_distance_within_5pct": sum(value <= 0.05 + 1e-12 for value in relative_errors),
        "max_relative_error": max(relative_errors, default=None),
    }


def _validate_sidecars(
    indexed: dict[tuple[str, float | None, str], dict[str, Any]],
    *,
    attack_root: Path,
    issues: Issues,
) -> dict[str, int]:
    expected_paths: set[Path] = set()
    checked = 0
    exact_rows = 0
    item_signs = sorted({(item_id, sign) for item_id, sign, _ in indexed if sign in EXPECTED_SIGNS})
    for item_id, sign in item_signs:
        assert sign is not None
        sidecar = (attack_root / "_metadata" / f"{item_id}_{_sign_tag(sign)}.json").resolve()
        expected_paths.add(sidecar)
        if not sidecar.is_file():
            issues.error("attack_sidecar_missing", f"item={item_id},sign={sign}")
            continue
        checked += 1
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except Exception as exc:
            issues.error("attack_sidecar_unreadable", f"{sidecar}: {type(exc).__name__}")
            continue
        expected_group = [
            indexed.get((item_id, sign, condition)) for condition in EXPECTED_CONDITIONS
        ]
        identity = payload.get("identity", {})
        standard = indexed.get((item_id, sign, "pv_standard"), {})
        expected_identity = {
            "operator_version": EXPECTED_OPERATOR_VERSION,
            "item_id": item_id,
            "sign": sign,
            "source_clean_sha256": standard.get("source_clean_sha256"),
            "processor_model_id": EXPECTED_PROCESSOR,
            "sample_rate": 16000,
        }
        if payload.get("schema_version") != 1 or identity != expected_identity:
            issues.error("attack_sidecar_identity_mismatch", f"item={item_id},sign={sign}")
        raw_sidecar_rows = payload.get("rows")
        sidecar_rows = (
            [_without_prompt_fields(row) for row in raw_sidecar_rows]
            if isinstance(raw_sidecar_rows, list)
            and all(isinstance(row, dict) for row in raw_sidecar_rows)
            else raw_sidecar_rows
        )
        if not isinstance(sidecar_rows, list) or sidecar_rows != expected_group:
            issues.error("attack_sidecar_rows_mismatch", f"item={item_id},sign={sign}")
        else:
            exact_rows += len(sidecar_rows)
    actual_paths = set((attack_root / "_metadata").glob("*.json"))
    unexpected = actual_paths - expected_paths
    if unexpected:
        issues.error("unexpected_attack_sidecar", f"count={len(unexpected)}")
    return {
        "expected": len(expected_paths),
        "checked": checked,
        "manifest_rows_exactly_matched": exact_rows,
        "unexpected": len(unexpected),
    }


def _validate_frozen_operator(
    indexed: dict[tuple[str, float | None, str], dict[str, Any]],
    *,
    run7_cells: Path,
    run7_root: Path,
    direction_doc: Path,
    run7_doc: Path,
    repo_root: Path,
    workers: int,
    issues: Issues,
) -> dict[str, Any]:
    from audio_safety.evaluation import phase_ops

    signature = inspect.signature(phase_ops.pitch_shift_custom)
    runtime_params = {
        "n_fft": phase_ops.PV_N_FFT,
        "hop_length": phase_ops.PV_HOP,
        "res_type": signature.parameters["res_type"].default,
        "signs": list(EXPECTED_SIGNS),
    }
    if runtime_params != {
        "n_fft": 2048,
        "hop_length": 512,
        "res_type": "soxr_hq",
        "signs": [-3.0, 3.0],
    }:
        issues.error("frozen_runtime_parameter_mismatch", str(runtime_params))

    direction_text = direction_doc.read_text(encoding="utf-8")
    run7_text = run7_doc.read_text(encoding="utf-8")
    direction_requirements = {
        "same_clean_dsp_no_extra_tts": "no extra TTS" in direction_text,
        "phase_excluded_from_almguard_training": (
            "channel/phase attack MUST be excluded" in direction_text
        ),
        "phase_and_eq_routes": "phase and EQ operators" in direction_text,
    }
    run7_requirements = {
        "signed_three_semitones": "{−3, +3}" in run7_text,
        "pv_standard": "`pv_standard`" in run7_text,
        "pv_locked": "`pv_locked`" in run7_text,
        "mel_matched_ctrl": "`mel_matched_ctrl`" in run7_text,
        "mel_distance_5pct": "within 5%" in run7_text,
    }
    for name, passed in {**direction_requirements, **run7_requirements}.items():
        if not passed:
            issues.error("frozen_document_requirement_missing", name)

    old_index: dict[tuple[str, float | None, str], dict[str, Any]] = {}
    for row in _load_jsonl(run7_cells):
        key = _key(row)
        if key[1] not in EXPECTED_SIGNS or key[2] not in EXPECTED_CONDITIONS:
            continue
        if key in old_index:
            issues.error("duplicate_run7_cell", str(key))
        old_index[key] = row
    common_keys = sorted(
        set(indexed) & set(old_index), key=lambda value: (value[0], value[1] or 0, value[2])
    )
    missing_new = set(old_index) - set(indexed)
    if missing_new:
        issues.error("run7_cells_missing_from_run9", f"count={len(missing_new)}")
    old_paths: dict[tuple[str, float | None, str], Path] = {}
    for key in common_keys:
        candidate = Path(str(old_index[key].get("variant_path", "")))
        path = (candidate if candidate.is_absolute() else run7_root / candidate).resolve()
        old_paths[key] = path
    old_audio = _inspect_paths(
        old_paths.values(),
        workers=workers,
        issues=issues,
        prefix="run7_audio",
        expected_sample_rate=16000,
    )

    byte_matches = 0
    metadata_matches = 0
    new_hash_by_key = {key: indexed[key].get("output_sha256") for key in common_keys}
    for key, old_path in old_paths.items():
        old_meta = old_audio.get(old_path)
        if old_meta is not None and old_meta["sha256"] == new_hash_by_key[key]:
            byte_matches += 1
        else:
            issues.error("run7_byte_reproduction_mismatch", str(key))
        old = old_index[key]
        new = indexed[key]
        comparable = ["d_pair", "mel_ctrl_realized_rms"]
        if all(old.get(field_name) == new.get(field_name) for field_name in comparable):
            metadata_matches += 1
        else:
            issues.error("run7_metadata_reproduction_mismatch", str(key))

    phase_source = repo_root / "src" / "audio_safety" / "evaluation" / "phase_ops.py"
    renderer_source = repo_root / "scripts" / "render_run9_fresh_attacks.py"
    return {
        "operator_version": EXPECTED_OPERATOR_VERSION,
        "runtime_parameters": runtime_params,
        "phase_ops_sha256": _sha256(phase_source),
        "renderer_sha256": _sha256(renderer_source),
        "direction_doc_sha256": _sha256(direction_doc),
        "run7_doc_sha256": _sha256(run7_doc),
        "direction_requirements": direction_requirements,
        "run7_requirements": run7_requirements,
        "run7_cells_eligible": len(old_index),
        "run7_cells_compared": len(common_keys),
        "run7_byte_exact_matches": byte_matches,
        "run7_metadata_exact_matches": metadata_matches,
    }


def validate(
    *,
    manifest: Path,
    clean_manifest: Path,
    data_dir: Path,
    attack_root: Path,
    run7_cells: Path,
    run7_root: Path,
    direction_doc: Path,
    run7_doc: Path,
    repo_root: Path,
    expected_items: int,
    workers: int,
) -> dict[str, Any]:
    issues = Issues()
    rows = _load_jsonl(manifest)
    clean_rows = _load_jsonl(clean_manifest)
    indexed, items = _validate_grid(rows, expected_items=expected_items, issues=issues)
    clean_audio, clean_summary = _validate_clean_provenance(
        rows,
        clean_rows,
        data_dir=data_dir,
        items=items,
        workers=workers,
        issues=issues,
    )
    _, attack_summary = _validate_attack_audio(
        rows,
        data_dir=data_dir,
        clean_audio=clean_audio,
        workers=workers,
        issues=issues,
    )
    pair_summary = _validate_pair_metadata(indexed, issues=issues)
    sidecar_summary = _validate_sidecars(indexed, attack_root=attack_root, issues=issues)
    frozen_summary = _validate_frozen_operator(
        indexed,
        run7_cells=run7_cells,
        run7_root=run7_root,
        direction_doc=direction_doc,
        run7_doc=run7_doc,
        repo_root=repo_root,
        workers=workers,
        issues=issues,
    )
    issue_report = issues.report()
    return {
        "schema_version": 1,
        "status": "pass" if issue_report["error_total"] == 0 else "fail",
        "prompt_text_included": False,
        "inputs": {
            "manifest": str(manifest.resolve()),
            "manifest_sha256": _sha256(manifest),
            "clean_manifest": str(clean_manifest.resolve()),
            "clean_manifest_sha256": _sha256(clean_manifest),
            "data_dir": str(data_dir.resolve()),
            "run7_cells": str(run7_cells.resolve()),
            "run7_cells_sha256": _sha256(run7_cells),
        },
        "grid": {
            "manifest_rows": len(rows),
            "unique_items": len(items),
            "stable_keys": len(indexed),
            "expected_conditions": list(EXPECTED_CONDITIONS),
            "expected_signs": list(EXPECTED_SIGNS),
            "condition_counts": dict(
                sorted(Counter(str(row.get("condition")) for row in rows).items())
            ),
            "sign_counts": dict(
                sorted(Counter(str(_sign(row.get("sign"))) for row in rows).items())
            ),
        },
        "clean_provenance": clean_summary,
        "attack_audio": attack_summary,
        "pair_metadata": pair_summary,
        "sidecars": sidecar_summary,
        "training_policy_semantics": {
            "primary_phase_rows_excluded": sum(
                row.get("condition") == "pv_standard"
                and row.get("almguard_training_policy") == "exclude_primary_phase_attack_under_test"
                for row in rows
            ),
            "phase_repaired_controls_eligible": sum(
                row.get("condition") == "pv_locked"
                and row.get("almguard_training_policy") == "eligible_non_target_control"
                for row in rows
            ),
            "eq_route_controls_eligible": sum(
                row.get("condition") == "mel_matched_ctrl"
                and row.get("almguard_training_policy") == "eligible_non_target_acoustic_route"
                for row in rows
            ),
            "only_pv_standard_phase_under_test": all(
                bool(row.get("phase_under_test")) == (row.get("condition") == "pv_standard")
                for row in rows
            ),
        },
        "frozen_operator_audit": frozen_summary,
        "issues": issue_report,
    }


def _atomic_write_json(value: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    docs_root = repo_root / "docs/experiments/exp1_refusal_cone_drift"
    data_dir = Path("/workspace/audio_safety_data/data")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=data_dir / "manifests/run9_fresh_attacks.jsonl"
    )
    parser.add_argument(
        "--clean-manifest", type=Path, default=data_dir / "manifests/run9_fresh_clean.jsonl"
    )
    parser.add_argument("--data-dir", type=Path, default=data_dir)
    parser.add_argument("--attack-root", type=Path, default=data_dir / "audio_run9/attacks")
    parser.add_argument(
        "--run7-cells",
        type=Path,
        default=Path(
            "/workspace/audio_safety_data/outputs/run7_20260714_phase_frontend/pitch_frontend/cells.jsonl"
        ),
    )
    parser.add_argument(
        "--run7-root",
        type=Path,
        default=Path("/workspace/audio_safety_data/outputs/run7_20260714_phase_frontend"),
    )
    parser.add_argument(
        "--direction-doc",
        type=Path,
        default=docs_root / "run9_advisor_defense_gate_direction_20260717.md",
    )
    parser.add_argument(
        "--run7-doc",
        type=Path,
        default=docs_root / "run7_phase_frontend_distortion_direction_20260714.md",
    )
    parser.add_argument("--repo-root", type=Path, default=repo_root)
    parser.add_argument("--expected-items", type=int, default=350)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/workspace/audio_safety_data/outputs/run9_fresh/audio_transform_integrity_report.json"
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.expected_items < 1 or args.workers < 1:
        raise ValueError("--expected-items and --workers must be positive")
    required = [
        args.manifest,
        args.clean_manifest,
        args.run7_cells,
        args.direction_doc,
        args.run7_doc,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"required audit inputs missing: {missing}")
    report = validate(
        manifest=args.manifest,
        clean_manifest=args.clean_manifest,
        data_dir=args.data_dir,
        attack_root=args.attack_root,
        run7_cells=args.run7_cells,
        run7_root=args.run7_root,
        direction_doc=args.direction_doc,
        run7_doc=args.run7_doc,
        repo_root=args.repo_root,
        expected_items=args.expected_items,
        workers=args.workers,
    )
    _atomic_write_json(report, args.output)
    print(
        json.dumps(
            {
                "status": report["status"],
                "manifest_rows": report["grid"]["manifest_rows"],
                "attack_audio_checked": report["attack_audio"]["unique_paths_checked"],
                "run7_byte_exact_matches": report["frozen_operator_audit"][
                    "run7_byte_exact_matches"
                ],
                "errors": report["issues"]["error_total"],
                "warnings": report["issues"]["warning_total"],
                "report": str(args.output.resolve()),
            },
            indent=2,
        )
    )
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
