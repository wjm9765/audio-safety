#!/usr/bin/env -S uv run python
"""Run a small, resumable SARSteer strength/layer diagnostic sweep.

The canonical paired input already contains the deterministic undefended response
and the current all-layer, alpha=0.1 response.  This script preserves that full
record and adds exactly one locally generated response for each requested
``record_id``/setting pair.  It never calls an external service or judge.

Settings use ``NAME:ALPHA:LAYERS`` where ``LAYERS`` is ``all``, one non-negative
integer, or a comma/range expression such as ``16-23`` or ``8,12,16-19``.
Completed JSONL rows are durably appended and a rerun resumes by the stable
``(record_id, sweep_setting)`` key without loading the model when nothing remains.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from audio_safety.config import load_experiment_config
from audio_safety.pipelines.sarsteer import (
    generate_audio_response_with_sarsteer,
    load_sarsteer_metadata,
    load_sarsteer_vectors,
    resolve_sarsteer_implementation,
    sarsteer_system_prompt,
)
from audio_safety.utils.io import load_jsonl

SCHEMA_VERSION = "sarsteer_diagnostic_sweep.v1"
SETTING_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
RANGE_PATTERN = re.compile(r"(\d+)-(\d+)")
INTEGER_PATTERN = re.compile(r"\d+")

SweepKey = tuple[str, str]

RESERVED_OUTPUT_FIELDS = frozenset(
    {
        "sarsteer_sweep_schema",
        "baseline_undefended_output",
        "current_all_alpha_0_1_output",
        "sweep_setting",
        "sweep_alpha",
        "sweep_layer_spec",
        "sweep_layers",
        "sweep_max_new_tokens",
        "sweep_do_sample",
        "sweep_input_sha256",
        "sweep_output",
    }
)


@dataclass(frozen=True)
class SweepSetting:
    name: str
    alpha: float
    layer_spec: str
    layers: tuple[int, ...] | None


@dataclass(frozen=True)
class ResolvedSweepSetting:
    name: str
    alpha: float
    layer_spec: str
    layers: tuple[int, ...]


@dataclass(frozen=True)
class SelectedRecord:
    record_id: str
    row: dict[str, Any]
    audio_path: Path
    input_sha256: str


@dataclass(frozen=True)
class SweepJob:
    record: SelectedRecord
    setting: ResolvedSweepSetting

    @property
    def key(self) -> SweepKey:
        return (self.record.record_id, self.setting.name)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--vectors", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--record-id",
        action="append",
        required=True,
        help="exact canonical paired record_id; repeat for each diagnostic input",
    )
    parser.add_argument(
        "--setting",
        action="append",
        required=True,
        metavar="NAME:ALPHA:LAYERS",
        help="repeatable steering setting, e.g. low:0.003:all or mid:0.01:16-23",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="deterministic generation horizon (default: config sarsteer.max_new_tokens)",
    )
    return parser.parse_args(argv)


def parse_layer_spec(raw: str) -> tuple[int, ...] | None:
    """Parse ``all`` or a fail-closed comma/range layer expression."""

    value = raw.strip()
    if value == "all":
        return None
    if not value:
        raise ValueError("layer specification is empty")

    layers: list[int] = []
    seen: set[int] = set()
    for raw_part in value.split(","):
        part = raw_part.strip()
        if not part:
            raise ValueError(f"invalid empty component in layer specification {raw!r}")
        if INTEGER_PATTERN.fullmatch(part):
            expanded = (int(part),)
        else:
            match = RANGE_PATTERN.fullmatch(part)
            if match is None:
                raise ValueError(f"invalid layer component {part!r} in {raw!r}")
            start, end = (int(match.group(1)), int(match.group(2)))
            if end < start:
                raise ValueError(f"descending layer range is not allowed: {part!r}")
            expanded = tuple(range(start, end + 1))
        for layer in expanded:
            if layer in seen:
                raise ValueError(f"duplicate layer {layer} in specification {raw!r}")
            seen.add(layer)
            layers.append(layer)
    return tuple(sorted(layers))


def parse_setting(raw: str) -> SweepSetting:
    """Parse one ``NAME:ALPHA:LAYERS`` setting without consulting the model."""

    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"setting must be NAME:ALPHA:LAYERS, got {raw!r}")
    name, raw_alpha, layer_spec = (part.strip() for part in parts)
    if SETTING_NAME_PATTERN.fullmatch(name) is None:
        raise ValueError(f"invalid setting name {name!r}; use letters, digits, '_', '-', or '.'")
    try:
        alpha = float(raw_alpha)
    except ValueError as exc:
        raise ValueError(f"setting {name!r} has non-numeric alpha {raw_alpha!r}") from exc
    if not math.isfinite(alpha):
        raise ValueError(f"setting {name!r} has non-finite alpha {raw_alpha!r}")
    layers = parse_layer_spec(layer_spec)
    canonical_spec = "all" if layers is None else ",".join(str(layer) for layer in layers)
    return SweepSetting(name=name, alpha=alpha, layer_spec=canonical_spec, layers=layers)


def parse_settings(raw_settings: Sequence[str]) -> list[SweepSetting]:
    if not raw_settings:
        raise ValueError("at least one --setting is required")
    settings: list[SweepSetting] = []
    names: set[str] = set()
    explicit_specs: set[tuple[float, tuple[int, ...] | None]] = set()
    for raw in raw_settings:
        setting = parse_setting(raw)
        if setting.name in names:
            raise ValueError(f"duplicate setting name: {setting.name!r}")
        effective = (setting.alpha, setting.layers)
        if effective in explicit_specs:
            raise ValueError(f"duplicate effective setting (alpha/layers) for {setting.name!r}")
        names.add(setting.name)
        explicit_specs.add(effective)
        settings.append(setting)
    return settings


def resolve_settings(
    settings: Sequence[SweepSetting], available_layers: Sequence[int]
) -> list[ResolvedSweepSetting]:
    """Resolve ``all`` and reject unavailable or effectively duplicate settings."""

    available = tuple(sorted(available_layers))
    if not available or len(set(available)) != len(available):
        raise ValueError("SARSteer vectors must expose a non-empty unique layer set")
    if any(
        isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in available
    ):
        raise ValueError(f"SARSteer vectors contain invalid layer keys: {available!r}")
    available_set = set(available)

    resolved: list[ResolvedSweepSetting] = []
    effective_specs: dict[tuple[float, tuple[int, ...]], str] = {}
    for setting in settings:
        layers = available if setting.layers is None else setting.layers
        missing = sorted(set(layers) - available_set)
        if missing:
            raise ValueError(
                f"setting {setting.name!r} requests layers absent from vectors: {missing}"
            )
        effective = (setting.alpha, layers)
        prior = effective_specs.get(effective)
        if prior is not None:
            raise ValueError(
                f"settings {prior!r} and {setting.name!r} resolve to the same alpha/layers"
            )
        effective_specs[effective] = setting.name
        resolved.append(
            ResolvedSweepSetting(
                name=setting.name,
                alpha=setting.alpha,
                layer_spec=setting.layer_spec,
                layers=layers,
            )
        )
    return resolved


def _required_text(row: Mapping[str, Any], field: str, *, role: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{role} requires a non-empty string field {field!r}")
    return value


def _record_fingerprint(row: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_paired_records(
    rows: Sequence[Mapping[str, Any]],
    record_ids: Sequence[str],
    *,
    data_dir: Path,
) -> list[SelectedRecord]:
    """Select exact record IDs in request order and resolve their audio paths."""

    if not record_ids:
        raise ValueError("at least one --record-id is required")
    requested: list[str] = []
    requested_set: set[str] = set()
    for index, value in enumerate(record_ids):
        if not isinstance(value, str) or not value.strip() or value != value.strip():
            raise ValueError(f"requested record_id {index} is empty or has surrounding whitespace")
        if value in requested_set:
            raise ValueError(f"duplicate requested record_id: {value!r}")
        requested.append(value)
        requested_set.add(value)

    by_id: dict[str, Mapping[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"paired row {index} is not a JSON object")
        record_id = _required_text(row, "record_id", role=f"paired row {index}")
        if record_id != record_id.strip():
            raise ValueError(f"paired row {index} record_id has surrounding whitespace")
        if record_id in by_id:
            raise ValueError(f"duplicate paired record_id: {record_id!r}")
        by_id[record_id] = row

    missing = [record_id for record_id in requested if record_id not in by_id]
    if missing:
        raise ValueError(f"requested record_ids missing from paired input: {missing}")

    selected: list[SelectedRecord] = []
    for record_id in requested:
        row = dict(by_id[record_id])
        collisions = sorted(RESERVED_OUTPUT_FIELDS.intersection(row))
        if collisions:
            raise ValueError(
                f"paired record {record_id!r} collides with sweep fields: {collisions}"
            )
        for field in ("undefended_output", "defended_output"):
            if field not in row or not isinstance(row[field], str):
                raise ValueError(
                    f"paired record {record_id!r} requires completed string field {field!r}"
                )
        if row.get("defense") != "sarsteer":
            raise ValueError(f"paired record {record_id!r} is not marked defense='sarsteer'")
        raw_audio = _required_text(row, "path", role=f"paired record {record_id!r}")
        candidate = Path(raw_audio)
        audio_path = (candidate if candidate.is_absolute() else data_dir / candidate).resolve()
        if not audio_path.is_file():
            raise ValueError(f"audio for paired record {record_id!r} not found: {audio_path}")
        selected.append(
            SelectedRecord(
                record_id=record_id,
                row=row,
                audio_path=audio_path,
                input_sha256=_record_fingerprint(row),
            )
        )
    return selected


def _expected_jobs(
    records: Sequence[SelectedRecord], settings: Sequence[ResolvedSweepSetting]
) -> list[SweepJob]:
    return [SweepJob(record, setting) for record in records for setting in settings]


def validate_completed_rows(
    rows: Sequence[Mapping[str, Any]],
    records: Sequence[SelectedRecord],
    settings: Sequence[ResolvedSweepSetting],
    *,
    max_new_tokens: int,
) -> dict[SweepKey, dict[str, Any]]:
    """Validate an existing resume file against the exact inputs and settings."""

    records_by_id = {record.record_id: record for record in records}
    settings_by_name = {setting.name: setting for setting in settings}
    valid_keys = {job.key for job in _expected_jobs(records, settings)}
    completed: dict[SweepKey, dict[str, Any]] = {}
    for index, raw_row in enumerate(rows):
        if not isinstance(raw_row, Mapping):
            raise ValueError(f"existing output row {index} is not a JSON object")
        row = dict(raw_row)
        record_id = _required_text(row, "record_id", role=f"existing output row {index}")
        setting_name = _required_text(row, "sweep_setting", role=f"existing output row {index}")
        key = (record_id, setting_name)
        if key not in valid_keys:
            raise ValueError(f"existing output contains foreign record/setting key: {key}")
        if key in completed:
            raise ValueError(f"duplicate existing output record/setting key: {key}")

        record = records_by_id[record_id]
        setting = settings_by_name[setting_name]
        for field, value in record.row.items():
            if row.get(field) != value:
                raise ValueError(
                    f"existing output key {key} does not preserve paired field {field!r}"
                )
        expected_fields: dict[str, Any] = {
            "sarsteer_sweep_schema": SCHEMA_VERSION,
            "baseline_undefended_output": record.row["undefended_output"],
            "current_all_alpha_0_1_output": record.row["defended_output"],
            "sweep_setting": setting.name,
            "sweep_alpha": setting.alpha,
            "sweep_layer_spec": setting.layer_spec,
            "sweep_layers": list(setting.layers),
            "sweep_max_new_tokens": max_new_tokens,
            "sweep_do_sample": False,
            "sweep_input_sha256": record.input_sha256,
        }
        for field, value in expected_fields.items():
            if row.get(field) != value:
                raise ValueError(
                    f"existing output key {key} has incompatible {field!r}: "
                    f"expected {value!r}, got {row.get(field)!r}"
                )
        if "sweep_output" not in row or not isinstance(row["sweep_output"], str):
            raise ValueError(f"existing output key {key} lacks string 'sweep_output'")
        completed[key] = row
    return completed


def plan_pending_jobs(
    records: Sequence[SelectedRecord],
    settings: Sequence[ResolvedSweepSetting],
    completed: Mapping[SweepKey, Mapping[str, Any]],
) -> list[SweepJob]:
    """Return deterministic record-major pending jobs and reject foreign keys."""

    jobs = _expected_jobs(records, settings)
    expected = {job.key for job in jobs}
    foreign = sorted(set(completed) - expected)
    if foreign:
        raise ValueError(f"completed rows contain foreign resume keys: {foreign}")
    return [job for job in jobs if job.key not in completed]


def _load_jsonl_checked(
    path: Path, *, role: str, allow_missing: bool = False
) -> list[dict[str, Any]]:
    if not path.exists():
        if allow_missing:
            return []
        raise SystemExit(f"{role} not found: {path}")
    if not path.is_file():
        raise SystemExit(f"{role} path is not a file: {path}")
    try:
        rows = load_jsonl(path)
    except Exception as exc:
        raise SystemExit(f"failed to read {role} {path}: {exc}") from exc
    if not rows and not allow_missing:
        raise SystemExit(f"{role} is empty: {path}")
    return rows


def _validate_output_collision(output: Path, protected_paths: Sequence[Path]) -> None:
    for protected in protected_paths:
        if output == protected.resolve():
            raise SystemExit(f"--output collides with protected input path: {protected.resolve()}")
    if output.exists() and not output.is_file():
        raise SystemExit(f"--output exists but is not a file: {output}")


def _build_output_row(job: SweepJob, output: str, *, max_new_tokens: int) -> dict[str, Any]:
    row = dict(job.record.row)
    row.update(
        {
            "sarsteer_sweep_schema": SCHEMA_VERSION,
            "baseline_undefended_output": job.record.row["undefended_output"],
            "current_all_alpha_0_1_output": job.record.row["defended_output"],
            "sweep_setting": job.setting.name,
            "sweep_alpha": job.setting.alpha,
            "sweep_layer_spec": job.setting.layer_spec,
            "sweep_layers": list(job.setting.layers),
            "sweep_max_new_tokens": max_new_tokens,
            "sweep_do_sample": False,
            "sweep_input_sha256": job.record.input_sha256,
            "sweep_output": output,
        }
    )
    return row


def append_durable_jsonl(handle: TextIO, row: Mapping[str, Any]) -> None:
    """Append one complete JSON object, then flush and fsync it before returning."""

    handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.max_new_tokens is not None and args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be >= 1")

    config_path = args.config.resolve()
    data_dir = args.data_dir.resolve()
    cache_dir = args.cache_dir.resolve()
    paired_path = args.paired.resolve()
    vectors_path = args.vectors.resolve()
    output_path = args.output.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"--data-dir is not a directory: {data_dir}")
    if not config_path.is_file():
        raise SystemExit(f"--config not found: {config_path}")
    if not vectors_path.is_file():
        raise SystemExit(f"--vectors not found: {vectors_path}")
    _validate_output_collision(
        output_path, (config_path, paired_path, vectors_path, data_dir, cache_dir)
    )

    try:
        settings = parse_settings(args.setting)
        paired_rows = _load_jsonl_checked(paired_path, role="canonical paired input")
        records = select_paired_records(paired_rows, args.record_id, data_dir=data_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    _validate_output_collision(output_path, [record.audio_path for record in records])

    cfg = load_experiment_config(config_path, overrides=args.override)
    if cfg.sarsteer is None or not cfg.sarsteer.enabled:
        raise SystemExit("cfg.sarsteer is disabled")
    if not math.isclose(cfg.sarsteer.alpha, 0.1, rel_tol=0.0, abs_tol=1e-12):
        raise SystemExit(
            "canonical current defended outputs are expected to be all-layer alpha=0.1; "
            f"config has alpha={cfg.sarsteer.alpha}"
        )
    max_new_tokens = args.max_new_tokens or cfg.sarsteer.max_new_tokens

    vectors = load_sarsteer_vectors(vectors_path)
    vector_meta = load_sarsteer_metadata(vectors_path)
    vector_implementation = resolve_sarsteer_implementation(vector_meta)
    if vector_implementation != cfg.sarsteer.implementation:
        raise SystemExit(
            "SARSteer vector/config implementation mismatch: "
            f"vectors={vector_implementation}, config={cfg.sarsteer.implementation}"
        )
    system_prompt = sarsteer_system_prompt(vector_implementation)
    try:
        resolved_settings = resolve_settings(settings, list(vectors))
        existing_rows = _load_jsonl_checked(
            output_path, role="existing sweep output", allow_missing=True
        )
        completed = validate_completed_rows(
            existing_rows,
            records,
            resolved_settings,
            max_new_tokens=max_new_tokens,
        )
        pending = plan_pending_jobs(records, resolved_settings, completed)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    print(
        f"[sarsteer-sweep] records={len(records)} settings={len(resolved_settings)} "
        f"completed={len(completed)} pending={len(pending)} output={output_path}",
        flush=True,
    )
    if not pending:
        print("[sarsteer-sweep] complete; model load skipped", flush=True)
        return

    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for index, job in enumerate(pending, start=1):
            selected_vectors = {layer: vectors[layer] for layer in job.setting.layers}
            generated = generate_audio_response_with_sarsteer(
                model,
                processor,
                job.record.audio_path,
                instruction,
                vectors=selected_vectors,
                alpha=job.setting.alpha,
                max_new_tokens=max_new_tokens,
                system_prompt=system_prompt,
                implementation=vector_implementation,
                do_sample=False,
            )
            output_row = _build_output_row(job, generated, max_new_tokens=max_new_tokens)
            append_durable_jsonl(handle, output_row)
            print(
                f"[sarsteer-sweep] wrote {index}/{len(pending)} "
                f"record_id={job.record.record_id} setting={job.setting.name}",
                flush=True,
            )
    print(
        f"[sarsteer-sweep] complete: {len(completed) + len(pending)} rows -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
