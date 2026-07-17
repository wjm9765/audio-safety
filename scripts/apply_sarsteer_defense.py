#!/usr/bin/env -S uv run python
"""Apply SARSteer and durably append paired generations for each eval row.

The legacy mode selects a configured split from the dataset target outputs.
With --manifest, rows are read directly from an explicit fresh manifest and the
configured target outputs are not touched. A completed row contains both
undefended and defended generations and is flushed immediately, so reruns resume
by the stable (item_id, safety_label, condition/style, sign) input-row key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.config import load_experiment_config
from audio_safety.pipelines.sarsteer import (
    generate_audio_response_with_sarsteer,
    load_sarsteer_metadata,
    load_sarsteer_vectors,
    resolve_sarsteer_implementation,
    sarsteer_system_prompt,
)
from audio_safety.utils.io import load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir

RowKey = tuple[str, str, str, str]


@dataclass(frozen=True)
class PreparedEvalRow:
    row: dict[str, Any]
    key: RowKey
    audio_path: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--split", type=str, default="heldout", help="legacy eval split name")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "explicit eval JSONL; relative paths resolve under data-dir and this mode "
            "does not read config target-generation outputs"
        ),
    )
    parser.add_argument(
        "--undefended-cache",
        type=Path,
        default=None,
        help=(
            "JSONL from run_sarsteer_undefended_baseline.py; reuse its undefended "
            "generations instead of regenerating them. The undefended arm does not "
            "depend on the vector or alpha, so an alpha sweep can share one baseline. "
            "A key missing from the cache is generated normally."
        ),
    )
    parser.add_argument(
        "--vectors",
        type=Path,
        default=None,
        help="SARSteer vectors npz (default: <run_dir>/<sarsteer.vectors_file>)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "paired JSONL override; an absolute path is used directly and a relative "
            "path resolves under the run directory"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="cap pending rows per invocation after resume filtering",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="truncate a prior output and regenerate selected rows instead of resuming",
    )
    return parser.parse_args(argv)


def _required_string(row: dict[str, Any], field: str, *, role: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{role} row {index} requires a non-empty string field '{field}'")
    return value.strip()


def _condition_name(row: dict[str, Any], *, role: str, index: int) -> str:
    value = row.get("condition")
    if value is None:
        value = row.get("style")
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{role} row {index} requires a non-empty 'condition' or legacy 'style'")
    return value.strip()


def _sign_token(value: Any, *, role: str, index: int) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        raise SystemExit(f"{role} row {index} has invalid boolean sign")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{role} row {index} has non-numeric sign: {value!r}") from exc
    if not math.isfinite(number):
        raise SystemExit(f"{role} row {index} has non-finite sign: {value!r}")
    if number == 0:
        number = 0.0
    return f"{number:.12g}"


def row_key(row: dict[str, Any], *, role: str = "input", index: int = 0) -> RowKey:
    return (
        _required_string(row, "item_id", role=role, index=index),
        _required_string(row, "safety_label", role=role, index=index),
        _condition_name(row, role=role, index=index),
        _sign_token(row.get("sign"), role=role, index=index),
    )


def record_id_for_key(key: RowKey) -> str:
    payload = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"sarsteer_{digest}"


def _resolve_manifest_path(path: Path, data_dir: Path) -> Path:
    return (path if path.is_absolute() else data_dir / path).resolve()


def _resolve_audio_path(path: str, data_dir: Path) -> Path:
    candidate = Path(path)
    return (candidate if candidate.is_absolute() else data_dir / candidate).resolve()


def _load_manifest(path: Path, data_dir: Path) -> tuple[Path, list[dict[str, Any]]]:
    resolved = _resolve_manifest_path(path, data_dir)
    if not resolved.is_file():
        raise SystemExit(f"eval manifest not found: {resolved}")
    try:
        rows = load_jsonl(resolved)
    except Exception as exc:
        raise SystemExit(f"failed to read eval manifest {resolved}: {exc}") from exc
    if not rows:
        raise SystemExit(f"eval manifest is empty: {resolved}")
    if any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"eval manifest must contain one JSON object per line: {resolved}")
    return resolved, rows


def prepare_eval_rows(rows: list[dict[str, Any]], *, data_dir: Path) -> list[PreparedEvalRow]:
    """Validate all input rows and resolve absolute/relative audio paths."""

    if not rows:
        raise SystemExit("eval rows are empty")
    seen_keys: set[RowKey] = set()
    seen_record_ids: set[str] = set()
    prepared: list[PreparedEvalRow] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"input row {index} is not a JSON object")
        key = row_key(row, role="input", index=index)
        if key in seen_keys:
            raise SystemExit(f"duplicate input row key: {key}")
        seen_keys.add(key)
        safety_label = _required_string(row, "safety_label", role="input", index=index)
        if safety_label not in {"harmful", "benign"}:
            raise SystemExit(f"input row {index} has unsupported safety_label: {safety_label!r}")
        raw_path = _required_string(row, "path", role="input", index=index)
        audio_path = _resolve_audio_path(raw_path, data_dir)
        if not audio_path.is_file():
            raise SystemExit(f"input audio file not found for row key {key}: {audio_path}")
        input_record_id = row.get("record_id")
        if input_record_id is not None:
            if not isinstance(input_record_id, str) or not input_record_id.strip():
                raise SystemExit(f"input row {index} has invalid record_id")
            if input_record_id in seen_record_ids:
                raise SystemExit(f"duplicate input record_id: {input_record_id}")
            seen_record_ids.add(input_record_id)
        prepared.append(PreparedEvalRow(row=dict(row), key=key, audio_path=audio_path))
    return prepared


def load_completed_rows(
    output_path: Path, *, valid_keys: set[RowKey]
) -> dict[RowKey, dict[str, Any]]:
    """Load only complete, unique paired rows suitable for resume."""

    if not output_path.exists():
        return {}
    if not output_path.is_file():
        raise SystemExit(f"output path exists but is not a file: {output_path}")
    try:
        rows = load_jsonl(output_path)
    except Exception as exc:
        raise SystemExit(
            f"existing output is not valid JSONL; inspect it or use --overwrite: "
            f"{output_path}: {exc}"
        ) from exc
    completed: dict[RowKey, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise SystemExit(f"existing output row {index} is not a JSON object")
        key = row_key(row, role="existing output", index=index)
        if key not in valid_keys:
            raise SystemExit(
                f"existing output row key is absent from the current input manifest: {key}"
            )
        if key in completed:
            raise SystemExit(f"duplicate existing output row key: {key}")
        for field in ("undefended_output", "defended_output"):
            if field not in row or not isinstance(row[field], str):
                raise SystemExit(
                    f"existing output row {index} lacks completed string field '{field}'"
                )
        completed[key] = row
    return completed


def load_undefended_cache(
    path: Path,
    *,
    prepared: Sequence[PreparedEvalRow],
    implementation: str,
    data_dir: Path,
) -> dict[RowKey, str]:
    """Undefended generations reusable for this run, keyed by input-row key.

    ``RowKey`` deliberately excludes the audio path (so a re-render keeps the same
    resume identity), which means the key alone cannot prove the cached generation
    was produced on the SAME audio. This loader therefore also compares the cached
    row's resolved audio path against the current eval row's path per key: if the
    audio was re-rendered between the baseline and this run, the cache is rejected
    rather than silently pairing old-audio undefended text against new-audio
    defended text. The implementation (hence system prompt) must also match.
    """

    if not path.is_file():
        raise SystemExit(f"undefended cache not found: {path}")
    expected_path = {p.key: p.audio_path for p in prepared}
    valid_keys = set(expected_path)
    cache: dict[RowKey, str] = {}
    for index, row in enumerate(load_jsonl(path)):
        if not isinstance(row, dict):
            raise SystemExit(f"undefended cache row {index} is not a JSON object")
        key = row_key(row, role="undefended cache", index=index)
        if key in cache:
            raise SystemExit(f"duplicate undefended cache row key: {key}")
        if key not in valid_keys:
            continue
        output = row.get("undefended_output")
        if not isinstance(output, str):
            raise SystemExit(f"undefended cache row {index} lacks string 'undefended_output'")
        row_impl = row.get("sarsteer_implementation")
        if row_impl != implementation:
            raise SystemExit(
                f"undefended cache row {index} was generated under implementation "
                f"{row_impl!r}, but this run uses {implementation!r}; the system prompt "
                "would differ and the paired contrast would be confounded"
            )
        cached_audio = _resolve_audio_path(str(row.get("path", "")), data_dir)
        if cached_audio != expected_path[key]:
            raise SystemExit(
                f"undefended cache row {index} key={key} was generated on audio "
                f"{cached_audio}, but this run evaluates {expected_path[key]}; the audio "
                "was re-rendered, so the cached undefended arm would be mispaired"
            )
        cache[key] = output
    return cache


def _resolve_output_path(path: Path | None, run_dir: Path) -> Path:
    if path is None:
        return (run_dir / "sarsteer_defended_outputs.jsonl").resolve()
    return (path if path.is_absolute() else run_dir / path).resolve()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be >= 1")
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.sarsteer is None or not cfg.sarsteer.enabled:
        raise SystemExit("cfg.sarsteer is disabled; enable it in the run9 config")
    sar = cfg.sarsteer
    paths = resolve_paths(
        cfg.paths, data_dir=args.data_dir, output_dir=args.output_dir, cache_dir=args.cache_dir
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)
    vectors_path = (
        args.vectors.resolve()
        if args.vectors is not None
        else (run_dir / sar.vectors_file).resolve()
    )
    output_path = _resolve_output_path(args.output, run_dir)

    if args.manifest is not None:
        manifest_path, raw_rows = _load_manifest(args.manifest, paths.data_dir)
        source_description = str(manifest_path)
    else:
        from audio_safety.data import load_audio_rdo_pairs
        from audio_safety.pipelines.rdo_gate import rows_for_split, split_ids

        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        split_map = split_ids(pairs, cfg)
        rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
        raw_rows = rows_for_split(rows, split_map, args.split)
        source_description = f"legacy split '{args.split}'"

    prepared_all = prepare_eval_rows(raw_rows, data_dir=paths.data_dir)
    valid_keys = {prepared.key for prepared in prepared_all}
    if len(valid_keys) != len(prepared_all):
        raise AssertionError("validated eval keys unexpectedly collapsed")
    if args.manifest is not None and output_path == manifest_path:
        raise SystemExit("--output must not overwrite the input --manifest")

    completed = {} if args.overwrite else load_completed_rows(output_path, valid_keys=valid_keys)
    pending_all = [prepared for prepared in prepared_all if prepared.key not in completed]
    pending = pending_all[: args.limit] if args.limit is not None else pending_all
    print(
        f"[sarsteer] apply {source_description}: input={len(prepared_all)} "
        f"completed={len(completed)} pending_total={len(pending_all)} "
        f"this_run={len(pending)}; output={output_path}",
        flush=True,
    )

    if not vectors_path.is_file():
        raise SystemExit(f"SARSteer vectors not found: {vectors_path}")
    vectors = load_sarsteer_vectors(vectors_path)
    vector_meta = load_sarsteer_metadata(vectors_path)
    vector_implementation = resolve_sarsteer_implementation(vector_meta)
    if vector_implementation != sar.implementation:
        raise SystemExit(
            "SARSteer vector/config implementation mismatch: "
            f"vectors={vector_implementation}, config={sar.implementation}"
        )
    system_prompt = sarsteer_system_prompt(vector_implementation)
    undefended_cache: dict[RowKey, str] = {}
    if args.undefended_cache is not None:
        undefended_cache = load_undefended_cache(
            args.undefended_cache.resolve(),
            prepared=prepared_all,
            implementation=vector_implementation,
            data_dir=paths.data_dir,
        )
        hits = sum(1 for prepared in pending if prepared.key in undefended_cache)
        print(
            f"[sarsteer] undefended cache: {len(undefended_cache)} usable rows, "
            f"{hits}/{len(pending)} pending rows reuse it",
            flush=True,
        )
    if not pending:
        print("[sarsteer] all input rows already contain complete pairs", flush=True)
        return

    from audio_safety.models.qwen2_audio import generate_audio_response, load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if args.overwrite else "a"
    with output_path.open(mode, encoding="utf-8") as handle:
        for index, prepared in enumerate(pending, start=1):
            row = prepared.row
            cached_undefended = undefended_cache.get(prepared.key)
            undefended = (
                cached_undefended
                if cached_undefended is not None
                else generate_audio_response(
                    model,
                    processor,
                    prepared.audio_path,
                    instruction,
                    max_new_tokens=sar.max_new_tokens,
                    do_sample=False,
                    system_prompt=system_prompt,
                )
            )
            defended = generate_audio_response_with_sarsteer(
                model,
                processor,
                prepared.audio_path,
                instruction,
                vectors=vectors,
                alpha=sar.alpha,
                max_new_tokens=sar.max_new_tokens,
                do_sample=False,
                system_prompt=system_prompt,
                implementation=vector_implementation,
            )
            record = dict(row)
            record.update(
                {
                    "record_id": row.get("record_id") or record_id_for_key(prepared.key),
                    "defense": "sarsteer",
                    "undefended_output": undefended,
                    "defended_output": defended,
                    "sarsteer_implementation": vector_implementation,
                    "sarsteer_alpha": sar.alpha,
                    "undefended_from_cache": cached_undefended is not None,
                }
            )
            handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            print(
                f"[sarsteer] wrote pair {index}/{len(pending)} key={prepared.key}",
                flush=True,
            )
    print(
        f"[sarsteer] complete: {len(completed) + len(pending)} paired rows -> {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
