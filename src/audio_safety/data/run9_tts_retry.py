"""Safe, resumable remediation for Run 9 clean-audio ASR failures.

The retry cohort is deliberately narrower than the ASR checkpoint: only rows
with ``asr_status='ok'`` and ``transcript_control_passed is False`` are selected.
Decode errors and rows that have not been checkpointed are audit counts, never
TTS retry candidates.

Retries render into a versioned directory.  A passing candidate is promoted by
atomically changing the clean manifest path; the original WAV is never replaced.
This keeps the original waveform available as the immutable backup and makes a
retry reversible by restoring the versioned manifest backup.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.evaluation.asr_faithfulness import asr_record_key

RETRY_CONTRACT_VERSION = "run9_clean_asr_retry_v1"
RETRY_STYLE = "neutral"
RETRY_SPEED = 1.0
_SAFE_RETRY_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
_HEX_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class RetryPlan:
    candidate_rows: list[dict[str, Any]]
    tts_jobs: list[dict[str, Any]]
    summary: dict[str, Any]


@dataclass(frozen=True)
class PromotionPlan:
    clean_rows: list[dict[str, Any]]
    promoted_item_ids: list[str]
    promoted_harmful_item_ids: list[str]
    summary: dict[str, Any]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _resolve_data_path(raw: object, data_dir: Path) -> Path:
    value = str(raw or "").strip()
    if not value:
        raise ValueError("row has no non-empty path")
    path = Path(value)
    resolved = (path if path.is_absolute() else data_dir / path).resolve()
    if not resolved.is_relative_to(data_dir.resolve()):
        raise ValueError(f"Run 9 audio path escapes data_dir: {resolved}")
    return resolved


def _relative_to_data(path: Path, data_dir: Path) -> str:
    try:
        return path.resolve().relative_to(data_dir.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"Run 9 retry path escapes data_dir: {path}") from exc


def _validate_retry_id(retry_id: str) -> None:
    if not _SAFE_RETRY_ID.fullmatch(retry_id):
        raise ValueError("retry_id must be 1..80 safe filename characters and start alphanumeric")


def deterministic_retry_seed(item_id: str, base_seed: int) -> int:
    """Derive a stable per-item CosyVoice sampling seed without Python hash()."""
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    payload = f"{RETRY_CONTRACT_VERSION}|{base_seed}|{item_id}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big")


def _index_unique(rows: Sequence[Mapping[str, Any]], key_name: str) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        key = str(row.get(key_name) or "").strip()
        if not key:
            raise ValueError(f"row has no non-empty {key_name}")
        if key in indexed:
            raise ValueError(f"duplicate {key_name}: {key}")
        indexed[key] = row
    return indexed


def _index_asr_rows(
    clean_rows: Sequence[Mapping[str, Any]], asr_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    manifest_keys = {asr_record_key(row) for row in clean_rows}
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in asr_rows:
        key = asr_record_key(row)
        if key not in manifest_keys:
            raise ValueError(f"ASR output key is absent from clean manifest: {key}")
        if key in indexed:
            raise ValueError(f"duplicate ASR output key: {key}")
        indexed[key] = row
    return indexed


def _index_tts_jobs(jobs: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return _index_unique(jobs, "item_id")


def audit_original_provenance(
    row: Mapping[str, Any],
    *,
    data_dir: Path,
    original_tts_jobs: Mapping[str, Mapping[str, Any]],
) -> tuple[str, str]:
    """Return provenance class and waveform digest without logging prompt bodies."""
    item_id = str(row.get("item_id") or "").strip()
    if not item_id:
        raise ValueError("clean row has no item_id")
    expected_text_hash = str(row.get("reference_sha256") or "").strip()
    if not _HEX_SHA256.fullmatch(expected_text_hash):
        raise ValueError(f"clean row {item_id} has invalid reference_sha256")
    audio_path = _resolve_data_path(row.get("path"), data_dir)
    if not audio_path.is_file():
        raise FileNotFoundError(f"clean WAV not found for {item_id}: {audio_path}")

    sidecar = audio_path.with_suffix(f"{audio_path.suffix}.sha256")
    if sidecar.is_file():
        actual = sidecar.read_text(encoding="utf-8").strip()
        if actual != expected_text_hash:
            raise ValueError(f"clean text-hash sidecar mismatch for {item_id}")
        provenance = "text_sha256_sidecar_verified"
    else:
        if row.get("render_status") == "reused_hash_verified":
            raise ValueError(f"hash-verified reused row lost its sidecar: {item_id}")
        job = original_tts_jobs.get(item_id)
        if job is None:
            raise ValueError(f"clean row has neither sidecar nor original TTS job: {item_id}")
        job_output = Path(str(job.get("output_path") or job.get("output") or ""))
        if not job_output.is_absolute():
            job_output = data_dir / job_output
        if job_output.resolve() != audio_path:
            raise ValueError(f"original TTS job output mismatch for {item_id}")
        job_text = str(job.get("text") or "")
        if _text_sha256(job_text) != expected_text_hash:
            raise ValueError(f"original TTS job text hash mismatch for {item_id}")
        if str(job.get("style") or "") != RETRY_STYLE:
            raise ValueError(f"original TTS job style mismatch for {item_id}")
        provenance = "original_tts_job_hash_path_style_verified"
    return provenance, _file_sha256(audio_path)


def build_retry_plan(
    clean_rows: Sequence[Mapping[str, Any]],
    asr_rows: Sequence[Mapping[str, Any]],
    original_tts_job_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    retry_root: Path,
    retry_id: str,
    base_seed: int,
) -> RetryPlan:
    """Build a seed-only, versioned CosyVoice retry cohort.

    The voice prompt, neutral instruction, renderer, and speed remain unchanged.
    Only a deterministic per-item RNG seed is introduced for the retry.
    """
    _validate_retry_id(retry_id)
    if base_seed < 0:
        raise ValueError("base_seed must be non-negative")
    data_dir = data_dir.resolve()
    retry_root = retry_root.resolve()
    if not retry_root.is_relative_to(data_dir):
        raise ValueError("retry_root must be under data_dir")
    clean_by_item = _index_unique(clean_rows, "item_id")
    del clean_by_item  # uniqueness is the validation; manifest order stays authoritative
    indexed_asr = _index_asr_rows(clean_rows, asr_rows)
    indexed_jobs = _index_tts_jobs(original_tts_job_rows)

    status_counts: Counter[str] = Counter()
    role_counts: Counter[str] = Counter()
    provenance_counts: Counter[str] = Counter()
    candidate_rows: list[dict[str, Any]] = []
    tts_jobs: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()

    for row in clean_rows:
        if str(row.get("condition") or "") != "clean":
            raise ValueError("Run 9 clean retry manifest may contain only condition=clean")
        current = indexed_asr.get(asr_record_key(row))
        if current is None:
            status_counts["missing"] += 1
            continue
        asr_status = str(current.get("asr_status") or "unknown")
        if asr_status == "error":
            status_counts["error_pending"] += 1
            continue
        if asr_status != "ok":
            status_counts[f"other_{asr_status}"] += 1
            continue
        if current.get("transcript_control_passed") is True:
            status_counts["ok_passed"] += 1
            continue
        if current.get("transcript_control_passed") is not False:
            raise ValueError("status=ok ASR row must have a boolean transcript_control_passed")
        status_counts["ok_failed_selected"] += 1

        item_id = str(row["item_id"])
        provenance, source_audio_hash = audit_original_provenance(
            row,
            data_dir=data_dir,
            original_tts_jobs=indexed_jobs,
        )
        provenance_counts[provenance] += 1
        role = str(row.get("gate_role") or "unknown")
        role_counts[role] += 1
        seed = deterministic_retry_seed(item_id, base_seed)
        output = retry_root / retry_id / role / f"{item_id}.wav"
        output = output.resolve()
        original = _resolve_data_path(row.get("path"), data_dir)
        if output == original:
            raise ValueError(f"retry output aliases original clean WAV for {item_id}")
        if output in seen_paths:
            raise ValueError(f"duplicate retry output path: {output}")
        seen_paths.add(output)
        reference_hash = str(row["reference_sha256"])
        provenance_path = output.with_suffix(f"{output.suffix}.render.json")
        candidate_path = _relative_to_data(output, data_dir)

        candidate = dict(row)
        candidate.update(
            {
                "path": candidate_path,
                "render_status": "retry_pending",
                "hash_path": _relative_to_data(
                    output.with_suffix(f"{output.suffix}.sha256"), data_dir
                ),
                "render_provenance_path": _relative_to_data(provenance_path, data_dir),
                "retry_contract_version": RETRY_CONTRACT_VERSION,
                "retry_id": retry_id,
                "retry_seed": seed,
                "retry_speed": RETRY_SPEED,
                "retry_of_path": str(row["path"]),
                "retry_of_audio_sha256": source_audio_hash,
                "retry_source_provenance": provenance,
            }
        )
        candidate_rows.append(candidate)
        tts_jobs.append(
            {
                "item_id": item_id,
                "safety_label": row.get("safety_label"),
                "gate_role": role,
                "style": RETRY_STYLE,
                "text": str(row.get("reference_text") or ""),
                "reference_sha256": reference_hash,
                "output_path": str(output),
                "provenance_path": str(provenance_path),
                "retry_contract_version": RETRY_CONTRACT_VERSION,
                "retry_id": retry_id,
                "seed": seed,
                "speed": RETRY_SPEED,
                "overwrite": False,
            }
        )

    summary = {
        "retry_contract_version": RETRY_CONTRACT_VERSION,
        "retry_id": retry_id,
        "base_seed": base_seed,
        "variant": "deterministic_seed_only",
        "style": RETRY_STYLE,
        "speed": RETRY_SPEED,
        "voice_prompt_changed": False,
        "clean_manifest_rows": len(clean_rows),
        "asr_checkpoint_rows": len(asr_rows),
        "status_counts": dict(sorted(status_counts.items())),
        "selected_rows": len(candidate_rows),
        "selected_role_counts": dict(sorted(role_counts.items())),
        "selected_provenance_counts": dict(sorted(provenance_counts.items())),
    }
    return RetryPlan(candidate_rows, tts_jobs, summary)


def _validate_candidate_artifacts(row: Mapping[str, Any], data_dir: Path) -> str:
    item_id = str(row.get("item_id") or "")
    output = _resolve_data_path(row.get("path"), data_dir)
    if not output.is_file():
        raise FileNotFoundError(f"retry candidate WAV not found for {item_id}: {output}")
    reference_hash = str(row.get("reference_sha256") or "")
    sidecar = output.with_suffix(f"{output.suffix}.sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").strip() != reference_hash:
        raise ValueError(f"retry candidate text-hash sidecar mismatch for {item_id}")
    provenance_path = _resolve_data_path(row.get("render_provenance_path"), data_dir)
    if not provenance_path.is_file():
        raise ValueError(f"retry candidate render provenance missing for {item_id}")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    audio_hash = _file_sha256(output)
    expected = {
        "reference_sha256": reference_hash,
        "output_sha256": audio_hash,
        "retry_contract_version": row.get("retry_contract_version"),
        "retry_id": row.get("retry_id"),
        "seed": row.get("retry_seed"),
        "speed": row.get("retry_speed"),
    }
    for key, value in expected.items():
        if provenance.get(key) != value:
            raise ValueError(f"retry candidate provenance {key} mismatch for {item_id}")
    return audio_hash


def build_promotion_plan(
    clean_rows: Sequence[Mapping[str, Any]],
    retry_rows: Sequence[Mapping[str, Any]],
    retry_asr_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    retry_id: str,
) -> PromotionPlan:
    """Validate passing retry artifacts and construct the updated clean manifest."""
    _validate_retry_id(retry_id)
    clean_by_item = _index_unique(clean_rows, "item_id")
    retry_by_item = _index_unique(retry_rows, "item_id")
    retry_asr_by_key: dict[str, Mapping[str, Any]] = {}
    for row in retry_asr_rows:
        key = asr_record_key(row)
        if key in retry_asr_by_key:
            raise ValueError(f"duplicate retry ASR key: {key}")
        retry_asr_by_key[key] = row

    replacements: dict[str, dict[str, Any]] = {}
    outcome_counts: Counter[str] = Counter()
    harmful_ids: list[str] = []
    for item_id, retry in retry_by_item.items():
        if retry.get("retry_id") != retry_id:
            raise ValueError(f"retry_id mismatch for {item_id}")
        original = clean_by_item.get(item_id)
        if original is None:
            raise ValueError(f"retry item absent from clean manifest: {item_id}")
        if str(retry.get("retry_of_path")) != str(original.get("path")):
            raise ValueError(f"clean manifest changed since retry preparation: {item_id}")
        asr = retry_asr_by_key.get(asr_record_key(retry))
        if asr is None:
            outcome_counts["missing"] += 1
            continue
        if asr.get("asr_status") == "error":
            outcome_counts["error_pending"] += 1
            continue
        if asr.get("asr_status") != "ok":
            outcome_counts["other_status"] += 1
            continue
        if asr.get("transcript_control_passed") is not True:
            outcome_counts["ok_failed"] += 1
            continue
        outcome_counts["ok_passed_promoted"] += 1
        candidate_hash = _validate_candidate_artifacts(retry, data_dir)
        promoted = dict(original)
        promoted.update(
            {
                "path": retry["path"],
                "hash_path": retry["hash_path"],
                "render_provenance_path": retry["render_provenance_path"],
                "render_status": "retry_promoted_asr_verified",
                "retry_contract_version": RETRY_CONTRACT_VERSION,
                "retry_id": retry_id,
                "retry_seed": retry["retry_seed"],
                "retry_speed": retry["retry_speed"],
                "retry_of_path": retry["retry_of_path"],
                "retry_of_audio_sha256": retry["retry_of_audio_sha256"],
                "retry_audio_sha256": candidate_hash,
                "retry_source_provenance": retry["retry_source_provenance"],
                "retry_asr_wer": asr.get("wer"),
                "retry_asr_token_overlap": asr.get("token_overlap"),
                "retry_asr_checkpoint": asr.get("asr_checkpoint"),
                "retry_asr_thresholds": asr.get("asr_thresholds"),
            }
        )
        replacements[item_id] = promoted
        if original.get("gate_role") == "harmful_eval":
            harmful_ids.append(item_id)

    updated = [dict(replacements.get(str(row["item_id"]), row)) for row in clean_rows]
    promoted_ids = sorted(replacements)
    summary = {
        "retry_contract_version": RETRY_CONTRACT_VERSION,
        "retry_id": retry_id,
        "retry_rows": len(retry_rows),
        "retry_asr_rows": len(retry_asr_rows),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "promoted_rows": len(promoted_ids),
        "promoted_harmful_rows": len(harmful_ids),
        "attack_rows_to_invalidate": 6 * len(harmful_ids),
    }
    return PromotionPlan(updated, promoted_ids, sorted(harmful_ids), summary)


def _copy_backup(source: Path, destination: Path) -> None:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"backup already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def filter_checkpoint_items(
    rows: Sequence[Mapping[str, Any]], item_ids: Sequence[str]
) -> tuple[list[dict[str, Any]], int]:
    targets = set(item_ids)
    kept = [dict(row) for row in rows if str(row.get("item_id")) not in targets]
    return kept, len(rows) - len(kept)


def invalidate_attack_artifacts(
    *,
    attack_rows: Sequence[Mapping[str, Any]],
    harmful_item_ids: Sequence[str],
    data_dir: Path,
    backup_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Move six attack WAVs and two item/sign sidecars per changed harmful row."""
    targets = set(harmful_item_ids)
    selected = [row for row in attack_rows if str(row.get("item_id")) in targets]
    counts = Counter(str(row.get("item_id")) for row in selected)
    incomplete = {item_id: count for item_id, count in counts.items() if count != 6}
    missing_items = targets - set(counts)
    if incomplete or missing_items:
        raise ValueError(
            "attack manifest must have exactly six rows per promoted harmful item; "
            f"incomplete={incomplete}, missing={sorted(missing_items)}"
        )

    artifacts: set[Path] = set()
    signs_by_item: dict[str, set[float]] = {}
    for row in selected:
        item_id = str(row["item_id"])
        artifacts.add(_resolve_data_path(row.get("path"), data_dir))
        signs_by_item.setdefault(item_id, set()).add(float(row["sign"]))
    for item_id, signs in signs_by_item.items():
        if signs != {-3.0, 3.0}:
            raise ValueError(f"attack signs are not frozen +/-3 for {item_id}: {signs}")
        for tag in ("m3", "p3"):
            artifacts.add(
                data_dir / "audio_run9" / "attacks" / "_metadata" / f"{item_id}_{tag}.json"
            )

    moved = 0
    missing = 0
    for source in sorted(artifacts):
        if not source.exists():
            missing += 1
            continue
        relative = source.resolve().relative_to(data_dir.resolve())
        destination = backup_root / "attack_artifacts" / relative
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"attack artifact backup already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        moved += 1

    kept = [dict(row) for row in attack_rows if str(row.get("item_id")) not in targets]
    return kept, {
        "invalidated_attack_rows": len(selected),
        "moved_attack_artifacts": moved,
        "missing_attack_artifacts": missing,
    }
