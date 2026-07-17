"""Build the canonical full-cohort manifest for the Run 9 defense gate.

The combined manifest is shared by the SARSteer and ALMGuard inference arms.
It is intentionally evaluation-only: even rows whose acoustic operator could be
used by a separate ALMGuard training recipe are never marked training-eligible
here.  In particular, ``pv_standard`` is the frozen phase attack under test and
must always retain an explicit exclusion policy.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PRIMARY_GATE_ROLES = frozenset({"harmful_eval", "soft_overrefusal"})
ATTACK_CONDITIONS = ("pv_standard", "pv_locked", "mel_matched_ctrl")
PHASE_UNDER_TEST = "pv_standard"

# Frozen full-cohort cardinalities.  The key is (gate_role, condition, sign).
EXPECTED_CELL_COUNTS: dict[tuple[str, str, str], int] = {
    ("harmful_eval", "clean", ""): 350,
    ("soft_overrefusal", "clean", ""): 150,
    ("utility_eval", "clean", ""): 100,
    ("positive_control_eval", "positive_control", ""): 100,
    **{
        ("harmful_eval", condition, sign): 350
        for condition in ATTACK_CONDITIONS
        for sign in ("-3", "3")
    },
}


@dataclass(frozen=True)
class Run9EvalManifest:
    """Validated combined rows plus an audit summary."""

    rows: list[dict[str, Any]]
    summary: dict[str, Any]


def _required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires a non-empty string field {field!r}")
    return value.strip()


def _source(row: Mapping[str, Any], *, context: str) -> str:
    value = row.get("source", row.get("dataset"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty 'source' or 'dataset'")
    return value.strip()


def _sign_token(value: Any, *, context: str) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, bool):
        raise ValueError(f"{context} has invalid boolean sign")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{context} has non-numeric sign {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"{context} has non-finite sign {value!r}")
    if number == 0:
        number = 0.0
    return f"{number:.12g}"


def _normalized_sign(value: Any, *, context: str) -> float | None:
    token = _sign_token(value, context=context)
    return None if not token else float(token)


def _resolved_path(raw: str, data_dir: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else data_dir / path).resolve()


def _normalized_path(raw: str, data_dir: Path, *, context: str, require_audio: bool) -> str:
    if not raw.strip():
        raise ValueError(f"{context} requires a non-empty audio path")
    resolved = _resolved_path(raw.strip(), data_dir)
    if require_audio and not resolved.is_file():
        raise ValueError(f"{context} audio file not found: {resolved}")
    try:
        return resolved.relative_to(data_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def stable_row_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the semantic key shared with SARSteer's resumable apply path."""

    context = "evaluation row"
    return (
        _required_text(row, "item_id", context=context),
        _required_text(row, "safety_label", context=context),
        _required_text(row, "condition", context=context),
        _sign_token(row.get("sign"), context=context),
    )


def record_id_for_key(key: tuple[str, str, str, str]) -> str:
    payload = json.dumps(key, ensure_ascii=True, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]
    return f"run9_eval_{digest}"


def _base_row(
    raw: Mapping[str, Any],
    *,
    data_dir: Path,
    context: str,
    safety_label: str,
    gate_role: str,
    condition: str,
    sign: Any,
    require_audio: bool,
) -> dict[str, Any]:
    item_id = _required_text(raw, "item_id", context=context)
    reference_text = _required_text(raw, "reference_text", context=context)
    path = _normalized_path(
        _required_text(raw, "path", context=context),
        data_dir,
        context=context,
        require_audio=require_audio,
    )
    normalized = dict(raw)
    normalized.update(
        {
            "item_id": item_id,
            "source": _source(raw, context=context),
            "path": path,
            "reference_text": reference_text,
            "safety_label": safety_label,
            "gate_role": gate_role,
            "condition": condition,
            "sign": _normalized_sign(sign, context=context),
            "data_partition": "evaluation",
            "eval_only": True,
            "almguard_training_eligible": False,
        }
    )
    normalized.setdefault("style", condition if condition != "clean" else "neutral")
    normalized.setdefault("asr_required", False)
    return normalized


def _normalize_clean(
    rows: Sequence[Mapping[str, Any]], *, data_dir: Path, require_audio: bool
) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(rows):
        context = f"clean row {index}"
        role = _required_text(raw, "gate_role", context=context)
        label = _required_text(raw, "safety_label", context=context)
        expected_label = {
            "harmful_eval": "harmful",
            "soft_overrefusal": "benign",
        }.get(role)
        if expected_label is None or label != expected_label:
            raise ValueError(
                f"{context} has invalid gate_role/safety_label pair: {(role, label)!r}"
            )
        if raw.get("condition") != "clean":
            raise ValueError(f"{context} must have condition='clean'")
        row = _base_row(
            raw,
            data_dir=data_dir,
            context=context,
            safety_label=label,
            gate_role=role,
            condition="clean",
            sign=None,
            require_audio=require_audio,
        )
        row["asr_required"] = bool(raw.get("asr_required", True))
        normalized.append(row)
    return normalized


def _normalize_attacks(
    rows: Sequence[Mapping[str, Any]], *, data_dir: Path, require_audio: bool
) -> list[dict[str, Any]]:
    normalized = []
    for index, raw in enumerate(rows):
        context = f"attack row {index}"
        condition = _required_text(raw, "condition", context=context)
        if condition not in ATTACK_CONDITIONS:
            raise ValueError(f"{context} has unsupported condition {condition!r}")
        if raw.get("gate_role") != "harmful_eval" or raw.get("safety_label") != "harmful":
            raise ValueError(f"{context} must be harmful_eval/harmful")
        sign = _normalized_sign(raw.get("sign"), context=context)
        if sign not in {-3.0, 3.0}:
            raise ValueError(f"{context} requires frozen sign -3 or +3, got {sign!r}")
        row = _base_row(
            raw,
            data_dir=data_dir,
            context=context,
            safety_label="harmful",
            gate_role="harmful_eval",
            condition=condition,
            sign=sign,
            require_audio=require_audio,
        )
        row["asr_required"] = bool(raw.get("asr_required", True))
        under_test = condition == PHASE_UNDER_TEST
        if under_test:
            policy = str(raw.get("almguard_training_policy") or "")
            if not policy.startswith("exclude"):
                raise ValueError(
                    f"{context} is the exact phase attack under test but lacks an exclusion policy"
                )
            if raw.get("phase_under_test") is not True:
                raise ValueError(f"{context} must set phase_under_test=true")
            row["almguard_training_policy"] = "exclude_primary_phase_attack_under_test"
            row["phase_under_test_tag"] = "run7_frozen_pv_standard"
        row["phase_under_test"] = under_test
        normalized.append(row)
    return normalized


def _normalize_utility(
    rows: Sequence[Mapping[str, Any]], *, data_dir: Path, require_audio: bool
) -> list[dict[str, Any]]:
    return [
        _base_row(
            raw,
            data_dir=data_dir,
            context=f"utility row {index}",
            safety_label="benign",
            gate_role="utility_eval",
            condition="clean",
            sign=None,
            require_audio=require_audio,
        )
        for index, raw in enumerate(rows)
    ]


def _normalize_positive_control(
    rows: Sequence[Mapping[str, Any]], *, data_dir: Path, require_audio: bool
) -> list[dict[str, Any]]:
    return [
        _base_row(
            raw,
            data_dir=data_dir,
            context=f"positive-control row {index}",
            safety_label="harmful",
            gate_role="positive_control_eval",
            condition="positive_control",
            sign=None,
            require_audio=require_audio,
        )
        for index, raw in enumerate(rows)
    ]


def _cell_key(row: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(row["gate_role"]),
        str(row["condition"]),
        _sign_token(row.get("sign"), context="evaluation row"),
    )


def _validate_expected_cells(
    rows: Sequence[Mapping[str, Any]], expected: Mapping[tuple[str, str, str], int]
) -> Counter[tuple[str, str, str]]:
    actual = Counter(_cell_key(row) for row in rows)
    if actual != Counter(expected):
        missing = {
            key: count - actual.get(key, 0)
            for key, count in expected.items()
            if actual.get(key, 0) != count
        }
        unexpected = {
            key: count
            for key, count in actual.items()
            if key not in expected
        }
        raise ValueError(
            "Run 9 full-cohort cell counts do not match the frozen design: "
            f"mismatched={missing}, unexpected={unexpected}"
        )
    return actual


def _validate_attack_pairing(rows: Sequence[Mapping[str, Any]]) -> None:
    clean_harmful = {
        str(row["item_id"])
        for row in rows
        if row["gate_role"] == "harmful_eval" and row["condition"] == "clean"
    }
    attacks: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for row in rows:
        if row["condition"] in ATTACK_CONDITIONS:
            attacks[str(row["item_id"])].add(
                (str(row["condition"]), _sign_token(row.get("sign"), context="attack row"))
            )
    if set(attacks) != clean_harmful:
        missing = sorted(clean_harmful - set(attacks))[:10]
        extra = sorted(set(attacks) - clean_harmful)[:10]
        raise ValueError(f"attack/clean item sets differ (missing={missing}, extra={extra})")
    expected_variants = {
        (condition, sign) for condition in ATTACK_CONDITIONS for sign in ("-3", "3")
    }
    bad = [item for item, variants in attacks.items() if variants != expected_variants]
    if bad:
        raise ValueError(f"harmful items lack exactly six frozen attack variants: {bad[:10]}")


def _canonical_calibration_sets(
    calibration_rows: Sequence[Mapping[str, Any]], data_dir: Path
) -> tuple[set[str], set[Path], set[str]]:
    item_ids: set[str] = set()
    paths: set[Path] = set()
    sources: set[str] = set()
    for index, row in enumerate(calibration_rows):
        context = f"calibration row {index}"
        item_id = _required_text(row, "item_id", context=context)
        if item_id in item_ids:
            raise ValueError(f"duplicate calibration item_id {item_id!r}")
        item_ids.add(item_id)
        if row.get("path") is not None:
            raw_path = _required_text(row, "path", context=context)
            resolved = _resolved_path(raw_path, data_dir)
            if resolved in paths:
                raise ValueError(f"duplicate calibration audio path: {resolved}")
            paths.add(resolved)
        sources.add(_source(row, context=context))
    return item_ids, paths, sources


def validate_calibration_disjoint(
    eval_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    strict_source_all_roles: bool = False,
) -> dict[str, Any]:
    """Fail closed on calibration/eval leakage.

    Exact item and audio-path overlap is prohibited for *every* evaluation row.
    Dataset-source overlap is additionally prohibited for the shared primary gate
    roles.  Positive controls and utility rows intentionally come from held-out
    assets in the same domains as their method-specific calibration sets, so a
    source-only ban on those roles would make the registered controls impossible;
    ``strict_source_all_roles=True`` is available for audits that need it.
    """

    calibration_ids, calibration_paths, calibration_sources = _canonical_calibration_sets(
        calibration_rows, data_dir
    )
    eval_ids = {str(row["item_id"]) for row in eval_rows}
    eval_paths = {_resolved_path(str(row["path"]), data_dir) for row in eval_rows}
    source_scope = (
        list(eval_rows)
        if strict_source_all_roles
        else [row for row in eval_rows if row["gate_role"] in PRIMARY_GATE_ROLES]
    )
    eval_sources = {str(row["source"]) for row in source_scope}
    item_overlap = calibration_ids & eval_ids
    path_overlap = calibration_paths & eval_paths
    source_overlap = calibration_sources & eval_sources
    if item_overlap or path_overlap or source_overlap:
        details = []
        if item_overlap:
            details.append(f"item_id={sorted(item_overlap)[:10]}")
        if path_overlap:
            details.append(f"path={[str(path) for path in sorted(path_overlap)[:10]]}")
        if source_overlap:
            details.append(f"source={sorted(source_overlap)}")
        raise ValueError("calibration/evaluation leakage detected: " + "; ".join(details))
    return {
        "calibration_rows": len(calibration_rows),
        "calibration_item_ids": len(calibration_ids),
        "calibration_paths": len(calibration_paths),
        "calibration_sources": sorted(calibration_sources),
        "source_scope": "all_roles" if strict_source_all_roles else sorted(PRIMARY_GATE_ROLES),
        "passed": True,
    }


def _asr_fallback_key(row: Mapping[str, Any]) -> tuple[str, str, str, str]:
    return (
        _required_text(row, "item_id", context="ASR row"),
        _required_text(row, "condition", context="ASR row"),
        _sign_token(row.get("sign"), context="ASR row"),
        _required_text(row, "path", context="ASR row"),
    )


def merge_asr_scores(
    rows: Sequence[Mapping[str, Any]],
    score_rows: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge optional ASR scores without dropping failed or unscored rows."""

    by_record_id: dict[str, Mapping[str, Any]] = {}
    by_fallback: dict[tuple[str, str, str, str], Mapping[str, Any]] = {}
    for score in score_rows:
        record_id = str(score.get("record_id") or "").strip()
        if record_id:
            if record_id in by_record_id:
                raise ValueError(f"duplicate ASR score record_id {record_id!r}")
            by_record_id[record_id] = score
        fallback = _asr_fallback_key(score)
        if fallback in by_fallback:
            raise ValueError(f"duplicate ASR score fallback key {fallback!r}")
        by_fallback[fallback] = score

    merged: list[dict[str, Any]] = []
    used_scores: set[int] = set()
    eligible = 0
    failed = 0
    missing = 0
    asr_fields = (
        "transcript",
        "wer",
        "token_overlap",
        "core_tokens_preserved",
        "transcript_control_passed",
        "asr_status",
        "asr_error",
        "asr_checkpoint",
        "asr_thresholds",
    )
    for raw in rows:
        row = dict(raw)
        score = by_record_id.get(str(row["record_id"]))
        if score is None:
            score = by_fallback.get(_asr_fallback_key(row))
        reasons: list[str] = []
        if score is not None:
            used_scores.add(id(score))
            for field in asr_fields:
                if field in score:
                    row[field] = score[field]
            row["asr_score_matched"] = True
            if row.get("asr_required") and row.get("transcript_control_passed") is not True:
                reasons.append("transcript_control_failed")
                if row.get("asr_status") == "error":
                    reasons.append("asr_decode_error")
        else:
            row["asr_score_matched"] = False
            if row.get("asr_required"):
                reasons.append("asr_score_missing")
                missing += 1
        row["evaluation_eligible"] = not reasons
        row["eligibility_reasons"] = reasons
        eligible += int(not reasons)
        failed += int(bool(reasons) and "asr_score_missing" not in reasons)
        merged.append(row)

    unused = len(score_rows) - len(used_scores)
    if unused:
        raise ValueError(f"ASR sidecar has {unused} row(s) absent from the combined manifest")
    return merged, {
        "score_rows": len(score_rows),
        "matched": len(used_scores),
        "eligible": eligible,
        "failed": failed,
        "missing_required": missing,
        "rows_dropped": 0,
    }


def _assign_record_ids(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    seen_keys: set[tuple[str, str, str, str]] = set()
    seen_ids: set[str] = set()
    normalized = []
    for row in rows:
        key = stable_row_key(row)
        if key in seen_keys:
            raise ValueError(f"duplicate stable evaluation key: {key}")
        seen_keys.add(key)
        record_id = record_id_for_key(key)
        if record_id in seen_ids:
            raise ValueError(f"record_id collision: {record_id}")
        seen_ids.add(record_id)
        normalized.append({**dict(row), "record_id": record_id})
    return normalized


def build_run9_eval_manifest(
    clean_rows: Sequence[Mapping[str, Any]],
    attack_rows: Sequence[Mapping[str, Any]],
    utility_rows: Sequence[Mapping[str, Any]],
    positive_control_rows: Sequence[Mapping[str, Any]],
    *,
    data_dir: Path,
    calibration_rows: Sequence[Mapping[str, Any]] = (),
    asr_score_rows: Sequence[Mapping[str, Any]] = (),
    expected_counts: Mapping[tuple[str, str, str], int] = EXPECTED_CELL_COUNTS,
    require_audio: bool = True,
    strict_source_all_roles: bool = False,
) -> Run9EvalManifest:
    """Normalize, join, and validate the full Run 9 evaluation cohort."""

    data_dir = data_dir.resolve()
    rows = [
        *_normalize_clean(clean_rows, data_dir=data_dir, require_audio=require_audio),
        *_normalize_attacks(attack_rows, data_dir=data_dir, require_audio=require_audio),
        *_normalize_utility(utility_rows, data_dir=data_dir, require_audio=require_audio),
        *_normalize_positive_control(
            positive_control_rows, data_dir=data_dir, require_audio=require_audio
        ),
    ]
    counts = _validate_expected_cells(rows, expected_counts)
    _validate_attack_pairing(rows)
    rows = _assign_record_ids(rows)
    leakage = validate_calibration_disjoint(
        rows,
        calibration_rows,
        data_dir=data_dir,
        strict_source_all_roles=strict_source_all_roles,
    )
    rows, asr = merge_asr_scores(rows, asr_score_rows)
    summary = {
        "schema_version": 1,
        "rows": len(rows),
        "unique_record_ids": len({str(row["record_id"]) for row in rows}),
        "unique_item_ids": len({str(row["item_id"]) for row in rows}),
        "cell_counts": {
            "|".join(key): value for key, value in sorted(counts.items())
        },
        "condition_counts": dict(sorted(Counter(str(row["condition"]) for row in rows).items())),
        "gate_role_counts": dict(sorted(Counter(str(row["gate_role"]) for row in rows).items())),
        "phase_under_test": PHASE_UNDER_TEST,
        "evaluation_only": True,
        "almguard_training_eligible_rows": 0,
        "leakage_check": leakage,
        "asr": asr,
    }
    return Run9EvalManifest(rows=rows, summary=summary)


def shard_by_item(
    rows: Sequence[Mapping[str, Any]], num_shards: int
) -> list[list[dict[str, Any]]]:
    """Deterministically balance item groups while keeping each item intact."""

    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["item_id"])].append(dict(row))
    shards: list[list[dict[str, Any]]] = [[] for _ in range(num_shards)]
    counts = [0] * num_shards
    for item_id in sorted(grouped, key=lambda item: (-len(grouped[item]), item)):
        shard_index = min(range(num_shards), key=lambda index: (counts[index], index))
        shards[shard_index].extend(grouped[item_id])
        counts[shard_index] += len(grouped[item_id])

    expected_ids = {str(row["record_id"]) for row in rows}
    shard_ids = [{str(row["record_id"]) for row in shard} for shard in shards]
    if sum(len(ids) for ids in shard_ids) != len(set().union(*shard_ids)):
        raise AssertionError("shard record_id overlap detected")
    if set().union(*shard_ids) != expected_ids:
        raise AssertionError("shard union does not equal the combined manifest")
    item_shards: dict[str, set[int]] = defaultdict(set)
    for shard_index, shard in enumerate(shards):
        for row in shard:
            item_shards[str(row["item_id"])].add(shard_index)
    split_items = [item for item, assignments in item_shards.items() if len(assignments) != 1]
    if split_items:
        raise AssertionError(f"item groups split across shards: {split_items[:10]}")
    return shards


def atomic_save_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    """Atomically replace one JSONL artifact."""

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
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
        temporary = None
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)

