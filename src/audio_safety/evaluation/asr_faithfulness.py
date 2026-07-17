"""CPU-only scoring and resume helpers for the Run 9 Whisper ASR gate."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from audio_safety.utils.text import token_overlap, word_error_rate

DEFAULT_WER_MAX = 0.20
DEFAULT_TOKEN_OVERLAP_MIN = 0.60


def asr_record_key(row: Mapping[str, Any]) -> str:
    """Stable key for clean/attack manifest rows and resumable ASR outputs."""
    if row.get("record_id") is not None:
        record_id = str(row["record_id"]).strip()
        if not record_id:
            raise ValueError("ASR record_id must be non-empty when provided")
        return f"record_id:{record_id}"
    item_id = str(row.get("item_id") or "").strip()
    path = str(row.get("path") or "").strip()
    condition = str(row.get("condition") or row.get("style") or "").strip()
    if not item_id or not path or not condition:
        raise ValueError("ASR rows need record_id or non-empty item_id, condition/style, and path")
    return f"metadata:{item_id}|{condition}|{path}"


def score_asr_record(
    row: Mapping[str, Any],
    transcript: str,
    *,
    wer_max: float = DEFAULT_WER_MAX,
    token_overlap_min: float = DEFAULT_TOKEN_OVERLAP_MIN,
    checkpoint: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """Attach WER/content-token scores and the frozen faithfulness pass flag."""
    if wer_max < 0.0:
        raise ValueError("wer_max must be non-negative")
    if not 0.0 <= token_overlap_min <= 1.0:
        raise ValueError("token_overlap_min must be in [0, 1]")
    reference = str(row.get("reference_text") or "").strip()
    if not reference:
        raise ValueError(f"ASR row has no reference_text: {asr_record_key(row)}")

    hypothesis = str(transcript or "").strip()
    wer = word_error_rate(reference, hypothesis)
    overlap = token_overlap(reference, hypothesis) if hypothesis else 0.0
    core_preserved = overlap >= token_overlap_min
    passed = error is None and wer <= wer_max and core_preserved
    updated = dict(row)
    updated.update(
        {
            "transcript": hypothesis,
            "wer": wer,
            "token_overlap": overlap,
            "core_tokens_preserved": core_preserved,
            "transcript_control_passed": passed,
            "asr_status": "error" if error is not None else "ok",
            "asr_error": error,
            "asr_checkpoint": checkpoint,
            "asr_thresholds": {
                "wer_max": wer_max,
                "token_overlap_min": token_overlap_min,
            },
        }
    )
    return updated


def merge_asr_resume_state(
    manifest_rows: Sequence[Mapping[str, Any]],
    existing_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, dict[str, Any]], list[Mapping[str, Any]]]:
    """Return current key->row state plus pending manifest rows.

    Completed rows are skipped regardless of line order.  Rows checkpointed with
    ``asr_status='error'`` remain pending on the next invocation so transient
    decode failures can be retried.
    """
    manifest_keys: list[str] = []
    seen_manifest: set[str] = set()
    for row in manifest_rows:
        key = asr_record_key(row)
        if key in seen_manifest:
            raise ValueError(f"duplicate ASR manifest key {key!r}")
        seen_manifest.add(key)
        manifest_keys.append(key)

    state: dict[str, dict[str, Any]] = {}
    for row in existing_rows:
        key = asr_record_key(row)
        if key in state:
            raise ValueError(f"duplicate existing ASR key {key!r}")
        if key not in seen_manifest:
            raise ValueError(f"existing ASR output key is absent from manifest: {key!r}")
        state[key] = dict(row)

    pending = []
    for key, row in zip(manifest_keys, manifest_rows, strict=True):
        current = state.get(key)
        complete = current is not None and current.get("asr_status") == "ok"
        if not complete:
            pending.append(row)
    return state, pending


def ordered_checkpoint_rows(
    manifest_rows: Sequence[Mapping[str, Any]],
    state: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Rows currently available in state, restored to manifest order."""
    return [dict(state[key]) for row in manifest_rows if (key := asr_record_key(row)) in state]


def atomic_save_jsonl(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    """Atomically replace a JSONL checkpoint in the destination directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temp_name).replace(path)
        temp_name = None
    finally:
        if temp_name is not None:
            Path(temp_name).unlink(missing_ok=True)
