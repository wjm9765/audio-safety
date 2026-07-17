"""Fresh SafeBench cohort planning for the Run 9 defense gate.

This module only plans clean TTS inputs.  Phase/EQ transforms and model
generation are deliberately downstream so the clean waveform is rendered once
and reused by every defense arm.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Run9FreshPlan:
    manifest_rows: list[dict[str, object]]
    tts_jobs: list[dict[str, object]]
    reused_count: int
    pending_count: int


def load_safebench_rows(path: Path) -> list[dict[str, str]]:
    """Load the canonical 500-row SafeBench CSV with strict required fields."""

    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"category_id", "category_name", "question"}
    missing = required - (set(rows[0]) if rows else set())
    if missing:
        raise ValueError(f"SafeBench CSV is missing required columns: {sorted(missing)}")
    return rows


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _matches_reusable_audio(wav_path: Path, text_sha256: str) -> bool:
    sidecar = wav_path.with_suffix(f"{wav_path.suffix}.sha256")
    if not wav_path.is_file() or not sidecar.is_file():
        return False
    return sidecar.read_text(encoding="utf-8").strip() == text_sha256


def _relative_to_data(path: Path, data_dir: Path) -> str:
    try:
        return str(path.relative_to(data_dir))
    except ValueError as exc:
        raise ValueError(f"Run 9 audio path must stay under data_dir: {path}") from exc


def build_run9_fresh_plan(
    seed_rows: Iterable[dict[str, str]],
    *,
    data_dir: Path,
    output_audio_root: Path,
    reuse_harmful_root: Path | None = None,
    harmful_category_max: int = 7,
    include_soft_controls: bool = True,
) -> Run9FreshPlan:
    """Plan the 350 harmful + 150 soft-topic clean cohort.

    Categories 1..7 are the harmful evaluation set. Categories 8..10 are kept
    as the pre-specified soft-topic over-refusal controls and are marked benign
    for defense-cost accounting, with ``control_type`` retaining their meaning.
    Existing neutral harmful TTS is reused only when its text SHA sidecar exactly
    matches the SafeBench question.
    """

    data_dir = data_dir.resolve()
    output_audio_root = output_audio_root.resolve()
    reuse_harmful_root = reuse_harmful_root.resolve() if reuse_harmful_root else None
    manifest_rows: list[dict[str, object]] = []
    tts_jobs: list[dict[str, object]] = []
    reused = 0
    pending = 0

    for index, raw in enumerate(seed_rows):
        category_id = int(raw["category_id"])
        if category_id > harmful_category_max and not include_soft_controls:
            continue
        text = raw["question"].strip()
        if not text:
            raise ValueError(f"empty SafeBench question at row {index}")
        item_id = f"figstep_safebench_{index:04d}"
        is_harmful = category_id <= harmful_category_max
        gate_role = "harmful_eval" if is_harmful else "soft_overrefusal"
        safety_label = "harmful" if is_harmful else "benign"
        digest = _text_sha256(text)

        reusable = reuse_harmful_root / f"{item_id}.wav" if reuse_harmful_root else None
        if is_harmful and reusable is not None and _matches_reusable_audio(reusable, digest):
            wav_path = reusable
            render_status = "reused_hash_verified"
            reused += 1
        else:
            wav_path = output_audio_root / gate_role / f"{item_id}.wav"
            render_status = "rendered" if wav_path.is_file() else "pending_tts"
            if render_status == "pending_tts":
                pending += 1
            tts_jobs.append(
                {
                    "item_id": item_id,
                    "safety_label": safety_label,
                    "gate_role": gate_role,
                    "style": "neutral",
                    "text": text,
                    "output_path": str(wav_path),
                    "overwrite": False,
                }
            )

        manifest_rows.append(
            {
                "item_id": item_id,
                "source": "figstep_safebench",
                "category_id": category_id,
                "category_name": raw["category_name"],
                "safety_label": safety_label,
                "gate_role": gate_role,
                "control_type": None if is_harmful else "safebench_soft_topic",
                "style": "neutral",
                "condition": "clean",
                "path": _relative_to_data(wav_path, data_dir),
                "reference_text": text,
                "reference_sha256": digest,
                "render_status": render_status,
                "asr_required": True,
            }
        )

    keys = [str(row["item_id"]) for row in manifest_rows]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate item_id in Run 9 fresh cohort")
    return Run9FreshPlan(
        manifest_rows=manifest_rows,
        tts_jobs=tts_jobs,
        reused_count=reused,
        pending_count=pending,
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_job_shards(
    jobs: list[dict[str, object]], base_path: Path, workers: int
) -> list[Path]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    shards = [[] for _ in range(workers)]
    for index, job in enumerate(jobs):
        shards[index % workers].append(job)
    paths = []
    for worker_index, shard in enumerate(shards):
        path = base_path.with_name(f"{base_path.stem}.worker{worker_index:02d}.jsonl")
        write_jsonl(path, shard)
        paths.append(path)
    return paths
