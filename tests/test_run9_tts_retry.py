"""CPU-only tests for safe Run 9 clean-TTS ASR remediation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from audio_safety.data.run9_tts_retry import (
    RETRY_CONTRACT_VERSION,
    build_promotion_plan,
    build_retry_plan,
    deterministic_retry_seed,
    filter_checkpoint_items,
    invalidate_attack_artifacts,
)


def _clean_row(
    data_dir: Path,
    item_id: str,
    *,
    role: str = "harmful_eval",
    reused: bool = False,
) -> tuple[dict, dict | None]:
    text = f"reference request {item_id}"
    digest = hashlib.sha256(text.encode()).hexdigest()
    root = "reused" if reused else "audio_run9/clean"
    path = data_dir / root / role / f"{item_id}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(f"wav-{item_id}".encode())
    if reused:
        path.with_suffix(".wav.sha256").write_text(digest, encoding="utf-8")
    row = {
        "item_id": item_id,
        "source": "figstep_safebench",
        "safety_label": "harmful" if role == "harmful_eval" else "benign",
        "gate_role": role,
        "style": "neutral",
        "condition": "clean",
        "path": path.relative_to(data_dir).as_posix(),
        "reference_text": text,
        "reference_sha256": digest,
        "render_status": "reused_hash_verified" if reused else "rendered",
        "asr_required": True,
    }
    job = None
    if not reused:
        job = {
            "item_id": item_id,
            "style": "neutral",
            "text": text,
            "output_path": str(path),
        }
    return row, job


def _asr(row: dict, status: str, passed: bool) -> dict:
    return {
        **row,
        "asr_status": status,
        "transcript_control_passed": passed,
        "wer": 0.0 if passed else 0.5,
        "token_overlap": 1.0 if passed else 0.5,
        "asr_checkpoint": "/models/large-v3.pt",
        "asr_thresholds": {"wer_max": 0.2, "token_overlap_min": 0.6},
    }


def test_retry_selects_only_completed_ok_failures_and_audits_provenance(tmp_path):
    data_dir = tmp_path / "data"
    fresh, fresh_job = _clean_row(data_dir, "fresh-fail")
    reused, _ = _clean_row(data_dir, "reused-fail", reused=True)
    passed, passed_job = _clean_row(data_dir, "passed")
    error, error_job = _clean_row(data_dir, "error")
    missing, missing_job = _clean_row(data_dir, "missing")
    rows = [fresh, reused, passed, error, missing]
    asr_rows = [
        _asr(fresh, "ok", False),
        _asr(reused, "ok", False),
        _asr(passed, "ok", True),
        _asr(error, "error", False),
    ]
    jobs = [job for job in [fresh_job, passed_job, error_job, missing_job] if job]

    plan = build_retry_plan(
        rows,
        asr_rows,
        jobs,
        data_dir=data_dir,
        retry_root=data_dir / "audio_run9/retries",
        retry_id="seed1709_v1",
        base_seed=1709,
    )

    assert [row["item_id"] for row in plan.candidate_rows] == [
        "fresh-fail",
        "reused-fail",
    ]
    assert plan.summary["status_counts"] == {
        "error_pending": 1,
        "missing": 1,
        "ok_failed_selected": 2,
        "ok_passed": 1,
    }
    assert plan.summary["selected_provenance_counts"] == {
        "original_tts_job_hash_path_style_verified": 1,
        "text_sha256_sidecar_verified": 1,
    }
    assert all(job["overwrite"] is False for job in plan.tts_jobs)
    assert all(job["speed"] == 1.0 and job["style"] == "neutral" for job in plan.tts_jobs)
    assert all("audio_run9/retries/seed1709_v1" in row["path"] for row in plan.candidate_rows)
    assert plan.tts_jobs[0]["seed"] == deterministic_retry_seed("fresh-fail", 1709)


def test_retry_rejects_unverifiable_original_provenance(tmp_path):
    data_dir = tmp_path / "data"
    row, job = _clean_row(data_dir, "bad", reused=True)
    (data_dir / row["path"]).with_suffix(".wav.sha256").write_text("0" * 64)

    with pytest.raises(ValueError, match="sidecar mismatch"):
        build_retry_plan(
            [row],
            [_asr(row, "ok", False)],
            [job] if job else [],
            data_dir=data_dir,
            retry_root=data_dir / "audio_run9/retries",
            retry_id="seed1709_v1",
            base_seed=1709,
        )


def _materialize_retry_candidate(data_dir: Path, row: dict) -> None:
    output = data_dir / row["path"]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(f"retry-{row['item_id']}".encode())
    output.with_suffix(".wav.sha256").write_text(row["reference_sha256"])
    provenance = {
        "reference_sha256": row["reference_sha256"],
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "retry_contract_version": RETRY_CONTRACT_VERSION,
        "retry_id": row["retry_id"],
        "seed": row["retry_seed"],
        "speed": row["retry_speed"],
    }
    (data_dir / row["render_provenance_path"]).write_text(json.dumps(provenance))


def test_promotion_changes_only_passing_candidate_path_and_keeps_original_wav(tmp_path):
    data_dir = tmp_path / "data"
    harmful, harmful_job = _clean_row(data_dir, "harmful")
    soft, soft_job = _clean_row(data_dir, "soft", role="soft_overrefusal")
    originals = {row["item_id"]: (data_dir / row["path"]).read_bytes() for row in [harmful, soft]}
    retry = build_retry_plan(
        [harmful, soft],
        [_asr(harmful, "ok", False), _asr(soft, "ok", False)],
        [harmful_job, soft_job],
        data_dir=data_dir,
        retry_root=data_dir / "audio_run9/retries",
        retry_id="seed1709_v1",
        base_seed=1709,
    )
    for row in retry.candidate_rows:
        _materialize_retry_candidate(data_dir, row)
    retry_asr = [
        _asr(retry.candidate_rows[0], "ok", True),
        _asr(retry.candidate_rows[1], "ok", False),
    ]

    promotion = build_promotion_plan(
        [harmful, soft],
        retry.candidate_rows,
        retry_asr,
        data_dir=data_dir,
        retry_id="seed1709_v1",
    )

    assert promotion.promoted_item_ids == ["harmful"]
    assert promotion.promoted_harmful_item_ids == ["harmful"]
    assert promotion.summary["attack_rows_to_invalidate"] == 6
    assert promotion.clean_rows[0]["path"] == retry.candidate_rows[0]["path"]
    assert promotion.clean_rows[1]["path"] == soft["path"]
    for row in [harmful, soft]:
        assert (data_dir / row["path"]).read_bytes() == originals[row["item_id"]]


def test_filter_checkpoint_and_invalidate_exact_six_attacks(tmp_path):
    data_dir = tmp_path / "data"
    rows = []
    for sign, tag in [(-3.0, "m3"), (3.0, "p3")]:
        for condition in ["pv_standard", "pv_locked", "mel_matched_ctrl"]:
            path = data_dir / "audio_run9/attacks" / condition / tag / "harmful.wav"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(f"{condition}-{tag}".encode())
            rows.append(
                {
                    "item_id": "harmful",
                    "condition": condition,
                    "sign": sign,
                    "path": path.relative_to(data_dir).as_posix(),
                }
            )
        metadata = data_dir / "audio_run9/attacks/_metadata" / f"harmful_{tag}.json"
        metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata.write_text("{}")
    rows.append({"item_id": "other", "condition": "pv_standard", "sign": -3.0})

    kept, counts = invalidate_attack_artifacts(
        attack_rows=rows,
        harmful_item_ids=["harmful"],
        data_dir=data_dir,
        backup_root=tmp_path / "backup",
    )

    assert [row["item_id"] for row in kept] == ["other"]
    assert counts == {
        "invalidated_attack_rows": 6,
        "moved_attack_artifacts": 8,
        "missing_attack_artifacts": 0,
    }
    assert not list((data_dir / "audio_run9/attacks").rglob("harmful*"))
    assert len(list((tmp_path / "backup/attack_artifacts").rglob("harmful*"))) == 8

    filtered, removed = filter_checkpoint_items(rows, ["harmful"])
    assert removed == 6
    assert [row["item_id"] for row in filtered] == ["other"]
