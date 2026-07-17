"""CPU-only tests for Run 9 ASR scoring and resumable checkpoints."""

from __future__ import annotations

import importlib.util
import json
import sys
import wave
from pathlib import Path

import pytest

from audio_safety.evaluation.asr_faithfulness import (
    asr_record_key,
    atomic_save_jsonl,
    merge_asr_resume_state,
    ordered_checkpoint_rows,
    score_asr_record,
)


def _load_asr_script():
    path = Path(__file__).parents[1] / "scripts" / "almguard" / "score_run9_asr.py"
    spec = importlib.util.spec_from_file_location("score_run9_asr_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


asr_script = _load_asr_script()


def _row(item: str, condition: str = "clean", *, path: str | None = None) -> dict:
    return {
        "item_id": item,
        "safety_label": "harmful",
        "condition": condition,
        "style": "neutral",
        "path": path or f"audio/{condition}/{item}.wav",
        "reference_text": "Explain how to repair a bicycle wheel safely",
        "asr_required": True,
    }


def test_score_exact_transcript_passes_frozen_thresholds():
    row = _row("q0")
    scored = score_asr_record(
        row,
        row["reference_text"],
        checkpoint="/models/large-v3.pt",
    )

    assert scored["wer"] == 0.0
    assert scored["token_overlap"] == 1.0
    assert scored["core_tokens_preserved"] is True
    assert scored["transcript_control_passed"] is True
    assert scored["asr_status"] == "ok"
    assert scored["asr_thresholds"] == {
        "wer_max": 0.20,
        "token_overlap_min": 0.60,
    }


def test_score_bad_or_failed_transcript_fails():
    row = _row("q0")
    bad = score_asr_record(row, "unrelated words only")
    failed = score_asr_record(row, "", error="RuntimeError: decode failed")

    assert bad["wer"] > 0.20
    assert bad["token_overlap"] < 0.60
    assert bad["core_tokens_preserved"] is False
    assert bad["transcript_control_passed"] is False
    assert failed["asr_status"] == "error"
    assert failed["transcript_control_passed"] is False
    assert failed["asr_error"] == "RuntimeError: decode failed"


def test_resume_merge_is_keyed_not_positional_and_retries_errors():
    manifest = [_row("q0"), _row("q1"), _row("q2", "phase")]
    complete = score_asr_record(manifest[1], manifest[1]["reference_text"])
    error = score_asr_record(manifest[0], "", error="temporary")

    state, pending = merge_asr_resume_state(manifest, [complete, error])

    assert [asr_record_key(row) for row in pending] == [
        asr_record_key(manifest[0]),
        asr_record_key(manifest[2]),
    ]
    assert [row["item_id"] for row in ordered_checkpoint_rows(manifest, state)] == [
        "q0",
        "q1",
    ]


def test_resume_merge_accepts_fresh_manifest_condition_over_style():
    clean = _row("q0", "clean")
    attack = _row("q0", "phase", path="audio/phase/q0.wav")

    assert asr_record_key(clean) != asr_record_key(attack)
    state, pending = merge_asr_resume_state([clean, attack], [])
    assert state == {}
    assert pending == [clean, attack]


def test_resume_merge_rejects_duplicate_or_extraneous_keys():
    row = _row("q0")
    with pytest.raises(ValueError, match="duplicate ASR manifest key"):
        merge_asr_resume_state([row, dict(row)], [])
    with pytest.raises(ValueError, match="absent from manifest"):
        merge_asr_resume_state([row], [_row("q9")])
    with pytest.raises(ValueError, match="duplicate existing ASR key"):
        merge_asr_resume_state([row], [row, dict(row)])


def test_atomic_checkpoint_replaces_valid_jsonl_in_manifest_order(tmp_path):
    path = tmp_path / "nested" / "asr.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text('{"stale": true}\n', encoding="utf-8")
    rows = [_row("q0"), _row("q1")]

    atomic_save_jsonl(rows, path)

    loaded = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["item_id"] for row in loaded] == ["q0", "q1"]
    assert not list(path.parent.glob(".asr.jsonl.*.tmp"))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wer_max": -0.1}, "wer_max must be non-negative"),
        ({"token_overlap_min": 1.1}, "token_overlap_min must be in"),
    ],
)
def test_score_rejects_invalid_thresholds(kwargs, message):
    with pytest.raises(ValueError, match=message):
        score_asr_record(_row("q0"), "text", **kwargs)


def test_score_requires_reference_text_and_stable_key():
    row = _row("q0")
    row["reference_text"] = ""
    with pytest.raises(ValueError, match="no reference_text"):
        score_asr_record(row, "text")
    with pytest.raises(ValueError, match="need record_id"):
        asr_record_key({"reference_text": "text"})


def test_worker_imports_vendored_whisper_once_and_serves_multiple_rows(tmp_path):
    repo = tmp_path / "ALMGuard"
    package = repo / "whisper"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        """from pathlib import Path
import numpy as np
LOADS = Path(__file__).with_name('loads.txt')
ARRAYS = Path(__file__).with_name('arrays.txt')
def load_model(checkpoint, device):
    with LOADS.open('a', encoding='utf-8') as handle:
        handle.write(f'{checkpoint}|{device}\\n')
    return Model()
class Model:
    def transcribe(self, audio, *, language, task, verbose, fp16):
        assert isinstance(audio, np.ndarray)
        assert audio.dtype == np.float32
        assert audio.ndim == 1
        assert audio.flags.c_contiguous
        with ARRAYS.open('a', encoding='utf-8') as handle:
            handle.write(f'{audio.shape[0]}|{audio.dtype}|{audio.ndim}\\n')
        return {'text': f'{language}:{task}:{verbose}:{fp16}:{audio.shape[0]}'}
""",
        encoding="utf-8",
    )
    checkpoint = tmp_path / "large-v3.pt"
    checkpoint.write_bytes(b"fake")
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    with wave.open(str(first), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8_000)
        wav.writeframes(b"\x00\x00" * 800)
    with wave.open(str(second), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(16_000)
        wav.writeframes(b"\x00\x00" * 2 * 1_600)

    worker = asr_script._WhisperWorker(
        Path(sys.executable),
        Path(asr_script.__file__).resolve(),
        checkpoint,
        "cpu",
        repo,
    )
    try:
        first_text, first_error = worker.transcribe("first", first)
        second_text, second_error = worker.transcribe("second", second)
    finally:
        worker.close()

    assert first_error is None
    assert second_error is None
    assert first_text == "en:transcribe:False:False:1600"
    assert second_text == "en:transcribe:False:False:1600"
    assert len((package / "loads.txt").read_text().splitlines()) == 1
    assert (package / "arrays.txt").read_text().splitlines() == [
        "1600|float32|1",
        "1600|float32|1",
    ]
