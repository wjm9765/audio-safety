import importlib.util
from pathlib import Path

import pytest


def _load_tts_script():
    path = Path(__file__).parents[1] / "scripts" / "cosyvoice2_tts.py"
    spec = importlib.util.spec_from_file_location("cosyvoice2_tts_script", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


tts_script = _load_tts_script()


def test_batch_progress_description_reports_worker_and_shard_style():
    jobs = [{"style": "jb_ica"} for _ in range(300)]
    assert tts_script.batch_progress_description(jobs, 1, 3) == "TTS worker 2/3 jb_ica"


def test_batch_worker_context_reads_parent_metadata(monkeypatch):
    monkeypatch.setenv("AUDIO_SAFETY_TTS_WORKER_INDEX", "2")
    monkeypatch.setenv("AUDIO_SAFETY_TTS_NUM_WORKERS", "3")
    assert tts_script.batch_worker_context() == (2, 3)


def test_batch_worker_context_rejects_invalid_index(monkeypatch):
    monkeypatch.setenv("AUDIO_SAFETY_TTS_WORKER_INDEX", "3")
    monkeypatch.setenv("AUDIO_SAFETY_TTS_NUM_WORKERS", "3")
    with pytest.raises(ValueError, match="between 0 and 2"):
        tts_script.batch_worker_context()
