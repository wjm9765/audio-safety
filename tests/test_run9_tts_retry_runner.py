"""Contract tests for the isolated Run 9 CosyVoice retry adapter."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts/run_run9_tts_retry.py"
    spec = importlib.util.spec_from_file_location("run9_tts_retry_runner", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


runner = _load_script()


def _job(tmp_path: Path) -> dict:
    text = "reference text"
    output = tmp_path / "retry.wav"
    return {
        "item_id": "q0",
        "style": "neutral",
        "text": text,
        "reference_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "output_path": str(output),
        "provenance_path": str(output.with_suffix(".wav.render.json")),
        "retry_contract_version": runner.RETRY_CONTRACT_VERSION,
        "retry_id": "seed1709_v1",
        "seed": 123,
        "speed": 1.0,
        "overwrite": False,
    }


def test_retry_runner_accepts_only_seed_only_neutral_contract(tmp_path):
    values = runner._validated_job(_job(tmp_path))
    assert values["seed"] == 123
    assert values["speed"] == 1.0

    bad_speed = {**_job(tmp_path), "speed": 0.97}
    with pytest.raises(ValueError, match="preserve speed=1.0"):
        runner._validated_job(bad_speed)
    bad_style = {**_job(tmp_path), "style": "sad"}
    with pytest.raises(ValueError, match="preserve neutral style"):
        runner._validated_job(bad_style)


def test_retry_runner_validates_resumed_audio_and_sidecars(tmp_path):
    values = runner._validated_job(_job(tmp_path))
    output = values["output"]
    output.write_bytes(b"candidate")
    values["hash_sidecar"].write_text(values["reference_sha256"])
    prompt = tmp_path / "prompt.wav"
    prompt.write_bytes(b"voice")
    provenance = runner._provenance(
        values,
        output=output,
        prompt_audio=prompt,
        model_id="FunAudioLLM/CosyVoice2-0.5B",
    )
    values["provenance"].write_text(json.dumps(provenance))

    runner._validate_existing(
        values,
        prompt_audio=prompt,
        model_id="FunAudioLLM/CosyVoice2-0.5B",
    )

    values["hash_sidecar"].write_text("0" * 64)
    with pytest.raises(ValueError, match="text-hash sidecar mismatch"):
        runner._validate_existing(
            values,
            prompt_audio=prompt,
            model_id="FunAudioLLM/CosyVoice2-0.5B",
        )
