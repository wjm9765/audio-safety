"""exp2 model-pool config and loader-dispatch tests (CPU-only, no weights).

These guard the two ways adding models could silently break exp1:
1. an exp1 config must still route through the pinned Qwen2-Audio path;
2. a new model config must not be loadable through that pinned path by accident.
"""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from audio_safety.config import load_experiment_config
from audio_safety.config.schema import ModelConfig
from audio_safety.models.hf_audio_lm import EXP2_MODEL_POOL
from audio_safety.models.loader import QWEN2_AUDIO_MODEL_ID, load_model_and_processor

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "configs" / "models"
EXP1 = REPO_ROOT / "configs" / "experiments" / "exp1_refusal_cone_drift.yaml"

EXP2_CONFIGS = {
    "minicpm_o_2_6.yaml": "openbmb/MiniCPM-o-2_6",
    "kimi_audio.yaml": "moonshotai/Kimi-Audio-7B-Instruct",
    "ultravox_v0_4_1.yaml": "fixie-ai/ultravox-v0_4_1-llama-3_1-8b",
}


def _load(name: str) -> ModelConfig:
    raw = yaml.safe_load((MODEL_DIR / name).read_text(encoding="utf-8")) or {}
    return ModelConfig(**raw)


def test_qwen_config_still_defaults_to_pinned_loader():
    cfg = _load("qwen2_audio.yaml")
    assert cfg.model_id == QWEN2_AUDIO_MODEL_ID
    assert cfg.loader == "qwen2_audio"
    assert cfg.trust_remote_code is False
    assert cfg.revision is None


def test_exp1_experiment_config_unchanged_by_new_fields():
    cfg = load_experiment_config(EXP1)
    assert cfg.model.model_id == QWEN2_AUDIO_MODEL_ID
    assert cfg.model.loader == "qwen2_audio"


@pytest.mark.parametrize(("filename", "model_id"), sorted(EXP2_CONFIGS.items()))
def test_exp2_model_configs_load(filename: str, model_id: str):
    cfg = _load(filename)
    assert cfg.model_id == model_id
    assert cfg.loader == "hf_auto_multimodal"
    assert cfg.trust_remote_code is True
    assert cfg.dtype == "bfloat16"


@pytest.mark.parametrize(("filename", "model_id"), sorted(EXP2_CONFIGS.items()))
def test_exp2_model_ids_are_declared_in_pool(filename: str, model_id: str):
    assert model_id in EXP2_MODEL_POOL, f"{filename} points at an undeclared model"


def test_resistant_control_is_still_in_the_pool():
    """Qwen2-Audio is retained as the specificity contrast, not dropped."""
    assert EXP2_MODEL_POOL[QWEN2_AUDIO_MODEL_ID] == "resistant_control"


def test_pinned_loader_rejects_non_qwen_model_id():
    cfg = ModelConfig(model_id="openbmb/MiniCPM-o-2_6", loader="qwen2_audio")
    with pytest.raises(ValueError, match="pinned to"):
        load_model_and_processor(cfg)


def test_unknown_loader_is_rejected():
    cfg = ModelConfig(model_id="some/model", loader="not_a_loader")
    with pytest.raises(ValueError, match="unknown loader"):
        load_model_and_processor(cfg)


def test_model_config_still_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        ModelConfig(model_id="some/model", nonexistent_field=1)
