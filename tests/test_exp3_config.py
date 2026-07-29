from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_safety.config import load_exp3_config

CONFIG = Path("configs/experiments/exp3_qwen_refusal_mechanism.yaml")


def test_exp3_config_loads_without_exp1_dataset_sections():
    cfg = load_exp3_config(CONFIG)
    assert cfg.name == "exp3_qwen_refusal_mechanism"
    assert cfg.model.model_id == "Qwen/Qwen2-Audio-7B-Instruct"
    assert [contrast.name for contrast in cfg.exp3.contrasts if contrast.enabled] == ["phase"]
    assert cfg.model.revision == "0a095220c30b7b31434169c3086508ef3ea5bf0a"
    assert cfg.exp3.span_patch.layers == [8, 10, 12, 14, 16, 18]
    assert cfg.exp3.escape_reset.windows[0].inject_layer == 10
    assert cfg.exp3.escape_reset.windows[0].reset_layer == 12


def test_exp3_dotted_override_is_validated():
    cfg = load_exp3_config(
        CONFIG,
        overrides=["exp3.input_dose.max_pairs=2", "exp3.span_patch.full_generation_layers=[10]"],
    )
    assert cfg.exp3.input_dose.max_pairs == 2
    assert cfg.exp3.span_patch.full_generation_layers == [10]


def test_exp3_rejects_unknown_stage_contrast():
    with pytest.raises(ValidationError, match="unknown contrasts"):
        load_exp3_config(CONFIG, overrides=["exp3.input_dose.contrast=missing"])


def test_exp3_rejects_unordered_escape_window():
    with pytest.raises(ValidationError, match="reset_layer must be later"):
        load_exp3_config(
            CONFIG,
            overrides=["exp3.escape_reset.windows=[{inject_layer: 12, reset_layer: 10}]"],
        )
