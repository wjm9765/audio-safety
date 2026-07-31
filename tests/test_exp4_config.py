from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_safety.config import load_exp4_config

CONFIG = Path("configs/experiments/exp4_audio_kv_routing.yaml")


def test_exp4_config_inherits_frozen_exp3_contract():
    cfg = load_exp4_config(CONFIG)
    assert cfg.name == "exp4_audio_kv_routing"
    assert cfg.model.model_id == "Qwen/Qwen2-Audio-7B-Instruct"
    assert cfg.model.revision == "0a095220c30b7b31434169c3086508ef3ea5bf0a"
    assert cfg.exp3.instruction == "Please respond to the request in the audio."
    assert cfg.exp4.inject_layer == 10
    assert cfg.exp4.relay_start_layer == 11
    assert cfg.exp4.expected_relay_positions == 15
    assert set(cfg.exp4.conditions) == {
        "audio_injected__tab_injected",
        "audio_host__tab_injected",
        "audio_injected__tab_host",
        "audio_host__tab_host",
    }


def test_exp4_dotted_override_is_applied_after_exp3_composition():
    cfg = load_exp4_config(
        CONFIG,
        overrides=["exp4.max_pairs=2", "exp4.n_bootstrap=100", "exp3.max_new_tokens=48"],
    )
    assert cfg.exp4.max_pairs == 2
    assert cfg.exp4.n_bootstrap == 100
    assert cfg.exp3.max_new_tokens == 48


def test_exp4_rejects_duplicated_factorial_cell():
    with pytest.raises(ValidationError, match="conditions must be unique"):
        load_exp4_config(
            CONFIG,
            overrides=[
                "exp4.conditions=[audio_injected__tab_injected,"
                "audio_host__tab_injected,audio_injected__tab_host,"
                "audio_injected__tab_host]"
            ],
        )


def test_exp4_rejects_unordered_layers_and_overlapping_thresholds():
    with pytest.raises(ValidationError, match="relay_start_layer must be later"):
        load_exp4_config(CONFIG, overrides=["exp4.relay_start_layer=10"])
    with pytest.raises(ValidationError, match="negligible_effect_max must be smaller"):
        load_exp4_config(CONFIG, overrides=["exp4.negligible_effect_max=0.20"])
