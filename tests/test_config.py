from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_safety.config import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP1 = REPO_ROOT / "configs" / "experiments" / "exp1_refusal_cone_drift.yaml"


def test_exp1_config_loads():
    cfg = load_experiment_config(EXP1)
    assert cfg.name == "exp1_refusal_cone_drift"
    assert cfg.model.model_id == "Qwen/Qwen2-Audio-7B-Instruct"
    assert cfg.drift.families == ["plain", "nonspeech", "style", "perturbed"]
    assert len(cfg.cone.categories) == 6


def test_preregistered_thresholds_match_design():
    """design.md §0: GO < 0.6, NO-GO >= 0.85, p < 0.05. If this test fails, someone
    touched the pre-registered thresholds — that requires an explicit protocol change."""
    cfg = load_experiment_config(EXP1)
    assert cfg.decision.go_max_cosine == 0.60
    assert cfg.decision.nogo_min_cosine == 0.85
    assert cfg.decision.p_threshold == 0.05


def test_dotted_overrides():
    cfg = load_experiment_config(
        EXP1, overrides=["stats.n_permutations=100", "seed=7", "drift.families=[plain,style]"]
    )
    assert cfg.stats.n_permutations == 100
    assert cfg.seed == 7
    assert cfg.drift.families == ["plain", "style"]


def test_unknown_key_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(EXP1.read_text() + "\ntypo_key: 1\n")
    with pytest.raises(ValidationError):
        load_experiment_config(bad)


def test_bad_override_path_rejected():
    with pytest.raises(KeyError):
        load_experiment_config(EXP1, overrides=["nonexistent.section=1"])
