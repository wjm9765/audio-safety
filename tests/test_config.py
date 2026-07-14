from pathlib import Path

import pytest
from pydantic import ValidationError

from audio_safety.config import load_experiment_config

REPO_ROOT = Path(__file__).resolve().parents[1]
EXP1 = REPO_ROOT / "configs" / "experiments" / "exp1_refusal_cone_drift.yaml"
EXP1_FAST = REPO_ROOT / "configs" / "experiments" / "exp1_refusal_cone_drift_fast.yaml"
RUN7_COAST_R = REPO_ROOT / "configs" / "experiments" / "run7_coast_r_stage_a.yaml"


def test_exp1_config_loads():
    cfg = load_experiment_config(EXP1)
    assert cfg.name == "exp1_audio_rdo_axis_gate"
    assert cfg.model.model_id == "Qwen/Qwen2-Audio-7B-Instruct"
    assert cfg.dataset.harmful_source == "figstep_safebench"
    assert str(cfg.dataset.seed_file) == "text/figstep/safebench.csv"
    assert str(cfg.dataset.source_file) == "text/figstep/audio_rdo_pairs.jsonl"
    assert cfg.dataset.pair_generation.model == "z-ai/glm-5.2"
    assert cfg.dataset.pair_generation.api_key_env == "OPENROUTER_API_KEY"
    assert cfg.dataset.pair_generation.max_concurrency == 8
    assert cfg.dataset.style_variant_generation.styles == ["sad", "angry"]
    assert cfg.dataset.style_variant_generation.max_concurrency == 8
    assert (
        str(cfg.dataset.style_variant_generation.output_file)
        == "text/figstep/audio_rdo_style_variants.jsonl"
    )
    assert cfg.dataset.tts.engine == "cosyvoice2"
    assert "scripts/cosyvoice2_tts.py" in str(cfg.dataset.tts.command_template)
    assert "--batch-jsonl" in str(cfg.dataset.tts.batch_command_template)
    assert cfg.dataset.tts.batch_workers == 2
    assert cfg.dataset.tts.batch_worker_cuda_devices == ["0"]
    assert cfg.dataset.tts.batch_worker_env["OMP_NUM_THREADS"] == "2"
    assert cfg.dataset.asr.mode == "skip"
    assert cfg.dataset.transcript_control.require_style_classifier_pass is False
    assert cfg.dataset.target_generation.max_new_tokens == 64
    assert cfg.dataset.styles == ["neutral", "sad", "angry"]
    assert cfg.hidden.layers == [8, 12, 16, 20, 24, 28]
    assert cfg.rdo.refusal_target.startswith("I'm sorry")


def test_exp1_fast_config_loads():
    cfg = load_experiment_config(EXP1_FAST)
    assert cfg.name == "exp1_audio_rdo_axis_gate_fast"
    assert cfg.hidden.layers == [12, 16, 20]
    assert cfg.hidden.positions == ["first_generation_prelogit"]
    assert cfg.rdo.train_steps == 50
    assert cfg.rdo.limit_per_site == 10
    assert cfg.baselines.random_vectors == 4


def test_run7_coast_r_config_loads_with_frozen_stage_a_contract():
    cfg = load_experiment_config(RUN7_COAST_R)
    coast = cfg.coast_r

    assert cfg.name == "run7_coast_r_stage_a"
    assert coast is not None and coast.enabled is True
    assert coast.source_run_name == "run5_20260714_0308_pitch_n150"
    assert coast.site == "llm_p2"
    assert coast.primary_layer == 18
    assert coast.sensitivity_layers == [16, 20]
    assert coast.operator_family == "librosa_phase_vocoder_compound"
    assert coast.operator_backend == "librosa"
    assert coast.severity_field == "pitch_semitones"
    assert coast.outer_folds == 5
    assert coast.reachable_rank_candidates == [4, 8, 16, 32]
    assert coast.reachable_rank_cap == 32
    assert coast.transport_ranks == [1, 2, 3, 4]
    assert coast.endpoint_kind == "continuation_curve"
    assert coast.max_continuation_tokens == 32
    assert len(coast.refusal_continuations) == 3
    assert len(coast.compliance_continuations) == 3
    assert coast.score_safety_labels == ["harmful", "benign"]
    assert str(coast.fit_artifact_file) == "coast_r/fit_artifacts.npz"
    assert coast.intervention_eligibility == "neutral_refusers"


def test_preregistered_thresholds_match_design():
    """design.md §0 Audio-RDO gate thresholds. If this test fails, someone touched
    the pre-registered thresholds — that requires an explicit protocol change."""
    cfg = load_experiment_config(EXP1)
    assert cfg.decision.min_genuine_style_gap_pp == 8.0
    assert cfg.decision.min_add_rr_pp == 20.0
    assert cfg.decision.max_benign_orr_pp == 3.0
    assert cfg.decision.min_ablation_asr_pp == 10.0
    assert cfg.decision.min_restoration_rr_pp == 20.0
    assert cfg.decision.min_restored_fraction == 0.25
    assert cfg.decision.min_escape_spearman == 0.30
    assert cfg.decision.min_escape_auroc == 0.65


def test_dotted_overrides():
    cfg = load_experiment_config(
        EXP1,
        overrides=[
            "stats.n_permutations=100",
            "seed=7",
            "dataset.styles=[neutral,angry]",
            "dataset.pair_generation.max_concurrency=3",
        ],
    )
    assert cfg.stats.n_permutations == 100
    assert cfg.seed == 7
    assert cfg.dataset.styles == ["neutral", "angry"]
    assert cfg.dataset.pair_generation.max_concurrency == 3


def test_unknown_key_rejected(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(EXP1.read_text() + "\ntypo_key: 1\n")
    with pytest.raises(ValidationError):
        load_experiment_config(bad)


def test_bad_override_path_rejected():
    with pytest.raises(KeyError):
        load_experiment_config(EXP1, overrides=["nonexistent.section=1"])
