import numpy as np
import pytest

from audio_safety.config.schema import AudioRdoDecisionConfig
from audio_safety.evaluation import AudioRdoGateMetrics, decide_audio_rdo_gate
from audio_safety.pipelines.audio_rdo import (
    behavior_counts,
    benign_controlled_safety_shift,
    coordinate_restore,
    dim_compliance_to_refusal,
    escape_scores,
    harmful_compliance_rate,
    projection_ablate,
    refusal_rate,
    sar_text_refusal_vector,
    signed_occupancy,
    unit_vector,
)


def test_signed_occupancy_and_ablation():
    axis = np.array([2.0, 0.0, 0.0])
    hidden = np.array([[3.0, 4.0, 0.0]])
    np.testing.assert_allclose(signed_occupancy(hidden, axis), np.array([3.0]))
    ablated = projection_ablate(hidden, axis)
    np.testing.assert_allclose(ablated, np.array([[0.0, 4.0, 0.0]]))


def test_coordinate_restore_only_changes_axis_coordinate():
    axis = np.array([1.0, 0.0])
    styled = np.array([[1.0, 10.0], [2.0, 20.0]])
    neutral = np.array([[5.0, -1.0], [7.0, -2.0]])
    restored = coordinate_restore(styled, neutral, axis)
    np.testing.assert_allclose(restored, np.array([[5.0, 10.0], [7.0, 20.0]]))


def test_escape_scores_benign_controlled():
    axis = np.array([1.0, 0.0])
    harmful_neutral = np.array([[5.0, 0.0]])
    harmful_style = np.array([[2.0, 0.0]])
    benign_neutral = np.array([[3.0, 0.0]])
    benign_style = np.array([[2.0, 0.0]])
    delta = benign_controlled_safety_shift(
        harmful_style,
        harmful_neutral,
        benign_style,
        benign_neutral,
    )
    np.testing.assert_allclose(delta, np.array([[-2.0, 0.0]]))
    np.testing.assert_allclose(
        escape_scores(harmful_style, harmful_neutral, benign_style, benign_neutral, axis),
        np.array([2.0]),
    )


def test_baseline_vectors_unit_norm():
    refused = np.array([[2.0, 0.0], [4.0, 0.0]])
    complied = np.array([[1.0, 0.0], [1.0, 0.0]])
    np.testing.assert_allclose(dim_compliance_to_refusal(refused, complied), np.array([1.0, 0.0]))

    base = np.array([[1.0, 1.0], [2.0, 2.0]])
    refusal = np.array([[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(sar_text_refusal_vector(refusal, base), np.array([0.0, 1.0]))
    np.testing.assert_allclose(np.linalg.norm(unit_vector(np.array([3.0, 4.0]))), 1.0)


def test_behavior_rates_exclude_decoding_failure():
    labels = [
        "policy_refusal",
        "harmful_compliance",
        "decoding_failure",
        "policy_refusal",
    ]
    assert behavior_counts(labels)["decoding_failure"] == 1
    assert refusal_rate(labels) == pytest.approx(2 / 3)
    assert harmful_compliance_rate(labels) == pytest.approx(1 / 3)


def test_audio_rdo_decision_go():
    result = decide_audio_rdo_gate(
        AudioRdoGateMetrics(
            genuine_style_gap_pp=10.0,
            decoding_failure_share=0.1,
            add_rr_pp=25.0,
            benign_orr_add_pp=2.0,
            ablation_asr_pp=12.0,
            rdo_beats_mdsteer_c2r=True,
            rdo_beats_sarsteer_text=True,
            escape_spearman=0.35,
            restoration_rr_pp=22.0,
            restored_fraction=0.30,
            benign_orr_restore_pp=1.0,
        ),
        AudioRdoDecisionConfig(),
        max_decoding_failure_share=0.5,
    )
    assert result.status == "GO"


def test_audio_rdo_decision_weak_go():
    result = decide_audio_rdo_gate(
        AudioRdoGateMetrics(
            genuine_style_gap_pp=3.0,
            decoding_failure_share=0.1,
            add_rr_pp=25.0,
            benign_orr_add_pp=2.0,
            ablation_asr_pp=12.0,
            rdo_beats_mdsteer_c2r=True,
            rdo_beats_sarsteer_text=True,
            escape_spearman=0.1,
            restoration_rr_pp=5.0,
            restored_fraction=0.05,
            benign_orr_restore_pp=1.0,
        ),
        AudioRdoDecisionConfig(),
        max_decoding_failure_share=0.5,
    )
    assert result.status == "WEAK-GO"


def test_audio_rdo_decision_nogo_on_benign_orr():
    result = decide_audio_rdo_gate(
        AudioRdoGateMetrics(
            genuine_style_gap_pp=10.0,
            decoding_failure_share=0.1,
            add_rr_pp=25.0,
            benign_orr_add_pp=4.0,
            ablation_asr_pp=12.0,
            rdo_beats_mdsteer_c2r=True,
            rdo_beats_sarsteer_text=True,
        ),
        AudioRdoDecisionConfig(),
        max_decoding_failure_share=0.5,
    )
    assert result.status == "NO-GO"
