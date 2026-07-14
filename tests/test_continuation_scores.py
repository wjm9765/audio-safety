"""CPU-only tests for Run 7 counterfactual continuation-score curves."""

import numpy as np
import pytest
from scipy.special import logsumexp

from audio_safety.evaluation.continuation_scores import (
    aggregate_class_curve,
    cumulative_mean_curve,
    refusal_compliance_curve,
    shifted_target_log_probs,
)


def test_shifted_target_log_probs_uses_preceding_logits_and_respects_limit():
    # Positions 0-1 are the prompt; tokens at positions 2-4 are the target.  Each
    # target must be scored from logits at the immediately preceding position.
    input_ids = np.array([0, 1, 2, 3, 1], dtype=np.int64)
    logits = np.zeros((len(input_ids), 4), dtype=np.float64)
    logits[1] = [0.0, -1.0, 3.0, 0.5]  # predicts target token 2 at position 2
    logits[2] = [-2.0, 0.0, 0.5, 2.0]  # predicts target token 3 at position 3
    logits[3] = [1.0, 2.5, -0.5, 0.0]  # would predict the third target token

    actual = shifted_target_log_probs(
        logits,
        input_ids,
        prompt_length=2,
        max_tokens=2,
    )
    expected = np.array(
        [
            logits[1, 2] - logsumexp(logits[1]),
            logits[2, 3] - logsumexp(logits[2]),
        ]
    )

    np.testing.assert_allclose(actual, expected)
    assert actual.shape == (2,)


@pytest.mark.parametrize("prompt_length", [0, 3])
def test_shifted_target_log_probs_requires_nonempty_prompt_and_target(prompt_length):
    with pytest.raises(ValueError, match="leave at least one target token"):
        shifted_target_log_probs(
            np.zeros((3, 4)),
            np.array([0, 1, 2]),
            prompt_length=prompt_length,
            max_tokens=2,
        )


def test_shifted_target_log_probs_rejects_misaligned_logits():
    with pytest.raises(ValueError, match="align with input_ids"):
        shifted_target_log_probs(
            np.zeros((2, 4)),
            np.array([0, 1, 2]),
            prompt_length=1,
            max_tokens=2,
        )


def test_cumulative_mean_curve_is_length_normalized_and_last_value_padded():
    curve = cumulative_mean_curve([-1.0, -3.0], steps=4)
    np.testing.assert_allclose(curve, [-1.0, -2.0, -2.0, -2.0])


def test_class_and_refusal_compliance_curves_average_paraphrases_separately():
    refusal = [[-1.0, -1.0], [-3.0]]
    compliance = [[-4.0, -2.0], [-2.0, -2.0]]

    refusal_curve = aggregate_class_curve(refusal, steps=3)
    result = refusal_compliance_curve(refusal, compliance, steps=3)

    np.testing.assert_allclose(refusal_curve, [-2.0, -2.0, -2.0])
    np.testing.assert_allclose(result["refusal_curve"], refusal_curve)
    np.testing.assert_allclose(result["compliance_curve"], [-3.0, -2.5, -2.5])
    np.testing.assert_allclose(result["continuation_curve"], [1.0, 0.5, 0.5])


def test_curve_helpers_reject_empty_targets_and_nonpositive_steps():
    with pytest.raises(ValueError, match="at least one scored token"):
        cumulative_mean_curve([], steps=2)
    with pytest.raises(ValueError, match="steps must be positive"):
        cumulative_mean_curve([-1.0], steps=0)
    with pytest.raises(ValueError, match="at least one continuation"):
        aggregate_class_curve([], steps=2)
