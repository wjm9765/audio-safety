"""Counterfactual multi-token continuation-score curves (CPU helpers).

Teacher-forced target token ``t_j`` is scored from the logit immediately before
it.  The functions here operate on NumPy arrays so indexing and curve aggregation
remain unit-testable without torch/transformers.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp


def shifted_target_log_probs(
    logits: np.ndarray,
    input_ids: np.ndarray,
    *,
    prompt_length: int,
    max_tokens: int,
) -> np.ndarray:
    """Return log p(target token | preceding prefix) for target positions only."""
    scores = np.asarray(logits, dtype=np.float64)
    tokens = np.asarray(input_ids, dtype=np.int64).reshape(-1)
    if scores.ndim != 2 or scores.shape[0] != len(tokens):
        raise ValueError("logits must be (sequence, vocab) and align with input_ids")
    if prompt_length < 1 or prompt_length >= len(tokens):
        raise ValueError("prompt_length must leave at least one target token")
    stop = min(len(tokens), prompt_length + int(max_tokens))
    target_positions = np.arange(prompt_length, stop)
    predicting = scores[target_positions - 1]
    normalizers = logsumexp(predicting, axis=-1)
    chosen = predicting[np.arange(len(target_positions)), tokens[target_positions]]
    return chosen - normalizers


def cumulative_mean_curve(token_log_probs: Sequence[float], *, steps: int) -> np.ndarray:
    """Length-normalized prefix log-likelihood, padded by its last value."""
    values = np.asarray(token_log_probs, dtype=np.float64).reshape(-1)
    if len(values) == 0:
        raise ValueError("a continuation must contain at least one scored token")
    if steps < 1:
        raise ValueError("steps must be positive")
    prefix = np.cumsum(values) / np.arange(1, len(values) + 1)
    curve = np.empty(steps, dtype=np.float64)
    used = min(steps, len(prefix))
    curve[:used] = prefix[:used]
    curve[used:] = prefix[used - 1]
    return curve


def aggregate_class_curve(
    target_token_log_probs: Sequence[Sequence[float]], *, steps: int
) -> np.ndarray:
    """Average paraphrase-level cumulative curves without lexical pooling."""
    if not target_token_log_probs:
        raise ValueError("at least one continuation is required per class")
    curves = [cumulative_mean_curve(row, steps=steps) for row in target_token_log_probs]
    return np.mean(np.stack(curves), axis=0)


def refusal_compliance_curve(
    refusal_scores: Sequence[Sequence[float]],
    compliance_scores: Sequence[Sequence[float]],
    *,
    steps: int,
) -> dict[str, list[float]]:
    refusal = aggregate_class_curve(refusal_scores, steps=steps)
    compliance = aggregate_class_curve(compliance_scores, steps=steps)
    return {
        "refusal_curve": refusal.tolist(),
        "compliance_curve": compliance.tolist(),
        "continuation_curve": (refusal - compliance).tolist(),
    }
