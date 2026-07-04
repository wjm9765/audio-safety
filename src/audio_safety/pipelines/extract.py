"""Activation extraction and layer selection (design.md §4.1-§4.2).

Requires the gpu dependency group. Heavy lifting is per-sample forwards with
ResidualStreamCapture; all downstream analysis consumes the saved .npz arrays and
stays torch-free.
"""

from pathlib import Path
from typing import Any

import numpy as np

from audio_safety.models.hooks import ResidualStreamCapture


def extract_last_token_activations(
    model: Any,
    inputs: dict[str, Any],
) -> dict[int, np.ndarray]:
    """One forward pass; returns {layer_idx: (d,) float32 array} at the last input
    token (readout position fixed across all conditions — design.md §3.3)."""
    import torch

    with ResidualStreamCapture(model) as cap, torch.no_grad():
        model(**inputs)
    return {i: t.numpy() for i, t in cap.states().items()}


def layer_separation_sweep(
    harmful_by_layer: dict[int, np.ndarray],
    benign_by_layer: dict[int, np.ndarray],
) -> dict[int, float]:
    """Diff-in-means separation per layer (design.md §4.2): distance between harmful
    and benign mean hidden states, normalized by pooled per-layer std so layers with
    different activation scales are comparable. Argmax -> primary refusal layer L*.

    Report the full sweep (Fig 1) — never pick a layer by fiat.
    """
    separation: dict[int, float] = {}
    for layer, h in harmful_by_layer.items():
        b = benign_by_layer[layer]
        gap = np.linalg.norm(h.mean(axis=0) - b.mean(axis=0))
        pooled_std = float(np.sqrt((h.var(axis=0).mean() + b.var(axis=0).mean()) / 2))
        separation[layer] = float(gap / pooled_std) if pooled_std > 0 else 0.0
    return separation


def pick_primary_layer(separation: dict[int, float]) -> int:
    """L* = argmax separation (design.md §4.2)."""
    return max(separation, key=separation.__getitem__)


def save_activations(acts: dict[str, np.ndarray], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **acts)
