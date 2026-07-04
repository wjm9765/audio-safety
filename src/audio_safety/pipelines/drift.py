"""Paired drift computation (design.md §1, §5.1).

d_f(c) = h_f(c) - h_text(c): content confounds cancel because family is a
within-item factor. Inputs are pre-extracted activations, so this stays numpy-only.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from audio_safety.pipelines.cone import project_onto_cone


def drift_vectors(h_family: np.ndarray, h_text: np.ndarray) -> np.ndarray:
    """d_f(c) = h_f(c) - h_text(c), rows aligned by content. Shapes must match —
    misaligned pairing is a silent-correctness bug, so it is checked loudly."""
    if h_family.shape != h_text.shape:
        raise ValueError(
            f"pairing mismatch: family activations {h_family.shape} vs text anchors "
            f"{h_text.shape} — contents must be aligned row-by-row"
        )
    return h_family - h_text


def project_drifts(
    h_by_family: Mapping[str, np.ndarray],
    h_text: np.ndarray,
    basis: np.ndarray,
    families: Sequence[str],
) -> np.ndarray:
    """Stack cone-projected paired drifts into the (n_contents, n_families, k)
    array consumed by audio_safety.evaluation.stats.

    All families must cover the same contents in the same row order. Samples that
    fail the comprehension filter must be dropped from ALL families (and the text
    anchor) before calling this, to keep the pairing intact.
    """
    stacks = []
    for family in families:
        d = drift_vectors(h_by_family[family], h_text)
        stacks.append(project_onto_cone(d, basis))
    return np.stack(stacks, axis=1)
