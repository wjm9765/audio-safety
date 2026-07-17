"""Faithful SARSteer defense (arXiv:2510.17633), reimplemented in-house.

SARSteer has NO public code release; this reconstructs Algorithm 1 from the paper
so we can run it as a GATE against our content-preserving channel attack. The
method has two ingredients, applied at inference with the model frozen:

1. **Text-derived refusal vector, per decoder layer.** From TEXT (not audio)
   harmful queries, contrast the activation with vs without an appended refusal
   response, averaged over a calibration set:

       v_ell = mean_i h_ell(harm_text_i + refusal) - mean_i h_ell(harm_text_i)

   The direction is text-derived precisely because SARSteer argues an audio
   difference-in-means direction is noisy; the text refusal direction is
   "modality-agnostic".

2. **Decomposed safe-space ablation.** PCA on benign-SPEECH activations gives the
   top-k directions U_ell of benign variance. Keep only the component of the
   refusal vector ORTHOGONAL to that safe space, so steering does not push benign
   speech toward refusal:

       v_perp_ell = (I - U_ell U_ell^T) v_ell

Inference steers every decoder layer at every generated position:

       h'_ell = h_ell + alpha * v_perp_ell        (alpha = 0.1, raw v_perp)

Fidelity NOTES (Codex cross-check 2026-07-17; confirm against the paper's
Appendix A.5 BEFORE any confirmatory run — these are the under-specified knobs):

- ``extraction_position``: how the per-layer activation is pooled for the refusal
  contrast. Default ``"mean_all"`` = mean over all sequence tokens, the literal
  reading of Eq. 4 (``mu = mean activation of the sequence``); non-degenerate.
  ``"last_token"`` is a control only — a naive last-token contrast confounds the
  refusal direction with terminal-token identity (Codex 2026-07-17). Confirm the
  exact pooling against Appendix A.5 before a confirmatory run.
- ``n_refusal_calib`` vs ``n_benign_pca``: the refusal-contrast count and the
  benign-PCA sample count are NOT necessarily the same set; keep them separate.
  Paper reports insensitivity from n=10..100; default 100.
- ``alpha`` is applied to the RAW orthogonal component (do NOT unit-normalize
  v_perp; that would change what alpha=0.1 means). Enforced by
  ``MultiLayerAdditiveSteering(normalize=False)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Pure numpy core (no torch) — unit-testable on CPU.
# ---------------------------------------------------------------------------


def safe_subspace(activations: np.ndarray, n_pcs: int) -> np.ndarray:
    """Top-``n_pcs`` PCA directions of benign-speech activations for one layer.

    ``activations`` is ``(n_samples, d)``. Returns an orthonormal basis ``U`` of
    shape ``(d, k)`` with ``k = min(n_pcs, n_samples-1, d)`` columns, computed on
    MEAN-CENTERED data (PCA, not raw SVD of the gram matrix). Columns are the
    principal directions of benign variance to be ablated out of the refusal
    vector.
    """
    if activations.ndim != 2:
        raise ValueError(f"expected (n_samples, d) activations, got {activations.shape}")
    n_samples, d = activations.shape
    if n_samples < 2:
        raise ValueError("safe-space PCA needs at least 2 benign samples")
    if n_pcs < 1:
        raise ValueError("n_pcs must be >= 1")
    data = activations.astype(np.float64)
    centered = data - data.mean(axis=0, keepdims=True)
    # Right singular vectors of the centered data = principal axes.
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    # Centered data has rank <= n_samples-1, so never keep more components than
    # that (an extra column would be numerical noise, not benign variance).
    k = min(n_pcs, n_samples - 1, d)
    basis = vt[:k].T  # (d, k), orthonormal columns
    return np.ascontiguousarray(basis.astype(np.float32))


def orthogonal_complement(vector: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Return ``(I - U U^T) v`` — the part of ``vector`` outside ``basis``' span.

    ``basis`` has orthonormal columns ``(d, k)``; ``vector`` is ``(d,)``. The raw
    magnitude of the result is preserved (no renormalization): SARSteer applies
    ``alpha`` to this raw component.
    """
    if vector.ndim != 1:
        raise ValueError(f"vector must be 1-D, got {vector.shape}")
    if basis.ndim != 2 or basis.shape[0] != vector.shape[0]:
        raise ValueError(
            f"basis {basis.shape} incompatible with vector dim {vector.shape[0]}"
        )
    v = vector.astype(np.float64)
    u = basis.astype(np.float64)
    projected = u @ (u.T @ v)  # U U^T v
    return (v - projected).astype(np.float32)


def build_sarsteer_vectors(
    refusal_vectors: dict[int, np.ndarray],
    benign_activations: dict[int, np.ndarray],
    *,
    n_pcs: int = 10,
) -> dict[int, np.ndarray]:
    """Assemble the per-layer defense vectors ``v_perp_ell``.

    ``refusal_vectors[ell]`` is the ``(d,)`` text-derived refusal vector;
    ``benign_activations[ell]`` is the ``(n, d)`` benign-speech activation matrix
    for the safe-space PCA. Layers present in both are used; a layer missing from
    either is skipped (and reported by the caller).
    """
    layers = sorted(set(refusal_vectors) & set(benign_activations))
    if not layers:
        raise ValueError("no layer is present in both refusal_vectors and benign_activations")
    out: dict[int, np.ndarray] = {}
    for ell in layers:
        basis = safe_subspace(benign_activations[ell], n_pcs)
        out[ell] = orthogonal_complement(refusal_vectors[ell], basis)
    return out


def save_sarsteer_vectors(path: Path, vectors: dict[int, np.ndarray], meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {f"layer_{ell}": vec for ell, vec in vectors.items()}
    payload["_layers"] = np.asarray(sorted(vectors), dtype=np.int64)
    payload["_meta"] = np.asarray(_encode_meta(meta))
    np.savez_compressed(path, **payload)


def load_sarsteer_vectors(path: Path) -> dict[int, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SARSteer vectors: {path}. Run scripts/build_sarsteer_defense.py first."
        )
    data = np.load(path, allow_pickle=False)
    layers = [int(x) for x in data["_layers"]]
    return {ell: data[f"layer_{ell}"].astype(np.float32) for ell in layers}


def _encode_meta(meta: dict[str, Any]) -> str:
    import json

    return json.dumps(meta, sort_keys=True)


# ---------------------------------------------------------------------------
# GPU extraction + defended generation (lazy torch; runs on the GPU box only).
# ---------------------------------------------------------------------------


def _capture_pooled_all_layers(model: Any, inputs: Any, pool: str) -> dict[int, np.ndarray]:
    """Capture every decoder layer's activation for one batch-1 forward, pooled.

    ``pool="mean_all"`` means over ALL sequence positions; ``pool="last_token"``
    takes the final position. Uses a local all-layer hook (not the single-token
    ResidualStreamCapture) so mean pooling is available.
    """
    import torch

    from audio_safety.models.hooks import get_decoder_layers

    layers = get_decoder_layers(model)
    acts: dict[int, np.ndarray] = {}
    handles = []

    def make(layer_idx: int):
        def hook(module: Any, inp: Any, out: Any) -> None:
            hidden = out[0] if isinstance(out, tuple) else out
            if hidden.shape[0] != 1:
                raise ValueError("refusal-vector extraction requires batch size 1")
            if pool == "mean_all":
                pooled = hidden[0].mean(dim=0)
            elif pool == "last_token":
                pooled = hidden[0, -1, :]
            else:
                raise ValueError(f"unsupported pool {pool!r}")
            acts[layer_idx] = pooled.detach().float().cpu().numpy().astype(np.float32)

        return hook

    for i, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make(i)))
    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return acts


def extract_text_refusal_vectors(
    model: Any,
    processor: Any,
    harmful_texts: Sequence[str],
    *,
    refusal_text: str = "I cannot assist with that.",
    system_prompt: str = "You are a helpful assistant.",
    extraction_position: str = "mean_all",
) -> dict[int, np.ndarray]:
    """Per-layer text-derived refusal vector ``mean(harm+refusal) - mean(harm)``.

    Pooling (Codex 2026-07-17): ``"mean_all"`` (default) pools each layer's
    activation over ALL sequence tokens, the literal reading of the paper's Eq. 4
    ``mu = mean activation of the sequence``. This is NON-degenerate: the
    harm+refusal arm adds the refusal tokens' contributions on top of the shared
    query prefix. Do NOT use ``"last_token"`` naively — reading each arm at its own
    final token contrasts DIFFERENT terminal tokens (refusal identity confounded
    with the vector), and reading both at the same prefix position is exactly zero
    under causal attention. ``"last_token"`` is retained only as a control; confirm
    the exact SARSteer pooling against Appendix A.5 before a confirmatory run.
    """
    from audio_safety.models.qwen2_audio import (
        build_text_conversation,
        model_input_device,
        move_inputs_to_device,
        prepare_qwen2_text_inputs,
    )

    if extraction_position not in {"mean_all", "last_token"}:
        raise ValueError(
            f"unsupported extraction_position {extraction_position!r}; "
            "expected 'mean_all' or 'last_token'"
        )
    device = model_input_device(model)

    harm_sum: dict[int, np.ndarray] = {}
    ref_sum: dict[int, np.ndarray] = {}
    count = 0
    for text in harmful_texts:
        conversation = build_text_conversation(text, system_prompt=system_prompt)
        # "harm" arm: the query prompt alone.
        harm_inputs = prepare_qwen2_text_inputs(processor, conversation, device=device)
        # "harm+refusal" arm: append the refusal response after the generation prompt.
        prompt_text = processor.apply_chat_template(
            list(conversation), add_generation_prompt=True, tokenize=False
        )
        full_ids = processor.tokenizer(prompt_text + refusal_text, return_tensors="pt")
        full_ids = move_inputs_to_device(full_ids, device)
        harm_states = _capture_pooled_all_layers(model, harm_inputs, extraction_position)
        ref_states = _capture_pooled_all_layers(model, full_ids, extraction_position)
        for ell, state in harm_states.items():
            harm_sum[ell] = harm_sum.get(ell, 0.0) + state
        for ell, state in ref_states.items():
            ref_sum[ell] = ref_sum.get(ell, 0.0) + state
        count += 1
    if count == 0:
        raise ValueError("no harmful_texts provided for refusal-vector extraction")
    return {ell: (ref_sum[ell] - harm_sum[ell]) / count for ell in harm_sum}


def extract_benign_speech_activations(
    model: Any,
    processor: Any,
    benign_audio_paths: Sequence[Path | str],
    instruction: str,
    *,
    position_name: str = "first_generation_prelogit",
    system_prompt: str = "You are a helpful assistant.",
) -> dict[int, np.ndarray]:
    """Per-layer benign-speech activation matrices ``(n_samples, d)`` for the PCA.

    Reads all decoder layers at the resolved ``position_name`` for each benign
    audio clip. Returned dict maps ``layer -> (n, d)`` stacked activations.
    """
    import torch

    from audio_safety.models.hooks import ResidualStreamCapture
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        model_input_device,
        prepare_qwen2_audio_inputs,
        resolve_audio_position_indices,
    )

    device = model_input_device(model)
    per_layer: dict[int, list[np.ndarray]] = {}
    for audio_path in benign_audio_paths:
        conversation = build_audio_analysis_conversation(
            audio_path, instruction, system_prompt=system_prompt
        )
        token_index = resolve_audio_position_indices(processor, conversation)[position_name]
        inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
        with torch.no_grad(), ResidualStreamCapture(
            model, token_index=token_index, layers=None
        ) as cap:
            model(**inputs)
        for ell, state in cap.states().items():
            per_layer.setdefault(ell, []).append(state.numpy().astype(np.float32))
    if not per_layer:
        raise ValueError("no benign audio provided for safe-space PCA")
    return {ell: np.stack(states) for ell, states in per_layer.items()}


def generate_audio_response_with_sarsteer(
    model: Any,
    processor: Any,
    audio_path: Path | str,
    instruction: str,
    *,
    vectors: dict[int, np.ndarray],
    alpha: float = 0.1,
    max_new_tokens: int = 256,
    system_prompt: str = "You are a helpful assistant.",
    do_sample: bool = False,
) -> str:
    """Generate one response with SARSteer steering active at all layers/positions.

    ``do_sample=False`` (greedy) by default so the gate's defended vs undefended
    contrast is not dominated by sampling noise.
    """
    from audio_safety.models.hooks import MultiLayerAdditiveSteering
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        model_input_device,
        prepare_qwen2_audio_inputs,
    )

    conversation = build_audio_analysis_conversation(
        audio_path, instruction, system_prompt=system_prompt
    )
    device = model_input_device(model)
    inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    prompt_len = inputs.input_ids.shape[1]
    with MultiLayerAdditiveSteering(model, vectors=vectors, alpha=alpha, normalize=False):
        generate_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=do_sample
        )
    generate_ids = generate_ids[:, prompt_len:]
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
