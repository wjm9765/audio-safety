"""SARSteer defense (arXiv:2510.17633), reproduced from the paper.

SARSteer released **no code** (verified 2026-07-17: the paper carries no
repository link and no public implementation exists). The authority for this
module is therefore the paper text alone — §3.3 (notation, Eq. 1, Eq. 2), §4.1
(Eq. 4), §4.2 (safe-space ablation) and Appendix A.5 (Algorithm 1).

``paper_faithful`` (default) implements those equations. ``legacy_reconstruction``
is retained ONLY to reload/reproduce the superseded 2026-07-17 artifacts; it
deviates from the paper in three ways and must not be used for a verdict (see
"Legacy deviations" below).

Two ingredients, applied at inference with the model frozen:

1. **Text-derived refusal vector, per decoder layer** (Eq. 4, Algorithm 1 Step 1).
   "Text-derived" means the refusal PROMPT ``p`` is text appended to the textual
   instruction while **the audio is unchanged and present in both arms** — it does
   NOT mean the query is text-only. With Q = (a, t) and Q' = (a, t + p):

       v_ell = mean_i h_ell(a_i, t + p) - mean_i h_ell(a_i, t)

   where, per §3.3, ``h_ell(Q)`` is "the activation at the last token position of
   layer l". SARSteer derives the direction from the refusal text because it shows
   (§3.3) that an audio difference-in-means direction degenerates into noise.

2. **Decomposed safe-space ablation** (§4.2, Algorithm 1 Step 2). PCA over the
   activations of the **paired safe queries** ``Q_safe`` — the LLM-purified,
   lexically-similar benign counterparts of the harmful calibration queries
   (§3.2), read at the same last-token position — gives the top-k directions
   ``U_ell`` of safe variance. Keep only the orthogonal component:

       v_perp_ell = (I - U_ell U_ell^T) v_ell

Inference (Eq. 2 + Algorithm 1 Step 3) steers every decoder layer, adding the raw
component at the last token position of each forward pass, i.e. at each generated
token position:

       h'_ell = h_ell + alpha * v_perp_ell        (alpha = 0.1, k = 10, raw v_perp)

Paper-pinned constants: alpha = 0.1, k = 10, n = 100 calibration pairs, refusal
prompt "I cannot assist with that.", every decoder layer. The paper reports (Fig.
5a) that n is insensitive over 10..100, and (Fig. 5c) that k is flat over 5..45.

``alpha`` multiplies the RAW orthogonal component — do NOT unit-normalize v_perp,
which would silently redefine alpha. Enforced by
``MultiLayerAdditiveSteering(normalize=False)``.

Legacy deviations (why ``legacy_reconstruction`` cannot support a verdict):
  1. it contrasts **text-only** queries with no audio, not Q=(a,t) vs Q=(a,t+p);
  2. it pools ``mean_all`` over every sequence token instead of the paper's last
     token position;
  3. it steers **all** positions (audio + prompt + generated) instead of the last
     position, which multiplies the injected norm by the sequence length and
     collapses generation at the paper's own alpha=0.1.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

SARSteerImplementation = Literal["legacy_reconstruction", "paper_faithful"]
SARSTEER_IMPLEMENTATIONS = frozenset({"legacy_reconstruction", "paper_faithful"})


def sarsteer_system_prompt(implementation: SARSteerImplementation) -> str | None:
    """System prompt for BOTH arms under a given implementation.

    The undefended and defended arms must be generated under an identical prompt
    or the contrast is confounded, so every caller resolves it here rather than
    hardcoding one. The paper prepends no system prompt: §3.2 fixes the textual
    input to the FigStep instruction alone.
    """

    if implementation not in SARSTEER_IMPLEMENTATIONS:
        raise ValueError(f"unsupported SARSteer implementation: {implementation!r}")
    return None if implementation == "paper_faithful" else "You are a helpful assistant."


def sarsteer_position_scope(implementation: SARSteerImplementation) -> str:
    """Residual-stream position scope for the steering hook.

    ``paper_faithful`` -> ``"last"``: Eq. 2 adds the vector "at each generated
    token position i", and Algorithm 1 Step 3 updates ``h(Q)``, which §3.3 defines
    as the last-token activation. Under a KV cache the last position of the
    prefill forward is the position whose logits emit the first generated token,
    and every decode forward carries exactly one (generated) position — so
    steering ``hidden[:, -1, :]`` on every forward pass is precisely that rule.
    ``legacy_reconstruction`` -> ``"all"`` reproduces the superseded artifacts.
    """

    if implementation not in SARSTEER_IMPLEMENTATIONS:
        raise ValueError(f"unsupported SARSteer implementation: {implementation!r}")
    return "last" if implementation == "paper_faithful" else "all"


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
        raise ValueError(f"basis {basis.shape} incompatible with vector dim {vector.shape[0]}")
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


def load_sarsteer_metadata(path: Path) -> dict[str, Any]:
    """Load and validate the provenance metadata embedded in a vector bundle."""

    if not path.exists():
        raise FileNotFoundError(f"Missing SARSteer vectors: {path}")
    data = np.load(path, allow_pickle=False)
    if "_meta" not in data:
        return {}
    import json

    raw = data["_meta"]
    try:
        meta = json.loads(str(raw.item()))
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid SARSteer metadata in {path}: {exc}") from exc
    if not isinstance(meta, dict):
        raise ValueError(f"SARSteer metadata in {path} must decode to an object")
    return meta


def resolve_sarsteer_implementation(meta: dict[str, Any]) -> SARSteerImplementation:
    """Resolve bundle mode, treating pre-mode artifacts as legacy."""

    value = meta.get("implementation") or "legacy_reconstruction"
    if value not in SARSTEER_IMPLEMENTATIONS:
        raise ValueError(f"unsupported SARSteer vector implementation metadata: {value!r}")
    return value


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
    """LEGACY (NOT paper-faithful): text-only refusal contrast. Do not use.

    Retained only to reload/reproduce the superseded 2026-07-17 artifacts. Use
    ``extract_paper_refusal_vectors`` instead. Verified against the paper on
    2026-07-17 (Claude+Codex, independently), this function deviates three ways:

    1. Both arms are TEXT-ONLY. The paper's Eq. 4 contrasts Q=(a,t) against
       Q'=(a,t+p) with the SAME harmful audio present in both arms; "text-derived"
       describes the appended refusal PROMPT, not a text-only query.
    2. The default ``mean_all`` pools over every sequence token, but §3.3 defines
       ``h^l(Q)`` as "the activation at the last token position of layer l", and
       Eq. 4's ``mu`` is the mean over calibration ITEMS of those last-token
       vectors — not a mean over positions.
    3. The refusal text is appended AFTER the assistant-generation marker, so the
       arm is not the paper's ``t + p`` textual instruction either.
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
    system_prompt: str | None = "You are a helpful assistant.",
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
        with (
            torch.no_grad(),
            ResidualStreamCapture(model, token_index=token_index, layers=None) as cap,
        ):
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
    system_prompt: str | None = None,
    implementation: SARSteerImplementation = "paper_faithful",
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

    if implementation not in SARSTEER_IMPLEMENTATIONS:
        raise ValueError(f"unsupported SARSteer implementation: {implementation!r}")

    conversation = build_audio_analysis_conversation(
        audio_path, instruction, system_prompt=system_prompt
    )
    device = model_input_device(model)
    inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    prompt_len = inputs.input_ids.shape[1]
    position_scope = sarsteer_position_scope(implementation)
    with MultiLayerAdditiveSteering(
        model, vectors=vectors, alpha=alpha, normalize=False, position_scope=position_scope
    ):
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    generate_ids = generate_ids[:, prompt_len:]
    return processor.batch_decode(
        generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


def extract_paper_refusal_vectors(
    model: Any,
    processor: Any,
    harmful_audio_paths: Sequence[Path | str],
    instruction: str,
    *,
    refusal_text: str = "I cannot assist with that.",
    system_prompt: str | None = None,
) -> dict[int, np.ndarray]:
    """Paper-faithful per-layer refusal vector (Eq. 4 / Algorithm 1 Step 1).

        v_l = mean_i [ h_l(a_i, t + p) - h_l(a_i, t) ]

    Both arms carry the SAME harmful audio ``a_i`` and the same
    assistant-generation marker; only the textual refusal prompt ``p`` is appended
    to the fixed instruction ``t``, exactly as Algorithm 1 Step 1 constructs
    ``Q' = (Q + p)``. Every decoder layer is read at the final prompt token, per
    §3.3's definition of ``h^l(Q)``.
    """
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        model_input_device,
        prepare_qwen2_audio_inputs,
    )

    if not instruction:
        raise ValueError("paper-faithful SARSteer requires a non-empty fixed instruction")
    if not refusal_text:
        raise ValueError("paper-faithful SARSteer requires a non-empty refusal_text")
    device = model_input_device(model)
    base_sum: dict[int, np.ndarray] = {}
    refusal_sum: dict[int, np.ndarray] = {}
    count = 0
    for audio_path in harmful_audio_paths:
        base_conversation = build_audio_analysis_conversation(
            audio_path, instruction, system_prompt=system_prompt
        )
        refusal_conversation = build_audio_analysis_conversation(
            audio_path,
            instruction + refusal_text,
            system_prompt=system_prompt,
        )
        base_inputs = prepare_qwen2_audio_inputs(processor, base_conversation, device=device)
        refusal_inputs = prepare_qwen2_audio_inputs(processor, refusal_conversation, device=device)
        base_states = _capture_pooled_all_layers(model, base_inputs, "last_token")
        refusal_states = _capture_pooled_all_layers(model, refusal_inputs, "last_token")
        if base_states.keys() != refusal_states.keys():
            raise RuntimeError("SARSteer refusal arms captured different decoder layers")
        for ell, state in base_states.items():
            base_sum[ell] = base_sum.get(ell, 0.0) + state
            refusal_sum[ell] = refusal_sum.get(ell, 0.0) + refusal_states[ell]
        count += 1
    if count == 0:
        raise ValueError("no harmful audio provided for refusal-vector extraction")
    return {ell: (refusal_sum[ell] - base_sum[ell]) / count for ell in base_sum}
