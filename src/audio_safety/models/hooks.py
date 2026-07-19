"""Residual-stream capture and intervention helpers.

Default readout remains the last input token, but the Audio-RDO gate also sweeps
assistant-start-pre and first-generation-prelogit positions. Callers must resolve
the token index from the same Qwen chat template used for inference.
"""

from collections.abc import Iterable
from typing import Any

DECODER_LAYER_PATHS = (
    "model.language_model.layers",
    "language_model.layers",
    "language_model.model.layers",
    "model.layers",
    "layers",
)

AUDIO_TOWER_PATHS = (
    "model.audio_tower",
    "audio_tower",
)

MULTIMODAL_PROJECTOR_PATHS = (
    "model.multi_modal_projector",
    "multi_modal_projector",
)


def _resolve_module(model: Any, paths: tuple[str, ...], description: str) -> Any:
    for path in paths:
        node = model
        try:
            for attr in path.split("."):
                node = getattr(node, attr)
        except AttributeError:
            continue
        return node
    tried = ", ".join(paths)
    raise AttributeError(
        f"could not locate {description} on {type(model).__name__}; tried: {tried}"
    )


def get_decoder_layers(model: Any) -> list[Any]:
    """Locate the decoder layer stack across wrapper variants.

    Qwen2-Audio wrappers have changed across transformers versions. Current HF
    Qwen2AudioForConditionalGeneration exposes the LLM stack at
    ``model.language_model.layers``.
    """
    for path in DECODER_LAYER_PATHS:
        node = model
        try:
            for attr in path.split("."):
                node = getattr(node, attr)
        except AttributeError:
            continue
        layers = list(node)
        if layers:
            return layers
    tried = ", ".join(DECODER_LAYER_PATHS)
    raise AttributeError(
        f"could not locate decoder layers on {type(model).__name__}; tried: {tried}"
    )


def get_audio_tower(model: Any) -> Any:
    """Locate the Qwen2-Audio encoder module across wrapper variants."""
    return _resolve_module(model, AUDIO_TOWER_PATHS, "audio tower")


def get_audio_encoder_layers(model: Any) -> list[Any]:
    """Return every transformer block in the Qwen2-Audio encoder."""
    tower = get_audio_tower(model)
    layers = list(getattr(tower, "layers", ()))
    if not layers:
        raise AttributeError(f"audio tower {type(tower).__name__} exposes no encoder layers")
    return layers


def get_multimodal_projector(model: Any) -> Any:
    """Locate the audio-to-language projector across wrapper variants."""
    return _resolve_module(model, MULTIMODAL_PROJECTOR_PATHS, "multimodal projector")


def _select_layers(all_layers: list[Any], layers: Iterable[int] | None) -> list[tuple[int, Any]]:
    if layers is None:
        return list(enumerate(all_layers))
    selected = []
    for idx in layers:
        if idx < 0 or idx >= len(all_layers):
            raise IndexError(f"layer index {idx} out of range for {len(all_layers)} decoder layers")
        selected.append((idx, all_layers[idx]))
    return selected


def _replace_hidden_output(output: Any, hidden: Any) -> Any:
    if isinstance(output, tuple):
        return (hidden, *output[1:])
    return hidden


class ResidualStreamCapture:
    """Context manager capturing selected decoder layer outputs at one token.

    Usage:
        with ResidualStreamCapture(model, token_index=-1, layers=[12, 16]) as cap:
            model(**inputs)
        acts = cap.states()  # {layer_idx: float32 cpu tensor of shape (d,)}

    Assumes batch size 1 for activation extraction. Batched generation can still
    use ResidualStreamIntervention below.
    """

    def __init__(
        self,
        model: Any,
        *,
        token_index: int = -1,
        layers: Iterable[int] | None = None,
    ):
        self._all_layers = get_decoder_layers(model)
        self._layers = _select_layers(self._all_layers, layers)
        self._token_index = token_index
        self._handles: list[Any] = []
        self._acts: dict[int, Any] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[0] != 1:
                raise ValueError("ResidualStreamCapture requires batch size 1")
            self._acts[layer_idx] = hidden[0, self._token_index, :].detach().float().cpu()

        return hook

    def __enter__(self) -> "ResidualStreamCapture":
        self._acts.clear()
        self._handles = [
            layer.register_forward_hook(self._make_hook(i)) for i, layer in self._layers
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def num_layers(self) -> int:
        return len(self._all_layers)

    def states(self) -> dict[int, Any]:
        if len(self._acts) != len(self._layers):
            raise RuntimeError(
                f"captured {len(self._acts)}/{len(self._layers)} selected layers — "
                "was a forward pass run inside the context?"
            )
        return dict(self._acts)


class AudioPathCapture:
    """Capture pooled Qwen2-Audio states through encoder, projector, and LLM.

    The context is intentionally restricted to a batch of one conversation with
    one audio. It stores only valid-frame/span means and last positions, avoiding
    the very large padded full-sequence activations produced by Qwen2-Audio.

    Decoder layer ``i`` is captured after block ``i``. P1/P2 must be resolved from
    the same processor-expanded prompt used to build ``audio_positions``.
    """

    def __init__(
        self,
        model: Any,
        *,
        audio_positions: Iterable[int],
        p1_index: int,
        p2_index: int,
        encoder_valid_length: int,
        projector_valid_length: int,
        decoder_layers: Iterable[int] | None = None,
    ):
        positions = sorted({int(position) for position in audio_positions})
        if not positions:
            raise ValueError("audio_positions must contain at least one projected audio token")
        if p1_index < 0 or p2_index < 0:
            raise ValueError("P1/P2 must be non-negative absolute prompt indices")
        if encoder_valid_length < 1 or projector_valid_length < 1:
            raise ValueError("encoder/projector valid lengths must be positive")
        if len(positions) != projector_valid_length:
            raise ValueError(
                "projected audio-token count does not match projector valid length: "
                f"{len(positions)} != {projector_valid_length}"
            )

        self._encoder_layers = list(enumerate(get_audio_encoder_layers(model)))
        self._projector = get_multimodal_projector(model)
        self._decoder_layers = _select_layers(get_decoder_layers(model), decoder_layers)
        self._audio_positions = positions
        self._p1_index = int(p1_index)
        self._p2_index = int(p2_index)
        self._encoder_valid_length = int(encoder_valid_length)
        self._projector_valid_length = int(projector_valid_length)

        self._handles: list[Any] = []
        self._encoder: dict[int, dict[str, Any]] = {}
        self._projected: dict[str, Any] = {}
        self._decoder: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _hidden(output: Any) -> Any:
        return output[0] if isinstance(output, tuple) else output

    @staticmethod
    def _valid_pool(hidden: Any, valid_length: int, *, name: str) -> dict[str, Any]:
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise ValueError(
                f"{name} capture expects hidden shape (1, time, dim), got {tuple(hidden.shape)}"
            )
        if valid_length > hidden.shape[1]:
            raise ValueError(
                f"{name} valid length {valid_length} exceeds sequence length {hidden.shape[1]}"
            )
        valid = hidden[0, :valid_length, :].detach().float()
        return {
            "mean": valid.mean(dim=0).cpu(),
            "last": valid[-1].cpu(),
        }

    def _encoder_hook(self, layer_idx: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            if layer_idx in self._encoder:
                raise RuntimeError(
                    "AudioPathCapture encoder hook fired more than once; use one direct forward"
                )
            hidden = self._hidden(output)
            self._encoder[layer_idx] = self._valid_pool(
                hidden,
                self._encoder_valid_length,
                name=f"audio encoder layer {layer_idx}",
            )

        return hook

    def _projector_hook(self, module: Any, inputs: Any, output: Any) -> None:
        if self._projected:
            raise RuntimeError(
                "AudioPathCapture projector hook fired more than once; use one direct forward"
            )
        hidden = self._hidden(output)
        self._projected = self._valid_pool(
            hidden,
            self._projector_valid_length,
            name="audio projector",
        )

    def _decoder_hook(self, layer_idx: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            import torch

            if layer_idx in self._decoder:
                raise RuntimeError(
                    "AudioPathCapture decoder hook fired more than once; use one direct forward"
                )
            hidden = self._hidden(output)
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError(
                    "decoder capture expects hidden shape (1, prompt, dim), "
                    f"got {tuple(hidden.shape)}"
                )
            required_last = max(self._audio_positions[-1], self._p1_index, self._p2_index)
            if required_last >= hidden.shape[1]:
                raise ValueError(
                    f"semantic position {required_last} exceeds decoder length {hidden.shape[1]}"
                )
            row = hidden[0].detach()
            audio_indices = torch.as_tensor(
                self._audio_positions,
                dtype=torch.long,
                device=row.device,
            )
            audio = row.index_select(0, audio_indices).float()
            self._decoder[layer_idx] = {
                "audio_mean": audio.mean(dim=0).cpu(),
                "audio_last": audio[-1].cpu(),
                "p1": row[self._p1_index].float().cpu(),
                "p2": row[self._p2_index].float().cpu(),
            }

        return hook

    def __enter__(self) -> "AudioPathCapture":
        self._encoder.clear()
        self._projected.clear()
        self._decoder.clear()
        self._handles = [
            layer.register_forward_hook(self._encoder_hook(layer_idx))
            for layer_idx, layer in self._encoder_layers
        ]
        self._handles.append(self._projector.register_forward_hook(self._projector_hook))
        self._handles.extend(
            layer.register_forward_hook(self._decoder_hook(layer_idx))
            for layer_idx, layer in self._decoder_layers
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def encoder_layer_indices(self) -> list[int]:
        return [layer_idx for layer_idx, _ in self._encoder_layers]

    @property
    def decoder_layer_indices(self) -> list[int]:
        return [layer_idx for layer_idx, _ in self._decoder_layers]

    def states(self) -> dict[str, Any]:
        """Return stacked CPU tensors after exactly one forward pass."""
        import torch

        if len(self._encoder) != len(self._encoder_layers):
            raise RuntimeError(
                f"captured {len(self._encoder)}/{len(self._encoder_layers)} encoder layers"
            )
        if not self._projected:
            raise RuntimeError("multimodal projector was not captured")
        if len(self._decoder) != len(self._decoder_layers):
            raise RuntimeError(
                f"captured {len(self._decoder)}/{len(self._decoder_layers)} decoder layers"
            )

        encoder_indices = self.encoder_layer_indices
        decoder_indices = self.decoder_layer_indices
        return {
            "encoder_mean": torch.stack(
                [self._encoder[layer]["mean"] for layer in encoder_indices]
            ),
            "encoder_last": torch.stack(
                [self._encoder[layer]["last"] for layer in encoder_indices]
            ),
            "projector_mean": self._projected["mean"],
            "projector_last": self._projected["last"],
            "llm_audio_mean": torch.stack(
                [self._decoder[layer]["audio_mean"] for layer in decoder_indices]
            ),
            "llm_audio_last": torch.stack(
                [self._decoder[layer]["audio_last"] for layer in decoder_indices]
            ),
            "llm_p1": torch.stack([self._decoder[layer]["p1"] for layer in decoder_indices]),
            "llm_p2": torch.stack([self._decoder[layer]["p2"] for layer in decoder_indices]),
        }


class ResidualStreamIntervention:
    """Context manager applying a residual-stream intervention.

    Modes:
    - ``add``: h[p] <- h[p] + scale * unit(vector)
    - ``ablate``: h[p] <- h[p] - <h[p], unit(vector)> unit(vector)
    - ``set_coordinate``: h[p] is moved only along vector so its signed coordinate
      equals ``target_coordinate``.
    - ``patch_state``: h[p] is REPLACED by ``replacement_state`` (a full d_model
      donor hidden state captured from another run). This is interchange /
      activation patching for causal tracing, NOT a directional edit: the donor
      is used verbatim and is never normalized. Single-position, prefill-only.

    Scope:
    - default (``all_positions=False``): intervene only at the single residual
      position ``token_index``. During KV-cached generation this edits only the
      prefill pass, so the perturbation reaches later generated tokens indirectly
      through attention and can wash out.
    - ``all_positions=True``: intervene at every position of every forward pass,
      including each length-1 KV-cached decode step, so the edit persists across
      all generated tokens. ``token_index`` is ignored in this mode. This matches
      the all-token activation-addition / directional-ablation scope used by
      Arditi et al. 2024 and RDO (Wollschlaeger et al. 2025).

    The vector may be a trainable torch tensor; it is intentionally not detached.
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_idx: int,
        token_index: int | None = None,
        vector: Any = None,
        mode: str,
        scale: float = 1.0,
        target_coordinate: float | Any | None = None,
        replacement_state: Any | None = None,
        all_positions: bool = False,
        eps: float = 1e-12,
    ):
        if mode not in {"add", "ablate", "set_coordinate", "patch_state"}:
            raise ValueError(f"unsupported intervention mode {mode!r}")
        if mode == "patch_state":
            # Interchange patching: a donor full-state replacement at one absolute
            # prefill position. A negative token_index would resolve to position 0
            # on every length-1 cached decode step and silently repatch every
            # generated token, so require a non-negative absolute index.
            if replacement_state is None:
                raise ValueError("patch_state mode requires replacement_state")
            if all_positions:
                raise ValueError("patch_state does not support all_positions")
            if token_index is None or token_index < 0:
                raise ValueError("patch_state requires a non-negative absolute token_index")
        else:
            if vector is None:
                raise ValueError(f"{mode!r} mode requires a vector")
            if not all_positions and token_index is None:
                raise ValueError("token_index is required unless all_positions is True")
        self._layer = get_decoder_layers(model)[layer_idx]
        self._layer_idx = layer_idx
        self._token_index = token_index
        self._vector = vector
        self._mode = mode
        self._scale = scale
        self._target_coordinate = target_coordinate
        self._replacement_state = replacement_state
        self._all_positions = all_positions
        self._eps = eps
        self._applied_count = 0
        self._handle: Any | None = None

    def _edit(self, current: Any, vector: Any) -> Any:
        """Apply the mode's operator along the last (d_model) dim of ``current``.

        ``current`` may be ``(batch, d)`` for a single position or ``(batch, T, d)``
        for the all-positions scope; ``current @ vector`` collapses the last dim in
        both cases.
        """
        import torch

        if self._mode == "add":
            return current + self._scale * vector
        if self._mode == "ablate":
            coord = current @ vector
            return current - coord.unsqueeze(-1) * vector

        if self._target_coordinate is None:
            raise ValueError("target_coordinate is required for set_coordinate")
        target = self._target_coordinate
        if not torch.is_tensor(target):
            target = torch.tensor(target, device=current.device, dtype=current.dtype)
        else:
            target = target.to(device=current.device, dtype=current.dtype)
        target = target.reshape(-1)
        coord = current @ vector
        # A per-row target is only meaningful for the (batch, d) single-position
        # case; broadcasting it across an all-positions (batch, T) coord would be
        # ambiguous, so reject it explicitly rather than steer silently wrong.
        if target.numel() != 1 and (current.ndim != 2 or target.shape[0] != current.shape[0]):
            raise ValueError(
                "target_coordinate must be scalar or match batch size "
                f"{current.shape[0]}, got {tuple(target.shape)}"
            )
        return current + (target - coord).unsqueeze(-1) * vector

    def _patch_state_hook(self, output: Any, hidden: Any) -> Any:
        import torch

        # Non-negative absolute prefill index. During KV-cached decode steps the
        # forward pass sees a length-1 slice, so this index is out of range and the
        # patch is skipped: the donor state is injected exactly once, at prefill.
        token_index = self._token_index
        if token_index >= hidden.shape[1]:
            return output
        if hidden.shape[0] != 1:
            raise ValueError("patch_state requires batch size 1")
        donor = self._replacement_state
        if not torch.is_tensor(donor):
            donor = torch.as_tensor(donor)
        donor = donor.reshape(-1).to(device=hidden.device, dtype=hidden.dtype)
        if donor.shape[0] != hidden.shape[-1]:
            raise ValueError(
                f"replacement_state dim {donor.shape[0]} != hidden dim {hidden.shape[-1]}"
            )
        current = hidden[:, token_index, :]
        token_mask = torch.nn.functional.one_hot(
            torch.tensor(token_index, device=hidden.device),
            num_classes=hidden.shape[1],
        ).to(dtype=hidden.dtype)
        edited = hidden + token_mask.view(1, -1, 1) * (donor.unsqueeze(0) - current).unsqueeze(1)
        self._applied_count += 1
        return _replace_hidden_output(output, edited)

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output

        if self._mode == "patch_state":
            return self._patch_state_hook(output, hidden)

        if torch.is_tensor(self._vector):
            vector = self._vector.to(device=hidden.device, dtype=hidden.dtype)
        else:
            vector = torch.as_tensor(self._vector, device=hidden.device, dtype=hidden.dtype)
        vector = vector / torch.clamp(torch.linalg.vector_norm(vector), min=self._eps)

        if self._all_positions:
            edited = self._edit(hidden, vector)
            return _replace_hidden_output(output, edited)

        token_index = self._token_index
        if token_index < 0:
            token_index = hidden.shape[1] + token_index
        if token_index < 0 or token_index >= hidden.shape[1]:
            return output

        current = hidden[:, token_index, :]
        replacement = self._edit(current, vector)

        token_mask = torch.nn.functional.one_hot(
            torch.tensor(token_index, device=hidden.device),
            num_classes=hidden.shape[1],
        ).to(dtype=hidden.dtype)
        edited = hidden + token_mask.view(1, -1, 1) * (replacement - current).unsqueeze(1)
        return _replace_hidden_output(output, edited)

    def __enter__(self) -> "ResidualStreamIntervention":
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None

    @property
    def applied_count(self) -> int:
        """How many forward passes actually applied the intervention.

        For ``patch_state`` this must equal 1 after a full generation (patched once
        at prefill, skipped on every cached decode step). Callers should assert it.
        """
        return self._applied_count


class ProjectedTransportIntervention:
    """Pair-specific projected transport at a set of prefill positions (one layer).

    At decoder layer ``layer_idx`` and a fixed set of absolute prefill ``positions``,
    replace ONLY the ``basis``-subspace component of each position's residual with the
    donor's, scaled by the dose::

        h[j] <- h[j] + scale * P_U( donor[j] - h[j] )      for j in positions

    where ``P_U(x) = (x @ basis.T) @ basis`` projects onto the row space of the
    orthonormal ``basis`` (shape ``(k, d)``), and ``donor`` is a ``(len(positions), d)``
    matrix of donor hidden states aligned row-for-row with **sorted** ``positions``.

    This is interchange / activation patching RESTRICTED TO A SUBSPACE — the Run 10
    channel-invariance operator — not a directional add. Properties:

    - **Reciprocity.** With ``scale=1`` and the same ``basis``, restoration
      (``donor=clean`` while the forward pass runs on the *attack* audio) and corruption
      (``donor=attack`` while the pass runs on the *clean* audio) are exact reciprocals
      within the subspace: each swaps in the donor's subspace coordinate.
    - **Prefill-only, batch-1.** On every length-1 KV-cached decode step the positions
      are out of range, so the transport applies exactly once. Assert
      ``applied_count == 1`` after a full forward/generation.
    - **Arm A** passes all audio-token positions (propagation support); **Arm B** passes
      the single readout ``t_AB`` position (expression support). Never both together.

    Projection is computed in float32 for numerical stability, then cast to the hidden
    dtype. ``donor``/``basis`` may be numpy arrays or torch tensors.

    **One-shot.** The transport is applied on the FIRST forward the hook sees (the full
    prefill) and skipped on every later forward (KV-cached decode steps). The driver MUST
    build a fresh object per (direction, arm, pair, dose), use it as a context manager so
    the handle is removed on exit, and assert ``applied_count == 1`` afterwards. Assumes
    the standard KV-cached path (no chunked/speculative prefill).
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_idx: int,
        positions: Iterable[int],
        donor: Any,
        basis: Any,
        scale: float = 1.0,
    ):
        positions = [int(p) for p in positions]
        if not positions:
            raise ValueError("positions must be non-empty")
        if len(set(positions)) != len(positions):
            raise ValueError("positions must be unique")
        if min(positions) < 0:
            raise ValueError("positions must be non-negative absolute prefill indices")
        ordered = sorted(positions)
        if ordered != positions:
            raise ValueError("positions must be provided sorted to stay aligned with donor rows")
        self._layer = get_decoder_layers(model)[layer_idx]
        self._layer_idx = layer_idx
        self._positions = positions
        self._max_pos = positions[-1]
        self._donor = donor
        self._basis = basis
        self._scale = float(scale)
        self._applied_count = 0
        self._handle: Any | None = None

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[0] != 1:
            raise ValueError("ProjectedTransportIntervention requires batch size 1")
        # One-shot: apply on the first forward only, skip every later (decode) forward.
        # Gating on _applied_count (not sequence length) is required — a support that
        # includes position 0 would otherwise be re-patched on every length-1 decode step.
        if self._applied_count:
            return output
        # Fail fast: an out-of-range support on the (first) prefill would otherwise be a
        # silent no-op read as a null causal effect — e.g. a misaligned attack sequence.
        if self._max_pos >= hidden.shape[1]:
            raise ValueError(
                f"intervention positions (max {self._max_pos}) exceed the prefill "
                f"sequence length {hidden.shape[1]}"
            )

        idx = torch.as_tensor(self._positions, dtype=torch.long, device=hidden.device)
        donor = torch.as_tensor(self._donor, device=hidden.device, dtype=torch.float32)
        basis = torch.as_tensor(self._basis, device=hidden.device, dtype=torch.float32)
        if donor.ndim != 2 or donor.shape[0] != len(self._positions):
            raise ValueError(
                f"donor must be (n_positions, d); got {tuple(donor.shape)} for "
                f"{len(self._positions)} positions"
            )
        if basis.ndim != 2:
            raise ValueError(f"basis must be (k, d); got {tuple(basis.shape)}")
        if donor.shape[1] != hidden.shape[-1] or basis.shape[1] != hidden.shape[-1]:
            raise ValueError(
                f"donor/basis dim must equal hidden dim {hidden.shape[-1]}; "
                f"got donor {donor.shape[1]}, basis {basis.shape[1]}"
            )
        if not torch.isfinite(donor).all() or not torch.isfinite(basis).all():
            raise ValueError("donor/basis contain non-finite values")
        gram = basis @ basis.transpose(0, 1)
        if not torch.allclose(
            gram, torch.eye(basis.shape[0], device=basis.device, dtype=basis.dtype),
            atol=1e-4, rtol=1e-4,
        ):
            raise ValueError("basis rows must be orthonormal (P_U assumes an orthonormal basis)")

        current = hidden[0].index_select(0, idx).to(torch.float32)  # (m, d)
        delta = donor - current                                     # (m, d)
        proj = (delta @ basis.transpose(0, 1)) @ basis              # (m, d), row-space of basis
        replacement = current + self._scale * proj                  # float32, one cast below
        edited = hidden.clone()
        edited[0, idx, :] = replacement.to(hidden.dtype)
        self._applied_count += 1
        return _replace_hidden_output(output, edited)

    def __enter__(self) -> "ProjectedTransportIntervention":
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None

    @property
    def applied_count(self) -> int:
        """Forward passes that applied the transport. Must be 1 after a full generation."""
        return self._applied_count


class MultiLayerAdditiveSteering:
    """Add a distinct per-layer vector at selected positions on every forward pass.

    This is the operator SARSteer (arXiv:2510.17633) uses as a defense:

        h'_ell = h_ell + alpha * v_ell   at all layers, all generated positions.

    It differs from ``ResidualStreamIntervention`` in three deliberate ways, all
    required for a faithful SARSteer reproduction:

    1. **Per-layer vectors.** SARSteer steers every decoder block with its own
       ``v_ell``, not one shared direction. ``vectors`` maps ``layer_idx -> vector``.
    2. **No implicit unit-normalization.** ``ResidualStreamIntervention`` divides the
       vector by its norm, which turns ``alpha`` into a fixed absolute step size.
       SARSteer applies ``alpha`` to the *raw* orthogonal component ``v_perp`` whose
       magnitude carries the refusal strength, so normalizing would silently change
       the method. ``normalize`` defaults to ``False`` for that reason; set it True
       only for controls.
    3. **Explicit position scope.** ``position_scope="last"`` matches the paper by
       editing only ``hidden[:, -1, :]``: Eq. 2 steers "each generated token
       position", and Algorithm 1 Step 3 updates ``h(Q)``, defined in §3.3 as the
       last-token activation. ``position_scope="all"`` preserves the legacy
       whole-hidden-tensor behavior, which additionally rewrites every audio and
       prompt position during prefill — and thus their cached keys/values — an
       intervention absent from the paper that over-injects by roughly the prompt
       length. Both modes run on prefill and every length-1 KV-cached decode call,
       so generated-token steering stays active throughout decoding.

    The vectors may be trainable torch tensors; they are not detached.
    """

    def __init__(
        self,
        model: Any,
        *,
        vectors: dict[int, Any],
        alpha: float = 0.1,
        normalize: bool = False,
        position_scope: str = "all",
        eps: float = 1e-12,
    ):
        if not vectors:
            raise ValueError("MultiLayerAdditiveSteering requires at least one layer vector")
        if position_scope not in {"all", "last"}:
            raise ValueError("position_scope must be 'all' or 'last'")
        all_layers = get_decoder_layers(model)
        n_layers = len(all_layers)
        self._layer_vectors: list[tuple[int, Any, Any]] = []
        for layer_idx, vector in vectors.items():
            if layer_idx < 0 or layer_idx >= n_layers:
                raise IndexError(
                    f"layer index {layer_idx} out of range for {n_layers} decoder layers"
                )
            if vector is None:
                raise ValueError(f"vector for layer {layer_idx} is None")
            self._layer_vectors.append((layer_idx, all_layers[layer_idx], vector))
        self._alpha = alpha
        self._normalize = normalize
        self._eps = eps
        self._position_scope = position_scope
        self._handles: list[Any] = []
        self._applied_counts: dict[int, int] = {}

    def _make_hook(self, layer_idx: int, raw_vector: Any):
        def hook(module: Any, inputs: Any, output: Any) -> Any:
            import torch

            hidden = output[0] if isinstance(output, tuple) else output
            if torch.is_tensor(raw_vector):
                vector = raw_vector.to(device=hidden.device, dtype=hidden.dtype)
            else:
                vector = torch.as_tensor(raw_vector, device=hidden.device, dtype=hidden.dtype)
            if vector.ndim != 1 or vector.shape[0] != hidden.shape[-1]:
                raise ValueError(
                    f"layer {layer_idx} steering vector dim {tuple(vector.shape)} "
                    f"!= hidden dim {hidden.shape[-1]}"
                )
            if self._normalize:
                vector = vector / torch.clamp(torch.linalg.vector_norm(vector), min=self._eps)
            if self._position_scope == "last":
                edited = hidden.clone()
                edited[:, -1, :] = hidden[:, -1, :] + self._alpha * vector
            else:
                edited = hidden + self._alpha * vector
            self._applied_counts[layer_idx] = self._applied_counts.get(layer_idx, 0) + 1
            return _replace_hidden_output(output, edited)

        return hook

    def __enter__(self) -> "MultiLayerAdditiveSteering":
        self._applied_counts = {}
        self._handles = [
            layer.register_forward_hook(self._make_hook(layer_idx, vector))
            for layer_idx, layer, vector in self._layer_vectors
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def applied_counts(self) -> dict[int, int]:
        """Forward-pass application count per layer (prefill + each decode step)."""
        return dict(self._applied_counts)
