"""Residual-stream capture and intervention helpers.

Default readout remains the last input token, but the Audio-RDO gate also sweeps
assistant-start-pre and first-generation-prelogit positions. Callers must resolve
the token index from the same Qwen chat template used for inference.
"""

from collections.abc import Iterable
from typing import Any


def get_decoder_layers(model: Any) -> list[Any]:
    """Locate the decoder layer stack across wrapper variants (Qwen2-Audio wraps the
    Qwen2 LM; attribute paths differ between transformers versions)."""
    for path in ("language_model.model.layers", "model.layers", "layers"):
        node = model
        try:
            for attr in path.split("."):
                node = getattr(node, attr)
        except AttributeError:
            continue
        return list(node)
    raise AttributeError(
        f"could not locate decoder layers on {type(model).__name__}; "
        "inspect the model and extend get_decoder_layers()"
    )


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


class ResidualStreamIntervention:
    """Context manager applying a single residual-stream intervention.

    Modes:
    - ``add``: h[p] <- h[p] + scale * unit(vector)
    - ``ablate``: h[p] <- h[p] - <h[p], unit(vector)> unit(vector)
    - ``set_coordinate``: h[p] is moved only along vector so its signed coordinate
      equals ``target_coordinate``.

    The vector may be a trainable torch tensor; it is intentionally not detached.
    """

    def __init__(
        self,
        model: Any,
        *,
        layer_idx: int,
        token_index: int,
        vector: Any,
        mode: str,
        scale: float = 1.0,
        target_coordinate: float | Any | None = None,
        eps: float = 1e-12,
    ):
        if mode not in {"add", "ablate", "set_coordinate"}:
            raise ValueError(f"unsupported intervention mode {mode!r}")
        self._layer = get_decoder_layers(model)[layer_idx]
        self._layer_idx = layer_idx
        self._token_index = token_index
        self._vector = vector
        self._mode = mode
        self._scale = scale
        self._target_coordinate = target_coordinate
        self._eps = eps
        self._handle: Any | None = None

    def _hook(self, module: Any, inputs: Any, output: Any) -> Any:
        import torch

        hidden = output[0] if isinstance(output, tuple) else output
        token_index = self._token_index
        if token_index < 0:
            token_index = hidden.shape[1] + token_index
        if token_index < 0 or token_index >= hidden.shape[1]:
            return output

        vector = self._vector.to(device=hidden.device, dtype=hidden.dtype)
        vector = vector / torch.clamp(torch.linalg.vector_norm(vector), min=self._eps)

        edited = hidden.clone()
        current = edited[:, token_index, :]

        if self._mode == "add":
            replacement = current + self._scale * vector
        elif self._mode == "ablate":
            coord = current @ vector
            replacement = current - coord.unsqueeze(-1) * vector
        else:
            if self._target_coordinate is None:
                raise ValueError("target_coordinate is required for set_coordinate")
            target = self._target_coordinate
            if not torch.is_tensor(target):
                target = torch.tensor(target, device=hidden.device, dtype=hidden.dtype)
            else:
                target = target.to(device=hidden.device, dtype=hidden.dtype)
            target = target.reshape(-1)
            if target.numel() == 1:
                target = target.expand(current.shape[0])
            if target.shape[0] != current.shape[0]:
                raise ValueError(
                    "target_coordinate must be scalar or match batch size "
                    f"{current.shape[0]}, got {target.shape[0]}"
                )
            coord = current @ vector
            replacement = current + (target - coord).unsqueeze(-1) * vector

        edited[:, token_index, :] = replacement
        return _replace_hidden_output(output, edited)

    def __enter__(self) -> "ResidualStreamIntervention":
        self._handle = self._layer.register_forward_hook(self._hook)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self._handle is not None:
            self._handle.remove()
        self._handle = None
