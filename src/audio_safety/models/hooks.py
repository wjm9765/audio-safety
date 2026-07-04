"""Residual-stream capture at the last input token (design.md §4.1, §3.3).

Readout convention: the hidden state at the LAST input token position (just before
generation), identical across text and audio conditions. Changing this position in
one condition but not the other is the classic silent-correctness bug here.
"""

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


class ResidualStreamCapture:
    """Context manager capturing each decoder layer's output hidden state at the
    last token position of a single forward pass.

    Usage:
        with ResidualStreamCapture(model) as cap:
            model(**inputs)
        acts = cap.states()  # {layer_idx: float32 cpu tensor of shape (d,)}

    Assumes batch size 1 and right-most position = last input token (no padding
    after the sequence). Callers must not use left-padded batches.
    """

    def __init__(self, model: Any):
        self._layers = get_decoder_layers(model)
        self._handles: list[Any] = []
        self._acts: dict[int, Any] = {}

    def _make_hook(self, layer_idx: int):
        def hook(module: Any, inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.shape[0] != 1:
                raise ValueError("ResidualStreamCapture requires batch size 1")
            self._acts[layer_idx] = hidden[0, -1, :].detach().float().cpu()

        return hook

    def __enter__(self) -> "ResidualStreamCapture":
        self._acts.clear()
        self._handles = [
            layer.register_forward_hook(self._make_hook(i)) for i, layer in enumerate(self._layers)
        ]
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    @property
    def num_layers(self) -> int:
        return len(self._layers)

    def states(self) -> dict[int, Any]:
        if len(self._acts) != len(self._layers):
            raise RuntimeError(
                f"captured {len(self._acts)}/{len(self._layers)} layers — "
                "was a forward pass run inside the context?"
            )
        return dict(self._acts)
