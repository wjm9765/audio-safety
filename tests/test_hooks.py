from types import SimpleNamespace

import numpy as np
import pytest
import torch

from audio_safety.models.hooks import ResidualStreamIntervention, get_decoder_layers


def test_get_decoder_layers_current_qwen2_audio_path():
    layers = [object(), object(), object()]
    model = SimpleNamespace(model=SimpleNamespace(language_model=SimpleNamespace(layers=layers)))

    assert get_decoder_layers(model) == layers


def test_get_decoder_layers_legacy_language_model_model_path():
    layers = [object()]
    model = SimpleNamespace(language_model=SimpleNamespace(model=SimpleNamespace(layers=layers)))

    assert get_decoder_layers(model) == layers


def test_get_decoder_layers_reports_tried_paths():
    with pytest.raises(AttributeError, match="model.language_model.layers"):
        get_decoder_layers(SimpleNamespace())


class _IdentityLayer(torch.nn.Module):
    def forward(self, hidden):
        return hidden


class _TinyLayerModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([_IdentityLayer()])

    def forward(self, hidden):
        return self.layers[0](hidden)


def test_residual_intervention_accepts_numpy_vector_and_backpropagates():
    model = _TinyLayerModel()
    hidden = torch.zeros(1, 3, 2, requires_grad=True)
    vector = np.array([1.0, 0.0], dtype=np.float32)

    with ResidualStreamIntervention(
        model,
        layer_idx=0,
        token_index=1,
        vector=vector,
        mode="add",
        scale=2.0,
    ):
        edited = model(hidden)

    assert torch.allclose(edited[0, 1], torch.tensor([2.0, 0.0]))
    edited.sum().backward()
    assert hidden.grad is not None
