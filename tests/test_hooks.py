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
    # Single-position scope leaves the other positions untouched.
    assert torch.allclose(edited[0, 0], torch.tensor([0.0, 0.0]))
    assert torch.allclose(edited[0, 2], torch.tensor([0.0, 0.0]))
    edited.sum().backward()
    assert hidden.grad is not None


def test_all_positions_add_edits_every_position():
    model = _TinyLayerModel()
    hidden = torch.zeros(1, 3, 2)
    vector = np.array([3.0, 0.0], dtype=np.float32)  # non-unit; hook normalizes

    with ResidualStreamIntervention(
        model,
        layer_idx=0,
        vector=vector,
        mode="add",
        scale=2.0,
        all_positions=True,
    ):
        edited = model(hidden)

    for pos in range(3):
        assert torch.allclose(edited[0, pos], torch.tensor([2.0, 0.0]))


def test_all_positions_edits_length_one_decode_step():
    # Simulates a KV-cached decode step where the forward pass sees a single new
    # token. The legacy single-position hook is a no-op here; all_positions must
    # still steer it so the edit persists across generated tokens.
    model = _TinyLayerModel()
    decode_step = torch.zeros(1, 1, 2)
    vector = np.array([1.0, 0.0], dtype=np.float32)

    with ResidualStreamIntervention(
        model,
        layer_idx=0,
        vector=vector,
        mode="add",
        scale=2.0,
        all_positions=True,
    ):
        edited = model(decode_step)

    assert torch.allclose(edited[0, 0], torch.tensor([2.0, 0.0]))


def test_all_positions_ablate_removes_coordinate_everywhere_and_backpropagates():
    model = _TinyLayerModel()
    hidden = torch.tensor([[[1.0, 5.0], [2.0, 6.0]]], requires_grad=True)
    vector = np.array([1.0, 0.0], dtype=np.float32)

    with ResidualStreamIntervention(
        model,
        layer_idx=0,
        vector=vector,
        mode="ablate",
        all_positions=True,
    ):
        edited = model(hidden)

    # The along-vector coordinate is removed at every position; the orthogonal
    # component is preserved.
    assert torch.allclose(edited[0, 0], torch.tensor([0.0, 5.0]))
    assert torch.allclose(edited[0, 1], torch.tensor([0.0, 6.0]))
    edited.sum().backward()
    assert hidden.grad is not None


def test_all_positions_set_coordinate_clamps_every_position_to_scalar():
    # Documents why restoration (H4) must NOT run with all_positions: a single
    # neutral-occupancy scalar target clamps the axis coordinate at every position,
    # not just the readout position, which is a different, stronger intervention
    # than the preregistered per-sample single-position restore. rdo_gate pins
    # set_coordinate to single-position for this reason.
    model = _TinyLayerModel()
    hidden = torch.tensor([[[5.0, 1.0], [-2.0, 3.0], [9.0, 4.0]]])
    vector = np.array([1.0, 0.0], dtype=np.float32)

    with ResidualStreamIntervention(
        model,
        layer_idx=0,
        vector=vector,
        mode="set_coordinate",
        target_coordinate=0.7,
        all_positions=True,
    ):
        edited = model(hidden)

    # Every position's along-vector coordinate is forced to 0.7 (orthogonal kept).
    assert torch.allclose(edited[0], torch.tensor([[0.7, 1.0], [0.7, 3.0], [0.7, 4.0]]))

    # Single-position scope touches only the readout position, per H4.
    with ResidualStreamIntervention(
        model,
        layer_idx=0,
        token_index=-1,
        vector=vector,
        mode="set_coordinate",
        target_coordinate=0.7,
    ):
        single = model(hidden)
    assert torch.allclose(single[0], torch.tensor([[5.0, 1.0], [-2.0, 3.0], [0.7, 4.0]]))


def test_all_positions_set_coordinate_rejects_per_row_target():
    model = _TinyLayerModel()
    hidden = torch.zeros(1, 3, 2)
    vector = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="scalar or match batch size"), ResidualStreamIntervention(
        model,
        layer_idx=0,
        vector=vector,
        mode="set_coordinate",
        target_coordinate=np.array([1.0, 2.0], dtype=np.float32),
        all_positions=True,
    ):
        model(hidden)


def test_missing_token_index_without_all_positions_raises():
    model = _TinyLayerModel()
    with pytest.raises(ValueError, match="token_index is required"):
        ResidualStreamIntervention(model, layer_idx=0, vector=np.zeros(2), mode="add")
