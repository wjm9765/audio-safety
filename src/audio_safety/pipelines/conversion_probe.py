"""Run 4 Stage B (fast): matched text/audio activation capture.

Captures residual-stream activations for the mechanism adjudication:

- raw hidden at the content position P1 across ``c_h_layers`` (for a cross-fitted
  harmfulness direction r_H),
- raw hidden at the decision position P2 at ``c_r_layer`` (for c_R via frozen r_A
  and the specificity contrast),
- per-layer c_R(l) = <out(l) at P2, r_A> for every decoder layer (block-writer).

Text and audio arms go through the same LLM backbone, chat template, and system
prompt so the only deliberate difference is the input modality (design §7.5).
Only stdlib/numpy are imported at module load; torch enters through the lazy
qwen2_audio helpers, so this file imports on a CPU-only box.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from audio_safety.config.schema import ExperimentConfig
from audio_safety.models.hooks import ResidualStreamCapture
from audio_safety.models.qwen2_audio import (
    build_audio_analysis_conversation,
    build_text_conversation,
    model_input_device,
    prepare_qwen2_audio_inputs,
    prepare_qwen2_text_inputs,
    resolve_audio_position_indices,
    resolve_chat_position_indices,
)
from audio_safety.pipelines.audio_rdo import unit_vector
from audio_safety.utils.io import save_jsonl

DEFAULT_TEXT_FRAMING = "Please answer the following question:\n\n{reference_text}"


def _text_framing(cfg: ExperimentConfig) -> str:
    if cfg.conversion_gap is not None:
        return cfg.conversion_gap.text_arm.framing_template
    return DEFAULT_TEXT_FRAMING


def _conversation_and_inputs(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    cfg: ExperimentConfig,
    data_dir: Path,
):
    """Build the modality-appropriate conversation, inputs, and P1/P2 indices."""
    modality = str(row["modality"])
    device = model_input_device(model)
    if modality == "audio":
        conversation = build_audio_analysis_conversation(
            data_dir / str(row["path"]),
            cfg.dataset.target_generation.instruction,
        )
        positions = resolve_audio_position_indices(processor, conversation)
        inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    elif modality == "text":
        prompt = _text_framing(cfg).format(reference_text=str(row.get("reference_text") or ""))
        conversation = build_text_conversation(prompt)
        positions = resolve_chat_position_indices(processor, conversation)
        inputs = prepare_qwen2_text_inputs(processor, conversation, device=device)
    else:
        raise ValueError(f"conversion probe expects modality text|audio, got {modality!r}")
    return inputs, positions


def _capture_at(model: Any, inputs: Any, *, token_index: int, layers) -> dict[int, np.ndarray]:
    # Inference-only capture: no_grad avoids building the autograd graph, which
    # otherwise ~doubles peak activation memory per forward and OOMs on longer
    # (e.g. jailbreak-wrapped audio) sequences. Captured hidden states are identical.
    import torch

    with ResidualStreamCapture(model, token_index=token_index, layers=layers) as cap:
        with torch.no_grad():
            model(**inputs)
    return {layer: tensor.numpy() for layer, tensor in cap.states().items()}


def capture_row(
    model: Any,
    processor: Any,
    row: dict[str, Any],
    cfg: ExperimentConfig,
    data_dir: Path,
    r_a: np.ndarray,
    probe_cfg,
) -> dict[str, np.ndarray]:
    """Capture P1 (c_h layers) + P2 (c_r layer raw and per-layer r_A projection)."""
    inputs, positions = _conversation_and_inputs(model, processor, row, cfg, data_dir)
    p1 = positions[probe_cfg.c_h_position]
    p2 = positions[probe_cfg.c_r_position]

    ch = _capture_at(model, inputs, token_index=p1, layers=list(probe_cfg.c_h_layers))
    ch_stack = np.stack([ch[layer] for layer in probe_cfg.c_h_layers]).astype(np.float32)

    p2_all = _capture_at(model, inputs, token_index=p2, layers=None)
    axis_u = unit_vector(r_a)
    layers_sorted = sorted(p2_all)
    cr_by_layer = np.array(
        [float(p2_all[layer] @ axis_u) for layer in layers_sorted], dtype=np.float32
    )
    cr_hidden = p2_all[probe_cfg.c_r_layer].astype(np.float32)
    return {"ch_stack": ch_stack, "cr_hidden": cr_hidden, "cr_by_layer": cr_by_layer}


def extract_conversion_activations(
    model: Any,
    processor: Any,
    rows: Sequence[dict[str, Any]],
    cfg: ExperimentConfig,
    data_dir: Path,
    run_dir: Path,
    r_a: np.ndarray,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Capture all rows and persist arrays + aligned metadata."""
    probe_cfg = cfg.conversion_probe
    if probe_cfg is None:
        raise ValueError("conversion_probe config is required for Stage B extraction")

    import torch

    ch_stacks, cr_hiddens, cr_layers, metadata = [], [], [], []
    for idx, row in enumerate(tqdm(rows, desc="conversion capture", unit="row")):
        captured = capture_row(model, processor, row, cfg, data_dir, r_a, probe_cfg)
        ch_stacks.append(captured["ch_stack"])
        cr_hiddens.append(captured["cr_hidden"])
        cr_layers.append(captured["cr_by_layer"])
        if torch.cuda.is_available() and idx % 20 == 0:
            torch.cuda.empty_cache()
        metadata.append(
            {
                "activation_index": idx,
                "item_id": row["item_id"],
                "modality": row["modality"],
                "safety_label": row["safety_label"],
                "style": row.get("style"),
                "behavior_label": row.get("behavior_label"),
            }
        )

    arrays = {
        "ch_stack": np.stack(ch_stacks),  # (N, n_c_h_layers, d)
        "cr_hidden": np.stack(cr_hiddens),  # (N, d)
        "cr_by_layer": np.stack(cr_layers),  # (N, n_decoder_layers)
        "c_h_layers": np.array(list(probe_cfg.c_h_layers)),
    }
    path = run_dir / probe_cfg.activations_file
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    save_jsonl(metadata, run_dir / probe_cfg.metadata_file)
    return arrays, metadata
