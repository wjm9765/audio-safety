"""Model loading (gpu dependency group).

torch/transformers are imported lazily so the base environment (stats, tests)
never needs them.
"""

from pathlib import Path
from typing import Any

from audio_safety.config.schema import ModelConfig
from audio_safety.models.hf_audio_lm import load_hf_audio_lm
from audio_safety.models.qwen2_audio import load_qwen2_audio

QWEN2_AUDIO_MODEL_ID = "Qwen/Qwen2-Audio-7B-Instruct"


def load_model_and_processor(
    cfg: ModelConfig,
    cache_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load the configured audio LM and its processor.

    Dispatch is by ``cfg.loader``, not by model id, so exp1 configs keep loading
    through the exact Qwen2-Audio path they always did while exp2 adds models.
    Layer count / hidden size must be read from ``model.config`` downstream.
    """
    if cfg.loader == "qwen2_audio":
        if cfg.model_id != QWEN2_AUDIO_MODEL_ID:
            raise ValueError(
                f"loader 'qwen2_audio' is pinned to {QWEN2_AUDIO_MODEL_ID!r}, "
                f"got {cfg.model_id!r}; use loader: hf_auto_multimodal instead"
            )
        return load_qwen2_audio(cfg, cache_dir=cache_dir)
    if cfg.loader == "hf_auto_multimodal":
        return load_hf_audio_lm(cfg, cache_dir=cache_dir)
    raise ValueError(
        f"unknown loader {cfg.loader!r}; expected 'qwen2_audio' or 'hf_auto_multimodal'"
    )
