"""Model loading (gpu dependency group).

torch/transformers are imported lazily so the base environment (stats, tests)
never needs them.
"""

from pathlib import Path
from typing import Any

from audio_safety.config.schema import ModelConfig
from audio_safety.models.qwen2_audio import load_qwen2_audio


def load_model_and_processor(
    cfg: ModelConfig,
    cache_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load the configured audio LM and its processor.

    Currently exp1 is pinned to Qwen2-Audio, using the official transformers path.
    Layer count / hidden size must be read from model.config downstream.
    """
    if cfg.model_id != "Qwen/Qwen2-Audio-7B-Instruct":
        raise ValueError(f"unsupported model_id for exp1 Audio-RDO gate: {cfg.model_id!r}")
    return load_qwen2_audio(cfg, cache_dir=cache_dir)
