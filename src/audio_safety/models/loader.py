"""Model loading (gpu dependency group). torch/transformers are imported lazily so
the base environment (stats, tests) never needs them."""

from pathlib import Path
from typing import Any

from audio_safety.config.schema import ModelConfig

_DTYPES = {"bfloat16", "float16", "float32"}


def load_model_and_processor(
    cfg: ModelConfig,
    cache_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load the audio LM and its processor per config.

    Returns (model, processor). Layer count / hidden size must be read from
    model.config downstream (design.md §4.1) — never hardcoded.
    """
    import torch
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    if cfg.dtype not in _DTYPES:
        raise ValueError(f"unsupported dtype {cfg.dtype!r}, expected one of {_DTYPES}")

    kwargs: dict[str, Any] = {
        "torch_dtype": getattr(torch, cfg.dtype),
        "device_map": cfg.device_map,
        "attn_implementation": cfg.attn_implementation,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.model_id, **kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        cfg.model_id, cache_dir=str(cache_dir) if cache_dir else None
    )
    return model, processor
