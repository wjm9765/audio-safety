"""Generic HuggingFace audio-LM loading for the exp2 model pool.

exp1 was pinned to Qwen2-Audio (see :mod:`audio_safety.models.qwen2_audio`).
exp2 needs additional audio LMs whose refusal behaviour actually moves under
content-preserving acoustic manipulation; see
``docs/experiments/exp2_acoustic_refusal_transport/direction.md`` for why the
primary model changed.

Scope of this module — deliberately narrow:

* It loads and caches models via the same ``from_pretrained`` path that later
  inference will use, so the cached revision is exactly what runs.
* It does **not** implement per-model conversation building, processor input
  packing, or readout-position resolution. Those differ per architecture and are
  raised as :class:`NotImplementedError` rather than silently reusing the
  Qwen2-Audio chat template, which would produce wrong prompts and wrong token
  indices without failing loudly.

Nothing here has been executed against real weights yet. The module-path
constants in :mod:`audio_safety.models.hooks` for the new architectures are
best-effort and must be confirmed on GPU before any activation work.
"""

from pathlib import Path
from typing import Any

from audio_safety.config.schema import ModelConfig

_DTYPES = {"bfloat16", "float16", "float32"}

# Loader families this module serves. "qwen2_audio" is handled elsewhere.
SUPPORTED_LOADERS = frozenset({"hf_auto_multimodal"})

# Documented model pool for exp2. Values are the exact HF repo ids that the
# configs under ``configs/models/`` point at, kept here so a typo in a config is
# visible next to the reason the model is in the pool at all.
EXP2_MODEL_POOL: dict[str, str] = {
    # Primary. Aligned at baseline (F1 neutral unsafe rate 3.27%) yet the largest
    # null-calibrated acoustic effect of the 10 models in F1 Table 1.
    "openbmb/MiniCPM-o-2_6": "primary",
    # Secondary. Discrete-semantic + continuous-acoustic dual stream gives a
    # privileged site for separating acoustic from lexical transport.
    "moonshotai/Kimi-Audio-7B-Instruct": "secondary",
    # Third. Llama-3.1-8B backbone = mature interpretability substrate;
    # attribution patching already demonstrated on it (arXiv 2606.18924).
    "fixie-ai/ultravox-v0_4_1-llama-3_1-8b": "interpretability_substrate",
    # Retained as the RESISTANT control, not discarded: three independent sources
    # report it is inert to single natural acoustic edits.
    "Qwen/Qwen2-Audio-7B-Instruct": "resistant_control",
}


def _torch_dtype(name: str) -> Any:
    import torch

    if name not in _DTYPES:
        raise ValueError(f"unsupported dtype {name!r}, expected one of {_DTYPES}")
    return getattr(torch, name)


def _from_pretrained_kwargs(cfg: ModelConfig, cache_dir: Path | None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if cfg.revision is not None:
        kwargs["revision"] = cfg.revision
    if cfg.trust_remote_code:
        kwargs["trust_remote_code"] = True
    if cache_dir is not None:
        kwargs["cache_dir"] = str(cache_dir)
    return kwargs


def load_hf_audio_lm(
    cfg: ModelConfig,
    cache_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load an audio LM and its processor through the transformers Auto* path.

    Uses ``AutoModel`` because the exp2 pool is a mix of wrapper classes
    (MiniCPM-o and Kimi-Audio register their own via ``trust_remote_code``;
    Ultravox exposes a standard multimodal wrapper). Falls back to
    ``AutoTokenizer`` when a repo ships no processor config.
    """
    from transformers import AutoModel, AutoProcessor, AutoTokenizer

    shared = _from_pretrained_kwargs(cfg, cache_dir)
    model = AutoModel.from_pretrained(
        cfg.model_id,
        torch_dtype=_torch_dtype(cfg.dtype),
        device_map=cfg.device_map,
        attn_implementation=cfg.attn_implementation,
        **shared,
    )
    model.eval()

    try:
        processor = AutoProcessor.from_pretrained(cfg.model_id, **shared)
    except (OSError, ValueError, KeyError):
        # Some audio-LM repos ship only a tokenizer plus their own feature
        # extractor inside the remote code. Surface the tokenizer so callers can
        # still resolve token indices; audio packing stays the caller's problem.
        processor = AutoTokenizer.from_pretrained(cfg.model_id, **shared)
    return model, processor


def download_hf_audio_lm(cfg: ModelConfig, cache_dir: Path | None = None) -> None:
    """Materialize weights and processor files into the configured cache.

    Intentionally goes through the same ``from_pretrained`` call as inference so
    the cached revision is exactly what later runs load.
    """
    model, processor = load_hf_audio_lm(cfg, cache_dir=cache_dir)
    del model, processor


def build_audio_analysis_conversation(*_args: Any, **_kwargs: Any) -> Any:
    """Not implemented — per-architecture chat templates differ.

    Reusing the Qwen2-Audio conversation format here would silently produce the
    wrong prompt (and therefore wrong readout token indices) on MiniCPM-o,
    Kimi-Audio and Ultravox. Implement one adapter per model, verified against
    its official model card, before any exp2 capture run.
    """
    raise NotImplementedError(
        "exp2 conversation adapters are not implemented yet; implement per-model "
        "against the official model card and verify token indices on GPU"
    )


def resolve_audio_position_indices(*_args: Any, **_kwargs: Any) -> Any:
    """Not implemented — see :func:`build_audio_analysis_conversation`."""
    raise NotImplementedError(
        "exp2 readout-position resolution is not implemented yet; it must be "
        "derived from each model's own processor-expanded prompt length"
    )
