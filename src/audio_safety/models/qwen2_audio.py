"""Qwen2-Audio loading and inference helpers.

The implementation follows the official Qwen/Hugging Face model-card path:
``AutoProcessor`` plus ``Qwen2AudioForConditionalGeneration`` with ChatML via
``processor.apply_chat_template``. Imports stay lazy so CPU-only tests can import
the package without torch/transformers installed.
"""

from collections.abc import Mapping, Sequence
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.request import urlopen

from audio_safety.config.schema import ModelConfig
from audio_safety.models.hooks import ResidualStreamIntervention

_DTYPES = {"bfloat16", "float16", "float32"}
PositionName = str


def _torch_dtype(name: str) -> Any:
    import torch

    if name not in _DTYPES:
        raise ValueError(f"unsupported dtype {name!r}, expected one of {_DTYPES}")
    return getattr(torch, name)


def load_qwen2_audio(
    cfg: ModelConfig,
    cache_dir: Path | None = None,
) -> tuple[Any, Any]:
    """Load Qwen2-Audio-7B-Instruct and processor from local cache/HF hub."""
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

    model_kwargs: dict[str, Any] = {
        "torch_dtype": _torch_dtype(cfg.dtype),
        "device_map": cfg.device_map,
        "attn_implementation": cfg.attn_implementation,
    }
    if cache_dir is not None:
        model_kwargs["cache_dir"] = str(cache_dir)

    model = Qwen2AudioForConditionalGeneration.from_pretrained(cfg.model_id, **model_kwargs)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        cfg.model_id,
        cache_dir=str(cache_dir) if cache_dir is not None else None,
    )
    return model, processor


def download_qwen2_audio(
    cfg: ModelConfig,
    cache_dir: Path | None = None,
) -> None:
    """Materialize model and processor files in the configured cache.

    This intentionally uses the same ``from_pretrained`` path as inference so the
    downloaded revision is exactly what later runs load.
    """
    model, processor = load_qwen2_audio(cfg, cache_dir=cache_dir)
    del model, processor


def build_audio_analysis_conversation(
    audio_path: Path | str,
    instruction: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
) -> list[dict[str, Any]]:
    """Build the Qwen2-Audio audio-analysis conversation format."""
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio_url": str(audio_path)},
                {"type": "text", "text": instruction},
            ],
        },
    ]


def build_text_conversation(
    text: str,
    *,
    system_prompt: str = "You are a helpful assistant.",
) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]


def _is_url(value: str) -> bool:
    scheme = urlparse(value).scheme
    return scheme in {"http", "https"}


def _audio_refs(conversation: Sequence[Mapping[str, Any]]) -> list[str]:
    refs: list[str] = []
    for message in conversation:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if item.get("type") == "audio":
                ref = item.get("audio_url") or item.get("audio") or item.get("path")
                if ref is None:
                    raise KeyError("audio content item requires audio_url, audio, or path")
                refs.append(str(ref))
    return refs


def load_audio_array(ref: Path | str, sampling_rate: int) -> Any:
    """Load local or URL audio as a mono float array at the processor sample rate."""
    import librosa

    value = str(ref)
    if _is_url(value):
        with urlopen(value) as response:
            return librosa.load(BytesIO(response.read()), sr=sampling_rate)[0]
    return librosa.load(Path(value), sr=sampling_rate)[0]


def model_input_device(model: Any) -> Any:
    """Best-effort first device for model inputs under device_map='auto'."""
    import torch

    device_map = getattr(model, "hf_device_map", None)
    if isinstance(device_map, dict):
        for device in device_map.values():
            if isinstance(device, int):
                return torch.device(f"cuda:{device}")
            if isinstance(device, str) and device not in {"cpu", "disk"}:
                return torch.device(device)
    return next(model.parameters()).device


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    """Move tensors in a transformers BatchEncoding/dict to one device."""
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()
    }


def prepare_qwen2_audio_inputs(
    processor: Any,
    conversation: Sequence[Mapping[str, Any]],
    *,
    device: Any | None = None,
    padding: bool = True,
) -> Any:
    text = processor.apply_chat_template(
        list(conversation),
        add_generation_prompt=True,
        tokenize=False,
    )
    sample_rate = processor.feature_extractor.sampling_rate
    audios = [load_audio_array(ref, sample_rate) for ref in _audio_refs(conversation)]
    inputs = processor(
        text=text,
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=padding,
    )
    return move_inputs_to_device(inputs, device) if device is not None else inputs


def _audio_prompt_text(
    processor: Any,
    conversation: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
) -> str:
    return processor.apply_chat_template(
        list(conversation),
        add_generation_prompt=add_generation_prompt,
        tokenize=False,
    )


def prepare_qwen2_audio_teacher_forced_inputs(
    processor: Any,
    conversation: Sequence[Mapping[str, Any]],
    target: str,
    *,
    device: Any | None = None,
    ignore_index: int = -100,
) -> tuple[Any, Any, int]:
    """Prepare prompt+target inputs and CE labels for target tokens only."""
    prompt_text = _audio_prompt_text(processor, conversation, add_generation_prompt=True)
    full_text = prompt_text + target
    sample_rate = processor.feature_extractor.sampling_rate
    audios = [load_audio_array(ref, sample_rate) for ref in _audio_refs(conversation)]
    prompt_inputs = processor(
        text=prompt_text,
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    full_inputs = processor(
        text=full_text,
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    )
    prompt_len = prompt_inputs.input_ids.shape[1]
    labels = full_inputs.input_ids.clone()
    labels[:, :prompt_len] = ignore_index
    if device is not None:
        full_inputs = move_inputs_to_device(full_inputs, device)
        labels = labels.to(device)
    return full_inputs, labels, prompt_len


def prepare_qwen2_text_inputs(
    processor: Any,
    conversation: Sequence[Mapping[str, Any]],
    *,
    device: Any | None = None,
) -> Any:
    text = processor.apply_chat_template(
        list(conversation),
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = processor.tokenizer(text, return_tensors="pt")
    return move_inputs_to_device(inputs, device) if device is not None else inputs


def resolve_chat_position_indices(
    processor: Any,
    conversation: Sequence[Mapping[str, Any]],
) -> dict[PositionName, int]:
    """Resolve the two preregistered readout positions from the chat template.

    ``assistant_start_pre`` is the final token before the assistant generation
    prompt is appended. ``first_generation_prelogit`` is the final token of the
    full prompt, i.e. the position whose logits produce the first generated token.
    """
    no_generation = processor.apply_chat_template(
        list(conversation),
        add_generation_prompt=False,
        tokenize=False,
    )
    with_generation = processor.apply_chat_template(
        list(conversation),
        add_generation_prompt=True,
        tokenize=False,
    )
    tokenizer = processor.tokenizer
    no_gen_len = tokenizer(no_generation, return_tensors="pt").input_ids.shape[1]
    gen_len = tokenizer(with_generation, return_tensors="pt").input_ids.shape[1]
    if no_gen_len < 1 or gen_len < no_gen_len:
        raise ValueError("unexpected Qwen2-Audio chat template token lengths")
    return {
        "assistant_start_pre": no_gen_len - 1,
        "first_generation_prelogit": gen_len - 1,
    }


def resolve_audio_position_indices(
    processor: Any,
    conversation: Sequence[Mapping[str, Any]],
) -> dict[PositionName, int]:
    """Resolve readout positions using processor-expanded audio prompt lengths."""
    sample_rate = processor.feature_extractor.sampling_rate
    audios = [load_audio_array(ref, sample_rate) for ref in _audio_refs(conversation)]
    no_generation = _audio_prompt_text(
        processor,
        conversation,
        add_generation_prompt=False,
    )
    with_generation = _audio_prompt_text(
        processor,
        conversation,
        add_generation_prompt=True,
    )
    no_gen_len = processor(
        text=no_generation,
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    ).input_ids.shape[1]
    gen_len = processor(
        text=with_generation,
        audio=audios,
        sampling_rate=sample_rate,
        return_tensors="pt",
        padding=True,
    ).input_ids.shape[1]
    if no_gen_len < 1 or gen_len < no_gen_len:
        raise ValueError("unexpected Qwen2-Audio prompt token lengths")
    return {
        "assistant_start_pre": no_gen_len - 1,
        "first_generation_prelogit": gen_len - 1,
    }


def generate_audio_response(
    model: Any,
    processor: Any,
    audio_path: Path | str,
    instruction: str,
    *,
    max_new_tokens: int = 128,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Run one local Qwen2-Audio audio-analysis generation."""
    conversation = build_audio_analysis_conversation(
        audio_path,
        instruction,
        system_prompt=system_prompt,
    )
    device = model_input_device(model)
    inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    prompt_len = inputs.input_ids.shape[1]
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generate_ids = generate_ids[:, prompt_len:]
    return processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_text_response(
    model: Any,
    processor: Any,
    text: str,
    *,
    max_new_tokens: int = 128,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    """Run one Qwen2-Audio generation on a pure-text ChatML prompt.

    Used by the Run 4 text arm so matched harmful/benign content can be sent as
    text instead of audio through the *same* LLM backbone, chat template, system
    prompt, and decoding as ``generate_audio_response``. No audio is attached, so
    inputs go through the tokenizer text path (``prepare_qwen2_text_inputs``).
    """
    conversation = build_text_conversation(text, system_prompt=system_prompt)
    device = model_input_device(model)
    inputs = prepare_qwen2_text_inputs(processor, conversation, device=device)
    prompt_len = inputs.input_ids.shape[1]
    generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generate_ids = generate_ids[:, prompt_len:]
    return processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def generate_audio_response_with_intervention(
    model: Any,
    processor: Any,
    audio_path: Path | str,
    instruction: str,
    *,
    layer_idx: int,
    position_name: str,
    vector: Any,
    mode: str,
    scale: float = 1.0,
    target_coordinate: float | Any | None = None,
    all_positions: bool = False,
    max_new_tokens: int = 128,
    system_prompt: str = "You are a helpful assistant.",
) -> str:
    conversation = build_audio_analysis_conversation(
        audio_path,
        instruction,
        system_prompt=system_prompt,
    )
    token_index = (
        None
        if all_positions
        else resolve_audio_position_indices(processor, conversation)[position_name]
    )
    device = model_input_device(model)
    inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    prompt_len = inputs.input_ids.shape[1]
    with ResidualStreamIntervention(
        model,
        layer_idx=layer_idx,
        token_index=token_index,
        vector=vector,
        mode=mode,
        scale=scale,
        target_coordinate=target_coordinate,
        all_positions=all_positions,
    ):
        generate_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generate_ids = generate_ids[:, prompt_len:]
    return processor.batch_decode(
        generate_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
