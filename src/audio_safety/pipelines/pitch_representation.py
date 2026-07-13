"""Fast pitch-only activation extraction and representation analysis pipeline.

GPU imports remain inside extraction functions. The analysis path loads saved
NumPy artifacts and can run in the base CPU environment.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from tqdm.auto import tqdm

from audio_safety.config.schema import ExperimentConfig, PitchRepresentationConfig
from audio_safety.data.acoustic import load_wav, pitch_shift, save_wav
from audio_safety.utils.io import load_jsonl, save_jsonl
from audio_safety.utils.text import token_overlap, word_error_rate


def _require_config(cfg: ExperimentConfig) -> PitchRepresentationConfig:
    gate = cfg.pitch_representation
    if gate is None or not gate.enabled:
        raise ValueError("pitch_representation config must be present and enabled")
    return gate


def _manifest_path(
    cfg: ExperimentConfig,
    gate: PitchRepresentationConfig,
    data_dir: Path,
) -> Path:
    relative = gate.source_manifest_file or cfg.dataset.tts.manifest_file
    path = Path(relative)
    return path if path.is_absolute() else data_dir / path


def _source_audio_path(data_dir: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else data_dir / path


def _pitch_grid(gate: PitchRepresentationConfig) -> list[float]:
    pitches = sorted({round(float(value), 6) for value in gate.pitch_semitones})
    if len(pitches) != len(gate.pitch_semitones):
        raise ValueError("pitch_representation.pitch_semitones must be unique")
    if 0.0 not in pitches:
        raise ValueError("pitch_representation.pitch_semitones must include neutral pitch 0")
    return pitches


def _select_complete_pairs(
    records: Sequence[dict[str, Any]],
    gate: PitchRepresentationConfig,
    *,
    item_limit: int | None,
) -> list[dict[str, dict[str, Any]]]:
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        if str(record.get("style")) != gate.source_style:
            continue
        label = str(record.get("safety_label"))
        if label not in {"harmful", "benign"}:
            continue
        key = (str(record["item_id"]), label)
        if key in cells:
            raise ValueError(f"duplicate source manifest cell for item/label {key}")
        cells[key] = dict(record)

    complete_ids = sorted(
        item_id
        for item_id, label in cells
        if label == "harmful" and (item_id, "benign") in cells
    )
    requested = item_limit if item_limit is not None else gate.n_items
    selected_ids = complete_ids[:requested]
    if len(selected_ids) < requested:
        raise ValueError(
            f"requested {requested} complete neutral pairs, found {len(selected_ids)} "
            f"in style {gate.source_style!r}"
        )
    return [
        {
            "harmful": cells[(item_id, "harmful")],
            "benign": cells[(item_id, "benign")],
        }
        for item_id in selected_ids
    ]


def _pitch_tag(pitch: float) -> str:
    sign = "m" if pitch < 0 else "p"
    magnitude = f"{abs(pitch):05.2f}".replace(".", "p")
    return f"{sign}{magnitude}st"


def _variant_audio(
    source: Path,
    *,
    item_id: str,
    safety_label: str,
    pitch: float,
    gate: PitchRepresentationConfig,
    run_dir: Path,
) -> tuple[Path, str | None]:
    if pitch == 0.0:
        return source, None
    relative = gate.variants_dir / safety_label / _pitch_tag(pitch) / f"{item_id}.wav"
    target = run_dir / relative
    if gate.overwrite or not target.exists():
        waveform = load_wav(source, sr=gate.sample_rate)
        shifted = pitch_shift(waveform, gate.sample_rate, pitch)
        save_wav(target, shifted, sr=gate.sample_rate)
    return target, str(relative)


def _first_token_ids(tokenizer: Any, prefixes: Sequence[str]) -> list[int]:
    ids: set[int] = set()
    for prefix in prefixes:
        for form in (str(prefix), " " + str(prefix)):
            token_ids = tokenizer(form, add_special_tokens=False).input_ids
            if token_ids:
                ids.add(int(token_ids[0]))
    if not ids:
        raise ValueError("first-token prefix bank produced no token ids")
    return sorted(ids)


def _audio_lengths(model: Any, inputs: Any) -> tuple[int, int]:
    from audio_safety.models.hooks import get_audio_tower

    feature_mask = inputs["feature_attention_mask"]
    if feature_mask.shape[0] != 1:
        raise ValueError("pitch representation capture requires batch size 1")
    tower = get_audio_tower(model)
    feature_lengths, projector_lengths = tower._get_feat_extract_output_lengths(
        feature_mask.sum(-1)
    )
    if feature_lengths.numel() != 1 or projector_lengths.numel() != 1:
        raise ValueError("pitch representation capture requires exactly one audio")
    return int(feature_lengths.item()), int(projector_lengths.item())


def _capture_cell(
    model: Any,
    processor: Any,
    audio_path: Path,
    cfg: ExperimentConfig,
    gate: PitchRepresentationConfig,
    refusal_ids: list[int],
    compliance_ids: list[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from audio_safety.models.hooks import AudioPathCapture
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        model_input_device,
        prepare_qwen2_audio_inputs,
        resolve_audio_position_indices,
    )

    conversation = build_audio_analysis_conversation(
        audio_path,
        cfg.dataset.target_generation.instruction,
        system_prompt=gate.system_prompt,
    )
    device = model_input_device(model)
    inputs = prepare_qwen2_audio_inputs(processor, conversation, device=device)
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("pitch representation capture requires one conversation per forward")
    positions = resolve_audio_position_indices(processor, conversation)
    p1 = int(positions["assistant_start_pre"])
    p2 = int(positions["first_generation_prelogit"])

    audio_token_id = getattr(model.config, "audio_token_id", None)
    if audio_token_id is None:
        raise AttributeError("Qwen2-Audio config exposes no audio_token_id")
    audio_mask = inputs["input_ids"][0].eq(int(audio_token_id))
    if "attention_mask" in inputs:
        audio_mask = audio_mask & inputs["attention_mask"][0].bool()
    audio_positions = audio_mask.nonzero(as_tuple=False).flatten().tolist()
    encoder_length, projector_length = _audio_lengths(model, inputs)
    decoder_layers = None if gate.llm_layers == "all" else list(gate.llm_layers)

    capture = AudioPathCapture(
        model,
        audio_positions=audio_positions,
        p1_index=p1,
        p2_index=p2,
        encoder_valid_length=encoder_length,
        projector_valid_length=projector_length,
        decoder_layers=decoder_layers,
    )
    with torch.inference_mode(), capture:
        outputs = model(**inputs, use_cache=False, return_dict=True)
    states = {
        name: tensor.numpy().astype(np.float32, copy=False)
        for name, tensor in capture.states().items()
    }
    logits = outputs.logits[0, p2, :].detach().float().cpu().numpy()
    refusal_logmass = float(np.logaddexp.reduce(logits[np.asarray(refusal_ids)]))
    compliance_logmass = float(np.logaddexp.reduce(logits[np.asarray(compliance_ids)]))
    trace = {
        "refusal_margin": refusal_logmass - compliance_logmass,
        "p1_index": p1,
        "p2_index": p2,
        "prompt_length": int(inputs["input_ids"].shape[1]),
        "audio_span_start": int(audio_positions[0]),
        "audio_span_end": int(audio_positions[-1]),
        "audio_token_count": len(audio_positions),
        "encoder_valid_length": encoder_length,
        "projector_valid_length": projector_length,
        "encoder_layers": capture.encoder_layer_indices,
        "llm_layers": capture.decoder_layer_indices,
    }
    return states, trace


def _generation_indices(
    cells: list[dict[str, Any]],
    gate: PitchRepresentationConfig,
) -> list[int]:
    selected: set[int] = set()
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    allowed_labels = set(gate.response_safety_labels)
    for cell in cells:
        if cell["safety_label"] in allowed_labels:
            grouped[(str(cell["item_id"]), str(cell["safety_label"]))].append(cell)

    for group in grouped.values():
        ordered = sorted(group, key=lambda cell: float(cell["pitch_semitones"]))
        neutral = next(
            (cell for cell in ordered if float(cell["pitch_semitones"]) == 0.0),
            None,
        )
        if neutral is not None:
            selected.add(int(neutral["activation_index"]))
        if gate.generate_endpoints and ordered:
            selected.add(int(ordered[0]["activation_index"]))
            selected.add(int(ordered[-1]["activation_index"]))
        if gate.generate_margin_crossings:
            for left, right in zip(ordered, ordered[1:], strict=False):
                left_margin = float(left["refusal_margin"])
                right_margin = float(right["refusal_margin"])
                if left_margin == 0.0 or right_margin == 0.0 or left_margin * right_margin < 0.0:
                    selected.add(int(left["activation_index"]))
                    selected.add(int(right["activation_index"]))
    return sorted(selected)


def _generate_selected_outputs(
    model: Any,
    processor: Any,
    cells: list[dict[str, Any]],
    audio_paths: dict[int, Path],
    cfg: ExperimentConfig,
    gate: PitchRepresentationConfig,
) -> None:
    from audio_safety.evaluation.labeling import label_output
    from audio_safety.models.qwen2_audio import generate_audio_response

    selected = _generation_indices(cells, gate)
    for activation_index in tqdm(selected, desc="pitch generation", unit="cell"):
        cell = cells[activation_index]
        audio_path = audio_paths[activation_index]
        output = generate_audio_response(
            model,
            processor,
            audio_path,
            cfg.dataset.target_generation.instruction,
            max_new_tokens=gate.max_new_tokens,
            system_prompt=gate.system_prompt,
            do_sample=False,
        )
        behavior, failure_mode, needs_review = label_output(
            output,
            safety_label=str(cell["safety_label"]),
            reference_text=str(cell.get("reference_text") or "") or None,
        )
        cell.update(
            {
                "output": output,
                "behavior_label": behavior,
                "decoding_failure_mode": failure_mode,
                "needs_manual_review": needs_review,
            }
        )
        if gate.transcribe_selected:
            transcript = generate_audio_response(
                model,
                processor,
                audio_path,
                gate.transcribe_instruction,
                max_new_tokens=gate.transcribe_max_new_tokens,
                system_prompt=gate.system_prompt,
                do_sample=False,
            )
            reference = str(cell.get("reference_text") or "")
            wer = word_error_rate(reference, transcript)
            overlap = token_overlap(reference, transcript)
            cell.update(
                {
                    "pitch_transcript": transcript,
                    "pitch_transcript_wer": wer,
                    "pitch_transcript_token_overlap": overlap,
                    "semantic_preserved": bool(
                        wer <= gate.transcript_wer_max
                        and overlap >= gate.transcript_token_overlap_min
                    ),
                }
            )


def extract_pitch_representation(
    model: Any,
    processor: Any,
    cfg: ExperimentConfig,
    data_dir: Path,
    run_dir: Path,
    *,
    item_limit: int | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Create pitch variants and capture the complete pooled model path."""
    import torch

    gate = _require_config(cfg)
    activation_path = run_dir / gate.activations_file
    cells_path = run_dir / gate.cells_file
    if not gate.overwrite and (activation_path.exists() or cells_path.exists()):
        raise FileExistsError(
            f"pitch artifacts already exist under {run_dir}; set pitch_representation.overwrite=true"
        )

    records = load_jsonl(_manifest_path(cfg, gate, data_dir))
    pairs = _select_complete_pairs(records, gate, item_limit=item_limit)
    pitches = _pitch_grid(gate)
    refusal_ids = _first_token_ids(processor.tokenizer, gate.refusal_prefixes)
    compliance_ids = _first_token_ids(processor.tokenizer, gate.compliance_prefixes)

    stored: dict[str, list[np.ndarray]] = defaultdict(list)
    cells: list[dict[str, Any]] = []
    audio_paths: dict[int, Path] = {}
    encoder_layers: list[int] | None = None
    llm_layers: list[int] | None = None

    total = len(pairs) * 2 * len(pitches)
    progress = tqdm(total=total, desc="pitch capture", unit="cell")
    for pair in pairs:
        for safety_label in ("harmful", "benign"):
            source_record = pair[safety_label]
            source_path = _source_audio_path(data_dir, source_record["path"])
            if not source_path.is_file():
                raise FileNotFoundError(f"source audio is missing: {source_path}")
            item_id = str(source_record["item_id"])
            for pitch in pitches:
                audio_path, variant_relative = _variant_audio(
                    source_path,
                    item_id=item_id,
                    safety_label=safety_label,
                    pitch=pitch,
                    gate=gate,
                    run_dir=run_dir,
                )
                states, trace = _capture_cell(
                    model,
                    processor,
                    audio_path,
                    cfg,
                    gate,
                    refusal_ids,
                    compliance_ids,
                )
                if encoder_layers is None:
                    encoder_layers = list(trace["encoder_layers"])
                    llm_layers = list(trace["llm_layers"])
                elif encoder_layers != trace["encoder_layers"] or llm_layers != trace["llm_layers"]:
                    raise RuntimeError("captured layer indices changed between cells")
                for name, values in states.items():
                    stored[name].append(values)

                activation_index = len(cells)
                audio_paths[activation_index] = audio_path
                cells.append(
                    {
                        "activation_index": activation_index,
                        "item_id": item_id,
                        "category": source_record.get("category"),
                        "safety_label": safety_label,
                        "source_style": gate.source_style,
                        "pitch_semitones": pitch,
                        "source_path": str(source_record["path"]),
                        "variant_path": variant_relative,
                        "reference_text": source_record.get("reference_text"),
                        "source_transcript": source_record.get("transcript"),
                        "source_wer": source_record.get("wer"),
                        "source_transcript_control_passed": source_record.get(
                            "transcript_control_passed"
                        ),
                        "refusal_margin": trace["refusal_margin"],
                        "p1_index": trace["p1_index"],
                        "p2_index": trace["p2_index"],
                        "prompt_length": trace["prompt_length"],
                        "audio_span_start": trace["audio_span_start"],
                        "audio_span_end": trace["audio_span_end"],
                        "audio_token_count": trace["audio_token_count"],
                        "encoder_valid_length": trace["encoder_valid_length"],
                        "projector_valid_length": trace["projector_valid_length"],
                        "output": None,
                        "behavior_label": None,
                        "needs_manual_review": None,
                        "semantic_preserved": None,
                    }
                )
                progress.update(1)
                if torch.cuda.is_available() and activation_index % 20 == 0:
                    torch.cuda.empty_cache()
    progress.close()

    if gate.generate_responses:
        _generate_selected_outputs(model, processor, cells, audio_paths, cfg, gate)

    arrays = {
        name: np.stack(values).astype(np.float32, copy=False)
        for name, values in stored.items()
    }
    arrays["encoder_layers"] = np.asarray(encoder_layers, dtype=np.int16)
    arrays["llm_layers"] = np.asarray(llm_layers, dtype=np.int16)
    arrays["refusal_token_ids"] = np.asarray(refusal_ids, dtype=np.int64)
    arrays["compliance_token_ids"] = np.asarray(compliance_ids, dtype=np.int64)
    activation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(activation_path, **arrays)
    save_jsonl(cells, cells_path)
    return arrays, cells


def analyze_pitch_artifacts(
    cfg: ExperimentConfig,
    run_dir: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    """Load saved extraction artifacts and run the CPU-only grouped analysis."""
    from audio_safety.evaluation.pitch_representation import (
        analyze_pitch_representation,
        save_pitch_analysis,
    )

    gate = _require_config(cfg)
    with np.load(run_dir / gate.activations_file) as archive:
        arrays = {name: archive[name] for name in archive.files}
    cells = load_jsonl(run_dir / gate.cells_file)
    metrics = analyze_pitch_representation(arrays, cells, gate, seed=seed)
    save_pitch_analysis(metrics, cells, gate, run_dir)
    return metrics
