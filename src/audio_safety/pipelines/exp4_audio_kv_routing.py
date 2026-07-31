"""Exp4 post-prefill audio/t_AB KV-cache routing experiment.

Torch and Transformers remain lazy imports. Source freezing and statistical
analysis therefore run in the CPU-only base environment.
"""

from __future__ import annotations

import contextlib
import copy
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from audio_safety.config.schema import Exp4RunConfig
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.evaluation.refusal_instability import (
    EXPLICIT_REFUSAL,
    classify_explicit_refusal,
)
from audio_safety.pipelines.exp3_qwen_mechanism import (
    _behavior_index,
    _checkpoint_mapping,
    _generate_from_inputs,
    _margin,
    _prepare_aligned_pair,
    _sha256_file,
    _stable_int,
    readout_token_ids,
)
from audio_safety.utils.io import load_json, load_jsonl, save_json
from audio_safety.utils.paths import ResolvedPaths

CONDITION_SOURCES = {
    "audio_injected__tab_injected": ("injected", "injected"),
    "audio_host__tab_injected": ("host", "injected"),
    "audio_injected__tab_host": ("injected", "host"),
    "audio_host__tab_host": ("host", "host"),
}
CONDITION_CODES = {
    "audio_injected__tab_injected": "II",
    "audio_host__tab_injected": "HI",
    "audio_injected__tab_host": "IH",
    "audio_host__tab_host": "HH",
}
# Exp5 adds the first generated token as a third factor. A condition name gains
# a `__y1_host` suffix; the bare Exp4 names keep meaning "y1 from the injected
# trajectory", so Exp4 configs and records are unaffected.
Y1_HOST_SUFFIX = "__y1_host"


def condition_parts(condition: str) -> tuple[str, str, str]:
    """(audio cache source, t_AB cache source, y1 source) for a condition name."""

    base, y1_source = condition, "injected"
    if condition.endswith(Y1_HOST_SUFFIX):
        base, y1_source = condition[: -len(Y1_HOST_SUFFIX)], "host"
    if base not in CONDITION_SOURCES:
        raise ValueError(f"unknown Exp4/Exp5 condition {condition!r}")
    audio_source, tab_source = CONDITION_SOURCES[base]
    return audio_source, tab_source, y1_source


def condition_code(condition: str) -> str:
    """Three-letter cell code, e.g. `IIH` = injected audio, injected t_AB, host y1."""

    audio_source, tab_source, y1_source = condition_parts(condition)
    letters = {"injected": "I", "host": "H"}
    return letters[audio_source] + letters[tab_source] + letters[y1_source]


@dataclass
class PrefillBundle:
    """Physical-arm prefill cache and residual states needed for composition."""

    cache: Any
    last_logits: Any
    audio_state: Any
    relay_states: dict[int, Any]
    prompt_length: int


def _artifact(run_dir: Path, relative: Path) -> Path:
    return run_dir / relative


def _relative_artifact(value: Path, *, label: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be a run-relative path, got {path}")
    return path


def _source_artifacts(cfg: Exp4RunConfig) -> dict[str, Path]:
    gate = cfg.exp4
    return {
        "config_snapshot": Path("config_snapshot.yaml"),
        "provenance": Path("provenance.json"),
        "pairs": cfg.exp3.artifacts.pairs_file,
        "behavior": cfg.exp3.artifacts.behavior_file,
        "cohort": gate.source_cohort_file,
        "span_patch": gate.source_span_patch_file,
        "relay_closure": gate.source_relay_file,
    }


def _validate_source_snapshot(cfg: Exp4RunConfig, snapshot_path: Path) -> dict[str, Any]:
    snapshot = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get("config"), dict):
        raise ValueError("source config_snapshot.yaml has no resolved config mapping")
    frozen = snapshot["config"]
    current = cfg.model_dump(mode="json")
    for key in ("model", "paths", "exp3"):
        if frozen.get(key) != current[key]:
            raise RuntimeError(
                f"Exp4 inherited {key!r} differs from the source Exp3 config snapshot"
            )
    return snapshot


def _validate_source_evidence(
    cfg: Exp4RunConfig,
    cohort: Sequence[Mapping[str, Any]],
    span_rows: Sequence[Mapping[str, Any]],
    relay_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Require source evidence for every pair/direction entering Exp4."""

    pair_ids = [str(row["pair_id"]) for row in cohort]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("source Exp3 cohort contains duplicate pair_id values")
    if any(str(row.get("contrast")) != cfg.exp4.contrast for row in cohort):
        raise ValueError("source cohort contains a contrast outside Exp4's frozen contrast")

    span_keys = {
        (str(row.get("pair_id")), str(row.get("direction")))
        for row in span_rows
        if int(row.get("layer", -1)) == cfg.exp4.inject_layer
        and row.get("condition") == "real"
        and bool(row.get("full_generation"))
    }
    relay_keys = {
        (str(row.get("pair_id")), str(row.get("direction")), str(row.get("condition")))
        for row in relay_rows
        if row.get("condition") in {"open", "relay_closed"}
    }
    missing_span: list[tuple[str, str]] = []
    missing_relay: list[tuple[str, str, str]] = []
    for pair_id in pair_ids:
        for direction in cfg.exp4.directions:
            if (pair_id, direction) not in span_keys:
                missing_span.append((pair_id, direction))
            for condition in ("open", "relay_closed"):
                if (pair_id, direction, condition) not in relay_keys:
                    missing_relay.append((pair_id, direction, condition))
    if missing_span:
        raise RuntimeError(f"source L10 real span-patch evidence is missing: {missing_span[:5]}")
    if missing_relay:
        raise RuntimeError(f"source relay-closure evidence is missing: {missing_relay[:5]}")


def freeze_source_inputs(
    cfg: Exp4RunConfig,
    paths: ResolvedPaths,
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Hash the source Exp3 run and freeze the exact Exp4 cohort.

    Re-running preflight against mutated source artifacts fails rather than
    silently updating the manifest under an existing run name.
    """

    source_dir = paths.output_dir / cfg.exp4.source_exp3_run
    artifacts = {
        name: _relative_artifact(relative, label=f"source artifact {name}")
        for name, relative in _source_artifacts(cfg).items()
    }
    missing = [
        str(relative) for relative in artifacts.values() if not (source_dir / relative).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"source Exp3 run {cfg.exp4.source_exp3_run!r} is incomplete; missing {missing}"
        )

    snapshot = _validate_source_snapshot(cfg, source_dir / artifacts["config_snapshot"])
    source_cohort = load_jsonl(source_dir / artifacts["cohort"])
    span_rows = load_jsonl(source_dir / artifacts["span_patch"])
    relay_rows = load_jsonl(source_dir / artifacts["relay_closure"])
    _validate_source_evidence(cfg, source_cohort, span_rows, relay_rows)

    files = []
    for name, relative in sorted(artifacts.items()):
        source_path = source_dir / relative
        files.append(
            {
                "name": name,
                "relative_path": relative.as_posix(),
                "sha256": _sha256_file(source_path),
                "size_bytes": source_path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": cfg.exp4.schema_version,
        "source_exp3_run": cfg.exp4.source_exp3_run,
        "source_git_commit": snapshot.get("git_commit"),
        "files": files,
    }
    manifest_path = _artifact(run_dir, cfg.exp4.artifacts.source_manifest_file)
    if manifest_path.is_file():
        if load_json(manifest_path) != manifest:
            raise RuntimeError("source Exp3 artifacts changed after the Exp4 manifest was frozen")
    else:
        save_json(manifest, manifest_path)

    selected = [dict(row) for row in source_cohort]
    if cfg.exp4.max_pairs is not None:
        selected = selected[: cfg.exp4.max_pairs]
    if not selected:
        raise ValueError("Exp4 source cohort is empty")
    cohort_path = _artifact(run_dir, cfg.exp4.artifacts.cohort_file)
    if cohort_path.is_file():
        frozen = load_jsonl(cohort_path)
        if frozen != selected:
            raise RuntimeError("Exp4 cohort changed after it was frozen")
        return frozen
    atomic_save_jsonl(selected, cohort_path)
    return selected


def _validate_shard(shard_index: int, shard_count: int) -> None:
    if shard_count < 1:
        raise ValueError("shard_count must be >= 1")
    if not 0 <= shard_index < shard_count:
        raise ValueError(f"shard_index must lie in [0, {shard_count}), got {shard_index}")


def shard_records_path(
    cfg: Exp4RunConfig,
    run_dir: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> Path:
    """Per-shard checkpoint path; a single shard keeps the canonical file name."""

    _validate_shard(shard_index, shard_count)
    records = _relative_artifact(cfg.exp4.artifacts.records_file, label="records_file")
    if shard_count == 1:
        return run_dir / records
    name = f"{records.stem}.shard{shard_index:02d}_of_{shard_count:02d}{records.suffix}"
    return run_dir / records.with_name(name)


def shard_assignment(
    cohort: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> list[tuple[int, Mapping[str, Any]]]:
    """Round-robin partition carrying each pair's global cohort index.

    The global index is preserved so that shard-invariant decisions -- notably
    which pair/direction cells receive the pre-registered standard-generate
    equivalence check -- are identical to a single-process run.
    """

    _validate_shard(shard_index, shard_count)
    return [
        (index, pair) for index, pair in enumerate(cohort) if index % shard_count == shard_index
    ]


def _expected_standard_checks(
    cfg: Exp4RunConfig,
    cohort: Sequence[Mapping[str, Any]],
    *,
    shard_index: int,
    shard_count: int,
) -> int:
    """Standard-generate equivalence checks owed by one shard.

    Each shard runs the pre-registered number against its own model replica, so
    a single-shard run performs exactly the pre-registered count and a sharded
    run performs that many per replica.
    """

    n_assigned = len(shard_assignment(cohort, shard_index=shard_index, shard_count=shard_count))
    return min(cfg.exp4.standard_generate_checks, n_assigned * len(cfg.exp4.directions))


def total_expected_standard_checks(
    cfg: Exp4RunConfig, cohort: Sequence[Mapping[str, Any]], *, shard_count: int
) -> int:
    return sum(
        _expected_standard_checks(cfg, cohort, shard_index=index, shard_count=shard_count)
        for index in range(shard_count)
    )


def _factorial_keys(
    cfg: Exp4RunConfig, pairs: Sequence[Mapping[str, Any]]
) -> set[tuple[str, str, str]]:
    return {
        (str(pair["pair_id"]), direction, condition)
        for pair in pairs
        for direction in cfg.exp4.directions
        for condition in cfg.exp4.conditions
    }


def merge_shard_records(
    cfg: Exp4RunConfig,
    run_dir: Path,
    *,
    shard_count: int,
) -> list[dict[str, Any]]:
    """Merge per-shard checkpoints into the canonical records file, fail closed.

    Rejects a missing shard, a row written outside its shard's pair assignment,
    a cell produced twice, and any deviation from the frozen cohort factorial.
    Re-merging is idempotent but refuses to rewrite a differing canonical file.
    """

    _validate_shard(0, shard_count)
    cohort_path = _artifact(run_dir, cfg.exp4.artifacts.cohort_file)
    if not cohort_path.is_file():
        raise FileNotFoundError("Exp4 frozen cohort is missing; run preflight first")
    cohort = load_jsonl(cohort_path)

    merged: dict[tuple[str, str, str], dict[str, Any]] = {}
    for shard_index in range(shard_count):
        path = shard_records_path(cfg, run_dir, shard_index=shard_index, shard_count=shard_count)
        if not path.is_file():
            raise FileNotFoundError(f"Exp4 shard checkpoint is missing: {path}")
        owned = {
            str(pair["pair_id"])
            for _index, pair in shard_assignment(
                cohort, shard_index=shard_index, shard_count=shard_count
            )
        }
        for row in load_jsonl(path):
            key = (str(row["pair_id"]), str(row["direction"]), str(row["condition"]))
            if key[0] not in owned:
                raise RuntimeError(
                    f"Exp4 shard {shard_index} wrote a row outside its pair assignment: {key}"
                )
            if key in merged:
                raise RuntimeError(f"duplicate Exp4 cell across shards: {key}")
            merged[key] = row

    expected = _factorial_keys(cfg, cohort)
    if set(merged) != expected:
        missing = sorted(expected - set(merged))
        extra = sorted(set(merged) - expected)
        raise RuntimeError(
            f"merged Exp4 shards do not cover the frozen cohort factorial: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )

    rows = [merged[key] for key in sorted(merged)]
    records_path = _artifact(run_dir, cfg.exp4.artifacts.records_file)
    if records_path.is_file() and load_jsonl(records_path) != rows:
        raise RuntimeError(
            "merged Exp4 shards differ from the existing canonical records checkpoint"
        )
    atomic_save_jsonl(rows, records_path)
    return rows


def _cache_layers(cache: Any) -> list[Any]:
    layers = getattr(cache, "layers", None)
    if layers is None:
        raise TypeError(
            "Exp4 requires a Transformers Cache exposing .layers; "
            f"got {type(cache).__module__}.{type(cache).__qualname__}"
        )
    layers = list(layers)
    if not layers:
        raise ValueError("KV cache has no layers")
    for index, layer in enumerate(layers):
        if getattr(layer, "keys", None) is None or getattr(layer, "values", None) is None:
            raise ValueError(f"KV cache layer {index} is uninitialized")
    return layers


def clone_kv_cache(cache: Any) -> Any:
    """Clone every K/V tensor while preserving the concrete cache/layer classes."""

    cloned_cache = copy.copy(cache)
    cloned_layers = []
    for layer in _cache_layers(cache):
        cloned_layer = copy.copy(layer)
        cloned_layer.keys = layer.keys.clone()
        cloned_layer.values = layer.values.clone()
        cloned_layers.append(cloned_layer)
    cloned_cache.layers = cloned_layers
    return cloned_cache


def cache_max_abs_diff(left: Any, right: Any) -> float:
    """Maximum absolute K/V difference across every layer; shapes must agree.

    Non-finite entries raise instead of propagating: a NaN difference compares
    false against every tolerance, so a silent NaN would let the tolerance-zero
    closure gates in design.md §5 pass on a corrupted cache.
    """

    import torch

    left_layers, right_layers = _cache_layers(left), _cache_layers(right)
    if len(left_layers) != len(right_layers):
        raise ValueError("KV caches have different layer counts")
    maximum = 0.0
    for index, (left_layer, right_layer) in enumerate(zip(left_layers, right_layers, strict=True)):
        for kind in ("keys", "values"):
            left_tensor, right_tensor = getattr(left_layer, kind), getattr(right_layer, kind)
            if tuple(left_tensor.shape) != tuple(right_tensor.shape):
                raise ValueError(f"KV cache {kind} shape differs at layer {index}")
            if not (torch.isfinite(left_tensor).all() and torch.isfinite(right_tensor).all()):
                raise ValueError(f"KV cache {kind} contains non-finite values at layer {index}")
            difference = float((left_tensor - right_tensor).detach().abs().max().item())
            maximum = max(maximum, difference)
    return maximum


def transplant_cache_positions(base: Any, donor: Any, positions: Sequence[int]) -> Any:
    """Return a cloned cache with donor K and V at absolute prompt positions."""

    import torch

    positions = [int(position) for position in positions]
    if not positions or positions != sorted(set(positions)) or positions[0] < 0:
        raise ValueError("cache positions must be sorted unique non-negative indices")
    result = clone_kv_cache(base)
    result_layers, donor_layers = _cache_layers(result), _cache_layers(donor)
    if len(result_layers) != len(donor_layers):
        raise ValueError("base and donor caches have different layer counts")
    for layer_index, (result_layer, donor_layer) in enumerate(
        zip(result_layers, donor_layers, strict=True)
    ):
        for kind in ("keys", "values"):
            target, source = getattr(result_layer, kind), getattr(donor_layer, kind)
            if tuple(target.shape) != tuple(source.shape):
                raise ValueError(f"base/donor {kind} shape differs at layer {layer_index}")
            if target.dtype != source.dtype or target.device != source.device:
                raise ValueError(
                    f"base/donor {kind} dtype or device differs at layer {layer_index}"
                )
            if target.ndim < 2 or positions[-1] >= target.shape[-2]:
                raise IndexError(f"cache position exceeds sequence length at layer {layer_index}")
            index = torch.as_tensor(positions, dtype=torch.long, device=target.device)
            target.index_copy_(-2, index, source.index_select(-2, index))
    return result


def _cache_sequence_length(cache: Any) -> int:
    lengths = {int(layer.keys.shape[-2]) for layer in _cache_layers(cache)}
    if len(lengths) != 1:
        raise ValueError(f"KV cache layers disagree on sequence length: {sorted(lengths)}")
    return lengths.pop()


def _physical_prefill(
    cfg: Exp4RunConfig,
    model: Any,
    prepared: Mapping[str, Any],
    *,
    relay_positions: Sequence[int],
    clamp_layers: Sequence[int],
) -> PrefillBundle:
    import torch

    from audio_safety.models.hooks import AudioSpanCapture

    audio_capture = AudioSpanCapture(
        model,
        layers=[cfg.exp4.inject_layer],
        positions=prepared["audio_positions"],
    )
    relay_capture = AudioSpanCapture(
        model,
        layers=clamp_layers,
        positions=relay_positions,
    )
    with torch.inference_mode(), audio_capture, relay_capture:
        outputs = model(**prepared["inputs"], use_cache=True, return_dict=True)
    cache = outputs.past_key_values
    prompt_length = int(prepared["inputs"]["input_ids"].shape[1])
    if cache is None or _cache_sequence_length(cache) != prompt_length:
        raise RuntimeError("physical prefill did not return a complete prompt KV cache")
    return PrefillBundle(
        cache=cache,
        last_logits=outputs.logits[0, -1].detach(),
        audio_state=audio_capture.states()[cfg.exp4.inject_layer],
        relay_states=relay_capture.states(),
        prompt_length=prompt_length,
    )


def _injected_prefill(
    cfg: Exp4RunConfig,
    model: Any,
    host: Mapping[str, Any],
    *,
    donor_audio_state: Any,
    host_relay_states: Mapping[int, Any],
    relay_positions: Sequence[int],
    clamp_layers: Sequence[int],
) -> tuple[Any, Any, list[int]]:
    import torch

    from audio_safety.models.hooks import SpanStateIntervention

    contexts = [
        SpanStateIntervention(
            model,
            layer_idx=cfg.exp4.inject_layer,
            positions=host["audio_positions"],
            replacement=donor_audio_state,
        )
    ]
    contexts.extend(
        SpanStateIntervention(
            model,
            layer_idx=layer,
            positions=relay_positions,
            replacement=host_relay_states[layer],
        )
        for layer in clamp_layers
    )
    with torch.inference_mode(), contextlib.ExitStack() as stack:
        for intervention in contexts:
            stack.enter_context(intervention)
        outputs = model(**host["inputs"], use_cache=True, return_dict=True)
    counts = [int(intervention.applied_count) for intervention in contexts]
    if any(count != 1 for count in counts):
        raise RuntimeError(f"Exp4 prefill hooks must fire exactly once, got {counts}")
    cache = outputs.past_key_values
    if cache is None or _cache_sequence_length(cache) != int(host["t_ab"]) + 1:
        raise RuntimeError("injected prefill did not return a complete prompt KV cache")
    return cache, outputs.logits[0, -1].detach(), counts


def greedy_logits_processors(model: Any) -> Any:
    """Rebuild the logits processors `generate(do_sample=False)` would apply.

    Qwen2-Audio ships `repetition_penalty: 1.1` in its generation config, so
    ordinary greedy `model.generate` is NOT plain argmax. Exp3's entire frozen
    behaviour corpus was produced through that path, so Exp4's manual decoder
    must reproduce it or it is measuring a different decoder. Sampling warpers
    (temperature/top_k/top_p) are deliberately excluded: `generate` applies them
    only when `do_sample=True`.

    Any other generation-config field that would introduce a processor is
    rejected rather than silently dropped; design.md §5 gate 7 is the empirical
    proof that this list is complete.
    """

    from transformers import LogitsProcessorList, RepetitionPenaltyLogitsProcessor

    generation_config = model.generation_config
    unsupported = {
        "no_repeat_ngram_size": 0,
        "encoder_repetition_penalty": 1.0,
        "bad_words_ids": None,
        "min_length": 0,
        "min_new_tokens": None,
        "forced_bos_token_id": None,
        "forced_eos_token_id": None,
        "exponential_decay_length_penalty": None,
        "suppress_tokens": None,
        "begin_suppress_tokens": None,
        "sequence_bias": None,
        "diversity_penalty": 0.0,
        "guidance_scale": None,
        "renormalize_logits": False,
    }
    active = [
        name
        for name, default in unsupported.items()
        if getattr(generation_config, name, default) not in (default, None)
    ]
    if active:
        raise NotImplementedError(
            f"Exp4's manual decoder does not replicate generation-config fields {active}; "
            "extend greedy_logits_processors before running"
        )

    processors = LogitsProcessorList()
    penalty = getattr(generation_config, "repetition_penalty", None)
    if penalty is not None and float(penalty) != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(penalty=float(penalty)))
    return processors


def select_next_token(processors: Any, context_ids: Any, logits: Any) -> int:
    """Greedy selection exactly as `generate` performs it.

    `generate` casts the next-token logits to float32 before the processors run
    and before argmax. Selecting in bfloat16 instead resolves near-ties
    arbitrarily, so the cast is part of the decoding rule, not a detail.
    """

    import torch

    scores = logits.reshape(1, -1).to(copy=True, dtype=torch.float32)
    scores = processors(context_ids, scores)
    return int(torch.argmax(scores, dim=-1)[0].item())


def _stop_token_ids(model: Any, processor: Any) -> set[int]:
    values = getattr(model.generation_config, "eos_token_id", None)
    if values is None:
        values = getattr(processor.tokenizer, "eos_token_id", None)
    if values is None:
        return set()
    if isinstance(values, int):
        return {values}
    return {int(value) for value in values}


def manual_greedy_decode(
    cfg: Exp4RunConfig,
    model: Any,
    processor: Any,
    *,
    cache: Any,
    prompt_attention_mask: Any,
    prompt_input_ids: Any,
    first_token_id: int,
) -> dict[str, Any]:
    """Decode y2 onward from a surgically edited prefill cache and fixed y1.

    `prompt_input_ids` is required because the repetition penalty is defined over
    the whole context `generate` would see, prompt included, not over the
    generated suffix alone.
    """

    import torch

    token_ids = [int(first_token_id)]
    stop_ids = _stop_token_ids(model, processor)
    attention_mask = prompt_attention_mask.detach().clone()
    device = attention_mask.device
    processors = greedy_logits_processors(model)
    context_ids = torch.cat(
        [
            prompt_input_ids.detach().to(device=device, dtype=torch.long),
            torch.tensor([[first_token_id]], dtype=torch.long, device=device),
        ],
        dim=1,
    )
    current = torch.tensor([[first_token_id]], dtype=torch.long, device=device)
    with torch.inference_mode():
        for _ in range(cfg.exp4.max_new_tokens - 1):
            if token_ids[-1] in stop_ids:
                break
            attention_mask = torch.cat(
                [
                    attention_mask,
                    torch.ones(
                        (attention_mask.shape[0], 1),
                        dtype=attention_mask.dtype,
                        device=device,
                    ),
                ],
                dim=1,
            )
            outputs = model(
                input_ids=current,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                return_dict=True,
            )
            cache = outputs.past_key_values
            next_token = select_next_token(processors, context_ids, outputs.logits[0, -1])
            token_ids.append(next_token)
            current = torch.tensor([[next_token]], dtype=torch.long, device=device)
            context_ids = torch.cat([context_ids, current], dim=1)
    response = processor.batch_decode(
        [token_ids],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "response": response,
        "explicit_refusal": classify_explicit_refusal(response, cfg.exp3.readout.refusal_patterns)
        == EXPLICIT_REFUSAL,
        "generated_token_ids": token_ids,
    }


def _oriented_prefill(
    direction: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    source_bundle: PrefillBundle,
    target_bundle: PrefillBundle,
) -> tuple[Mapping[str, Any], Mapping[str, Any], PrefillBundle, PrefillBundle]:
    # Preserve Exp3's established direction convention: source_to_target starts
    # at the source host and transports toward the target donor.
    if direction == "source_to_target":
        return source, target, source_bundle, target_bundle
    if direction == "target_to_source":
        return target, source, target_bundle, source_bundle
    raise ValueError(f"unknown Exp4 direction {direction!r}")


def _fresh_interventions(
    cfg: Exp4RunConfig,
    model: Any,
    host: Mapping[str, Any],
    donor_bundle: PrefillBundle,
    host_bundle: PrefillBundle,
    relay_positions: Sequence[int],
    clamp_layers: Sequence[int],
) -> list[Any]:
    from audio_safety.models.hooks import SpanStateIntervention

    contexts = [
        SpanStateIntervention(
            model,
            layer_idx=cfg.exp4.inject_layer,
            positions=host["audio_positions"],
            replacement=donor_bundle.audio_state,
        )
    ]
    contexts.extend(
        SpanStateIntervention(
            model,
            layer_idx=layer,
            positions=relay_positions,
            replacement=host_bundle.relay_states[layer],
        )
        for layer in clamp_layers
    )
    return contexts


def _condition_cache(
    condition: str,
    *,
    injected_cache: Any,
    host_cache: Any,
    audio_positions: Sequence[int],
    t_ab: int,
) -> Any:
    audio_source, tab_source, _y1_source = condition_parts(condition)
    result = clone_kv_cache(injected_cache)
    if audio_source == "host":
        result = transplant_cache_positions(result, host_cache, audio_positions)
    if tab_source == "host":
        result = transplant_cache_positions(result, host_cache, [t_ab])
    return result


def _endpoint_role(source_refusal: bool, target_refusal: bool) -> str:
    return "discordant" if bool(source_refusal) != bool(target_refusal) else "stable_control"


def revalidate_physical_endpoints(
    cfg: Exp4RunConfig,
    paths: ResolvedPaths,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    """Regenerate both physical arms of every cohort pair on the current device.

    The frozen Exp3 markers were measured on other hardware. Greedy decoding is
    deterministic within a device but not across devices, so importing those
    markers would score this device's generations against labels that may not
    hold here. This stage re-measures `H` and `D` where the cells are actually
    run and records the frozen-vs-current transition as a diagnostic.
    """

    cohort_path = _artifact(run_dir, cfg.exp4.artifacts.cohort_file)
    if not cohort_path.is_file():
        raise FileNotFoundError("Exp4 cohort is not frozen; run preflight first")
    cohort = load_jsonl(cohort_path)
    source_dir = paths.output_dir / cfg.exp4.source_exp3_run
    frozen_behavior = _behavior_index(cfg, source_dir)
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)

    endpoints_path = _artifact(run_dir, cfg.exp4.artifacts.endpoints_file)
    existing = load_jsonl(endpoints_path) if endpoints_path.is_file() else []
    state = {(str(row["pair_id"]), str(row["arm"])): row for row in existing}
    if len(state) != len(existing):
        raise ValueError("Exp4 endpoint checkpoint contains duplicate pair/arm rows")

    completed = 0
    for index, pair in enumerate(cohort):
        keys = [(str(pair["pair_id"]), str(pair[f"{side}_arm"])) for side in ("source", "target")]
        if all(key in state for key in keys):
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        for side, prepared in (("source", source), ("target", target)):
            arm = str(pair[f"{side}_arm"])
            generated = _generate_from_inputs(
                cfg,
                model,
                processor,
                prepared["inputs"],
                refusal_ids,
                nonrefusal_ids,
                max_new_tokens=cfg.exp3.max_new_tokens,
            )
            frozen = frozen_behavior[(str(pair["item_id"]), arm)]
            state[(str(pair["pair_id"]), arm)] = {
                "schema_version": cfg.exp4.schema_version,
                "stage": "physical_endpoints",
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "arm": arm,
                "side": side,
                "frozen_selection_role": pair.get("selection_role"),
                "response": generated["response"],
                "explicit_refusal": bool(generated["explicit_refusal"]),
                "r_tab_margin": generated["r_tab_margin"],
                "r_tab_margin_raw": generated["r_tab_margin_raw"],
                "generated_token_ids": generated["generated_token_ids"],
                "frozen_response": frozen["response"],
                "frozen_explicit_refusal": bool(frozen["explicit_refusal"]),
                "marker_agrees_with_frozen": bool(generated["explicit_refusal"])
                == bool(frozen["explicit_refusal"]),
                "text_agrees_with_frozen": generated["response"] == frozen["response"],
            }
            completed += 1
        _checkpoint_mapping(state, endpoints_path, completed_since_resume=completed, every=10)
        print(
            f"[exp4] endpoints pair {index + 1}/{len(cohort)} measured={completed}",
            flush=True,
        )

    expected = {
        (str(pair["pair_id"]), str(pair[f"{side}_arm"]))
        for pair in cohort
        for side in ("source", "target")
    }
    if set(state) != expected:
        raise RuntimeError("Exp4 physical endpoints do not exactly cover the frozen cohort")
    _checkpoint_mapping(state, endpoints_path, completed_since_resume=completed, force=True)
    return [state[key] for key in sorted(state)]


def endpoint_index(
    cfg: Exp4RunConfig, run_dir: Path
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, str]]:
    """Load current-device endpoints and the roles they imply, per pair."""

    endpoints_path = _artifact(run_dir, cfg.exp4.artifacts.endpoints_file)
    if not endpoints_path.is_file():
        raise FileNotFoundError(
            "Exp4 physical endpoints are missing; run the 'endpoints' stage first"
        )
    rows = load_jsonl(endpoints_path)
    index = {(str(row["pair_id"]), str(row["arm"])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("Exp4 endpoint records contain duplicate pair/arm rows")
    sides: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in rows:
        sides[str(row["pair_id"])][str(row["side"])] = bool(row["explicit_refusal"])
    roles = {}
    for pair_id, by_side in sides.items():
        if set(by_side) != {"source", "target"}:
            raise ValueError(f"Exp4 endpoints for {pair_id} do not cover both arms")
        roles[pair_id] = _endpoint_role(by_side["source"], by_side["target"])
    return index, roles


def summarize_endpoints(cfg: Exp4RunConfig, run_dir: Path) -> dict[str, Any]:
    """Frozen-vs-current endpoint transition table, recorded with every run."""

    index, roles = endpoint_index(cfg, run_dir)
    rows = list(index.values())
    frozen_roles = {str(row["pair_id"]): str(row["frozen_selection_role"]) for row in rows}
    transitions: dict[str, int] = defaultdict(int)
    for pair_id, role in roles.items():
        transitions[f"{frozen_roles[pair_id]}->{role}"] += 1
    return {
        "endpoint_policy": cfg.exp4.endpoint_policy,
        "n_arms": len(rows),
        "marker_agrees_with_frozen": sum(bool(row["marker_agrees_with_frozen"]) for row in rows),
        "text_agrees_with_frozen": sum(bool(row["text_agrees_with_frozen"]) for row in rows),
        "n_pairs": len(roles),
        "current_discordant": sum(role == "discordant" for role in roles.values()),
        "frozen_discordant": sum(role == "discordant" for role in frozen_roles.values()),
        "role_transitions": dict(sorted(transitions.items())),
        "changed_pairs": sorted(
            pair_id for pair_id, role in roles.items() if role != frozen_roles[pair_id]
        ),
    }


def run_cache_routing(
    cfg: Exp4RunConfig,
    paths: ResolvedPaths,
    run_dir: Path,
    model: Any,
    processor: Any,
    *,
    shard_index: int = 0,
    shard_count: int = 1,
) -> list[dict[str, Any]]:
    """Run the frozen pair x direction x 2x2 post-prefill intervention.

    Sharding only partitions which cohort pairs this process executes. Pairs are
    causally independent, global cell numbering is preserved, and every shard
    writes its own checkpoint, so a sharded run reproduces the single-process
    records exactly once `merge_shard_records` has validated coverage.
    """

    _validate_shard(shard_index, shard_count)
    cohort_path = _artifact(run_dir, cfg.exp4.artifacts.cohort_file)
    if not cohort_path.is_file():
        raise FileNotFoundError("Exp4 cohort is not frozen; run preflight first")
    cohort = load_jsonl(cohort_path)
    assigned = shard_assignment(cohort, shard_index=shard_index, shard_count=shard_count)
    source_dir = paths.output_dir / cfg.exp4.source_exp3_run
    frozen_behavior = _behavior_index(cfg, source_dir)
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    if cfg.exp4.endpoint_policy == "current_hardware":
        endpoints, current_roles = endpoint_index(cfg, run_dir)
    else:
        endpoints, current_roles = {}, {}

    from audio_safety.models.hooks import get_decoder_layers

    n_layers = len(get_decoder_layers(model))
    if cfg.exp4.inject_layer >= n_layers or cfg.exp4.relay_start_layer >= n_layers:
        raise ValueError(f"Exp4 layers must lie inside the model's {n_layers} decoder blocks")
    clamp_layers = list(range(cfg.exp4.relay_start_layer, n_layers))
    records_path = shard_records_path(
        cfg, run_dir, shard_index=shard_index, shard_count=shard_count
    )
    existing = load_jsonl(records_path) if records_path.is_file() else []
    state = {
        (str(row["pair_id"]), str(row["direction"]), str(row["condition"])): row for row in existing
    }
    if len(state) != len(existing):
        raise ValueError("Exp4 checkpoint contains duplicate pair/direction/condition rows")

    completed = 0
    for shard_position, (pair_index, pair) in enumerate(assigned):
        pending_pair = any(
            (str(pair["pair_id"]), direction, condition) not in state
            for direction in cfg.exp4.directions
            for condition in cfg.exp4.conditions
        )
        if not pending_pair:
            continue
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        audio_positions = [int(value) for value in source["audio_positions"]]
        if audio_positions != list(range(audio_positions[0], audio_positions[-1] + 1)):
            raise RuntimeError(f"audio span is non-contiguous for {pair['pair_id']}")
        t_ab = int(source["t_ab"])
        relay_positions = list(range(audio_positions[-1] + 1, t_ab))
        if len(relay_positions) != cfg.exp4.expected_relay_positions:
            raise RuntimeError(
                f"{pair['pair_id']} has {len(relay_positions)} relay positions, "
                f"expected {cfg.exp4.expected_relay_positions}"
            )

        source_bundle = _physical_prefill(
            cfg,
            model,
            source,
            relay_positions=relay_positions,
            clamp_layers=clamp_layers,
        )
        target_bundle = _physical_prefill(
            cfg,
            model,
            target,
            relay_positions=relay_positions,
            clamp_layers=clamp_layers,
        )

        for direction_index, direction in enumerate(cfg.exp4.directions):
            pending = [
                condition
                for condition in cfg.exp4.conditions
                if (str(pair["pair_id"]), direction, condition) not in state
            ]
            if not pending:
                continue
            host, donor, host_bundle, donor_bundle = _oriented_prefill(
                direction, source, target, source_bundle, target_bundle
            )
            host_arm = (
                str(pair["source_arm"])
                if direction == "source_to_target"
                else str(pair["target_arm"])
            )
            donor_arm = (
                str(pair["target_arm"])
                if direction == "source_to_target"
                else str(pair["source_arm"])
            )
            frozen_baseline = frozen_behavior[(str(pair["item_id"]), host_arm)]
            frozen_donor = frozen_behavior[(str(pair["item_id"]), donor_arm)]
            if cfg.exp4.endpoint_policy == "current_hardware":
                baseline = endpoints[(str(pair["pair_id"]), host_arm)]
                donor_baseline = endpoints[(str(pair["pair_id"]), donor_arm)]
                selection_role = current_roles[str(pair["pair_id"])]
            else:
                baseline, donor_baseline = frozen_baseline, frozen_donor
                selection_role = str(pair.get("selection_role"))

            injected_cache, injected_logits, hook_counts = _injected_prefill(
                cfg,
                model,
                host,
                donor_audio_state=donor_bundle.audio_state,
                host_relay_states=host_bundle.relay_states,
                relay_positions=relay_positions,
                clamp_layers=clamp_layers,
            )
            # y1 must be selected by the same rule `generate` uses, penalty and
            # float32 cast included; a raw bfloat16 argmax is a different decoder.
            processors = greedy_logits_processors(model)
            prompt_ids = host["inputs"]["input_ids"]
            fixed_y1 = select_next_token(processors, prompt_ids, injected_logits)
            host_y1 = select_next_token(processors, prompt_ids, host_bundle.last_logits)
            donor_y1 = select_next_token(
                processors, donor["inputs"]["input_ids"], donor_bundle.last_logits
            )
            first_margin = _margin(injected_logits, refusal_ids, nonrefusal_ids)

            clone_error = cache_max_abs_diff(clone_kv_cache(injected_cache), injected_cache)
            if clone_error != 0.0:
                raise RuntimeError(f"KV cache clone is not exact (max error {clone_error})")
            fully_closed = _condition_cache(
                "audio_host__tab_host",
                injected_cache=injected_cache,
                host_cache=host_bundle.cache,
                audio_positions=audio_positions,
                t_ab=t_ab,
            )
            closure_error = cache_max_abs_diff(fully_closed, host_bundle.cache)
            if closure_error > cfg.exp4.cache_atol:
                raise RuntimeError(
                    "audio+t_AB transplant did not reconstruct the full physical host cache; "
                    f"max error={closure_error:.6g}"
                )
            del fully_closed

            # Counted within the shard, so every independently loaded model
            # replica proves its own decoder against `generate`. With one shard
            # this is exactly the pre-registered single-process assignment.
            check_standard = (
                shard_position * len(cfg.exp4.directions) + direction_index
                < cfg.exp4.standard_generate_checks
            )
            standard_result = None
            if check_standard:
                standard_result = _generate_from_inputs(
                    cfg,
                    model,
                    processor,
                    host["inputs"],
                    refusal_ids,
                    nonrefusal_ids,
                    contexts=_fresh_interventions(
                        cfg,
                        model,
                        host,
                        donor_bundle,
                        host_bundle,
                        relay_positions,
                        clamp_layers,
                    ),
                    max_new_tokens=cfg.exp4.max_new_tokens,
                )

            for condition in pending:
                cell_cache = _condition_cache(
                    condition,
                    injected_cache=injected_cache,
                    host_cache=host_bundle.cache,
                    audio_positions=audio_positions,
                    t_ab=t_ab,
                )
                _a_src, _t_src, y1_source = condition_parts(condition)
                cell_y1 = fixed_y1 if y1_source == "injected" else host_y1
                generated = manual_greedy_decode(
                    cfg,
                    model,
                    processor,
                    cache=cell_cache,
                    prompt_attention_mask=host["inputs"]["attention_mask"],
                    prompt_input_ids=host["inputs"]["input_ids"],
                    first_token_id=cell_y1,
                )
                standard_exact = None
                standard_margin_error = None
                if check_standard and condition == "audio_injected__tab_injected":
                    assert standard_result is not None
                    # Token IDs, not decoded text: distinct token sequences can
                    # decode to the same string once specials are stripped.
                    standard_exact = (
                        generated["generated_token_ids"] == standard_result["generated_token_ids"]
                        and generated["response"] == standard_result["response"]
                    )
                    standard_margin_error = abs(first_margin - standard_result["r_tab_margin_raw"])
                    if not standard_exact or standard_margin_error > cfg.exp4.cache_atol:
                        raise RuntimeError(
                            "manual II decoding did not reproduce standard model.generate"
                        )

                # Gate 8a (design.md §9, 2026-07-31): the reference is an
                # independently recomputed physical-host generation on THIS
                # replica, not the source run's text. Comparing across devices
                # tests floating-point portability, not cache closure.
                # Under a host cache AND a host y1 the cell is the physical host
                # computation, so it must reproduce it regardless of whether the
                # injection moved y1. With an injected y1 the check only applies
                # where the injection left y1 unchanged (Exp4 gate 8a).
                host_reproduction_exact = None
                if (
                    condition_code(condition).startswith("HH")
                    and (y1_source == "host" or fixed_y1 == host_y1)
                    and cfg.exp4.max_new_tokens == cfg.exp3.max_new_tokens
                ):
                    host_reproduction_exact = (
                        generated["generated_token_ids"] == baseline["generated_token_ids"]
                        and generated["response"] == baseline["response"]
                    )
                    if not host_reproduction_exact:
                        raise RuntimeError(
                            "fully host-closed cell did not reproduce the current-device "
                            f"physical host generation for {pair['pair_id']} {direction} "
                            f"{condition}"
                        )

                audio_source, tab_source, _ = condition_parts(condition)
                key = (str(pair["pair_id"]), direction, condition)
                state[key] = {
                    "schema_version": cfg.exp4.schema_version,
                    "stage": "post_prefill_cache_routing",
                    "pair_id": pair["pair_id"],
                    "item_id": pair["item_id"],
                    "role": pair["role"],
                    "category_name": pair.get("category_name"),
                    "transition": pair.get("transition"),
                    "selection_role": selection_role,
                    "frozen_selection_role": pair.get("selection_role"),
                    "endpoint_policy": cfg.exp4.endpoint_policy,
                    "frozen_baseline_explicit_refusal": bool(frozen_baseline["explicit_refusal"]),
                    "frozen_donor_explicit_refusal": bool(frozen_donor["explicit_refusal"]),
                    "contrast": cfg.exp4.contrast,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "direction": direction,
                    "condition": condition,
                    "condition_code": condition_code(condition),
                    "host_arm": host_arm,
                    "donor_arm": donor_arm,
                    "audio_cache_source": audio_source,
                    "t_ab_cache_source": tab_source,
                    "y1_source": y1_source,
                    "cell_y1_id": cell_y1,
                    "inject_layer": cfg.exp4.inject_layer,
                    "clamp_layers": [clamp_layers[0], clamp_layers[-1]],
                    "n_audio_positions": len(audio_positions),
                    "n_relay_positions": len(relay_positions),
                    "t_ab": t_ab,
                    "fixed_y1_id": fixed_y1,
                    "host_y1_id": host_y1,
                    "donor_y1_id": donor_y1,
                    "fixed_y1_text": processor.tokenizer.decode([fixed_y1]),
                    "injected_y1_changed_from_host": fixed_y1 != host_y1,
                    "baseline_response": baseline["response"],
                    "baseline_explicit_refusal": bool(baseline["explicit_refusal"]),
                    "donor_response": donor_baseline["response"],
                    "donor_explicit_refusal": bool(donor_baseline["explicit_refusal"]),
                    "response": generated["response"],
                    "explicit_refusal": generated["explicit_refusal"],
                    "generated_token_ids": generated["generated_token_ids"],
                    "injected_r_tab_margin": first_margin,
                    "hook_counts": hook_counts,
                    "hook_counts_ok": all(count == 1 for count in hook_counts),
                    "cache_clone_max_abs_error": clone_error,
                    "full_host_cache_max_abs_error": (
                        closure_error if condition == "audio_host__tab_host" else None
                    ),
                    "standard_generate_checked": (
                        check_standard and condition == "audio_injected__tab_injected"
                    ),
                    "standard_generate_exact": standard_exact,
                    "standard_margin_abs_error": standard_margin_error,
                    "host_reproduction_exact": host_reproduction_exact,
                }
                completed += 1
                _checkpoint_mapping(
                    state,
                    records_path,
                    completed_since_resume=completed,
                    every=8,
                )
                del cell_cache

            # Gate 6 (Exp5 design.md §3): where the injection left y1 unchanged
            # the two y1 levels are the same intervention, so their cells must be
            # token-identical. A mismatch means the y1 factor is not wired to the
            # decode it claims to control.
            if fixed_y1 == host_y1:
                for condition in cfg.exp4.conditions:
                    if not condition.endswith(Y1_HOST_SUFFIX):
                        continue
                    injected_key = (
                        str(pair["pair_id"]),
                        direction,
                        condition[: -len(Y1_HOST_SUFFIX)],
                    )
                    host_key = (str(pair["pair_id"]), direction, condition)
                    if injected_key not in state or host_key not in state:
                        continue
                    if (
                        state[injected_key]["generated_token_ids"]
                        != state[host_key]["generated_token_ids"]
                    ):
                        raise RuntimeError(
                            "y1-degenerate cells diverged despite an identical first token: "
                            f"{pair['pair_id']} {direction} {condition}"
                        )
        print(
            f"[exp4] shard {shard_index}/{shard_count} "
            f"pair {pair_index + 1}/{len(cohort)} completed_cells={completed}",
            flush=True,
        )

    expected = _factorial_keys(cfg, [pair for _index, pair in assigned])
    actual = set(state)
    if expected != actual:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"Exp4 checkpoint is incomplete/incompatible: missing={missing[:5]}, extra={extra[:5]}"
        )
    _checkpoint_mapping(state, records_path, completed_since_resume=completed, force=True)

    rows = [state[key] for key in sorted(state)]
    if any(not row["hook_counts_ok"] for row in rows):
        raise RuntimeError("at least one Exp4 row has a failed hook-count gate")
    if any(float(row["cache_clone_max_abs_error"]) != 0.0 for row in rows):
        raise RuntimeError("at least one Exp4 row has an inexact cache clone")
    hh_rows = [row for row in rows if row["condition"] == "audio_host__tab_host"]
    if any(float(row["full_host_cache_max_abs_error"]) > cfg.exp4.cache_atol for row in hh_rows):
        raise RuntimeError("at least one Exp4 HH row failed full host-cache closure")
    designated = [
        row
        for row in rows
        if row.get("standard_generate_checked")
        and row["condition"] == "audio_injected__tab_injected"
    ]
    expected_checks = _expected_standard_checks(
        cfg, cohort, shard_index=shard_index, shard_count=shard_count
    )
    if len(designated) != expected_checks or any(
        row.get("standard_generate_exact") is not True for row in designated
    ):
        raise RuntimeError("Exp4 standard-generate equivalence gate is incomplete")
    return rows


def _mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_bootstrap: int,
    ci_alpha: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"estimate": None, "ci_low": None, "ci_high": None, "n_pairs": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [ci_alpha / 2.0, 1.0 - ci_alpha / 2.0])
    return {
        "estimate": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n_pairs": len(array),
    }


def _effect_decision(cfg: Exp4RunConfig, summary: Mapping[str, Any]) -> str:
    estimate, low, high = summary["estimate"], summary["ci_low"], summary["ci_high"]
    if estimate is None or low is None or high is None:
        return "not_estimable"
    if high < 0.0:
        return "reverse_effect"
    if estimate >= cfg.exp4.material_effect_min and low > 0.0:
        return "material_conditional_contributor"
    if high < cfg.exp4.negligible_effect_max:
        return "negligible_at_pilot_resolution"
    return "ambiguous"


def _donorward(row: Mapping[str, Any]) -> float:
    host = float(bool(row["baseline_explicit_refusal"]))
    donor = float(bool(row["donor_explicit_refusal"]))
    if host == donor:
        raise ValueError("donorward score is undefined for a stable physical pair")
    outcome = float(bool(row["explicit_refusal"]))
    return (outcome - host) * (donor - host)


def _clustered_values(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pair_id"])].append(_donorward(row))
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def _clustered_host_marker_change(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        host = bool(row["baseline_explicit_refusal"])
        grouped[str(row["pair_id"])].append(float(bool(row["explicit_refusal"]) != host))
    return [float(np.mean(values)) for _, values in sorted(grouped.items())]


def analyze_records(cfg: Exp4RunConfig, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute pair-clustered cell means and frozen 2x2 causal contrasts."""

    primary = [row for row in rows if row.get("selection_role") == "discordant"]
    if not primary:
        raise ValueError("Exp4 analysis has no discordant primary rows")
    direction_cells: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in primary:
        key = (str(row["pair_id"]), str(row["direction"]))
        condition = str(row["condition"])
        if condition in direction_cells[key]:
            raise ValueError(f"duplicate primary Exp4 cell {key + (condition,)}")
        direction_cells[key][condition] = _donorward(row)
    expected_conditions = set(cfg.exp4.conditions)
    incomplete = [
        key
        for key, direction_values in direction_cells.items()
        if set(direction_values) != expected_conditions
    ]
    if incomplete:
        raise ValueError(f"incomplete primary Exp4 2x2 cells: {incomplete[:5]}")

    # Exp5 primary population: directions where the injection actually moved y1.
    # Where it did not, the y1 factor is degenerate and carries no contrast.
    has_y1_factor = any(
        condition_parts(condition)[2] == "host" for condition in cfg.exp4.conditions
    )
    y1_changed_by_direction = {
        (str(row["pair_id"]), str(row["direction"])): bool(row["injected_y1_changed_from_host"])
        for row in primary
    }
    if has_y1_factor:
        restricted = {
            key: values for key, values in direction_cells.items() if y1_changed_by_direction[key]
        }
        if not restricted:
            raise ValueError("Exp5 analysis has no changed-y1 primary directions")
        unchanged_cells = {
            key: values
            for key, values in direction_cells.items()
            if not y1_changed_by_direction[key]
        }
        direction_cells = restricted
    else:
        unchanged_cells = {}

    pair_vectors: dict[str, list[dict[str, float]]] = defaultdict(list)
    for (pair_id, _direction), direction_values in direction_cells.items():
        pair_vectors[pair_id].append(direction_values)
    frozen_directions = set(cfg.exp4.directions)
    observed_by_pair = {
        pair_id: {str(row["direction"]) for row in primary if str(row["pair_id"]) == pair_id}
        for pair_id in pair_vectors
    }
    missing_directions = [
        pair_id for pair_id, observed in observed_by_pair.items() if observed != frozen_directions
    ]
    if missing_directions:
        raise ValueError(f"primary pairs lack a frozen direction: {missing_directions[:5]}")

    pair_means: dict[str, dict[str, float]] = {}
    for pair_id, vectors in pair_vectors.items():
        pair_means[pair_id] = {
            condition: float(np.mean([vector[condition] for vector in vectors]))
            for condition in cfg.exp4.conditions
        }
    ordered = [pair_means[pair_id] for pair_id in sorted(pair_means)]
    # Exp4 keeps its two-letter cache codes so its metrics stay comparable;
    # Exp5 uses three-letter codes that name the y1 level as well.
    by_code = {
        (condition_code(condition) if has_y1_factor else CONDITION_CODES[condition]): condition
        for condition in cfg.exp4.conditions
    }

    cells: dict[str, dict[str, Any]] = {}
    for code, condition in sorted(by_code.items()):
        cells[f"T_{code}"] = _mean_ci(
            [vector[condition] for vector in ordered],
            seed=_stable_int(cfg.seed, "exp4", code),
            n_bootstrap=cfg.exp4.n_bootstrap,
            ci_alpha=cfg.exp4.ci_alpha,
        )

    def diff(left: str, right: str) -> list[float]:
        return [vector[by_code[left]] - vector[by_code[right]] for vector in ordered]

    if has_y1_factor:
        contrast_values = {
            # Exp5 design.md §0 primary
            "D_joint_y1H": diff("IIH", "HHH"),
            "D_y1_at_HH": diff("HHI", "HHH"),
            # secondary route contrasts under a host first token
            "D_audio_y1H": diff("IIH", "HIH"),
            "D_tAB_y1H": diff("IIH", "IHH"),
            # y1 simple effects at each cache state
            "D_y1_at_II": diff("III", "IIH"),
            "D_y1_at_HI": diff("HII", "HIH"),
            "D_y1_at_IH": diff("IHI", "IHH"),
            # the Exp4 cache contrasts, recomputed on this population
            "D_audio_y1I": diff("III", "HII"),
            "D_tAB_y1I": diff("III", "IHI"),
            "D_joint_y1I": diff("III", "HHI"),
            "three_way_interaction": [
                (a - b - c + d) - (e - f - g + h)
                for a, b, c, d, e, f, g, h in zip(
                    *(
                        [vector[by_code[code]] for vector in ordered]
                        for code in ("III", "HII", "IHI", "HHI", "IIH", "HIH", "IHH", "HHH")
                    ),
                    strict=True,
                )
            ],
        }
        decision_names = {"D_joint_y1H", "D_y1_at_HH"}
    else:
        contrast_values = {
            "D_audio": diff("II", "HI"),
            "D_tAB": diff("II", "IH"),
            "D_joint": diff("II", "HH"),
            "D_audio_given_tAB_host": diff("IH", "HH"),
            "D_tAB_given_audio_host": diff("HI", "HH"),
            "factorial_interaction": [
                a - b - c + d
                for a, b, c, d in zip(
                    *(
                        [vector[by_code[code]] for vector in ordered]
                        for code in ("II", "HI", "IH", "HH")
                    ),
                    strict=True,
                )
            ],
        }
        decision_names = {"D_audio", "D_tAB", "D_joint"}
    contrasts = {}
    for name, values in contrast_values.items():
        summary = _mean_ci(
            values,
            seed=_stable_int(cfg.seed, "exp4", name),
            n_bootstrap=cfg.exp4.n_bootstrap,
            ci_alpha=cfg.exp4.ci_alpha,
        )
        if name in decision_names:
            summary["decision"] = _effect_decision(cfg, summary)
        contrasts[name] = summary

    # Descriptive: the cache contrasts in the subpopulation where y1 was already
    # unchanged, i.e. where the cache is the only channel for injected content.
    unchanged_y1_cells: dict[str, dict[str, Any]] = {}
    if has_y1_factor and unchanged_cells:
        grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
        for (pair_id, _direction), values in unchanged_cells.items():
            grouped[pair_id].append(values)
        vectors = [
            {
                condition: float(np.mean([v[condition] for v in per_pair]))
                for condition in cfg.exp4.conditions
            }
            for _pair_id, per_pair in sorted(grouped.items())
        ]
        for code, condition in sorted(by_code.items()):
            unchanged_y1_cells[f"T_{code}"] = _mean_ci(
                [vector[condition] for vector in vectors],
                seed=_stable_int(cfg.seed, "exp5", "unchanged", code),
                n_bootstrap=cfg.exp4.n_bootstrap,
                ci_alpha=cfg.exp4.ci_alpha,
            )

    hh_primary = [row for row in primary if row["condition"] == "audio_host__tab_host"]
    y1_strata = {}
    for changed in (False, True):
        stratum = [
            row for row in hh_primary if bool(row["injected_y1_changed_from_host"]) is changed
        ]
        y1_strata["changed" if changed else "unchanged"] = _mean_ci(
            _clustered_values(stratum),
            seed=_stable_int(cfg.seed, "exp4", "y1", changed),
            n_bootstrap=cfg.exp4.n_bootstrap,
            ci_alpha=cfg.exp4.ci_alpha,
        )

    stable = [row for row in rows if row.get("selection_role") != "discordant"]
    if any(
        bool(row["baseline_explicit_refusal"]) != bool(row["donor_explicit_refusal"])
        for row in stable
    ):
        raise ValueError("a non-discordant Exp4 row has discordant physical endpoints")
    stable_controls = {}
    for condition in cfg.exp4.conditions:
        condition_rows = [row for row in stable if row["condition"] == condition]
        code = condition_code(condition) if has_y1_factor else CONDITION_CODES[condition]
        stable_controls[code] = _mean_ci(
            _clustered_host_marker_change(condition_rows),
            seed=_stable_int(cfg.seed, "exp4", "stable", condition),
            n_bootstrap=cfg.exp4.n_bootstrap,
            ci_alpha=cfg.exp4.ci_alpha,
        )

    standard_rows = [row for row in rows if row.get("standard_generate_checked")]
    # Closure error is recorded once per direction on the injected-y1 HH cell;
    # host reproduction is checked on every fully host-closed cell.
    closure_rows = [row for row in rows if row.get("full_host_cache_max_abs_error") is not None]
    hh_rows = [row for row in rows if row.get("host_reproduction_exact") is not None]
    return {
        "schema_version": cfg.exp4.schema_version,
        "endpoint": "frozen literal explicit-refusal marker",
        "estimand_scope": (
            "y2 onward, changed-y1 directions, y1 randomised as a third factor"
            if has_y1_factor
            else "y2 onward conditional on one fixed injected y1"
        ),
        "design": "exp5.2x2x2" if has_y1_factor else "exp4.2x2",
        "n_rows": len(rows),
        "n_primary_pairs": len(pair_means),
        "n_primary_directions": len(direction_cells),
        "n_unchanged_y1_directions": len(unchanged_cells),
        "n_directions_per_pair": len(cfg.exp4.directions),
        "cells": cells,
        "unchanged_y1_cells": unchanged_y1_cells,
        "contrasts": contrasts,
        "hh_by_injected_y1_change": y1_strata,
        "stable_control_host_marker_change": stable_controls,
        "integrity": {
            "all_hook_counts_ok": all(bool(row.get("hook_counts_ok")) for row in rows),
            "max_cache_clone_abs_error": max(
                float(row.get("cache_clone_max_abs_error", float("inf"))) for row in rows
            ),
            "max_full_host_cache_abs_error": max(
                float(row["full_host_cache_max_abs_error"]) for row in closure_rows
            ),
            "standard_generate_checks": len(standard_rows),
            "all_standard_generate_exact": all(
                row.get("standard_generate_exact") is True for row in standard_rows
            ),
            "host_reproduction_checks": sum(
                row.get("host_reproduction_exact") is not None for row in hh_rows
            ),
            "all_host_reproductions_exact": all(
                row.get("host_reproduction_exact") is not False for row in hh_rows
            ),
        },
        "thresholds": {
            "material_effect_min": cfg.exp4.material_effect_min,
            "negligible_effect_max": cfg.exp4.negligible_effect_max,
            "ci_alpha": cfg.exp4.ci_alpha,
            "n_bootstrap": cfg.exp4.n_bootstrap,
        },
        "non_claim": "conditional capacities are not unique mediation shares",
    }


def _format_interval(summary: Mapping[str, Any]) -> str:
    if summary.get("estimate") is None:
        return "not estimable"
    return (
        f"{float(summary['estimate']):.3f} "
        f"[{float(summary['ci_low']):.3f}, {float(summary['ci_high']):.3f}]"
    )


def _render_analysis(metrics: Mapping[str, Any]) -> str:
    lines = [
        "# Exp4 cache-routing analysis",
        "",
        "> Generated from the frozen records. Conditional on the injected first token `y1`; ",
        "> contrasts are causal capacities, not mediation shares.",
        "",
        "## Integrity",
        "",
        "```json",
        yaml.safe_dump(dict(metrics["integrity"]), sort_keys=False).strip(),
        "```",
        "",
        "## Donorward cell means",
        "",
        "| cell | estimate [95% CI] |",
        "|---|---:|",
    ]
    for name, summary in metrics["cells"].items():
        lines.append(f"| `{name}` | {_format_interval(summary)} |")
    lines.extend(
        [
            "",
            "## Frozen contrasts",
            "",
            "| contrast | estimate [95% CI] | decision |",
            "|---|---:|---|",
        ]
    )
    for name, summary in metrics["contrasts"].items():
        lines.append(
            f"| `{name}` | {_format_interval(summary)} | "
            f"{summary.get('decision', 'descriptive only')} |"
        )
    if metrics.get("unchanged_y1_cells"):
        lines.extend(
            [
                "",
                "## Descriptive: unchanged-`y1` directions",
                "",
                "The subpopulation in which the cache is the only channel for injected ",
                "content. Not a pre-registered estimand.",
                "",
                "| cell | donorward [95% CI] |",
                "|---|---:|",
            ]
        )
        for name, summary in metrics["unchanged_y1_cells"].items():
            lines.append(f"| `{name}` | {_format_interval(summary)} |")
    lines.extend(
        [
            "",
            "## All-host cache (`T_HH`) by injected y1 change",
            "",
            "| injected y1 vs host y1 | donorward [95% CI] |",
            "|---|---:|",
        ]
    )
    for name, summary in metrics["hh_by_injected_y1_change"].items():
        lines.append(f"| {name} | {_format_interval(summary)} |")
    lines.extend(
        [
            "",
            "## Stable-pair controls",
            "",
            "| cache cell | marker changed from physical host [95% CI] |",
            "|---|---:|",
        ]
    )
    for name, summary in metrics["stable_control_host_marker_change"].items():
        lines.append(f"| `{name}` | {_format_interval(summary)} |")
    lines.extend(
        [
            "",
            "`D_audio` tests decode-time audio-cache rereading; `D_tAB` tests cached ",
            "`t_AB` broadcast; `D_joint` tests their combined prompt-cache capacity. ",
            "Residual donorward behaviour in `T_HH` is attributable to the fixed `y1` ",
            "and subsequent autoregressive continuation under the validated cache closure.",
            "",
        ]
    )
    return "\n".join(lines)


def analyze_cache_routing(cfg: Exp4RunConfig, run_dir: Path) -> dict[str, Any]:
    records_path = _artifact(run_dir, cfg.exp4.artifacts.records_file)
    if not records_path.is_file():
        raise FileNotFoundError("Exp4 cache-routing records are missing")
    cohort_path = _artifact(run_dir, cfg.exp4.artifacts.cohort_file)
    if not cohort_path.is_file():
        raise FileNotFoundError("Exp4 frozen cohort is missing")
    rows = load_jsonl(records_path)
    cohort = load_jsonl(cohort_path)
    keys = [(str(row["pair_id"]), str(row["direction"]), str(row["condition"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Exp4 records contain duplicate cells")
    expected = {
        (str(pair["pair_id"]), direction, condition)
        for pair in cohort
        for direction in cfg.exp4.directions
        for condition in cfg.exp4.conditions
    }
    if set(keys) != expected:
        raise RuntimeError("Exp4 records do not exactly cover the frozen cohort factorial")
    hh_rows = [row for row in rows if row["condition"] == "audio_host__tab_host"]
    standard_rows = [row for row in rows if row.get("standard_generate_checked")]
    shard_counts = {int(row.get("shard_count", 1)) for row in rows}
    if len(shard_counts) != 1:
        raise RuntimeError(f"Exp4 records disagree on shard_count: {sorted(shard_counts)}")
    expected_checks = total_expected_standard_checks(cfg, cohort, shard_count=shard_counts.pop())
    integrity_ok = (
        all(bool(row.get("hook_counts_ok")) for row in rows)
        and all(float(row.get("cache_clone_max_abs_error", float("inf"))) == 0.0 for row in rows)
        and all(
            float(row.get("full_host_cache_max_abs_error", float("inf"))) <= cfg.exp4.cache_atol
            for row in hh_rows
        )
        and len(standard_rows) == expected_checks
        and all(row.get("standard_generate_exact") is True for row in standard_rows)
        and all(
            float(row.get("standard_margin_abs_error", float("inf"))) <= cfg.exp4.cache_atol
            for row in standard_rows
        )
        and all(row.get("host_reproduction_exact") is not False for row in hh_rows)
    )
    if not integrity_ok:
        raise RuntimeError("Exp4 record integrity gates do not pass; analysis is invalid")
    metrics = analyze_records(cfg, rows)
    if cfg.exp4.endpoint_policy == "current_hardware":
        metrics["physical_endpoints"] = summarize_endpoints(cfg, run_dir)
    save_json(metrics, _artifact(run_dir, cfg.exp4.artifacts.metrics_file))
    analysis_path = _artifact(run_dir, cfg.exp4.artifacts.analysis_file)
    analysis_path.parent.mkdir(parents=True, exist_ok=True)
    analysis_path.write_text(_render_analysis(metrics), encoding="utf-8")
    return metrics
