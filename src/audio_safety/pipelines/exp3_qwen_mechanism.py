"""Config-driven Qwen2-Audio Exp3 factor-to-path mechanism pipeline.

Heavy torch/transformers imports stay inside GPU functions. CPU stages can build
and analyze manifests in a base environment, as required by AGENTS.md.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from audio_safety.config.schema import Exp3ContrastConfig, Exp3RunConfig
from audio_safety.evaluation.asr_faithfulness import atomic_save_jsonl
from audio_safety.evaluation.refusal_instability import (
    EXPLICIT_REFUSAL,
    classify_explicit_refusal,
    normalize_refusal_text,
    select_transition_cohort,
    summarize_paired_transitions,
)
from audio_safety.utils.io import load_jsonl, save_json
from audio_safety.utils.paths import ResolvedPaths
from audio_safety.utils.text import token_overlap, word_error_rate


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def refusal_matcher_sha(patterns: Sequence[str]) -> str:
    normalized = [normalize_refusal_text(pattern) for pattern in patterns]
    payload = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def behavior_contract_sha(cfg: Exp3RunConfig) -> str:
    """Hash every setting that can change a frozen physical-arm response."""

    payload = {
        "model": cfg.model.model_dump(mode="json"),
        "instruction": cfg.exp3.instruction,
        "system_prompt": cfg.exp3.system_prompt,
        "max_new_tokens": cfg.exp3.max_new_tokens,
        "decoding": "greedy",
        "matcher_sha256": refusal_matcher_sha(cfg.exp3.readout.refusal_patterns),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def runtime_behavior_contract_sha(
    cfg: Exp3RunConfig,
    model: Any,
    processor: Any,
) -> str:
    """Add package and resolved model/processor identities to the static contract."""

    import torch
    import transformers

    tokenizer = processor.tokenizer
    payload = {
        "static_contract_sha256": behavior_contract_sha(cfg),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_commit": getattr(tokenizer, "init_kwargs", {}).get("_commit_hash"),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "processor_class": f"{type(processor).__module__}.{type(processor).__qualname__}",
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _stable_int(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _artifact(run_dir: Path, relative: Path) -> Path:
    return run_dir / relative


def _checkpoint_mapping(
    state: Mapping[Any, Mapping[str, Any]],
    path: Path,
    *,
    completed_since_resume: int,
    every: int = 25,
    force: bool = False,
) -> None:
    """Atomically checkpoint in bounded batches, avoiding O(n²) per-cell I/O."""

    if every < 1:
        raise ValueError("checkpoint interval must be positive")
    if force or (completed_since_resume > 0 and completed_since_resume % every == 0):
        atomic_save_jsonl([state[key] for key in sorted(state)], path)


def _assert_expected_keys(
    stage: str,
    expected: set[Any],
    actual: set[Any],
) -> None:
    missing = sorted(expected - actual, key=str)
    extra = sorted(actual - expected, key=str)
    if missing or extra:
        raise RuntimeError(
            f"{stage} checkpoint is incomplete/incompatible: "
            f"missing={missing[:5]} ({len(missing)}), extra={extra[:5]} ({len(extra)})"
        )


def _enabled_contrasts(cfg: Exp3RunConfig) -> list[Exp3ContrastConfig]:
    return [contrast for contrast in cfg.exp3.contrasts if contrast.enabled]


def _contrast(cfg: Exp3RunConfig, name: str) -> Exp3ContrastConfig:
    matches = [contrast for contrast in cfg.exp3.contrasts if contrast.name == name]
    if len(matches) != 1:
        raise ValueError(f"expected one Exp3 contrast named {name!r}, got {len(matches)}")
    if not matches[0].enabled:
        raise ValueError(f"Exp3 contrast {name!r} is disabled")
    return matches[0]


def _resolve_audio_path(data_dir: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else data_dir / path


def _manifest_arm_paths(cfg: Exp3RunConfig, paths: ResolvedPaths) -> tuple[list[dict], dict]:
    clean_path = paths.data_dir / cfg.exp3.clean_manifest_file
    arm_path = paths.data_dir / cfg.exp3.arm_manifest_file
    if not clean_path.is_file():
        raise FileNotFoundError(f"clean manifest is missing: {clean_path}")
    if not arm_path.is_file():
        raise FileNotFoundError(f"arm manifest is missing: {arm_path}")
    clean_rows = load_jsonl(clean_path)
    arm_index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(arm_path):
        if row.get("status") not in {"rendered", "reused"}:
            continue
        key = (str(row.get("item_id") or ""), str(row.get("arm") or ""))
        if not all(key):
            raise ValueError(f"arm manifest row lacks item_id/arm: {row}")
        if key in arm_index:
            raise ValueError(f"duplicate arm manifest key: {key}")
        arm_index[key] = row
    return clean_rows, arm_index


def build_pair_manifest(
    cfg: Exp3RunConfig,
    paths: ResolvedPaths,
    run_dir: Path,
) -> list[dict[str, Any]]:
    """Join frozen Exp2 manifests into Exp3 paired acoustic contrasts."""

    clean_rows, arm_index = _manifest_arm_paths(cfg, paths)
    roles = set(cfg.exp3.behavior.roles)
    rows: list[dict[str, Any]] = []
    sha_cache: dict[Path, str] = {}

    for clean in clean_rows:
        item_id = str(clean.get("item_id") or "")
        role = str(clean.get("safety_label") or "")
        if not item_id or role not in roles:
            continue

        def arm_record(
            arm: str,
            clean_path_value: object = clean.get("path"),
            current_item: str = item_id,
        ) -> dict[str, Any] | None:
            if arm == "clean":
                return {"path": clean_path_value, "status": "clean_manifest"}
            return arm_index.get((current_item, arm))

        for contrast in _enabled_contrasts(cfg):
            source = arm_record(contrast.source_arm)
            target = arm_record(contrast.target_arm)
            if source is None or target is None or not source.get("path") or not target.get("path"):
                continue
            source_path = _resolve_audio_path(paths.data_dir, str(source["path"]))
            target_path = _resolve_audio_path(paths.data_dir, str(target["path"]))
            if not source_path.is_file() or not target_path.is_file():
                missing = source_path if not source_path.is_file() else target_path
                raise FileNotFoundError(f"paired audio is missing: {missing}")
            for audio_path in (source_path, target_path):
                if audio_path not in sha_cache:
                    sha_cache[audio_path] = _sha256_file(audio_path)
            rows.append(
                {
                    "schema_version": cfg.exp3.schema_version,
                    "pair_id": f"{contrast.name}:{item_id}",
                    "item_id": item_id,
                    "role": role,
                    "category_name": clean.get("category_name") or clean.get("category"),
                    "reference_text": clean.get("reference_text"),
                    "harmful_anchors": clean.get("harmful_anchors"),
                    "contrast": contrast.name,
                    "purpose": contrast.purpose,
                    "source_arm": contrast.source_arm,
                    "target_arm": contrast.target_arm,
                    "source_path": str(source_path),
                    "target_path": str(target_path),
                    "source_sha256": sha_cache[source_path],
                    "target_sha256": sha_cache[target_path],
                }
            )
    rows.sort(key=lambda row: (row["contrast"], row["role"], row["item_id"]))
    if not rows:
        raise ValueError("no complete Exp3 pairs were found")
    pair_ids = [row["pair_id"] for row in rows]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Exp3 pair_id collision")
    output_path = _artifact(run_dir, cfg.exp3.artifacts.pairs_file)
    if output_path.is_file():
        frozen = load_jsonl(output_path)
        if frozen != rows:
            raise RuntimeError("Exp3 pair manifest changed after it was frozen")
        return frozen
    atomic_save_jsonl(rows, output_path)
    return rows


def load_pairs(cfg: Exp3RunConfig, run_dir: Path) -> list[dict[str, Any]]:
    path = _artifact(run_dir, cfg.exp3.artifacts.pairs_file)
    if not path.is_file():
        raise FileNotFoundError(f"pair manifest is missing; run preflight first: {path}")
    return load_jsonl(path)


def _failure_mode(output: str, reference_text: str | None) -> str | None:
    from audio_safety.evaluation.labeling import _is_decoding_failure

    failed, mode = _is_decoding_failure(output, reference_text)
    return str(mode or "decoding_failure") if failed else None


def score_behavior_output(
    output: str,
    *,
    reference_text: str | None,
    patterns: Sequence[str],
    declared_failure: str | None = None,
) -> dict[str, Any]:
    """Return marker-only primary status plus a failure-aware sensitivity status."""

    marker_status = classify_explicit_refusal(output, patterns)
    failure = declared_failure or _failure_mode(output, reference_text)
    failure_status = f"decoding_failure:{failure}" if failure else None
    sensitivity_status = classify_explicit_refusal(
        output,
        patterns,
        decoding_failure_status=failure_status,
    )
    return {
        "marker_status": marker_status,
        "refusal_status": sensitivity_status,
        "explicit_refusal": marker_status == EXPLICIT_REFUSAL,
        "decoding_failure": failure_status,
    }


def _prior_behavior_index(cfg: Exp3RunConfig, paths: ResolvedPaths) -> dict[tuple[str, str], dict]:
    if not cfg.exp3.behavior.reuse_prior_generations:
        return {}
    prior_path = paths.output_dir / cfg.exp3.behavior.prior_generations_file
    if not prior_path.is_file():
        return {}
    index: dict[tuple[str, str], dict[str, Any]] = {}
    for row in load_jsonl(prior_path):
        key = (str(row.get("item_id") or ""), str(row.get("arm") or ""))
        if all(key):
            index[key] = row
    return index


def _behavior_tasks(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    tasks: dict[tuple[str, str], dict[str, Any]] = {}
    for pair in pairs:
        for side in ("source", "target"):
            key = (str(pair["item_id"]), str(pair[f"{side}_arm"]))
            candidate = {
                "item_id": key[0],
                "arm": key[1],
                "role": pair["role"],
                "category_name": pair.get("category_name"),
                "reference_text": pair.get("reference_text"),
                "path": pair[f"{side}_path"],
                "audio_sha256": pair[f"{side}_sha256"],
            }
            previous = tasks.get(key)
            if previous is not None and previous != candidate:
                raise ValueError(f"inconsistent behavior task metadata for {key}")
            tasks[key] = candidate
    return [tasks[key] for key in sorted(tasks)]


def collect_behavior(
    cfg: Exp3RunConfig,
    paths: ResolvedPaths,
    run_dir: Path,
    *,
    model: Any | None = None,
    processor: Any | None = None,
    generate_missing: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Import Exp2 generations, optionally filling missing arms with Qwen."""

    pairs = load_pairs(cfg, run_dir)
    if cfg.exp3.behavior.max_pairs_per_contrast is not None:
        limited: list[dict[str, Any]] = []
        for contrast in _enabled_contrasts(cfg):
            contrast_pairs = [row for row in pairs if row["contrast"] == contrast.name]
            limited.extend(
                _stratified_limit(
                    contrast_pairs,
                    cfg.exp3.behavior.max_pairs_per_contrast,
                    seed=cfg.seed,
                )
            )
        pairs = limited
    tasks = _behavior_tasks(pairs)
    prior = _prior_behavior_index(cfg, paths)
    output_path = _artifact(run_dir, cfg.exp3.artifacts.behavior_file)
    existing = load_jsonl(output_path) if output_path.is_file() else []
    state = {(str(row["item_id"]), str(row["arm"])): row for row in existing}
    matcher_sha = refusal_matcher_sha(cfg.exp3.readout.refusal_patterns)
    contract_sha = behavior_contract_sha(cfg)
    runtime_contract_sha = (
        runtime_behavior_contract_sha(cfg, model, processor)
        if model is not None and processor is not None
        else None
    )
    pending = 0
    completed = 0

    for task in tasks:
        key = (task["item_id"], task["arm"])
        current = state.get(key)
        if current is not None:
            expected = {
                "audio_sha256": task["audio_sha256"],
                "matcher_sha256": matcher_sha,
                "behavior_contract_sha256": contract_sha,
            }
            if runtime_contract_sha is not None:
                expected["runtime_behavior_contract_sha256"] = runtime_contract_sha
            mismatched = {
                field: (current.get(field), value)
                for field, value in expected.items()
                if current.get(field) != value
            }
            if mismatched:
                raise RuntimeError(f"frozen behavior row {key} violates its contract: {mismatched}")
            continue
        source = prior.get(key)
        response: str | None = None
        origin: str | None = None
        declared_failure: str | None = None
        if (
            source is not None
            and source.get("response") is not None
            and runtime_contract_sha is not None
        ):
            if source.get("audio_sha256") != task["audio_sha256"]:
                raise RuntimeError(f"prior behavior audio hash does not match {key}")
            if source.get("behavior_contract_sha256") != contract_sha:
                raise RuntimeError(f"prior behavior model/prompt contract does not match {key}")
            if source.get("runtime_behavior_contract_sha256") != runtime_contract_sha:
                raise RuntimeError(f"prior behavior runtime contract does not match {key}")
            response = str(source["response"])
            origin = "imported_verified_prior"
            if source.get("screen") == "decoding_failure":
                declared_failure = str(source.get("failure_mode") or "exp2_screen")
        elif generate_missing:
            if model is None or processor is None:
                raise ValueError("model and processor are required to generate missing behavior")
            from audio_safety.models.qwen2_audio import generate_audio_response

            response = generate_audio_response(
                model,
                processor,
                task["path"],
                cfg.exp3.instruction,
                max_new_tokens=cfg.exp3.max_new_tokens,
                system_prompt=cfg.exp3.system_prompt,
                do_sample=False,
            )
            origin = "generated_exp3"
        else:
            pending += 1
            continue
        score = score_behavior_output(
            response,
            reference_text=task.get("reference_text"),
            patterns=cfg.exp3.readout.refusal_patterns,
            declared_failure=declared_failure,
        )
        state[key] = {
            "schema_version": cfg.exp3.schema_version,
            **task,
            "response": response,
            "origin": origin,
            "matcher_sha256": matcher_sha,
            "behavior_contract_sha256": contract_sha,
            "runtime_behavior_contract_sha256": runtime_contract_sha,
            **score,
        }
        completed += 1
        _checkpoint_mapping(
            state,
            output_path,
            completed_since_resume=completed,
            every=10,
        )
    if pending == 0:
        _assert_expected_keys(
            "behavior", set((task["item_id"], task["arm"]) for task in tasks), set(state)
        )
    _checkpoint_mapping(
        state,
        output_path,
        completed_since_resume=completed,
        force=True,
    )
    rows = [state[key] for key in sorted(state)]
    analyze_behavior(cfg, run_dir, pairs=pairs, behavior_rows=rows)
    return rows, pending


def _paired_behavior_rows(
    pairs: Sequence[Mapping[str, Any]],
    behavior_rows: Sequence[Mapping[str, Any]],
    *,
    contrast: str,
    role: str,
    status_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = {(str(row["item_id"]), str(row["arm"])): row for row in behavior_rows}
    source: list[dict[str, Any]] = []
    target: list[dict[str, Any]] = []
    for pair in pairs:
        if pair["contrast"] != contrast or pair["role"] != role:
            continue
        source_row = index.get((str(pair["item_id"]), str(pair["source_arm"])))
        target_row = index.get((str(pair["item_id"]), str(pair["target_arm"])))
        if source_row is None or target_row is None:
            continue
        common = {
            "item_id": pair["item_id"],
            "role": pair["role"],
            "category_name": pair.get("category_name"),
        }
        source.append({**common, "refusal_status": source_row[status_key]})
        target.append({**common, "refusal_status": target_row[status_key]})
    return source, target


def _net_refusal_difference(
    source: Sequence[Mapping[str, Any]],
    target: Sequence[Mapping[str, Any]],
    *,
    seed: int,
    n_bootstrap: int,
    ci_alpha: float,
) -> dict[str, Any]:
    src = {str(row["item_id"]): row["refusal_status"] for row in source}
    tgt = {str(row["item_id"]): row["refusal_status"] for row in target}
    shared = sorted(src.keys() & tgt.keys())
    delta = np.asarray(
        [
            float(tgt[item] == EXPLICIT_REFUSAL) - float(src[item] == EXPLICIT_REFUSAL)
            for item in shared
        ],
        dtype=float,
    )
    if not len(delta):
        return {"estimate": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(delta), size=(n_bootstrap, len(delta)))
    draws = delta[indices].mean(axis=1)
    low, high = np.quantile(draws, [ci_alpha / 2, 1 - ci_alpha / 2])
    return {
        "estimate": float(delta.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(delta),
    }


def analyze_behavior(
    cfg: Exp3RunConfig,
    run_dir: Path,
    *,
    pairs: Sequence[Mapping[str, Any]] | None = None,
    behavior_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    pairs = list(pairs) if pairs is not None else load_pairs(cfg, run_dir)
    behavior_path = _artifact(run_dir, cfg.exp3.artifacts.behavior_file)
    behavior_rows = (
        list(behavior_rows)
        if behavior_rows is not None
        else (load_jsonl(behavior_path) if behavior_path.is_file() else [])
    )
    metrics: dict[str, Any] = {
        "schema_version": cfg.exp3.schema_version,
        "primary_endpoint": "paired explicit-refusal marker disagreement",
        "nonrefusal_semantics": "explicit refusal absent; not acceptance or compliance",
        "matcher_sha256": refusal_matcher_sha(cfg.exp3.readout.refusal_patterns),
        "contrasts": {},
    }
    failure_statuses = sorted(
        {
            str(row["refusal_status"])
            for row in behavior_rows
            if str(row.get("refusal_status", "")).startswith("decoding_failure:")
        }
    )
    for contrast in _enabled_contrasts(cfg):
        by_role: dict[str, Any] = {}
        for role in cfg.exp3.behavior.roles:
            source, target = _paired_behavior_rows(
                pairs,
                behavior_rows,
                contrast=contrast.name,
                role=role,
                status_key="marker_status",
            )
            primary = summarize_paired_transitions(
                source,
                target,
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
                seed=cfg.seed,
            )
            sensitivity_source, sensitivity_target = _paired_behavior_rows(
                pairs,
                behavior_rows,
                contrast=contrast.name,
                role=role,
                status_key="refusal_status",
            )
            sensitivity = summarize_paired_transitions(
                sensitivity_source,
                sensitivity_target,
                failure_statuses=failure_statuses,
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
                seed=cfg.seed,
            )
            by_role[role] = {
                "all_completed_outputs_primary": primary,
                "excluding_decoding_failures_sensitivity": sensitivity,
                "net_refusal_rate_difference_secondary": _net_refusal_difference(
                    source,
                    target,
                    seed=cfg.seed,
                    n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                    ci_alpha=cfg.exp3.behavior.ci_alpha,
                ),
            }
        metrics["contrasts"][contrast.name] = {
            "purpose": contrast.purpose,
            "source_arm": contrast.source_arm,
            "target_arm": contrast.target_arm,
            "roles": by_role,
        }
    save_json(metrics, _artifact(run_dir, cfg.exp3.artifacts.behavior_metrics_file))
    return metrics


def _stratified_limit(
    rows: Sequence[Mapping[str, Any]],
    limit: int | None,
    *,
    seed: int,
) -> list[dict[str, Any]]:
    """Deterministic round-robin over role/category buckets, avoiding ID-order bias."""

    copied = [dict(row) for row in rows]
    if limit is None or len(copied) <= limit:
        return copied
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in copied:
        key = (str(row.get("role") or "unknown"), str(row.get("category_name") or "unknown"))
        buckets[key].append(row)
    for key, values in buckets.items():
        values.sort(key=lambda row: _stable_int(seed, key, row["item_id"]))
    ordered: list[dict[str, Any]] = []
    keys = sorted(buckets)
    while len(ordered) < limit and keys:
        next_keys = []
        for key in keys:
            if buckets[key] and len(ordered) < limit:
                ordered.append(buckets[key].pop(0))
            if buckets[key]:
                next_keys.append(key)
        keys = next_keys
    return ordered


def _tensor_sha256(value: Any) -> str:
    array = value.detach().float().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def _clone_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if hasattr(value, "clone") else value for key, value in inputs.items()
    }


def _first_token_ids(tokenizer: Any, prefixes: Sequence[str]) -> list[int]:
    ids: set[int] = set()
    special_ids = {int(value) for value in getattr(tokenizer, "all_special_ids", [])}
    for prefix in prefixes:
        for form in (str(prefix), " " + str(prefix)):
            tokens = tokenizer(form, add_special_tokens=False).input_ids
            if tokens:
                token_id = int(tokens[0])
                decoded = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                # Prefixing a numeric/string form can yield Qwen's standalone
                # whitespace token (e.g. ``" 1"`` -> ID 220).  Such a token is
                # generic formatting, not refusal/non-refusal evidence.
                if token_id not in special_ids and str(decoded).strip():
                    ids.add(token_id)
    if not ids:
        raise ValueError("first-token prefix bank produced no token IDs")
    return sorted(ids)


def readout_token_ids(cfg: Exp3RunConfig, processor: Any) -> tuple[list[int], list[int]]:
    refusal = _first_token_ids(processor.tokenizer, cfg.exp3.readout.refusal_prefixes)
    nonrefusal = _first_token_ids(processor.tokenizer, cfg.exp3.readout.nonrefusal_prefixes)
    overlap = sorted(set(refusal) & set(nonrefusal))
    if overlap:
        raise ValueError(f"refusal/non-refusal first-token banks overlap: {overlap}")
    return refusal, nonrefusal


def _margin(logits: Any, refusal_ids: Sequence[int], nonrefusal_ids: Sequence[int]) -> float:
    from audio_safety.pipelines.channel_patching import refusal_margin

    values = logits.detach().float().cpu().numpy()
    return refusal_margin(
        values,
        refusal_ids=np.asarray(refusal_ids, dtype=int),
        compliance_ids=np.asarray(nonrefusal_ids, dtype=int),
    )


def _audio_lengths(model: Any, inputs: Mapping[str, Any]) -> tuple[int, int]:
    from audio_safety.models.hooks import get_audio_tower

    feature_mask = inputs["feature_attention_mask"]
    tower = get_audio_tower(model)
    encoder_lengths, projector_lengths = tower._get_feat_extract_output_lengths(
        feature_mask.sum(-1)
    )
    if encoder_lengths.numel() != 1 or projector_lengths.numel() != 1:
        raise ValueError("Exp3 requires exactly one audio per forward")
    return int(encoder_lengths.item()), int(projector_lengths.item())


def _feature_valid_length(inputs: Mapping[str, Any]) -> int:
    mask = inputs.get("feature_attention_mask")
    if mask is None:
        raise KeyError("Qwen inputs expose no feature_attention_mask")
    values = mask.detach().bool().cpu().numpy()
    if values.ndim != 2 or values.shape[0] != 1:
        raise ValueError("feature_attention_mask must have shape (1, frames)")
    row = values[0]
    valid_length = int(row.sum())
    if valid_length < 2 or valid_length > int(inputs["input_features"].shape[-1]):
        raise ValueError("invalid feature_attention_mask length")
    expected = np.arange(len(row)) < valid_length
    if not np.array_equal(row, expected):
        raise ValueError("Exp3 requires a contiguous valid-frame prefix")
    return valid_length


def _prepare_arm(
    cfg: Exp3RunConfig,
    model: Any,
    processor: Any,
    audio_path: str | Path,
    *,
    instruction: str | None = None,
    need_p1: bool = False,
) -> dict[str, Any]:
    from audio_safety.models.qwen2_audio import (
        build_audio_analysis_conversation,
        model_input_device,
        prepare_qwen2_audio_inputs,
        resolve_audio_position_indices,
    )

    conversation = build_audio_analysis_conversation(
        audio_path,
        instruction or cfg.exp3.instruction,
        system_prompt=cfg.exp3.system_prompt,
    )
    inputs = prepare_qwen2_audio_inputs(
        processor,
        conversation,
        device=model_input_device(model),
    )
    if inputs["input_ids"].shape[0] != 1:
        raise ValueError("Exp3 requires batch size one")
    t_ab = int(inputs["input_ids"].shape[1] - 1)
    p1 = None
    if need_p1:
        positions = resolve_audio_position_indices(processor, conversation)
        p1 = int(positions["assistant_start_pre"])
        if int(positions["first_generation_prelogit"]) != t_ab:
            raise ValueError("resolved t_AB disagrees with the prepared prompt length")
    audio_token_id = getattr(model.config, "audio_token_id", None)
    if audio_token_id is None:
        raise AttributeError("Qwen2-Audio config exposes no audio_token_id")
    audio_mask = inputs["input_ids"][0].eq(int(audio_token_id))
    if "attention_mask" in inputs:
        audio_mask = audio_mask & inputs["attention_mask"][0].bool()
    audio_positions = audio_mask.nonzero(as_tuple=False).flatten().tolist()
    if not audio_positions:
        raise ValueError("prepared prompt contains no expanded audio tokens")
    encoder_length, projector_length = _audio_lengths(model, inputs)
    feature_valid_length = _feature_valid_length(inputs)
    return {
        "path": str(audio_path),
        "conversation": conversation,
        "inputs": inputs,
        "t_ab": t_ab,
        "p1": p1,
        "audio_positions": audio_positions,
        "encoder_length": encoder_length,
        "projector_length": projector_length,
        "feature_valid_length": feature_valid_length,
        "input_ids_sha256": _tensor_sha256(inputs["input_ids"]),
        "input_features_sha256": _tensor_sha256(inputs["input_features"]),
    }


def _prepare_aligned_pair(
    cfg: Exp3RunConfig,
    model: Any,
    processor: Any,
    pair: Mapping[str, Any],
    *,
    need_p1: bool = False,
    instruction: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from audio_safety.pipelines.channel_patching import assert_pair_alignment

    source = _prepare_arm(
        cfg,
        model,
        processor,
        pair["source_path"],
        need_p1=need_p1,
        instruction=instruction,
    )
    target = _prepare_arm(
        cfg,
        model,
        processor,
        pair["target_path"],
        need_p1=need_p1,
        instruction=instruction,
    )
    inputs_source, inputs_target = source["inputs"], target["inputs"]
    positions, t_ab = assert_pair_alignment(
        inputs_source["input_ids"].detach().cpu().numpy(),
        inputs_source["attention_mask"].detach().cpu().numpy(),
        inputs_target["input_ids"].detach().cpu().numpy(),
        inputs_target["attention_mask"].detach().cpu().numpy(),
        audio_token_id=int(model.config.audio_token_id),
        clean_t_ab=source["t_ab"],
        attack_t_ab=target["t_ab"],
    )
    if positions != source["audio_positions"] or positions != target["audio_positions"]:
        raise ValueError("audio positions disagree after alignment validation")
    if t_ab != source["t_ab"] or t_ab != target["t_ab"]:
        raise ValueError("t_AB disagrees after alignment validation")
    source_feature_mask = inputs_source.get("feature_attention_mask")
    target_feature_mask = inputs_target.get("feature_attention_mask")
    if source_feature_mask is None or target_feature_mask is None:
        raise KeyError("Qwen inputs expose no feature_attention_mask")
    if not np.array_equal(
        source_feature_mask.detach().cpu().numpy(),
        target_feature_mask.detach().cpu().numpy(),
    ):
        raise ValueError("feature_attention_mask differs across the pair")
    if tuple(inputs_source["input_features"].shape) != tuple(inputs_target["input_features"].shape):
        raise ValueError("input_features shapes differ across the pair")
    return source, target


def _generate_from_inputs(
    cfg: Exp3RunConfig,
    model: Any,
    processor: Any,
    inputs: Mapping[str, Any],
    refusal_ids: Sequence[int],
    nonrefusal_ids: Sequence[int],
    *,
    contexts: Iterable[Any] = (),
    max_new_tokens: int | None = None,
) -> dict[str, Any]:
    """Greedy generation plus its first-token t_AB margin in one model call."""

    max_tokens = int(max_new_tokens or cfg.exp3.max_new_tokens)
    prompt_length = int(inputs["input_ids"].shape[1])
    active = list(contexts)
    with contextlib.ExitStack() as stack:
        for intervention in active:
            stack.enter_context(intervention)
        generated = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            return_dict_in_generate=True,
            output_scores=True,
        )
    counts = [
        int(intervention.applied_count)
        for intervention in active
        if hasattr(intervention, "applied_count")
    ]
    if any(count != 1 for count in counts):
        raise RuntimeError(f"intervention applied_count must be one, got {counts}")
    if not generated.scores:
        raise RuntimeError("generation produced no first-token scores")
    continuation = generated.sequences[:, prompt_length:]
    response = processor.batch_decode(
        continuation,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "response": response,
        "explicit_refusal": classify_explicit_refusal(response, cfg.exp3.readout.refusal_patterns)
        == EXPLICIT_REFUSAL,
        "r_tab_margin": _margin(generated.scores[0][0], refusal_ids, nonrefusal_ids),
        "applied_counts": counts,
    }


def _behavior_index(cfg: Exp3RunConfig, run_dir: Path) -> dict[tuple[str, str], dict[str, Any]]:
    path = _artifact(run_dir, cfg.exp3.artifacts.behavior_file)
    if not path.is_file():
        raise FileNotFoundError("frozen physical behavior is required before mechanism stages")
    rows = load_jsonl(path)
    index = {(str(row["item_id"]), str(row["arm"])): row for row in rows}
    if len(index) != len(rows):
        raise ValueError("behavior checkpoint contains duplicate item/arm keys")
    return index


def _validated_physical_baselines(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
    pair: Mapping[str, Any],
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    refusal_ids: Sequence[int],
    nonrefusal_ids: Sequence[int],
    *,
    max_new_tokens: int | None = None,
    frozen_behavior: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
    validate_marker: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Regenerate both physical arms and require their frozen marker states."""

    frozen = frozen_behavior if frozen_behavior is not None else _behavior_index(cfg, run_dir)
    results = []
    for side, prepared in (("source", source), ("target", target)):
        result = _generate_from_inputs(
            cfg,
            model,
            processor,
            prepared["inputs"],
            refusal_ids,
            nonrefusal_ids,
            max_new_tokens=max_new_tokens,
        )
        key = (str(pair["item_id"]), str(pair[f"{side}_arm"]))
        expected = frozen.get(key)
        if expected is None:
            raise RuntimeError(f"physical behavior row is missing for {key}")
        if validate_marker and bool(result["explicit_refusal"]) != bool(
            expected["explicit_refusal"]
        ):
            raise RuntimeError(
                f"physical {side} marker no longer reproduces frozen behavior for {key}"
            )
        results.append(result)
    return results[0], results[1]


def run_content_audit(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    """Transcribe both physical arms; retain all rows and stratify later."""

    gate = cfg.exp3.content_audit
    if not gate.enabled:
        return []
    contrast = _contrast(cfg, cfg.exp3.representation.contrast)
    pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast.name]
    pairs = _stratified_limit(pairs, gate.max_pairs, seed=cfg.seed)
    path = _artifact(run_dir, cfg.exp3.artifacts.content_audit_file)
    existing = load_jsonl(path) if path.is_file() else []
    state = {(str(row["pair_id"]), str(row["side"])): row for row in existing}
    eligible = [pair for pair in pairs if str(pair.get("reference_text") or "").strip()]
    completed = 0

    from audio_safety.models.qwen2_audio import generate_audio_response

    for pair in eligible:
        reference = str(pair.get("reference_text") or "").strip()
        for side in ("source", "target"):
            key = (str(pair["pair_id"]), side)
            if key in state:
                continue
            transcript = generate_audio_response(
                model,
                processor,
                pair[f"{side}_path"],
                gate.instruction,
                max_new_tokens=gate.max_new_tokens,
                system_prompt=cfg.exp3.system_prompt,
                do_sample=False,
            )
            wer = word_error_rate(reference, transcript)
            overlap = token_overlap(reference, transcript)
            state[key] = {
                "schema_version": cfg.exp3.schema_version,
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "role": pair["role"],
                "contrast": pair["contrast"],
                "side": side,
                "arm": pair[f"{side}_arm"],
                "transcript": transcript,
                "wer": wer,
                "token_overlap": overlap,
                "content_faithful": bool(wer <= gate.wer_max and overlap >= gate.token_overlap_min),
            }
            completed += 1
            _checkpoint_mapping(state, path, completed_since_resume=completed, every=10)
    expected = {(str(pair["pair_id"]), side) for pair in eligible for side in ("source", "target")}
    _assert_expected_keys("content-audit", expected, set(state))
    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    return [state[key] for key in sorted(state)]


def run_content_patch_audit(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    """Apply the L10 audio-span transport under a transcription prompt.

    This directly tests whether the causal refusal intervention also repairs the
    model's content readout, instead of inferring that from physical-arm ASR.
    """

    gate = cfg.exp3.content_audit
    if not gate.enabled or not gate.patch_enabled:
        return []
    contrast = _contrast(cfg, cfg.exp3.span_patch.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    cohort = _stratified_limit(cohort, gate.patch_max_pairs, seed=cfg.seed)
    path = _artifact(run_dir, cfg.exp3.artifacts.content_patch_file)
    existing = load_jsonl(path) if path.is_file() else []
    state = {(str(row["pair_id"]), str(row["direction"])): row for row in existing}
    eligible = [pair for pair in cohort if str(pair.get("reference_text") or "").strip()]
    completed = 0
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)

    from audio_safety.models.hooks import SpanStateIntervention

    for pair in eligible:
        reference = str(pair.get("reference_text") or "").strip()
        source, target = _prepare_aligned_pair(
            cfg,
            model,
            processor,
            pair,
            instruction=gate.instruction,
        )
        source_states = _capture_audio_spans(model, source, [gate.patch_layer])
        target_states = _capture_audio_spans(model, target, [gate.patch_layer])
        source_baseline = _generate_from_inputs(
            cfg,
            model,
            processor,
            source["inputs"],
            refusal_ids,
            nonrefusal_ids,
            max_new_tokens=gate.max_new_tokens,
        )
        target_baseline = _generate_from_inputs(
            cfg,
            model,
            processor,
            target["inputs"],
            refusal_ids,
            nonrefusal_ids,
            max_new_tokens=gate.max_new_tokens,
        )
        for direction in cfg.exp3.span_patch.directions:
            key = (str(pair["pair_id"]), direction)
            if key in state:
                continue
            host, _donor, _host_states, donor_states = _oriented(
                direction,
                source,
                target,
                source_states,
                target_states,
            )
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )
            intervention = SpanStateIntervention(
                model,
                layer_idx=gate.patch_layer,
                positions=host["audio_positions"],
                replacement=donor_states[gate.patch_layer],
            )
            patched = _generate_from_inputs(
                cfg,
                model,
                processor,
                host["inputs"],
                refusal_ids,
                nonrefusal_ids,
                contexts=[intervention],
                max_new_tokens=gate.max_new_tokens,
            )
            baseline_wer = word_error_rate(reference, baseline["response"])
            patched_wer = word_error_rate(reference, patched["response"])
            baseline_overlap = token_overlap(reference, baseline["response"])
            patched_overlap = token_overlap(reference, patched["response"])
            baseline_donor_overlap = token_overlap(donor_baseline["response"], baseline["response"])
            patched_donor_overlap = token_overlap(donor_baseline["response"], patched["response"])
            state[key] = {
                "schema_version": cfg.exp3.schema_version,
                "pair_id": pair["pair_id"],
                "item_id": pair["item_id"],
                "role": pair["role"],
                "contrast": contrast.name,
                "direction": direction,
                "layer": gate.patch_layer,
                "baseline_transcript": baseline["response"],
                "donor_transcript": donor_baseline["response"],
                "patched_transcript": patched["response"],
                "baseline_wer": baseline_wer,
                "patched_wer": patched_wer,
                "delta_wer": patched_wer - baseline_wer,
                "baseline_token_overlap": baseline_overlap,
                "patched_token_overlap": patched_overlap,
                "delta_token_overlap": patched_overlap - baseline_overlap,
                "baseline_donor_token_overlap": baseline_donor_overlap,
                "patched_donor_token_overlap": patched_donor_overlap,
                "delta_donor_token_overlap": patched_donor_overlap - baseline_donor_overlap,
                "baseline_content_faithful": bool(
                    baseline_wer <= gate.wer_max and baseline_overlap >= gate.token_overlap_min
                ),
                "patched_content_faithful": bool(
                    patched_wer <= gate.wer_max and patched_overlap >= gate.token_overlap_min
                ),
                "applied_count": patched["applied_counts"][0],
            }
            completed += 1
            _checkpoint_mapping(state, path, completed_since_resume=completed, every=10)
    expected = {
        (str(pair["pair_id"]), direction)
        for pair in eligible
        for direction in cfg.exp3.span_patch.directions
    }
    _assert_expected_keys("content-patch-audit", expected, set(state))
    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    return [state[state_key] for state_key in sorted(state)]


def _wrong_pair_lookup(pairs: Sequence[Mapping[str, Any]], *, seed: int) -> dict[str, dict]:
    """Map each pair to a deterministic different item in the same role."""

    by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_role[str(pair["role"])].append(dict(pair))
    result: dict[str, dict[str, Any]] = {}
    for role, values in by_role.items():
        values.sort(key=lambda row: _stable_int(seed, role, row["item_id"]))
        if len(values) < 2:
            continue
        for index, pair in enumerate(values):
            result[str(pair["pair_id"])] = values[(index + 1) % len(values)]
    return result


def run_input_dose(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    """Intervene on the exact post-processor ``input_features`` tensor."""

    gate = cfg.exp3.input_dose
    if not gate.enabled:
        return []
    contrast = _contrast(cfg, gate.contrast)
    all_pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast.name]
    pairs = _stratified_limit(all_pairs, gate.max_pairs, seed=cfg.seed)
    wrong_lookup = _wrong_pair_lookup(all_pairs, seed=cfg.seed)
    if "wrong_item" in gate.components:
        missing_wrong = [pair["pair_id"] for pair in pairs if pair["pair_id"] not in wrong_lookup]
        if missing_wrong:
            raise RuntimeError(
                f"wrong-item M1 control has no same-role donor for {missing_wrong[:5]}"
            )
    path = _artifact(run_dir, cfg.exp3.artifacts.input_dose_file)
    existing = load_jsonl(path) if path.is_file() else []
    state = {
        (str(row["pair_id"]), str(row["component"]), float(row["dose"])): row for row in existing
    }
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    frozen_behavior = _behavior_index(cfg, run_dir)
    completed = 0

    from audio_safety.evaluation.model_eye import model_eye_component

    for pair in pairs:
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        source_features = source["inputs"]["input_features"].detach().float().cpu().numpy()
        target_features = target["inputs"]["input_features"].detach().float().cpu().numpy()
        delta = target_features - source_features
        valid_length = int(source["feature_valid_length"])
        if int(target["feature_valid_length"]) != valid_length:
            raise ValueError("paired physical arms have different valid feature lengths")
        padding_delta = delta[..., valid_length:]
        if padding_delta.size and not np.allclose(padding_delta, 0.0, rtol=0.0, atol=1e-7):
            raise ValueError(
                "paired input-feature padding differs; M1 endpoints are not isolatable"
            )
        wrong_delta = None
        wrong_valid_length = None
        wrong_pair = wrong_lookup.get(str(pair["pair_id"]))
        if wrong_pair is not None:
            wrong_source, wrong_target = _prepare_aligned_pair(cfg, model, processor, wrong_pair)
            wrong_delta = (
                wrong_target["inputs"]["input_features"].detach().float().cpu().numpy()
                - wrong_source["inputs"]["input_features"].detach().float().cpu().numpy()
            )
            wrong_valid_length = int(wrong_source["feature_valid_length"])
            if int(wrong_target["feature_valid_length"]) != wrong_valid_length:
                raise ValueError("wrong-item arms have different valid feature lengths")

        generation_tokens = cfg.exp3.max_new_tokens if gate.generate_text else 1
        physical_source, physical_target = _validated_physical_baselines(
            cfg,
            run_dir,
            model,
            processor,
            pair,
            source,
            target,
            refusal_ids,
            nonrefusal_ids,
            max_new_tokens=generation_tokens,
            frozen_behavior=frozen_behavior,
            validate_marker=gate.generate_text,
        )
        for component_name in gate.components:
            if component_name == "wrong_item" and wrong_delta is None:
                raise RuntimeError(f"wrong-item M1 delta is unavailable for {pair['pair_id']}")
            component = model_eye_component(
                delta,
                component_name,
                frame_rate_hz=gate.feature_frame_rate_hz,
                cutoff_hz=gate.modulation_cutoff_hz,
                timeshift_frames=gate.timeshift_frames,
                wrong_item_delta=wrong_delta,
                valid_length=valid_length,
                wrong_item_valid_length=wrong_valid_length,
            )
            doses = gate.doses if component_name == "all" else [1.0]
            for dose in doses:
                key = (str(pair["pair_id"]), component_name, float(dose))
                if key in state:
                    continue
                mixed = source_features + np.float32(dose) * component
                inputs = _clone_inputs(source["inputs"])
                inputs["input_features"] = source["inputs"]["input_features"].new_tensor(mixed)
                generated = _generate_from_inputs(
                    cfg,
                    model,
                    processor,
                    inputs,
                    refusal_ids,
                    nonrefusal_ids,
                    max_new_tokens=generation_tokens,
                )
                endpoint_expected = None
                endpoint_exact = None
                margin_error = None
                if component_name == "all" and dose in {0.0, 1.0}:
                    physical_expected = physical_source if dose == 0.0 else physical_target
                    endpoint_expected = "physical_source" if dose == 0.0 else "physical_target"
                    margin_error = abs(
                        generated["r_tab_margin"] - physical_expected["r_tab_margin"]
                    )
                    endpoint_exact = (
                        generated["response"] == physical_expected["response"]
                        and margin_error <= 1e-6
                    )
                state[key] = {
                    "schema_version": cfg.exp3.schema_version,
                    "pair_id": pair["pair_id"],
                    "item_id": pair["item_id"],
                    "role": pair["role"],
                    "contrast": contrast.name,
                    "component": component_name,
                    "dose": float(dose),
                    "source_input_features_sha256": source["input_features_sha256"],
                    "target_input_features_sha256": target["input_features_sha256"],
                    "wrong_item_id": wrong_pair["item_id"] if wrong_pair else None,
                    "valid_feature_frames": valid_length,
                    "wrong_item_valid_feature_frames": wrong_valid_length,
                    "delta_fro": float(np.linalg.norm(delta.reshape(-1))),
                    "component_fro": float(np.linalg.norm(component.reshape(-1))),
                    "response": generated["response"] if gate.generate_text else None,
                    "explicit_refusal": (
                        generated["explicit_refusal"] if gate.generate_text else None
                    ),
                    "r_tab_margin": generated["r_tab_margin"],
                    "baseline_explicit_refusal": physical_source["explicit_refusal"],
                    "baseline_r_tab_margin": physical_source["r_tab_margin"],
                    "donor_explicit_refusal": physical_target["explicit_refusal"],
                    "donor_r_tab_margin": physical_target["r_tab_margin"],
                    "endpoint_expected": endpoint_expected,
                    "endpoint_exact": endpoint_exact,
                    "endpoint_margin_abs_error": margin_error,
                    "physical_source_response": physical_source["response"],
                    "physical_target_response": physical_target["response"],
                }
                completed += 1
                _checkpoint_mapping(state, path, completed_since_resume=completed, every=20)
    expected = {
        (str(pair["pair_id"]), component, float(dose))
        for pair in pairs
        for component in gate.components
        for dose in (gate.doses if component == "all" else [1.0])
    }
    _assert_expected_keys("input-dose", expected, set(state))
    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    assert_input_dose_gate(cfg, run_dir)
    return [state[key] for key in sorted(state)]


def _save_npz_atomic(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.npz")
    # NumPy's stubs reserve named keywords such as ``allow_pickle`` even though
    # arbitrary array names are valid at runtime.
    np.savez_compressed(temporary, **dict(arrays))  # type: ignore[arg-type]
    temporary.replace(path)


def _representation_cell(
    cfg: Exp3RunConfig,
    model: Any,
    prepared: Mapping[str, Any],
    refusal_ids: Sequence[int],
    nonrefusal_ids: Sequence[int],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    import torch

    from audio_safety.models.hooks import AudioPathCapture

    gate = cfg.exp3.representation
    decoder_layers = None if gate.llm_layers == "all" else list(gate.llm_layers)
    capture = AudioPathCapture(
        model,
        audio_positions=prepared["audio_positions"],
        p1_index=int(prepared["p1"]),
        p2_index=int(prepared["t_ab"]),
        encoder_valid_length=int(prepared["encoder_length"]),
        projector_valid_length=int(prepared["projector_length"]),
        decoder_layers=decoder_layers,
    )
    with torch.inference_mode(), capture:
        outputs = model(**prepared["inputs"], use_cache=False, return_dict=True)
    states = {
        name: tensor.numpy().astype(np.float32, copy=False)
        for name, tensor in capture.states().items()
    }
    logits = outputs.logits[0, int(prepared["t_ab"]), :]
    trace = {
        "r_tab_margin": _margin(logits, refusal_ids, nonrefusal_ids),
        "encoder_layers": capture.encoder_layer_indices,
        "llm_layers": capture.decoder_layer_indices,
        "p1_index": int(prepared["p1"]),
        "t_ab": int(prepared["t_ab"]),
        "audio_span_start": int(prepared["audio_positions"][0]),
        "audio_span_end": int(prepared["audio_positions"][-1]),
        "audio_token_count": len(prepared["audio_positions"]),
        "encoder_valid_length": int(prepared["encoder_length"]),
        "projector_valid_length": int(prepared["projector_length"]),
        "input_ids_sha256": prepared["input_ids_sha256"],
        "input_features_sha256": prepared["input_features_sha256"],
    }
    return states, trace


def run_representation_capture(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    gate = cfg.exp3.representation
    if not gate.enabled:
        return {}, []
    contrast = _contrast(cfg, gate.contrast)
    pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast.name]
    pairs = _stratified_limit(pairs, gate.max_pairs, seed=cfg.seed)
    behavior_path = _artifact(run_dir, cfg.exp3.artifacts.behavior_file)
    if not behavior_path.is_file():
        raise FileNotFoundError("behavior records are required before representation capture")
    behavior = {(str(row["item_id"]), str(row["arm"])): row for row in load_jsonl(behavior_path)}
    cells_path = _artifact(run_dir, cfg.exp3.artifacts.representation_cells_file)
    arrays_path = _artifact(run_dir, cfg.exp3.artifacts.representations_file)
    if cells_path.is_file() and not arrays_path.is_file():
        raise RuntimeError("orphan representation cells checkpoint has no authoritative NPZ")
    cells: list[dict[str, Any]] = []
    stored: dict[str, list[np.ndarray]] = defaultdict(list)
    encoder_layers: list[int] | None = None
    llm_layers: list[int] | None = None
    if arrays_path.is_file():
        with np.load(arrays_path) as archive:
            if "cells_json" not in archive.files:
                raise RuntimeError("representation checkpoint predates the atomic cells contract")
            cells = [json.loads(str(value)) for value in archive["cells_json"].tolist()]
            expected_indices = list(range(len(cells)))
            actual_indices = [int(cell["activation_index"]) for cell in cells]
            if actual_indices != expected_indices:
                raise RuntimeError("representation activation_index is not sequential")
            for name in (
                "encoder_mean",
                "encoder_last",
                "projector_mean",
                "projector_last",
                "llm_audio_mean",
                "llm_audio_last",
                "llm_p1",
                "llm_p2",
            ):
                if archive[name].shape[0] != len(cells):
                    raise RuntimeError(
                        f"representation {name} rows={archive[name].shape[0]} "
                        f"but cells={len(cells)}"
                    )
                stored[name] = [row for row in archive[name]]
            encoder_layers = [int(value) for value in archive["encoder_layers"]]
            llm_layers = [int(value) for value in archive["llm_layers"]]
        # NPZ is the single atomic authority.  A crash between NPZ and the
        # human-readable JSONL write is repaired deterministically on resume.
        disk_cells = load_jsonl(cells_path) if cells_path.is_file() else None
        if disk_cells != cells:
            atomic_save_jsonl(cells, cells_path)
    done = {(str(cell["item_id"]), str(cell["arm"])) for cell in cells}
    if len(done) != len(cells):
        raise RuntimeError("representation checkpoint contains duplicate item/arm cells")
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)

    def checkpoint() -> dict[str, np.ndarray]:
        arrays = {
            name: np.stack(values).astype(np.float32, copy=False) for name, values in stored.items()
        }
        arrays["encoder_layers"] = np.asarray(encoder_layers, dtype=np.int16)
        arrays["llm_layers"] = np.asarray(llm_layers, dtype=np.int16)
        arrays["refusal_token_ids"] = np.asarray(refusal_ids, dtype=np.int64)
        arrays["nonrefusal_token_ids"] = np.asarray(nonrefusal_ids, dtype=np.int64)
        arrays["cells_json"] = np.asarray(
            [json.dumps(cell, sort_keys=True, ensure_ascii=False) for cell in cells],
            dtype=np.str_,
        )
        _save_npz_atomic(arrays_path, arrays)
        atomic_save_jsonl(cells, cells_path)
        return arrays

    newly_captured = 0
    for pair in pairs:
        source, target = _prepare_aligned_pair(cfg, model, processor, pair, need_p1=True)
        for side, prepared in (("source", source), ("target", target)):
            arm = str(pair[f"{side}_arm"])
            key = (str(pair["item_id"]), arm)
            if key in done:
                continue
            states, trace = _representation_cell(cfg, model, prepared, refusal_ids, nonrefusal_ids)
            if encoder_layers is None:
                encoder_layers = list(trace["encoder_layers"])
                llm_layers = list(trace["llm_layers"])
            elif encoder_layers != trace["encoder_layers"] or llm_layers != trace["llm_layers"]:
                raise RuntimeError("captured layer indices changed between cells")
            for name, values in states.items():
                stored[name].append(values)
            behavior_row = behavior.get(key)
            if behavior_row is None:
                raise ValueError(f"behavior output is missing for {key}")
            cells.append(
                {
                    "activation_index": len(cells),
                    "schema_version": cfg.exp3.schema_version,
                    "pair_id": pair["pair_id"],
                    "item_id": pair["item_id"],
                    "role": pair["role"],
                    "category_name": pair.get("category_name"),
                    "contrast": contrast.name,
                    "side": side,
                    "arm": arm,
                    "marker_status": behavior_row["marker_status"],
                    "refusal_status": behavior_row["refusal_status"],
                    "explicit_refusal": behavior_row["explicit_refusal"],
                    **{key: value for key, value in trace.items() if not key.endswith("layers")},
                }
            )
            done.add(key)
            newly_captured += 1
            if newly_captured % 10 == 0:
                checkpoint()
    expected = {
        (str(pair["item_id"]), str(pair[f"{side}_arm"]))
        for pair in pairs
        for side in ("source", "target")
    }
    _assert_expected_keys("representation-capture", expected, done)
    arrays = checkpoint()
    analyze_representations(cfg, run_dir, arrays=arrays, cells=cells)
    return arrays, cells


def _crossfit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    n_folds: int,
    c_value: float,
    seed: int,
) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=int)
    groups = np.asarray(groups)
    if features.ndim != 2 or len(features) != len(labels) or len(labels) != len(groups):
        raise ValueError("probe features/labels/groups have incompatible shapes")
    if len(np.unique(labels)) < 2:
        return {"auroc": None, "n": len(labels), "reason": "one_class"}
    unique_groups = np.unique(groups)
    folds = min(n_folds, len(unique_groups))
    if folds < 2:
        return {"auroc": None, "n": len(labels), "reason": "too_few_groups"}
    splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
    scores = np.full(len(labels), np.nan, dtype=float)
    try:
        for train, test in splitter.split(features, labels, groups):
            if len(np.unique(labels[train])) < 2:
                continue
            estimator = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    C=c_value,
                    class_weight="balanced",
                    solver="liblinear",
                    max_iter=2000,
                    random_state=seed,
                ),
            )
            estimator.fit(features[train], labels[train])
            scores[test] = estimator.decision_function(features[test])
    except ValueError as exc:
        return {"auroc": None, "n": len(labels), "reason": f"split_failed:{exc}"}
    valid = np.isfinite(scores)
    if valid.sum() < 2 or len(np.unique(labels[valid])) < 2:
        return {"auroc": None, "n": int(valid.sum()), "reason": "invalid_folds"}
    return {
        "auroc": float(roc_auc_score(labels[valid], scores[valid])),
        "n": int(valid.sum()),
        "n_folds": folds,
    }


def analyze_representations(
    cfg: Exp3RunConfig,
    run_dir: Path,
    *,
    arrays: Mapping[str, np.ndarray] | None = None,
    cells: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    gate = cfg.exp3.representation
    arrays_path = _artifact(run_dir, cfg.exp3.artifacts.representations_file)
    if arrays is None:
        with np.load(arrays_path) as archive:
            arrays = {name: archive[name] for name in archive.files}
    if cells is None:
        cells = load_jsonl(_artifact(run_dir, cfg.exp3.artifacts.representation_cells_file))
    labels = np.asarray([bool(cell["explicit_refusal"]) for cell in cells], dtype=int)
    groups = np.asarray([str(cell["item_id"]) for cell in cells])
    roles = np.asarray([str(cell["role"]) for cell in cells])
    probes: dict[str, Any] = {}

    def probe_stack(
        destination: dict[str, Any],
        name: str,
        values: np.ndarray,
        layer_ids: Sequence[int] | None,
        mask: np.ndarray | None = None,
    ) -> None:
        selected_values = values if mask is None else values[mask]
        selected_labels = labels if mask is None else labels[mask]
        selected_groups = groups if mask is None else groups[mask]
        if values.ndim == 2:
            destination[name] = _crossfit_probe(
                selected_values,
                selected_labels,
                selected_groups,
                n_folds=gate.probe_folds,
                c_value=gate.probe_c,
                seed=cfg.seed,
            )
            return
        destination[name] = {}
        for index, layer in enumerate(layer_ids or range(values.shape[1])):
            destination[name][str(layer)] = _crossfit_probe(
                selected_values[:, index, :],
                selected_labels,
                selected_groups,
                n_folds=gate.probe_folds,
                c_value=gate.probe_c,
                seed=cfg.seed,
            )

    encoder_layers = [int(value) for value in arrays["encoder_layers"]]
    llm_layers = [int(value) for value in arrays["llm_layers"]]
    probe_specs = (
        ("encoder_mean", arrays["encoder_mean"], encoder_layers),
        ("projector_mean", arrays["projector_mean"], None),
        ("llm_audio_mean", arrays["llm_audio_mean"], llm_layers),
        ("llm_t_ab", arrays["llm_p2"], llm_layers),
    )
    for name, values, layer_ids in probe_specs:
        probe_stack(probes, name, values, layer_ids)
    probes_by_role: dict[str, Any] = {}
    for role in sorted(set(roles)):
        role_probes: dict[str, Any] = {}
        role_mask = roles == role
        for name, values, layer_ids in probe_specs:
            probe_stack(role_probes, name, values, layer_ids, role_mask)
        probes_by_role[role] = role_probes

    by_cell = {
        (str(cell["item_id"]), str(cell["side"])): int(cell["activation_index"]) for cell in cells
    }
    pair_indices = [
        (by_cell[(item, "source")], by_cell[(item, "target")])
        for item in sorted({str(cell["item_id"]) for cell in cells})
        if (item, "source") in by_cell and (item, "target") in by_cell
    ]
    displacement: dict[str, Any] = {}
    for name in ("encoder_mean", "projector_mean", "llm_audio_mean", "llm_p2"):
        values = arrays[name]
        if not pair_indices:
            displacement[name] = {"n": 0}
            continue
        deltas = np.stack([values[target] - values[source] for source, target in pair_indices])
        norms = np.linalg.norm(deltas, axis=-1)
        displacement[name] = {
            "n": len(pair_indices),
            "mean_l2": np.mean(norms, axis=0).tolist() if norms.ndim > 1 else float(norms.mean()),
            "median_l2": np.median(norms, axis=0).tolist()
            if norms.ndim > 1
            else float(np.median(norms)),
        }
    metrics = {
        "schema_version": cfg.exp3.schema_version,
        "endpoint": "cross-fitted explicit-refusal readout; descriptive localization only",
        "n_cells": len(cells),
        "n_items": len(set(groups)),
        "encoder_layers": encoder_layers,
        "llm_layers": llm_layers,
        "probes": probes,
        "probes_by_role": probes_by_role,
        "paired_displacement": displacement,
    }
    save_json(metrics, _artifact(run_dir, cfg.exp3.artifacts.representation_metrics_file))
    return metrics


def _mechanism_cohort(
    cfg: Exp3RunConfig,
    run_dir: Path,
    *,
    contrast_name: str,
) -> list[dict[str, Any]]:
    pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast_name]
    behavior_path = _artifact(run_dir, cfg.exp3.artifacts.behavior_file)
    behavior_rows = load_jsonl(behavior_path)
    behavior = {(str(row["item_id"]), str(row["arm"])): row for row in behavior_rows}
    source_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    pair_by_item: dict[str, dict[str, Any]] = {}
    failures: set[str] = set()
    for pair in pairs:
        item = str(pair["item_id"])
        source = behavior.get((item, str(pair["source_arm"])))
        target = behavior.get((item, str(pair["target_arm"])))
        if source is None or target is None:
            continue
        for row in (source, target):
            status = str(row["refusal_status"])
            if status.startswith("decoding_failure:"):
                failures.add(status)
        common = {
            "item_id": item,
            "role": pair["role"],
            "category_name": pair.get("category_name"),
        }
        source_rows.append({**common, "refusal_status": source["refusal_status"]})
        target_rows.append({**common, "refusal_status": target["refusal_status"]})
        pair_by_item[item] = {
            **pair,
            "source_explicit_refusal": bool(source["explicit_refusal"]),
            "target_explicit_refusal": bool(target["explicit_refusal"]),
            "source_behavior_contract_sha256": source["behavior_contract_sha256"],
            "target_behavior_contract_sha256": target["behavior_contract_sha256"],
            "source_runtime_behavior_contract_sha256": source["runtime_behavior_contract_sha256"],
            "target_runtime_behavior_contract_sha256": target["runtime_behavior_contract_sha256"],
        }
    cohort_cfg = cfg.exp3.span_patch.cohort
    selected = select_transition_cohort(
        source_rows,
        target_rows,
        failure_statuses=sorted(failures),
        stratum_keys=["role"],
        stable_refusal_cap_per_stratum=cohort_cfg.max_stable_refusal,
        stable_nonrefusal_cap_per_stratum=cohort_cfg.max_stable_nonrefusal,
        seed=cfg.exp3.span_patch.seed,
    )
    # The utility deliberately retains every discordant pair. Exp3 additionally
    # freezes a GPU-budget cap per direction and role before interventions.
    grouped_discordant: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    controls: list[dict[str, Any]] = []
    for row in selected:
        if row["selection_role"] == "discordant":
            # select_transition_cohort nests the stratum values it was given under
            # "stratum"; the role is not promoted to the top level of the row.
            stratum_role = str(dict(row["stratum"])["role"])
            grouped_discordant[(stratum_role, str(row["transition"]))].append(row)
        else:
            controls.append(row)
    capped: list[dict[str, Any]] = []
    for key, values in sorted(grouped_discordant.items()):
        values.sort(key=lambda row: _stable_int(cfg.seed, key, row["item_id"]))
        capped.extend(values[: cohort_cfg.max_discordant_each_direction])
    selected_rows = capped + controls
    result = []
    for selection in selected_rows:
        pair = dict(pair_by_item[str(selection["item_id"])])
        pair["transition"] = selection["transition"]
        pair["selection_role"] = selection["selection_role"]
        result.append(pair)
    result.sort(
        key=lambda row: (
            0 if row["selection_role"] == "discordant" else 1,
            row["role"],
            row["transition"],
            _stable_int(cfg.seed, row["item_id"]),
        )
    )
    cohort_path = run_dir / "inputs" / "mechanism_cohort.jsonl"
    if cohort_path.is_file():
        frozen = load_jsonl(cohort_path)
        if frozen != result:
            raise RuntimeError("mechanism cohort changed after it was frozen")
        return frozen
    atomic_save_jsonl(result, cohort_path)
    return result


def _capture_audio_spans(
    model: Any,
    prepared: Mapping[str, Any],
    layers: Sequence[int],
) -> dict[int, np.ndarray]:
    import torch

    from audio_safety.models.hooks import AudioSpanCapture

    capture = AudioSpanCapture(
        model,
        layers=layers,
        positions=prepared["audio_positions"],
    )
    with torch.inference_mode(), capture:
        model(**prepared["inputs"], use_cache=False, return_dict=True)
    return {
        layer: state.numpy().astype(np.float32, copy=False)
        for layer, state in capture.states().items()
    }


def _capture_t_ab_states(
    model: Any,
    prepared: Mapping[str, Any],
    layers: Sequence[int],
) -> dict[int, np.ndarray]:
    import torch

    from audio_safety.models.hooks import ResidualStreamCapture

    capture = ResidualStreamCapture(
        model,
        token_index=int(prepared["t_ab"]),
        layers=layers,
    )
    with torch.inference_mode(), capture:
        model(**prepared["inputs"], use_cache=False, return_dict=True)
    return {
        layer: state.numpy().astype(np.float32, copy=False)
        for layer, state in capture.states().items()
    }


def _oriented(
    direction: str,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    source_states: Mapping[int, np.ndarray],
    target_states: Mapping[int, np.ndarray],
) -> tuple[
    Mapping[str, Any],
    Mapping[str, Any],
    Mapping[int, np.ndarray],
    Mapping[int, np.ndarray],
]:
    if direction == "source_to_target":
        return source, target, source_states, target_states
    if direction == "target_to_source":
        return target, source, target_states, source_states
    raise ValueError(f"unknown Exp3 direction {direction!r}")


def _oriented_wrong_delta(
    direction: str,
    source_states: Mapping[int, np.ndarray],
    target_states: Mapping[int, np.ndarray],
    layer: int,
) -> np.ndarray:
    if direction == "source_to_target":
        return target_states[layer] - source_states[layer]
    if direction == "target_to_source":
        return source_states[layer] - target_states[layer]
    raise ValueError(f"unknown Exp3 direction {direction!r}")


def run_span_patch(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    gate = cfg.exp3.span_patch
    if not gate.enabled:
        return []
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    all_pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast.name]
    wrong_lookup = _wrong_pair_lookup(all_pairs, seed=gate.seed)
    path = _artifact(run_dir, cfg.exp3.artifacts.span_patch_file)
    existing = load_jsonl(path) if path.is_file() else []
    state = {
        (
            str(row["pair_id"]),
            str(row["direction"]),
            int(row["layer"]),
            str(row["condition"]),
        ): row
        for row in existing
    }
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    frozen_behavior = _behavior_index(cfg, run_dir)
    if "wrong_item" in gate.conditions:
        missing_wrong = [pair["pair_id"] for pair in cohort if pair["pair_id"] not in wrong_lookup]
        if missing_wrong:
            raise RuntimeError(f"span-patch wrong-item donors are missing for {missing_wrong[:5]}")
    completed = 0

    from audio_safety.evaluation.model_eye import controlled_span_delta
    from audio_safety.models.hooks import SpanStateIntervention

    for pair in cohort:
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        source_states = _capture_audio_spans(model, source, gate.layers)
        target_states = _capture_audio_spans(model, target, gate.layers)
        wrong_pair = wrong_lookup.get(str(pair["pair_id"]))
        wrong_source_states: dict[int, np.ndarray] | None = None
        wrong_target_states: dict[int, np.ndarray] | None = None
        if wrong_pair is not None:
            wrong_source, wrong_target = _prepare_aligned_pair(cfg, model, processor, wrong_pair)
            wrong_source_states = _capture_audio_spans(model, wrong_source, gate.layers)
            wrong_target_states = _capture_audio_spans(model, wrong_target, gate.layers)

        source_baseline, target_baseline = _validated_physical_baselines(
            cfg,
            run_dir,
            model,
            processor,
            pair,
            source,
            target,
            refusal_ids,
            nonrefusal_ids,
            frozen_behavior=frozen_behavior,
        )

        for direction in gate.directions:
            host, donor, host_states, donor_states = _oriented(
                direction, source, target, source_states, target_states
            )
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )
            for layer in gate.layers:
                real_delta = donor_states[layer] - host_states[layer]
                wrong_delta = None
                if wrong_source_states is not None and wrong_target_states is not None:
                    wrong_delta = _oriented_wrong_delta(
                        direction, wrong_source_states, wrong_target_states, layer
                    )
                for condition in gate.conditions:
                    key = (str(pair["pair_id"]), direction, int(layer), condition)
                    if key in state:
                        continue
                    if condition == "wrong_item" and wrong_delta is None:
                        raise RuntimeError(
                            f"span-patch wrong-item delta is unavailable for {pair['pair_id']}"
                        )
                    delta = controlled_span_delta(
                        real_delta,
                        condition,
                        wrong_item_delta=wrong_delta,
                        seed=_stable_int(gate.seed, pair["pair_id"], direction, layer, condition),
                    )
                    if condition == "real":
                        replacement = donor_states[layer]
                    elif condition == "identity":
                        replacement = host_states[layer]
                    else:
                        replacement = host_states[layer] + delta
                    intervention = SpanStateIntervention(
                        model,
                        layer_idx=layer,
                        positions=host["audio_positions"],
                        replacement=replacement,
                    )
                    full_generation = layer in gate.full_generation_layers
                    generated = _generate_from_inputs(
                        cfg,
                        model,
                        processor,
                        host["inputs"],
                        refusal_ids,
                        nonrefusal_ids,
                        contexts=[intervention],
                        max_new_tokens=cfg.exp3.max_new_tokens if full_generation else 1,
                    )
                    state[key] = {
                        "schema_version": cfg.exp3.schema_version,
                        "pair_id": pair["pair_id"],
                        "item_id": pair["item_id"],
                        "role": pair["role"],
                        "category_name": pair.get("category_name"),
                        "transition": pair.get("transition"),
                        "selection_role": pair.get("selection_role"),
                        "contrast": contrast.name,
                        "stage": "llm_audio_span",
                        "direction": direction,
                        "host_arm": (
                            pair["source_arm"]
                            if direction == "source_to_target"
                            else pair["target_arm"]
                        ),
                        "donor_arm": (
                            pair["target_arm"]
                            if direction == "source_to_target"
                            else pair["source_arm"]
                        ),
                        "layer": int(layer),
                        "condition": condition,
                        "wrong_item_id": wrong_pair["item_id"] if wrong_pair else None,
                        "span_len": len(host["audio_positions"]),
                        "delta_fro": float(np.linalg.norm(delta.reshape(-1))),
                        "real_delta_fro": float(np.linalg.norm(real_delta.reshape(-1))),
                        "baseline_response": baseline["response"],
                        "baseline_explicit_refusal": baseline["explicit_refusal"],
                        "baseline_r_tab_margin": baseline["r_tab_margin"],
                        "donor_explicit_refusal": donor_baseline["explicit_refusal"],
                        "donor_r_tab_margin": donor_baseline["r_tab_margin"],
                        "full_generation": full_generation,
                        "response": generated["response"] if full_generation else None,
                        "explicit_refusal": (
                            generated["explicit_refusal"] if full_generation else None
                        ),
                        "r_tab_margin": generated["r_tab_margin"],
                        "applied_count": generated["applied_counts"][0],
                        "identity_response_exact": (
                            generated["response"] == baseline["response"]
                            if condition == "identity" and full_generation
                            else None
                        ),
                        "identity_margin_abs_error": (
                            abs(generated["r_tab_margin"] - baseline["r_tab_margin"])
                            if condition == "identity"
                            else None
                        ),
                    }
                    completed += 1
                    _checkpoint_mapping(state, path, completed_since_resume=completed, every=25)
    expected = {
        (str(pair["pair_id"]), direction, int(layer), condition)
        for pair in cohort
        for direction in gate.directions
        for layer in gate.layers
        for condition in gate.conditions
    }
    _assert_expected_keys("span-patch", expected, set(state))
    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    assert_span_patch_gate(cfg, run_dir)
    return [state[key] for key in sorted(state)]


def run_readout_patch(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    """Patch the full t_AB residual state as a downstream positive control."""

    gate = cfg.exp3.readout_patch
    if not gate.enabled:
        return []
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    all_pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast.name]
    wrong_lookup = _wrong_pair_lookup(all_pairs, seed=cfg.seed)
    path = _artifact(run_dir, cfg.exp3.artifacts.readout_patch_file)
    existing = load_jsonl(path) if path.is_file() else []
    state = {
        (
            str(row["pair_id"]),
            str(row["direction"]),
            int(row["layer"]),
            str(row["condition"]),
        ): row
        for row in existing
    }
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    frozen_behavior = _behavior_index(cfg, run_dir)

    from audio_safety.models.hooks import get_decoder_layers

    decoder_count = len(get_decoder_layers(model))
    if not gate.layers or min(gate.layers) < 0 or max(gate.layers) >= decoder_count:
        raise ValueError(f"readout patch layers must lie in [0, {decoder_count - 1}]")
    final_layer = decoder_count - 1
    if final_layer not in gate.layers:
        raise ValueError(
            f"readout patch must include final decoder layer {final_layer} as a pipeline control"
        )
    if "wrong_item" in gate.conditions:
        missing_wrong = [pair["pair_id"] for pair in cohort if pair["pair_id"] not in wrong_lookup]
        if missing_wrong:
            raise RuntimeError(
                f"readout-patch wrong-item donors are missing for {missing_wrong[:5]}"
            )
    completed = 0

    from audio_safety.evaluation.model_eye import controlled_span_delta
    from audio_safety.models.hooks import ResidualStreamIntervention

    for pair in cohort:
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        source_states = _capture_t_ab_states(model, source, gate.layers)
        target_states = _capture_t_ab_states(model, target, gate.layers)
        wrong_pair = wrong_lookup.get(str(pair["pair_id"]))
        wrong_source_states: dict[int, np.ndarray] | None = None
        wrong_target_states: dict[int, np.ndarray] | None = None
        if wrong_pair is not None:
            wrong_source, wrong_target = _prepare_aligned_pair(cfg, model, processor, wrong_pair)
            wrong_source_states = _capture_t_ab_states(model, wrong_source, gate.layers)
            wrong_target_states = _capture_t_ab_states(model, wrong_target, gate.layers)

        source_baseline, target_baseline = _validated_physical_baselines(
            cfg,
            run_dir,
            model,
            processor,
            pair,
            source,
            target,
            refusal_ids,
            nonrefusal_ids,
            frozen_behavior=frozen_behavior,
        )

        for direction in gate.directions:
            host, donor, host_states, donor_states = _oriented(
                direction, source, target, source_states, target_states
            )
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )
            for layer in gate.layers:
                real_delta = (donor_states[layer] - host_states[layer]).reshape(1, -1)
                wrong_delta = None
                if wrong_source_states is not None and wrong_target_states is not None:
                    wrong_delta = _oriented_wrong_delta(
                        direction, wrong_source_states, wrong_target_states, layer
                    ).reshape(1, -1)
                for condition in gate.conditions:
                    key = (str(pair["pair_id"]), direction, int(layer), condition)
                    if key in state:
                        continue
                    if condition == "wrong_item" and wrong_delta is None:
                        raise RuntimeError(
                            f"readout wrong-item delta is unavailable for {pair['pair_id']}"
                        )
                    delta = controlled_span_delta(
                        real_delta,
                        condition,
                        wrong_item_delta=wrong_delta,
                        seed=_stable_int(
                            cfg.seed,
                            "readout",
                            pair["pair_id"],
                            direction,
                            layer,
                            condition,
                        ),
                    )[0]
                    if condition == "real":
                        replacement = donor_states[layer]
                    elif condition == "identity":
                        replacement = host_states[layer]
                    else:
                        replacement = host_states[layer] + delta
                    intervention = ResidualStreamIntervention(
                        model,
                        layer_idx=layer,
                        token_index=int(host["t_ab"]),
                        mode="patch_state",
                        replacement_state=replacement,
                    )
                    generated = _generate_from_inputs(
                        cfg,
                        model,
                        processor,
                        host["inputs"],
                        refusal_ids,
                        nonrefusal_ids,
                        contexts=[intervention],
                    )
                    state[key] = {
                        "schema_version": cfg.exp3.schema_version,
                        "pair_id": pair["pair_id"],
                        "item_id": pair["item_id"],
                        "role": pair["role"],
                        "transition": pair.get("transition"),
                        "selection_role": pair.get("selection_role"),
                        "contrast": contrast.name,
                        "stage": "t_ab_full_state",
                        "direction": direction,
                        "layer": int(layer),
                        "condition": condition,
                        "wrong_item_id": wrong_pair["item_id"] if wrong_pair else None,
                        "delta_l2": float(np.linalg.norm(delta)),
                        "real_delta_l2": float(np.linalg.norm(real_delta)),
                        "t_ab": int(host["t_ab"]),
                        "baseline_response": baseline["response"],
                        "baseline_explicit_refusal": baseline["explicit_refusal"],
                        "baseline_r_tab_margin": baseline["r_tab_margin"],
                        "donor_explicit_refusal": donor_baseline["explicit_refusal"],
                        "donor_r_tab_margin": donor_baseline["r_tab_margin"],
                        "response": generated["response"],
                        "explicit_refusal": generated["explicit_refusal"],
                        "r_tab_margin": generated["r_tab_margin"],
                        "applied_count": generated["applied_counts"][0],
                        "identity_response_exact": (
                            generated["response"] == baseline["response"]
                            if condition == "identity"
                            else None
                        ),
                        "identity_margin_abs_error": (
                            abs(generated["r_tab_margin"] - baseline["r_tab_margin"])
                            if condition == "identity"
                            else None
                        ),
                        "final_control_margin_abs_error": (
                            abs(generated["r_tab_margin"] - donor_baseline["r_tab_margin"])
                            if condition == "real" and layer == final_layer
                            else None
                        ),
                    }
                    completed += 1
                    _checkpoint_mapping(state, path, completed_since_resume=completed, every=20)
    expected = {
        (str(pair["pair_id"]), direction, int(layer), condition)
        for pair in cohort
        for direction in gate.directions
        for layer in gate.layers
        for condition in gate.conditions
    }
    _assert_expected_keys("readout-patch", expected, set(state))
    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    final_controls = [
        row
        for row in state.values()
        if int(row["layer"]) == final_layer and row["condition"] == "real"
    ]
    if not final_controls or any(
        float(row["final_control_margin_abs_error"]) > 1e-4 for row in final_controls
    ):
        raise RuntimeError("final-layer t_AB control failed to reproduce donor R_tAB margin")
    return [state[key] for key in sorted(state)]


def run_escape_reset(
    cfg: Exp3RunConfig,
    run_dir: Path,
    model: Any,
    processor: Any,
) -> list[dict[str, Any]]:
    """Inject at Lk, then reset only audio positions later to test path escape."""

    gate = cfg.exp3.escape_reset
    if not gate.enabled:
        return []
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    layers = sorted(
        {layer for window in gate.windows for layer in (window.inject_layer, window.reset_layer)}
    )
    path = _artifact(run_dir, cfg.exp3.artifacts.escape_reset_file)
    existing = load_jsonl(path) if path.is_file() else []
    state = {
        (
            str(row["pair_id"]),
            str(row["direction"]),
            int(row["inject_layer"]),
            int(row["reset_layer"]),
            str(row["condition"]),
        ): row
        for row in existing
    }
    refusal_ids, nonrefusal_ids = readout_token_ids(cfg, processor)
    frozen_behavior = _behavior_index(cfg, run_dir)
    completed = 0

    from audio_safety.models.hooks import SpanStateIntervention

    for pair in cohort:
        source, target = _prepare_aligned_pair(cfg, model, processor, pair)
        source_states = _capture_audio_spans(model, source, layers)
        target_states = _capture_audio_spans(model, target, layers)
        source_baseline, target_baseline = _validated_physical_baselines(
            cfg,
            run_dir,
            model,
            processor,
            pair,
            source,
            target,
            refusal_ids,
            nonrefusal_ids,
            frozen_behavior=frozen_behavior,
        )
        for direction in gate.directions:
            host, donor, host_states, donor_states = _oriented(
                direction, source, target, source_states, target_states
            )
            baseline, donor_baseline = (
                (source_baseline, target_baseline)
                if direction == "source_to_target"
                else (target_baseline, source_baseline)
            )
            for window in gate.windows:
                inject_layer = window.inject_layer
                reset_layer = window.reset_layer
                for condition in gate.conditions:
                    key = (
                        str(pair["pair_id"]),
                        direction,
                        inject_layer,
                        reset_layer,
                        condition,
                    )
                    if key in state:
                        continue
                    contexts: list[Any] = []
                    if condition in {"inject_only", "inject_reset"}:
                        contexts.append(
                            SpanStateIntervention(
                                model,
                                layer_idx=inject_layer,
                                positions=host["audio_positions"],
                                replacement=donor_states[inject_layer],
                            )
                        )
                    elif condition == "identity":
                        contexts.append(
                            SpanStateIntervention(
                                model,
                                layer_idx=inject_layer,
                                positions=host["audio_positions"],
                                replacement=host_states[inject_layer],
                            )
                        )
                    if condition in {"inject_reset", "reset_only"}:
                        contexts.append(
                            SpanStateIntervention(
                                model,
                                layer_idx=reset_layer,
                                positions=host["audio_positions"],
                                replacement=host_states[reset_layer],
                            )
                        )
                    generated = _generate_from_inputs(
                        cfg,
                        model,
                        processor,
                        host["inputs"],
                        refusal_ids,
                        nonrefusal_ids,
                        contexts=contexts,
                    )
                    state[key] = {
                        "schema_version": cfg.exp3.schema_version,
                        "pair_id": pair["pair_id"],
                        "item_id": pair["item_id"],
                        "role": pair["role"],
                        "transition": pair.get("transition"),
                        "selection_role": pair.get("selection_role"),
                        "contrast": contrast.name,
                        "stage": "audio_escape_reset",
                        "direction": direction,
                        "inject_layer": inject_layer,
                        "reset_layer": reset_layer,
                        "condition": condition,
                        "span_len": len(host["audio_positions"]),
                        "baseline_response": baseline["response"],
                        "baseline_explicit_refusal": baseline["explicit_refusal"],
                        "baseline_r_tab_margin": baseline["r_tab_margin"],
                        "donor_explicit_refusal": donor_baseline["explicit_refusal"],
                        "donor_r_tab_margin": donor_baseline["r_tab_margin"],
                        "response": generated["response"],
                        "explicit_refusal": generated["explicit_refusal"],
                        "r_tab_margin": generated["r_tab_margin"],
                        "applied_counts": generated["applied_counts"],
                        "identity_response_exact": (
                            generated["response"] == baseline["response"]
                            if condition in {"identity", "reset_only"}
                            else None
                        ),
                        "identity_margin_abs_error": (
                            abs(generated["r_tab_margin"] - baseline["r_tab_margin"])
                            if condition in {"identity", "reset_only"}
                            else None
                        ),
                    }
                    completed += 1
                    _checkpoint_mapping(state, path, completed_since_resume=completed, every=20)
    expected = {
        (
            str(pair["pair_id"]),
            direction,
            int(window.inject_layer),
            int(window.reset_layer),
            condition,
        )
        for pair in cohort
        for direction in gate.directions
        for window in gate.windows
        for condition in gate.conditions
    }
    _assert_expected_keys("escape-reset", expected, set(state))
    _checkpoint_mapping(state, path, completed_since_resume=completed, force=True)
    return [state[key] for key in sorted(state)]


def _mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    n_bootstrap: int,
    ci_alpha: float,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if not len(array):
        return {"mean": None, "ci_low": None, "ci_high": None, "n": 0}
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(array), size=(n_bootstrap, len(array)))
    draws = array[indices].mean(axis=1)
    low, high = np.quantile(draws, [ci_alpha / 2.0, 1.0 - ci_alpha / 2.0])
    return {
        "mean": float(array.mean()),
        "ci_low": float(low),
        "ci_high": float(high),
        "n": len(array),
    }


def _group_patch_records(
    cfg: Exp3RunConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> dict[str, Any]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output: dict[str, Any] = {}
    for group_key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        margin_delta = [
            float(row["r_tab_margin"]) - float(row["baseline_r_tab_margin"]) for row in values
        ]
        refusal_delta = [
            float(bool(row["explicit_refusal"])) - float(bool(row["baseline_explicit_refusal"]))
            for row in values
            if row.get("explicit_refusal") is not None
            and row.get("baseline_explicit_refusal") is not None
        ]
        name = "|".join(
            f"{field}={value}" for field, value in zip(group_fields, group_key, strict=True)
        )
        output[name] = {
            "fields": dict(zip(group_fields, group_key, strict=True)),
            "margin_delta": _mean_ci(
                margin_delta,
                seed=_stable_int(cfg.seed, name, "margin"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "explicit_refusal_delta": _mean_ci(
                refusal_delta,
                seed=_stable_int(cfg.seed, name, "refusal"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
        }
    return output


def _behavior_transport(
    cfg: Exp3RunConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> dict[str, Any]:
    """Aggregate binary output changes in the paired donor's direction.

    On a discordant pair, ``donorward_score`` is one exactly when the patched
    marker matches the donor marker and zero when it remains at the host state.
    Stable pairs are kept as controls and never mixed into that denominator.
    """

    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("explicit_refusal") is None or row.get("donor_explicit_refusal") is None:
            continue
        groups[tuple(row.get(field) for field in group_fields)].append(row)
    output: dict[str, Any] = {}
    for group_key, values in sorted(groups.items(), key=lambda item: str(item[0])):
        transitions: Counter[str] = Counter()
        donorward: list[float] = []
        donor_match_gain: list[float] = []
        for row in values:
            host = bool(row["baseline_explicit_refusal"])
            patched = bool(row["explicit_refusal"])
            donor = bool(row["donor_explicit_refusal"])
            transitions[f"{'R' if host else 'NR'}_to_{'R' if patched else 'NR'}"] += 1
            donor_match_gain.append(float(patched == donor) - float(host == donor))
            desired = int(donor) - int(host)
            if desired != 0:
                donorward.append(float((int(patched) - int(host)) * desired))
        name = "|".join(
            f"{field}={value}" for field, value in zip(group_fields, group_key, strict=True)
        )
        output[name] = {
            "fields": dict(zip(group_fields, group_key, strict=True)),
            "host_to_patched": {
                key: transitions[key] for key in ("R_to_R", "R_to_NR", "NR_to_R", "NR_to_NR")
            },
            "marker_flip_rate": _mean_ci(
                [
                    float(bool(row["explicit_refusal"]) != bool(row["baseline_explicit_refusal"]))
                    for row in values
                ],
                seed=_stable_int(cfg.seed, name, "marker-flip"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "donor_match_gain": _mean_ci(
                donor_match_gain,
                seed=_stable_int(cfg.seed, name, "donor-match"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "donorward_score_discordant_only": _mean_ci(
                donorward,
                seed=_stable_int(cfg.seed, name, "donorward"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "n": len(values),
            "n_discordant_host_donor": len(donorward),
        }
    return output


def _transport_alignment(
    cfg: Exp3RunConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    group_fields: Sequence[str],
) -> dict[str, Any]:
    """Positive means the patch moved t_AB margin toward the paired donor arm."""

    host_margin: dict[tuple[str, str], float] = {}
    for row in rows:
        host_margin[(str(row["pair_id"]), str(row["direction"]))] = float(
            row["baseline_r_tab_margin"]
        )
    opposite = {
        "source_to_target": "target_to_source",
        "target_to_source": "source_to_target",
    }
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    fractions: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in rows:
        direction = str(row["direction"])
        host = float(row["baseline_r_tab_margin"])
        donor_value = row.get("donor_r_tab_margin")
        donor = (
            float(donor_value)
            if donor_value is not None
            else host_margin.get((str(row["pair_id"]), opposite[direction]))
        )
        if donor is None:
            continue
        desired = donor - host
        observed = float(row["r_tab_margin"]) - host
        key = tuple(row.get(field) for field in group_fields)
        grouped[key].append(observed * desired)
        if abs(desired) > 1e-6:
            fractions[key].append(observed / desired)
    output: dict[str, Any] = {}
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        name = "|".join(f"{field}={value}" for field, value in zip(group_fields, key, strict=True))
        output[name] = {
            "fields": dict(zip(group_fields, key, strict=True)),
            "signed_alignment": _mean_ci(
                values,
                seed=_stable_int(cfg.seed, name, "alignment"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "transport_fraction": _mean_ci(
                fractions.get(key, []),
                seed=_stable_int(cfg.seed, name, "fraction"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
        }
    return output


def _real_minus_wrong(
    cfg: Exp3RunConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    site_fields: Sequence[str],
) -> dict[str, Any]:
    index = {
        (
            str(row["pair_id"]),
            str(row["direction"]),
            *(row.get(field) for field in site_fields),
            str(row["condition"]),
        ): row
        for row in rows
    }
    grouped: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    fractions: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    binary: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for key, real in index.items():
        if key[-1] != "real":
            continue
        wrong_key = (*key[:-1], "wrong_item")
        wrong = index.get(wrong_key)
        if wrong is None:
            continue
        site = (key[1], *key[2:-1])
        host = float(real["baseline_r_tab_margin"])
        donor = float(real["donor_r_tab_margin"])
        desired = donor - host
        real_delta = float(real["r_tab_margin"]) - host
        wrong_delta = float(wrong["r_tab_margin"]) - host
        grouped[site].append((real_delta - wrong_delta) * desired)
        if abs(desired) > 1e-6:
            fractions[site].append((real_delta - wrong_delta) / desired)
        if (
            real.get("explicit_refusal") is not None
            and wrong.get("explicit_refusal") is not None
            and real.get("donor_explicit_refusal") is not None
        ):
            host_r = int(bool(real["baseline_explicit_refusal"]))
            donor_r = int(bool(real["donor_explicit_refusal"]))
            desired_r = donor_r - host_r
            if desired_r != 0:
                real_score = (int(bool(real["explicit_refusal"])) - host_r) * desired_r
                wrong_score = (int(bool(wrong["explicit_refusal"])) - host_r) * desired_r
                binary[site].append(float(real_score - wrong_score))
    output = {}
    fields = ["direction", *site_fields]
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        name = "|".join(f"{field}={value}" for field, value in zip(fields, key, strict=True))
        output[name] = {
            "fields": dict(zip(fields, key, strict=True)),
            "donor_aligned_real_minus_wrong": _mean_ci(
                values,
                seed=_stable_int(cfg.seed, name, "real-minus-wrong"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "transport_fraction_real_minus_wrong": _mean_ci(
                fractions.get(key, []),
                seed=_stable_int(cfg.seed, name, "real-minus-wrong-fraction"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
            "binary_donorward_real_minus_wrong_discordant_only": _mean_ci(
                binary.get(key, []),
                seed=_stable_int(cfg.seed, name, "real-minus-wrong-binary"),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            ),
        }
    return output


def _escape_survival(
    cfg: Exp3RunConfig,
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Paired inject-only versus inject+reset survival at each tested window."""

    index = {
        (
            str(row["pair_id"]),
            str(row["direction"]),
            int(row["inject_layer"]),
            int(row["reset_layer"]),
            str(row["condition"]),
        ): row
        for row in rows
    }
    groups: dict[tuple[Any, ...], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for key, injected in index.items():
        if key[-1] != "inject_only":
            continue
        reset = index.get((*key[:-1], "inject_reset"))
        if reset is None:
            continue
        group = (
            key[2],
            key[3],
            injected.get("role"),
            injected.get("selection_role"),
            injected.get("transition"),
        )
        host = float(injected["baseline_r_tab_margin"])
        donor = float(injected["donor_r_tab_margin"])
        desired = donor - host
        inject_change = float(injected["r_tab_margin"]) - host
        reset_change = float(reset["r_tab_margin"]) - host
        groups[group]["inject_alignment"].append(inject_change * desired)
        groups[group]["inject_reset_alignment"].append(reset_change * desired)
        groups[group]["reset_minus_inject_alignment"].append(
            (reset_change - inject_change) * desired
        )
        if abs(inject_change) > 1e-6:
            groups[group]["survival_fraction"].append(reset_change / inject_change)
        if (
            injected.get("explicit_refusal") is not None
            and reset.get("explicit_refusal") is not None
        ):
            host_r = int(bool(injected["baseline_explicit_refusal"]))
            donor_r = int(bool(injected["donor_explicit_refusal"]))
            desired_r = donor_r - host_r
            if desired_r != 0:
                inject_score = (int(bool(injected["explicit_refusal"])) - host_r) * desired_r
                reset_score = (int(bool(reset["explicit_refusal"])) - host_r) * desired_r
                groups[group]["binary_inject_donorward"].append(float(inject_score))
                groups[group]["binary_inject_reset_donorward"].append(float(reset_score))
                groups[group]["binary_reset_minus_inject"].append(float(reset_score - inject_score))
    fields = ["inject_layer", "reset_layer", "role", "selection_role", "transition"]
    output: dict[str, Any] = {}
    for group, values in sorted(groups.items(), key=lambda item: str(item[0])):
        name = "|".join(f"{field}={value}" for field, value in zip(fields, group, strict=True))
        summary = {
            metric: _mean_ci(
                metric_values,
                seed=_stable_int(cfg.seed, name, metric),
                n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                ci_alpha=cfg.exp3.behavior.ci_alpha,
            )
            for metric, metric_values in sorted(values.items())
        }
        output[name] = {"fields": dict(zip(fields, group, strict=True)), **summary}
    return output


def _integrity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    identity = [row for row in rows if row.get("identity_response_exact") is not None]
    applied = []
    for row in rows:
        if row.get("applied_count") is not None:
            applied.append(int(row["applied_count"]))
        applied.extend(int(value) for value in row.get("applied_counts") or [])
    identity_margins = [
        float(row["identity_margin_abs_error"])
        for row in rows
        if row.get("identity_margin_abs_error") is not None
    ]
    return {
        "n_rows": len(rows),
        "identity_exact": sum(bool(row["identity_response_exact"]) for row in identity),
        "identity_checked": len(identity),
        "identity_pass": bool(identity)
        and all(bool(row["identity_response_exact"]) for row in identity),
        "identity_margin_checked": len(identity_margins),
        "identity_margin_max_abs_error": max(identity_margins) if identity_margins else None,
        "identity_margin_pass": bool(identity_margins)
        and all(value <= 1e-4 for value in identity_margins),
        "applied_count_all_one": bool(applied) and all(value == 1 for value in applied),
        "applied_count_values": dict(
            sorted({str(value): applied.count(value) for value in set(applied)}.items())
        ),
    }


def assert_input_dose_gate(cfg: Exp3RunConfig, run_dir: Path) -> None:
    """Fail closed unless every M1 cell exists and both physical endpoints reproduce."""

    gate = cfg.exp3.input_dose
    if not gate.enabled:
        return
    path = _artifact(run_dir, cfg.exp3.artifacts.input_dose_file)
    if not path.is_file():
        raise RuntimeError("M1 input-dose artifact is missing")
    rows = load_jsonl(path)
    contrast = _contrast(cfg, gate.contrast)
    all_pairs = [row for row in load_pairs(cfg, run_dir) if row["contrast"] == contrast.name]
    pairs = _stratified_limit(all_pairs, gate.max_pairs, seed=cfg.seed)
    expected = {
        (str(pair["pair_id"]), component, float(dose))
        for pair in pairs
        for component in gate.components
        for dose in (gate.doses if component == "all" else [1.0])
    }
    actual = {(str(row["pair_id"]), str(row["component"]), float(row["dose"])) for row in rows}
    _assert_expected_keys("input-dose gate", expected, actual)
    endpoints = [row for row in rows if row.get("endpoint_expected") is not None]
    if len(endpoints) != 2 * len(pairs) or not all(
        bool(row["endpoint_exact"]) for row in endpoints
    ):
        raise RuntimeError("M1 physical endpoint integrity failed; downstream stages are blocked")


def assert_span_patch_gate(cfg: Exp3RunConfig, run_dir: Path) -> None:
    """Fail closed unless M2 is complete and every identity/hook check passes."""

    gate = cfg.exp3.span_patch
    if not gate.enabled:
        return
    path = _artifact(run_dir, cfg.exp3.artifacts.span_patch_file)
    if not path.is_file():
        raise RuntimeError("M2 span-patch artifact is missing")
    rows = load_jsonl(path)
    contrast = _contrast(cfg, gate.contrast)
    cohort = _mechanism_cohort(cfg, run_dir, contrast_name=contrast.name)
    expected = {
        (str(pair["pair_id"]), direction, int(layer), condition)
        for pair in cohort
        for direction in gate.directions
        for layer in gate.layers
        for condition in gate.conditions
    }
    actual = {
        (
            str(row["pair_id"]),
            str(row["direction"]),
            int(row["layer"]),
            str(row["condition"]),
        )
        for row in rows
    }
    _assert_expected_keys("span-patch gate", expected, actual)
    integrity = _integrity(rows)
    if not (
        integrity["identity_pass"]
        and integrity["identity_margin_pass"]
        and integrity["applied_count_all_one"]
    ):
        raise RuntimeError(f"M2 identity/hook integrity failed: {integrity}")


def analyze_all(cfg: Exp3RunConfig, run_dir: Path) -> dict[str, Any]:
    """Combine all available stages without inventing a result for missing stages."""

    behavior = analyze_behavior(cfg, run_dir)
    metrics: dict[str, Any] = {
        "schema_version": cfg.exp3.schema_version,
        "claim_scope": "explicit-refusal instability, not jailbreak success",
        "behavior": behavior,
    }
    representation_path = _artifact(run_dir, cfg.exp3.artifacts.representation_metrics_file)
    if representation_path.is_file():
        metrics["representation"] = json.loads(representation_path.read_text())
    content_path = _artifact(run_dir, cfg.exp3.artifacts.content_audit_file)
    both_faithful_ids: set[str] = set()
    if content_path.is_file():
        content = load_jsonl(content_path)
        by_pair: dict[str, dict[str, bool]] = defaultdict(dict)
        for row in content:
            by_pair[str(row["pair_id"])][str(row["side"])] = bool(row["content_faithful"])
        both_faithful_ids = {
            pair_id
            for pair_id, sides in by_pair.items()
            if sides.get("source", False) and sides.get("target", False)
        }
        metrics["content_audit"] = {
            "n_rows": len(content),
            "n_pairs": len(by_pair),
            "both_faithful": len(both_faithful_ids),
            "note": "stratification audit; not a primary-endpoint exclusion filter",
        }
        pairs = [
            pair for pair in load_pairs(cfg, run_dir) if str(pair["pair_id"]) in both_faithful_ids
        ]
        behavior_rows = load_jsonl(_artifact(run_dir, cfg.exp3.artifacts.behavior_file))
        faithful_behavior: dict[str, Any] = {}
        for contrast in _enabled_contrasts(cfg):
            roles: dict[str, Any] = {}
            for role in cfg.exp3.behavior.roles:
                source, target = _paired_behavior_rows(
                    pairs,
                    behavior_rows,
                    contrast=contrast.name,
                    role=role,
                    status_key="marker_status",
                )
                if source and target:
                    roles[role] = summarize_paired_transitions(
                        source,
                        target,
                        n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                        ci_alpha=cfg.exp3.behavior.ci_alpha,
                        seed=cfg.seed,
                    )
            if roles:
                faithful_behavior[contrast.name] = roles
        metrics["behavior_both_content_faithful_audited_subset"] = faithful_behavior
    content_patch_path = _artifact(run_dir, cfg.exp3.artifacts.content_patch_file)
    if content_patch_path.is_file():
        content_patch_rows = load_jsonl(content_patch_path)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for row in content_patch_rows:
            grouped[(str(row["direction"]), str(row["role"]))].append(row)
        content_patch_metrics: dict[str, Any] = {}
        for key, rows in sorted(grouped.items()):
            name = f"direction={key[0]}|role={key[1]}"
            content_patch_metrics[name] = {
                "delta_wer": _mean_ci(
                    [float(row["delta_wer"]) for row in rows],
                    seed=_stable_int(cfg.seed, name, "wer"),
                    n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                    ci_alpha=cfg.exp3.behavior.ci_alpha,
                ),
                "delta_token_overlap": _mean_ci(
                    [float(row["delta_token_overlap"]) for row in rows],
                    seed=_stable_int(cfg.seed, name, "overlap"),
                    n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                    ci_alpha=cfg.exp3.behavior.ci_alpha,
                ),
                "delta_donor_token_overlap": _mean_ci(
                    [float(row["delta_donor_token_overlap"]) for row in rows],
                    seed=_stable_int(cfg.seed, name, "donor-overlap"),
                    n_bootstrap=cfg.exp3.behavior.n_bootstrap,
                    ci_alpha=cfg.exp3.behavior.ci_alpha,
                ),
                "content_faithfulness_changed": sum(
                    bool(row["baseline_content_faithful"]) != bool(row["patched_content_faithful"])
                    for row in rows
                ),
                "n": len(rows),
            }
        metrics["content_patch_audit"] = {
            "layer": cfg.exp3.content_audit.patch_layer,
            "groups": content_patch_metrics,
        }
    dose_path = _artifact(run_dir, cfg.exp3.artifacts.input_dose_file)
    if dose_path.is_file():
        dose_rows = load_jsonl(dose_path)
        endpoints = [row for row in dose_rows if row.get("endpoint_exact") is not None]
        metrics["input_dose"] = {
            "n_rows": len(dose_rows),
            "endpoint_exact": sum(bool(row["endpoint_exact"]) for row in endpoints),
            "endpoint_checked": len(endpoints),
            "endpoint_integrity_pass": bool(endpoints)
            and all(bool(row["endpoint_exact"]) for row in endpoints),
            "by_component_dose": _group_patch_records(
                cfg,
                dose_rows,
                group_fields=["component", "dose", "role"],
            ),
            "donorward_behavior": _behavior_transport(
                cfg, dose_rows, group_fields=["component", "dose", "role"]
            ),
            "donorward_r_tab": _transport_alignment(
                cfg, dose_rows, group_fields=["component", "dose", "role"]
            ),
        }
        if both_faithful_ids:
            faithful_dose = [row for row in dose_rows if str(row["pair_id"]) in both_faithful_ids]
            metrics["input_dose"]["both_content_faithful"] = _group_patch_records(
                cfg,
                faithful_dose,
                group_fields=["component", "dose", "role"],
            )
            metrics["input_dose"]["both_content_faithful_donorward"] = _behavior_transport(
                cfg, faithful_dose, group_fields=["component", "dose", "role"]
            )
    span_path = _artifact(run_dir, cfg.exp3.artifacts.span_patch_file)
    if span_path.is_file():
        rows = load_jsonl(span_path)
        metrics["span_patch"] = {
            "integrity": _integrity(rows),
            "effects": _group_patch_records(
                cfg,
                rows,
                group_fields=[
                    "direction",
                    "layer",
                    "condition",
                    "role",
                    "selection_role",
                    "transition",
                ],
            ),
            "donorward_behavior": _behavior_transport(
                cfg,
                rows,
                group_fields=["layer", "condition", "role", "selection_role", "transition"],
            ),
            "transport_alignment": _transport_alignment(
                cfg, rows, group_fields=["layer", "condition", "role"]
            ),
            "real_minus_wrong": _real_minus_wrong(
                cfg,
                rows,
                site_fields=["layer", "role", "selection_role", "transition"],
            ),
        }
        if both_faithful_ids:
            faithful = [row for row in rows if str(row["pair_id"]) in both_faithful_ids]
            metrics["span_patch"]["both_content_faithful"] = {
                "effects": _group_patch_records(
                    cfg,
                    faithful,
                    group_fields=[
                        "direction",
                        "layer",
                        "condition",
                        "role",
                        "selection_role",
                        "transition",
                    ],
                ),
                "transport_alignment": _transport_alignment(
                    cfg, faithful, group_fields=["layer", "condition", "role"]
                ),
                "donorward_behavior": _behavior_transport(
                    cfg,
                    faithful,
                    group_fields=["layer", "condition", "role", "selection_role", "transition"],
                ),
            }
        if content_patch_path.is_file():
            content_patch_rows = load_jsonl(content_patch_path)
            answer_real = {
                (str(row["pair_id"]), str(row["direction"])): row
                for row in rows
                if int(row["layer"]) == cfg.exp3.content_audit.patch_layer
                and row["condition"] == "real"
                and row.get("explicit_refusal") is not None
            }
            joined = [
                (content, answer_real[(str(content["pair_id"]), str(content["direction"]))])
                for content in content_patch_rows
                if (str(content["pair_id"]), str(content["direction"])) in answer_real
            ]
            margin_delta = np.asarray(
                [
                    (float(answer["r_tab_margin"]) - float(answer["baseline_r_tab_margin"]))
                    * (float(answer["donor_r_tab_margin"]) - float(answer["baseline_r_tab_margin"]))
                    for _content, answer in joined
                ],
                dtype=float,
            )
            overlap_delta = np.asarray(
                [float(content["delta_donor_token_overlap"]) for content, _answer in joined],
                dtype=float,
            )
            correlation = None
            if len(joined) >= 3 and np.std(margin_delta) > 0 and np.std(overlap_delta) > 0:
                correlation = float(np.corrcoef(margin_delta, overlap_delta)[0, 1])
            metrics["span_patch"]["refusal_content_decoupling"] = {
                "n": len(joined),
                "corr_donorward_margin_vs_donor_transcript_overlap": correlation,
                "refusal_marker_changed_content_faithfulness_unchanged": sum(
                    bool(answer["explicit_refusal"]) != bool(answer["baseline_explicit_refusal"])
                    and bool(content["patched_content_faithful"])
                    == bool(content["baseline_content_faithful"])
                    for content, answer in joined
                ),
            }
    readout_path = _artifact(run_dir, cfg.exp3.artifacts.readout_patch_file)
    if readout_path.is_file():
        rows = load_jsonl(readout_path)
        metrics["readout_patch"] = {
            "integrity": _integrity(rows),
            "final_layer_pipeline_control": {
                "layer": max(cfg.exp3.readout_patch.layers),
                "n": sum(
                    int(row["layer"]) == max(cfg.exp3.readout_patch.layers)
                    and row["condition"] == "real"
                    for row in rows
                ),
                "max_donor_margin_abs_error": max(
                    (
                        float(row["final_control_margin_abs_error"])
                        for row in rows
                        if row.get("final_control_margin_abs_error") is not None
                    ),
                    default=None,
                ),
            },
            "effects": _group_patch_records(
                cfg,
                rows,
                group_fields=[
                    "direction",
                    "layer",
                    "condition",
                    "role",
                    "selection_role",
                    "transition",
                ],
            ),
            "donorward_behavior": _behavior_transport(
                cfg,
                rows,
                group_fields=["layer", "condition", "role", "selection_role", "transition"],
            ),
            "transport_alignment": _transport_alignment(
                cfg, rows, group_fields=["layer", "condition", "role"]
            ),
            "real_minus_wrong": _real_minus_wrong(
                cfg,
                rows,
                site_fields=["layer", "role", "selection_role", "transition"],
            ),
        }
        if both_faithful_ids:
            faithful = [row for row in rows if str(row["pair_id"]) in both_faithful_ids]
            metrics["readout_patch"]["both_content_faithful"] = {
                "effects": _group_patch_records(
                    cfg,
                    faithful,
                    group_fields=["direction", "layer", "condition", "role"],
                ),
                "transport_alignment": _transport_alignment(
                    cfg, faithful, group_fields=["layer", "condition", "role"]
                ),
            }
    escape_path = _artifact(run_dir, cfg.exp3.artifacts.escape_reset_file)
    if escape_path.is_file():
        rows = load_jsonl(escape_path)
        metrics["escape_reset"] = {
            "integrity": _integrity(rows),
            "inject_reset_survival": _escape_survival(cfg, rows),
            "effects": _group_patch_records(
                cfg,
                rows,
                group_fields=[
                    "direction",
                    "inject_layer",
                    "reset_layer",
                    "condition",
                    "role",
                    "selection_role",
                    "transition",
                ],
            ),
            "donorward_behavior": _behavior_transport(
                cfg,
                rows,
                group_fields=[
                    "inject_layer",
                    "reset_layer",
                    "condition",
                    "role",
                    "selection_role",
                    "transition",
                ],
            ),
            "transport_alignment": _transport_alignment(
                cfg,
                rows,
                group_fields=["inject_layer", "reset_layer", "condition", "role"],
            ),
        }
        if both_faithful_ids:
            faithful = [row for row in rows if str(row["pair_id"]) in both_faithful_ids]
            metrics["escape_reset"]["both_content_faithful"] = {
                "inject_reset_survival": _escape_survival(cfg, faithful),
                "effects": _group_patch_records(
                    cfg,
                    faithful,
                    group_fields=[
                        "direction",
                        "inject_layer",
                        "reset_layer",
                        "condition",
                        "role",
                        "selection_role",
                        "transition",
                    ],
                ),
                "transport_alignment": _transport_alignment(
                    cfg,
                    faithful,
                    group_fields=["inject_layer", "reset_layer", "condition", "role"],
                ),
                "donorward_behavior": _behavior_transport(
                    cfg,
                    faithful,
                    group_fields=[
                        "inject_layer",
                        "reset_layer",
                        "condition",
                        "role",
                        "selection_role",
                        "transition",
                    ],
                ),
            }
    integrity_blocks = [
        value["integrity"]
        for key, value in metrics.items()
        if key in {"span_patch", "readout_patch", "escape_reset"}
    ]
    metrics["integrity_summary"] = {
        "all_available_identity_checks_pass": bool(integrity_blocks)
        and all(
            block["identity_pass"] and block["identity_margin_pass"] for block in integrity_blocks
        ),
        "all_available_hooks_applied_once": bool(integrity_blocks)
        and all(block["applied_count_all_one"] for block in integrity_blocks),
        "input_endpoints_pass": metrics.get("input_dose", {}).get("endpoint_integrity_pass"),
    }
    save_json(metrics, _artifact(run_dir, cfg.exp3.artifacts.metrics_file))
    _write_analysis_markdown(cfg, run_dir, metrics)
    return metrics


def _write_analysis_markdown(
    cfg: Exp3RunConfig,
    run_dir: Path,
    metrics: Mapping[str, Any],
) -> None:
    lines = [
        "# Exp3 Qwen refusal-instability mechanism analysis",
        "",
        "> 이 자동 보고서는 explicit-refusal 패턴의 존재 여부만 다룬다. "
        "패턴 부재를 승낙·유해한 답변 성공으로 해석하지 않는다.",
        "",
        "## Integrity",
        "",
        "- identity checks: "
        f"`{metrics['integrity_summary']['all_available_identity_checks_pass']}`",
        "- hook applied exactly once: "
        f"`{metrics['integrity_summary']['all_available_hooks_applied_once']}`",
        f"- input dose endpoints reproduce physical arms: "
        f"`{metrics['integrity_summary']['input_endpoints_pass']}`",
        "",
        "## Reading order",
        "",
        "1. `behavior`: any paired refusal-marker disagreement and both directions",
        "2. `input_dose`: exact model-input sufficiency and temporal component controls",
        "3. `representation`: encoder/projector/LLM/t_AB cross-fitted readout localization",
        "4. `span_patch`: same-item transport versus wrong-item/random/shuffle controls",
        "5. `escape_reset`: whether the effect survives after later audio-token reset",
        "6. `readout_patch`: L10/L18 comparison plus final-layer t_AB pipeline control",
        "",
        "## Claim discipline",
        "",
        "- 내용 감사 실패를 숨기거나 primary 표본에서 사후 삭제하지 않는다.",
        "- real과 wrong-item이 같으면 item-specific mechanism 주장을 generic transport로 낮춘다.",
        "- inject+reset 효과가 사라지면 non-audio path escape 주장을 하지 않는다.",
        "- harmful/benign 차이가 없더라도 refusal-instability 현상과 모순되지 않는다.",
    ]
    path = _artifact(run_dir, cfg.exp3.artifacts.analysis_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
