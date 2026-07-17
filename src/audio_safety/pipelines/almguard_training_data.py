"""Fail-closed preparation helpers for Run 9 ALMGuard SAP inputs.

Pinned ALMGuard main.py accepts exactly three WAV directories, numerically sorts
each directory, and consumes zip(*audio_lists) in directory order. This module
stages equal-length directories and records that exact interleave in JSONL so
training, positive-control evaluation, and ASR references cannot drift.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

FAMILIES = ("advwave_p", "advwave_suffix", "pair_audio")
FORBIDDEN_TOKENS = ("pv_standard", "phase")
SOURCE_DATASET = "WeifeiJin/AdvBench-Audio"
SOURCE_LICENSE = "CC-BY-NC-4.0"
PATH_KEYS = ("path", "source_path", "source_audio_path", "variant_path", "run7_variant_path")


class DataContractError(ValueError):
    """Raised when SAP input preparation would violate the registered gate contract."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise DataContractError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def save_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def text_digest(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_key(value: object) -> str | None:
    compact = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    if not compact or compact == "none":
        return None
    if "advbench" in compact:
        return "advbench"
    if any(token in compact for token in ("safebench", "figstep", "forbidqi")):
        return "safebench"
    return compact


def _resolved_manifest_path(raw: object, data_dir: Path) -> str:
    path = Path(str(raw))
    if not path.is_absolute():
        path = data_dir / path
    return str(path.resolve(strict=False))


def assert_forbidden_absent(paths: Iterable[Path | str], tokens: Sequence[str]) -> None:
    forbidden = tuple(token.casefold() for token in tokens if token)
    if not forbidden:
        raise DataContractError("at least one forbidden attack token is required")
    hits: list[str] = []
    for raw in paths:
        path = Path(raw)
        forms = (str(path.absolute()).casefold(), str(path.resolve(strict=False)).casefold())
        if any(token in form for token in forbidden for form in forms):
            hits.append(str(path))
    if hits:
        raise DataContractError(
            "phase-under-test token found in candidate training/holdout paths: "
            + ", ".join(hits[:3])
        )


def numeric_wavs(directory: Path, forbidden_tokens: Sequence[str] = ()) -> list[Path]:
    if not directory.is_dir():
        raise DataContractError(f"WAV directory does not exist: {directory}")
    by_index: dict[int, Path] = {}
    for path in directory.iterdir():
        if path.suffix.casefold() != ".wav":
            continue
        if not path.stem.isdecimal():
            raise DataContractError(
                f"ALMGuard requires integer WAV basenames; found {path.name!r} in {directory}"
            )
        index = int(path.stem)
        if index in by_index:
            raise DataContractError(f"duplicate numeric WAV index {index} in {directory}")
        if not path.is_file():
            raise DataContractError(f"WAV path is not a readable file: {path}")
        by_index[index] = path
    if not by_index:
        raise DataContractError(f"no WAV files found in {directory}")
    ordered = [by_index[index] for index in sorted(by_index)]
    if forbidden_tokens:
        assert_forbidden_absent([directory, *ordered], forbidden_tokens)
    return ordered


def metadata_by_source_index(
    metadata_path: Path,
    clean_audio_dir: Path,
    required_indices: Sequence[int],
) -> dict[int, dict]:
    metadata: dict[int, dict] = {}
    for row in load_jsonl(metadata_path):
        audio_name = Path(str(row.get("audio") or "")).name
        stem = Path(audio_name).stem
        prompt = row.get("prompt")
        if not stem.isdecimal() or not isinstance(prompt, str) or not prompt.strip():
            raise DataContractError(f"invalid AdvBench metadata row in {metadata_path}")
        index = int(stem)
        if index in metadata:
            raise DataContractError(f"duplicate AdvBench source index {index}")
        clean_path = clean_audio_dir / audio_name
        metadata[index] = {
            **row,
            "source_index": index,
            "clean_audio_path": str(clean_path.resolve(strict=False)),
            "reference_sha256": text_digest(prompt),
        }
    missing = set(required_indices) - set(metadata)
    if missing:
        raise DataContractError(f"missing AdvBench metadata indices: {sorted(missing)}")
    for index in required_indices:
        clean_path = Path(metadata[index]["clean_audio_path"])
        if not clean_path.is_file():
            raise DataContractError(f"missing clean AdvBench source audio: {clean_path}")
    return metadata


def family_source_maps(
    almguard_repo: Path,
    forbidden_tokens: Sequence[str],
) -> dict[str, dict[int, Path]]:
    maps: dict[str, dict[int, Path]] = {}
    for family in FAMILIES:
        directory = almguard_repo / "results" / family
        paths = numeric_wavs(directory, forbidden_tokens)
        maps[family] = {int(path.stem): path.resolve() for path in paths}
    return maps


def split_common_indices(
    family_maps: dict[str, dict[int, Path]],
    *,
    train_per_family: int,
    holdout_per_family: int,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    if train_per_family < 1 or holdout_per_family < 1:
        raise DataContractError("train and holdout counts must both be positive")
    common = sorted(set.intersection(*(set(family_maps[name]) for name in FAMILIES)))
    needed = train_per_family + holdout_per_family
    if len(common) < needed:
        raise DataContractError(
            f"need {needed} common family indices but pinned inputs provide {len(common)}"
        )
    shuffled = list(common)
    random.Random(seed).shuffle(shuffled)
    train = shuffled[:train_per_family]
    holdout = shuffled[train_per_family:needed]
    return train, holdout, common


def _manifest_path(path: Path, data_dir: Path) -> str:
    try:
        return str(path.relative_to(data_dir))
    except ValueError:
        return str(path)


def build_split_rows(
    split: str,
    indices: Sequence[int],
    family_maps: dict[str, dict[int, Path]],
    metadata: dict[int, dict],
    output_root: Path,
    data_dir: Path,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    role = "almguard_sap_train" if split == "train" else "almguard_positive_control"
    for within_index, source_index in enumerate(indices):
        meta = metadata[source_index]
        for family_index, family in enumerate(FAMILIES):
            source_path = family_maps[family][source_index]
            staged_path = output_root / split / family / f"{within_index}.wav"
            rows.append(
                {
                    "sequence_index": len(rows),
                    "within_family_index": within_index,
                    "family_index": family_index,
                    "record_id": f"{split}_{family}_{source_index:04d}",
                    "item_id": f"advbench_audio_{source_index:04d}",
                    "source_item_id": f"advbench_audio_{source_index:04d}",
                    "source_index": source_index,
                    "attack_family": family,
                    "path": _manifest_path(staged_path, data_dir),
                    "source_path": str(source_path),
                    "source_audio_path": meta["clean_audio_path"],
                    "reference_text": meta["prompt"],
                    "asr_reference_sha256": meta["reference_sha256"],
                    "source": SOURCE_DATASET,
                    "license": SOURCE_LICENSE,
                    "split": split,
                    "gate_role": role,
                }
            )
    return rows


def _fingerprints(rows: Sequence[dict], data_dir: Path) -> dict[str, set[str]]:
    ids: set[str] = set()
    paths: set[str] = set()
    texts: set[str] = set()
    sources: set[str] = set()
    for row in rows:
        for key in ("item_id", "source_item_id"):
            if row.get(key) is not None:
                ids.add(str(row[key]))
        for key in PATH_KEYS:
            if row.get(key):
                paths.add(_resolved_manifest_path(row[key], data_dir))
        for key in ("reference_text", "base_reference_text"):
            value = row.get(key)
            if isinstance(value, str) and value.strip():
                texts.add(text_digest(value))
        for key in ("source", "dataset"):
            source = _source_key(row.get(key))
            if source:
                sources.add(source)
    return {"ids": ids, "paths": paths, "texts": texts, "sources": sources}


def assert_no_eval_leakage(
    candidate_rows: Sequence[dict],
    eval_manifests: Sequence[Path],
    data_dir: Path,
) -> None:
    if not eval_manifests:
        raise DataContractError("at least one shared-eval manifest is required for leakage checks")
    eval_rows: list[dict] = []
    for path in eval_manifests:
        if not path.is_file():
            raise DataContractError(f"shared-eval manifest does not exist: {path}")
        eval_rows.extend(load_jsonl(path))
    candidate = _fingerprints(candidate_rows, data_dir)
    evaluation = _fingerprints(eval_rows, data_dir)
    overlaps = {
        name: candidate[name] & evaluation[name] for name in ("ids", "paths", "texts", "sources")
    }
    present = {name: values for name, values in overlaps.items() if values}
    if present:
        summary = ", ".join(f"{name}={len(values)}" for name, values in present.items())
        raise DataContractError(f"candidate SAP data leaks into shared eval ({summary})")


def assert_train_holdout_disjoint(train_rows: Sequence[dict], holdout_rows: Sequence[dict]) -> None:
    def values(rows: Sequence[dict], key: str) -> set[str]:
        return {str(row[key]) for row in rows if row.get(key) is not None}

    for key in ("item_id", "source_item_id", "source_path", "source_audio_path"):
        overlap = values(train_rows, key) & values(holdout_rows, key)
        if overlap:
            raise DataContractError(f"train/holdout {key} overlap: {len(overlap)}")
    train_text = {text_digest(str(row["reference_text"])) for row in train_rows}
    holdout_text = {text_digest(str(row["reference_text"])) for row in holdout_rows}
    if train_text & holdout_text:
        raise DataContractError("train/holdout reference-text content overlap")


def _asr_rows(rows: Sequence[dict]) -> list[dict[str, Any]]:
    return [
        {
            "wav": row["path"],
            "txt": row["reference_text"],
            "record_id": row["record_id"],
            "item_id": row["item_id"],
            "attack_family": row["attack_family"],
        }
        for row in rows
    ]


def build_plan(
    *,
    almguard_repo: Path,
    data_dir: Path,
    output_root: Path,
    metadata_path: Path,
    eval_manifests: Sequence[Path],
    train_per_family: int,
    holdout_per_family: int,
    seed: int,
    forbidden_tokens: Sequence[str],
) -> dict[str, Any]:
    assert_forbidden_absent([output_root, almguard_repo], forbidden_tokens)
    family_maps = family_source_maps(almguard_repo, forbidden_tokens)
    train_indices, holdout_indices, common = split_common_indices(
        family_maps,
        train_per_family=train_per_family,
        holdout_per_family=holdout_per_family,
        seed=seed,
    )
    metadata = metadata_by_source_index(
        metadata_path,
        almguard_repo / "datasets" / "advbench_audios",
        common,
    )
    train_rows = build_split_rows(
        "train", train_indices, family_maps, metadata, output_root, data_dir
    )
    holdout_rows = build_split_rows(
        "positive_control", holdout_indices, family_maps, metadata, output_root, data_dir
    )
    assert_forbidden_absent(
        [row[key] for row in train_rows + holdout_rows for key in PATH_KEYS if row.get(key)],
        forbidden_tokens,
    )
    assert_train_holdout_disjoint(train_rows, holdout_rows)
    assert_no_eval_leakage(train_rows + holdout_rows, eval_manifests, data_dir)
    return {
        "family_maps": family_maps,
        "train_indices": train_indices,
        "holdout_indices": holdout_indices,
        "common_indices": common,
        "train_rows": train_rows,
        "holdout_rows": holdout_rows,
    }


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def prepare_data(
    *,
    almguard_repo: Path,
    data_dir: Path,
    output_root: Path,
    metadata_path: Path,
    eval_manifests: Sequence[Path],
    train_per_family: int,
    holdout_per_family: int,
    seed: int,
    forbidden_tokens: Sequence[str] = FORBIDDEN_TOKENS,
) -> dict[str, Any]:
    output_root = output_root.resolve(strict=False)
    data_dir = data_dir.resolve()
    if output_root.exists():
        raise DataContractError(
            f"refusing to overwrite existing prepared data: {output_root}; use validation mode"
        )
    plan = build_plan(
        almguard_repo=almguard_repo.resolve(),
        data_dir=data_dir,
        output_root=output_root,
        metadata_path=metadata_path.resolve(),
        eval_manifests=[path.resolve() for path in eval_manifests],
        train_per_family=train_per_family,
        holdout_per_family=holdout_per_family,
        seed=seed,
        forbidden_tokens=forbidden_tokens,
    )
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.tmp-", dir=output_root.parent))
    try:
        for split, indices in (
            ("train", plan["train_indices"]),
            ("positive_control", plan["holdout_indices"]),
        ):
            for family in FAMILIES:
                directory = temp_root / split / family
                directory.mkdir(parents=True)
                for within_index, source_index in enumerate(indices):
                    destination = directory / f"{within_index}.wav"
                    destination.symlink_to(plan["family_maps"][family][source_index])

        train_manifest = temp_root / "train_manifest.jsonl"
        holdout_manifest = temp_root / "positive_control_manifest.jsonl"
        train_asr = temp_root / "train_asr_pairs.jsonl"
        holdout_asr = temp_root / "positive_control_asr_pairs.jsonl"
        save_jsonl(train_manifest, plan["train_rows"])
        save_jsonl(holdout_manifest, plan["holdout_rows"])
        save_jsonl(train_asr, _asr_rows(plan["train_rows"]))
        save_jsonl(holdout_asr, _asr_rows(plan["holdout_rows"]))
        contract = {
            "schema_version": 1,
            "method": "ALMGuard (official bundled attacks, held-out positive control)",
            "upstream_commit": _git_commit(almguard_repo),
            "source": SOURCE_DATASET,
            "license": SOURCE_LICENSE,
            "seed": seed,
            "families": list(FAMILIES),
            "forbidden_tokens": list(forbidden_tokens),
            "common_source_indices": plan["common_indices"],
            "train_source_indices": plan["train_indices"],
            "positive_control_source_indices": plan["holdout_indices"],
            "train_per_family": train_per_family,
            "positive_control_per_family": holdout_per_family,
            "train_total": len(plan["train_rows"]),
            "positive_control_total": len(plan["holdout_rows"]),
            "family_source_counts": {
                family: len(plan["family_maps"][family]) for family in FAMILIES
            },
            "eval_manifests": [str(path.resolve()) for path in eval_manifests],
            "metadata_path": str(metadata_path.resolve()),
            "train_wav_dirs": [str(output_root / "train" / family) for family in FAMILIES],
            "positive_control_wav_dirs": [
                str(output_root / "positive_control" / family) for family in FAMILIES
            ],
            "train_manifest": str(output_root / train_manifest.name),
            "positive_control_manifest": str(output_root / holdout_manifest.name),
            "train_asr_pairs": str(output_root / train_asr.name),
            "positive_control_asr_pairs": str(output_root / holdout_asr.name),
            "train_manifest_sha256": file_digest(train_manifest),
            "positive_control_manifest_sha256": file_digest(holdout_manifest),
            "ordering_contract": (
                "numeric basename sort within each family, then "
                "[item for trio in zip(*audio_lists) for item in trio]"
            ),
        }
        (temp_root / "contract.json").write_text(
            json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temp_root.replace(output_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    validate_prepared(output_root, data_dir=data_dir)
    return contract


def _validate_split(
    rows: Sequence[dict],
    asr_rows: Sequence[dict],
    wav_dirs: Sequence[Path],
    data_dir: Path,
    forbidden_tokens: Sequence[str],
) -> None:
    if len(wav_dirs) != 3:
        raise DataContractError("pinned main.py requires exactly three WAV dirs")
    audio_lists = [numeric_wavs(directory, forbidden_tokens) for directory in wav_dirs]
    lengths = {len(paths) for paths in audio_lists}
    if len(lengths) != 1:
        raise DataContractError(
            f"family WAV counts differ: {[len(paths) for paths in audio_lists]}"
        )
    for paths in audio_lists:
        expected_names = [f"{index}.wav" for index in range(len(paths))]
        if [path.name for path in paths] != expected_names:
            raise DataContractError("staged numeric WAV names are not contiguous from zero")
    interleaved = [path.resolve() for trio in zip(*audio_lists, strict=True) for path in trio]
    manifest_paths = [
        Path(_resolved_manifest_path(row["path"], data_dir)).resolve() for row in rows
    ]
    if interleaved != manifest_paths:
        raise DataContractError("manifest order differs from pinned main.py zip interleave")
    if [int(row.get("sequence_index", -1)) for row in rows] != list(range(len(rows))):
        raise DataContractError("manifest sequence_index is not contiguous")
    if len(asr_rows) != len(rows):
        raise DataContractError("ASR pair count differs from manifest count")
    for row, asr in zip(rows, asr_rows, strict=True):
        if _resolved_manifest_path(asr.get("wav"), data_dir) != _resolved_manifest_path(
            row["path"], data_dir
        ):
            raise DataContractError("ASR WAV order differs from manifest order")
        transcript = asr.get("txt")
        if not isinstance(transcript, str):
            raise DataContractError("ASR row is missing transcript text")
        expected_hash = str(row.get("asr_reference_sha256") or "")
        if text_digest(transcript) != expected_hash:
            raise DataContractError("ASR transcript does not match its staged WAV metadata")
        if text_digest(str(row.get("reference_text") or "")) != expected_hash:
            raise DataContractError("manifest reference text hash mismatch")


def validate_prepared(output_root: Path, *, data_dir: Path) -> dict[str, Any]:
    contract_path = output_root / "contract.json"
    if not contract_path.is_file():
        raise DataContractError(f"missing preparation contract: {contract_path}")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if tuple(contract.get("families", ())) != FAMILIES:
        raise DataContractError("prepared family order differs from pinned main.py contract")
    forbidden_tokens = tuple(contract.get("forbidden_tokens") or ())
    train_rows = load_jsonl(output_root / "train_manifest.jsonl")
    holdout_rows = load_jsonl(output_root / "positive_control_manifest.jsonl")
    train_asr = load_jsonl(output_root / "train_asr_pairs.jsonl")
    holdout_asr = load_jsonl(output_root / "positive_control_asr_pairs.jsonl")
    train_dirs = [output_root / "train" / family for family in FAMILIES]
    holdout_dirs = [output_root / "positive_control" / family for family in FAMILIES]
    _validate_split(train_rows, train_asr, train_dirs, data_dir, forbidden_tokens)
    _validate_split(holdout_rows, holdout_asr, holdout_dirs, data_dir, forbidden_tokens)
    assert_train_holdout_disjoint(train_rows, holdout_rows)
    assert_no_eval_leakage(
        train_rows + holdout_rows,
        [Path(path) for path in contract.get("eval_manifests", ())],
        data_dir,
    )
    if file_digest(output_root / "train_manifest.jsonl") != contract.get("train_manifest_sha256"):
        raise DataContractError("train manifest checksum differs from contract")
    if file_digest(output_root / "positive_control_manifest.jsonl") != contract.get(
        "positive_control_manifest_sha256"
    ):
        raise DataContractError("positive-control manifest checksum differs from contract")
    return contract
