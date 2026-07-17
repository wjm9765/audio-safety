#!/usr/bin/env -S uv run python
"""Build the SARSteer defense vectors (Run 9 defense gate).

``paper_faithful`` reads the refusal manifest's ``path`` as harmful audio and
contrasts Q=(a,t) against Q'=(a,t+p) at the last prompt token (Eq. 4), then builds
the safe space by PCA over the paired purified-safe counterparts (§4.2). The
legacy mode accepts refusal text only and is kept for artifact reproduction. Both
fail fast on calibration/eval leakage before model weights are loaded.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from audio_safety.config import load_experiment_config
from audio_safety.data import load_audio_rdo_pairs
from audio_safety.pipelines.rdo_gate import rows_for_split, split_ids
from audio_safety.pipelines.sarsteer import (
    build_sarsteer_vectors,
    extract_benign_speech_activations,
    extract_paper_refusal_vectors,
    extract_text_refusal_vectors,
    sarsteer_position_scope,
    sarsteer_system_prompt,
    save_sarsteer_vectors,
)
from audio_safety.utils.io import get_git_commit, load_jsonl
from audio_safety.utils.paths import resolve_paths, run_output_dir


@dataclass(frozen=True)
class CalibrationSelection:
    """Validated, ordered SARSteer calibration inputs."""

    harmful_texts: list[str]
    harmful_paths: list[Path]
    benign_paths: list[Path]
    refusal_item_ids: frozenset[str]
    benign_item_ids: frozenset[str]
    refusal_sources: frozenset[str]
    benign_sources: frozenset[str]
    # Normalized reference texts of every calibration row (harmful + benign), so
    # eval leakage is caught even when an item is re-keyed under a new item_id.
    calibration_texts: frozenset[str] = frozenset()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--run-name", type=str, required=True)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument(
        "--clean-style",
        type=str,
        default="neutral",
        help="style whose rows are the clean/undefended calibration source",
    )
    parser.add_argument(
        "--refusal-manifest",
        type=Path,
        default=None,
        help=(
            "JSONL rows with item_id/reference_text in legacy mode or item_id/path "
            "(harmful audio) in paper_faithful; pair with --benign-manifest"
        ),
    )
    parser.add_argument(
        "--benign-manifest",
        type=Path,
        default=None,
        help=(
            "explicit JSONL benign-PCA rows with item_id/path; must be paired with "
            "--refusal-manifest"
        ),
    )
    parser.add_argument(
        "--eval-manifest",
        type=Path,
        default=None,
        help="optional JSONL used only for calibration/eval item_id and source leakage checks",
    )
    return parser.parse_args(argv)


def _resolve_manifest_path(path: Path, data_dir: Path) -> Path:
    return (path if path.is_absolute() else data_dir / path).resolve()


def _load_manifest(path: Path, data_dir: Path, *, role: str) -> tuple[Path, list[dict[str, Any]]]:
    resolved = _resolve_manifest_path(path, data_dir)
    if not resolved.is_file():
        raise SystemExit(f"{role} manifest not found: {resolved}")
    try:
        rows = load_jsonl(resolved)
    except Exception as exc:
        raise SystemExit(f"failed to read {role} manifest {resolved}: {exc}") from exc
    if not rows:
        raise SystemExit(f"{role} manifest is empty: {resolved}")
    if any(not isinstance(row, dict) for row in rows):
        raise SystemExit(f"{role} manifest must contain one JSON object per line: {resolved}")
    return resolved, rows


def _required_string(row: dict[str, Any], field: str, *, role: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"{role} row {index} requires a non-empty string field '{field}'")
    return value.strip()


def _normalize_text(text: str) -> str:
    """Casefolded, whitespace-collapsed text for identity comparison."""

    return " ".join(text.split()).casefold()


def _row_text(row: dict[str, Any]) -> str | None:
    value = row.get("reference_text")
    if isinstance(value, str) and value.strip():
        return _normalize_text(value)
    return None


def _source_name(row: dict[str, Any]) -> str | None:
    value = row.get("source", row.get("dataset"))
    if value is None or not isinstance(value, str) or not value.strip():
        return None
    return value.strip()


def _require_requested_count(*, role: str, actual: int, requested: int) -> None:
    if actual < requested:
        raise SystemExit(
            f"{role} calibration requested {requested} unique rows but only {actual} are available"
        )


def select_explicit_calibration(
    refusal_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    *,
    data_dir: Path,
    n_refusal: int,
    n_benign: int,
) -> CalibrationSelection:
    """Validate explicit manifests and select exactly the configured counts."""

    refusal_seen: set[str] = set()
    refusal_candidates: list[tuple[str, str, str | None]] = []
    for index, row in enumerate(refusal_rows):
        item_id = _required_string(row, "item_id", role="refusal", index=index)
        text = _required_string(row, "reference_text", role="refusal", index=index)
        if item_id in refusal_seen:
            raise SystemExit(f"duplicate refusal item_id: {item_id}")
        refusal_seen.add(item_id)
        refusal_candidates.append((item_id, text, _source_name(row)))
    _require_requested_count(role="refusal", actual=len(refusal_candidates), requested=n_refusal)
    refusal_candidates = refusal_candidates[:n_refusal]

    benign_seen: set[str] = set()
    benign_path_seen: set[Path] = set()
    benign_candidates: list[tuple[str, Path, str | None]] = []
    for index, row in enumerate(benign_rows):
        item_id = _required_string(row, "item_id", role="benign", index=index)
        raw_path = _required_string(row, "path", role="benign", index=index)
        if item_id in benign_seen:
            raise SystemExit(f"duplicate benign item_id: {item_id}")
        benign_seen.add(item_id)
        audio_path = _resolve_manifest_path(Path(raw_path), data_dir)
        if audio_path in benign_path_seen:
            raise SystemExit(f"duplicate benign audio path: {audio_path}")
        if not audio_path.is_file():
            raise SystemExit(f"benign audio file not found for {item_id}: {audio_path}")
        benign_path_seen.add(audio_path)
        benign_candidates.append((item_id, audio_path, _source_name(row)))
    _require_requested_count(role="benign PCA", actual=len(benign_candidates), requested=n_benign)
    benign_candidates = benign_candidates[:n_benign]

    refusal_ids = frozenset(item_id for item_id, _, _ in refusal_candidates)
    benign_ids = frozenset(item_id for item_id, _, _ in benign_candidates)
    calibration_overlap = refusal_ids & benign_ids
    if calibration_overlap:
        raise SystemExit(
            f"refusal and benign calibration item_id overlap: {sorted(calibration_overlap)[:10]}"
        )
    return CalibrationSelection(
        harmful_texts=[text for _, text, _ in refusal_candidates],
        harmful_paths=[],
        benign_paths=[path for _, path, _ in benign_candidates],
        refusal_item_ids=refusal_ids,
        benign_item_ids=benign_ids,
        refusal_sources=frozenset(
            source for _, _, source in refusal_candidates if source is not None
        ),
        benign_sources=frozenset(
            source for _, _, source in benign_candidates if source is not None
        ),
    )


def select_paper_audio_calibration(
    refusal_rows: list[dict[str, Any]],
    benign_rows: list[dict[str, Any]],
    *,
    data_dir: Path,
    n_refusal: int,
    n_benign: int,
) -> CalibrationSelection:
    """Select the paper's paired harmful/safe calibration audio (Algorithm 1)."""

    texts: set[str] = set()

    def _collect(rows: list[dict[str, Any]], *, role: str, requested: int):
        seen: set[str] = set()
        path_seen: set[Path] = set()
        out: list[tuple[str, Path, str | None]] = []
        for index, row in enumerate(rows):
            item_id = _required_string(row, "item_id", role=role, index=index)
            raw_path = _required_string(row, "path", role=role, index=index)
            if item_id in seen:
                raise SystemExit(f"duplicate {role} item_id: {item_id}")
            path = _resolve_manifest_path(Path(raw_path), data_dir)
            if path in path_seen:
                raise SystemExit(f"duplicate {role} audio path: {path}")
            if not path.is_file():
                raise SystemExit(f"{role} audio not found for {item_id}: {path}")
            seen.add(item_id)
            path_seen.add(path)
            out.append((item_id, path, _source_name(row)))
        _require_requested_count(role=role, actual=len(out), requested=requested)
        out = out[:requested]
        kept = {item_id for item_id, _, _ in out}
        for row in rows:
            if row.get("item_id") in kept:
                text = _row_text(row)
                if text is not None:
                    texts.add(text)
        return out

    harmful = _collect(refusal_rows, role="paper refusal", requested=n_refusal)
    benign = _collect(benign_rows, role="paper benign PCA", requested=n_benign)

    selected_harmful_paths = {path for _, path, _ in harmful}
    selected_benign_paths = {path for _, path, _ in benign}
    path_overlap = selected_harmful_paths & selected_benign_paths
    if path_overlap:
        raise SystemExit(
            "paper harmful/benign calibration audio path overlap: "
            f"{sorted(map(str, path_overlap))[:10]}"
        )
    return CalibrationSelection(
        harmful_texts=[],
        harmful_paths=[path for _, path, _ in harmful],
        benign_paths=[path for _, path, _ in benign],
        refusal_item_ids=frozenset(item_id for item_id, _, _ in harmful),
        benign_item_ids=frozenset(item_id for item_id, _, _ in benign),
        refusal_sources=frozenset(source for _, _, source in harmful if source is not None),
        benign_sources=frozenset(source for _, _, source in benign if source is not None),
        calibration_texts=frozenset(texts),
    )


def select_legacy_calibration(
    train_rows: list[dict[str, Any]],
    *,
    data_dir: Path,
    clean_style: str,
    n_refusal: int,
    n_benign: int,
) -> CalibrationSelection:
    """Preserve the configured train/split selection while enforcing requested counts."""

    refusal: list[tuple[str, str, str | None]] = []
    refusal_seen: set[str] = set()
    benign: list[tuple[str, Path, str | None]] = []
    benign_seen: set[str] = set()
    for index, row in enumerate(train_rows):
        if row.get("style") != clean_style:
            continue
        if row.get("safety_label") == "harmful":
            item_id = _required_string(row, "item_id", role="legacy refusal", index=index)
            text = str(row.get("reference_text") or "").strip()
            if text and item_id not in refusal_seen:
                refusal_seen.add(item_id)
                refusal.append((item_id, text, _source_name(row)))
        elif row.get("safety_label") == "benign":
            item_id = _required_string(row, "item_id", role="legacy benign", index=index)
            raw_path = _required_string(row, "path", role="legacy benign", index=index)
            if item_id not in benign_seen:
                benign_seen.add(item_id)
                benign.append(
                    (
                        item_id,
                        _resolve_manifest_path(Path(raw_path), data_dir),
                        _source_name(row),
                    )
                )
    _require_requested_count(role="refusal", actual=len(refusal), requested=n_refusal)
    _require_requested_count(role="benign PCA", actual=len(benign), requested=n_benign)
    refusal = refusal[:n_refusal]
    benign = benign[:n_benign]
    missing = [str(path) for _, path, _ in benign if not path.is_file()]
    if missing:
        raise SystemExit(f"legacy benign audio files not found: {missing[:10]}")
    return CalibrationSelection(
        harmful_texts=[text for _, text, _ in refusal],
        harmful_paths=[],
        benign_paths=[path for _, path, _ in benign],
        refusal_item_ids=frozenset(item_id for item_id, _, _ in refusal),
        benign_item_ids=frozenset(item_id for item_id, _, _ in benign),
        refusal_sources=frozenset(source for _, _, source in refusal if source is not None),
        benign_sources=frozenset(source for _, _, source in benign if source is not None),
    )


def validate_eval_disjoint(
    selection: CalibrationSelection,
    eval_rows: list[dict[str, Any]],
    *,
    data_dir: Path,
) -> None:
    """Fail before model loading if eval rows overlap calibration.

    Leakage is checked at ITEM level — item_id, audio path, and reference-text
    identity — not at SOURCE level. SARSteer itself calibrates and evaluates on the
    same corpus: §3.2 samples 100 FigStep harmful/safe pairs for alignment "while
    the remaining pairs are reserved for evaluation". A source-level check would
    therefore reject the paper's own protocol and force calibration onto an
    off-method corpus, which is exactly what corrupted the superseded runs.
    """

    eval_ids: set[str] = set()
    eval_paths: set[Path] = set()
    eval_texts: set[str] = set()
    for index, row in enumerate(eval_rows):
        eval_ids.add(_required_string(row, "item_id", role="eval", index=index))
        raw_path = row.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            eval_paths.add(_resolve_manifest_path(Path(raw_path.strip()), data_dir))
        text = row.get("reference_text")
        if isinstance(text, str) and text.strip():
            eval_texts.add(_normalize_text(text))

    item_overlap = (selection.refusal_item_ids | selection.benign_item_ids) & eval_ids
    path_overlap = (set(selection.harmful_paths) | set(selection.benign_paths)) & eval_paths
    text_overlap = {t for t in selection.calibration_texts if t in eval_texts}
    pieces = []
    if item_overlap:
        pieces.append(f"item_id={sorted(item_overlap)[:10]}")
    if path_overlap:
        pieces.append(f"path={sorted(map(str, path_overlap))[:10]}")
    if text_overlap:
        pieces.append(f"reference_text={sorted(text_overlap)[:3]}")
    if pieces:
        raise SystemExit("calibration/eval overlap detected: " + "; ".join(pieces))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = load_experiment_config(args.config, overrides=args.override)
    if cfg.sarsteer is None or not cfg.sarsteer.enabled:
        raise SystemExit("cfg.sarsteer is disabled; enable it in the run9 config")
    sar = cfg.sarsteer
    paper_mode = sar.implementation == "paper_faithful"
    if paper_mode and sar.n_refusal_calib != sar.n_benign_pca:
        # Algorithm 1 draws Q_harm and its purified Q_safe counterpart per pair.
        raise SystemExit("paper_faithful requires equal harmful and benign sample counts")
    if paper_mode and sar.benign_position != "first_generation_prelogit":
        raise SystemExit("paper_faithful requires benign_position='first_generation_prelogit'")
    if paper_mode and sar.layers is not None:
        raise SystemExit("paper_faithful requires sarsteer.layers=null (all layers)")

    paths = resolve_paths(
        cfg.paths, data_dir=args.data_dir, output_dir=args.output_dir, cache_dir=args.cache_dir
    )
    run_dir = run_output_dir(paths.output_dir, args.run_name)

    explicit_mode = args.refusal_manifest is not None or args.benign_manifest is not None
    if (args.refusal_manifest is None) != (args.benign_manifest is None):
        raise SystemExit("--refusal-manifest and --benign-manifest must be provided together")

    refusal_manifest_path: Path | None = None
    benign_manifest_path: Path | None = None
    if explicit_mode:
        refusal_manifest_path, refusal_rows = _load_manifest(
            args.refusal_manifest, paths.data_dir, role="refusal"
        )
        benign_manifest_path, benign_rows = _load_manifest(
            args.benign_manifest, paths.data_dir, role="benign"
        )
        selector = select_paper_audio_calibration if paper_mode else select_explicit_calibration
        selection = selector(
            refusal_rows,
            benign_rows,
            data_dir=paths.data_dir,
            n_refusal=sar.n_refusal_calib,
            n_benign=sar.n_benign_pca,
        )
    else:
        if paper_mode:
            raise SystemExit("paper_faithful requires explicit refusal and benign manifests")
        pairs = load_audio_rdo_pairs(paths.data_dir, cfg.dataset)
        split_map = split_ids(pairs, cfg)
        rows = load_jsonl(paths.data_dir / cfg.dataset.target_generation.outputs_file)
        train_rows = rows_for_split(rows, split_map, "train")
        selection = select_legacy_calibration(
            train_rows,
            data_dir=paths.data_dir,
            clean_style=args.clean_style,
            n_refusal=sar.n_refusal_calib,
            n_benign=sar.n_benign_pca,
        )

    eval_manifest_path: Path | None = None
    if args.eval_manifest is not None:
        eval_manifest_path, eval_rows = _load_manifest(
            args.eval_manifest, paths.data_dir, role="eval"
        )
        validate_eval_disjoint(selection, eval_rows, data_dir=paths.data_dir)

    harmful_count = len(selection.harmful_paths if paper_mode else selection.harmful_texts)
    harmful_kind = "harmful audio" if paper_mode else "harmful texts"
    calibration_label = (
        "paper paired audio manifests"
        if paper_mode
        else ("explicit" if explicit_mode else "legacy train split")
    )
    print(
        f"[sarsteer] calibrate ({calibration_label}): "
        f"{harmful_count} {harmful_kind}, "
        f"{len(selection.benign_paths)} benign clips; run dir {run_dir}",
        flush=True,
    )

    from audio_safety.models.qwen2_audio import load_qwen2_audio

    model, processor = load_qwen2_audio(cfg.model, cache_dir=paths.cache_dir)
    instruction = cfg.dataset.target_generation.instruction
    system_prompt = sarsteer_system_prompt(sar.implementation)
    if paper_mode:
        refusal_vectors = extract_paper_refusal_vectors(
            model,
            processor,
            selection.harmful_paths,
            instruction,
            refusal_text=sar.refusal_prompt,
            system_prompt=system_prompt,
        )
        benign_acts = extract_benign_speech_activations(
            model, processor, selection.benign_paths, instruction, system_prompt=system_prompt
        )
    else:
        refusal_vectors = extract_text_refusal_vectors(
            model,
            processor,
            selection.harmful_texts,
            refusal_text=sar.refusal_prompt,
            extraction_position=sar.extraction_position,
        )
        benign_acts = extract_benign_speech_activations(
            model,
            processor,
            selection.benign_paths,
            instruction,
            position_name=sar.benign_position,
        )
    vectors = build_sarsteer_vectors(refusal_vectors, benign_acts, n_pcs=sar.n_pcs)
    if sar.layers is not None:
        keep = set(sar.layers)
        vectors = {ell: vec for ell, vec in vectors.items() if ell in keep}
        if not vectors:
            raise SystemExit(f"sarsteer.layers {sorted(keep)} selected no available layer")

    import numpy as np

    norm_report = {}
    for ell in sorted(vectors):
        v_norm = float(np.linalg.norm(refusal_vectors[ell]))
        vperp_norm = float(np.linalg.norm(vectors[ell]))
        norm_report[str(ell)] = {
            "v_norm": v_norm,
            "v_perp_norm": vperp_norm,
            "retained_ratio": vperp_norm / v_norm if v_norm > 0 else float("nan"),
        }
    ratios = [report["retained_ratio"] for report in norm_report.values()]
    print(
        "[sarsteer] v_perp/v retained ratio: "
        f"min={min(ratios):.3f} median={float(np.median(ratios)):.3f} "
        f"max={max(ratios):.3f}",
        flush=True,
    )

    out_path = run_dir / sar.vectors_file
    save_sarsteer_vectors(
        out_path,
        vectors,
        {
            "method": "sarsteer_paper_2510.17633" if paper_mode else "sarsteer_legacy",
            "seed": cfg.seed,
            "git_commit": get_git_commit() or "unknown",
            "implementation": sar.implementation,
            "refusal_input_modality": "audio" if paper_mode else "text",
            "system_prompt": system_prompt,
            "safe_ablation_lambda": 1.0,
            "generation_position_scope": sarsteer_position_scope(sar.implementation),
            "alpha": sar.alpha,
            "n_pcs": sar.n_pcs,
            "refusal_prompt": sar.refusal_prompt,
            "extraction_position": "last_prompt_token" if paper_mode else sar.extraction_position,
            "benign_position": sar.benign_position,
            "n_refusal_calib": harmful_count,
            "n_benign_pca": len(selection.benign_paths),
            "calibration_mode": "paper_paired_audio_manifests"
            if paper_mode
            else ("explicit_manifests" if explicit_mode else "legacy_train_split"),
            "refusal_manifest": (
                str(refusal_manifest_path) if refusal_manifest_path is not None else None
            ),
            "benign_manifest": (
                str(benign_manifest_path) if benign_manifest_path is not None else None
            ),
            "eval_manifest_checked": (
                str(eval_manifest_path) if eval_manifest_path is not None else None
            ),
            "refusal_item_ids": sorted(selection.refusal_item_ids),
            "benign_item_ids": sorted(selection.benign_item_ids),
            "refusal_sources": sorted(selection.refusal_sources),
            "benign_sources": sorted(selection.benign_sources),
            "layers": sorted(vectors),
            "retained_norm_ratio": norm_report,
        },
    )
    print(f"[sarsteer] saved {len(vectors)} per-layer defense vectors -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
