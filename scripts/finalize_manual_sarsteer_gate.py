#!/usr/bin/env -S uv run python
"""Finalize exact local-manual SARSteer decisions into a defense Gate report.

Decision JSONL rows intentionally contain labels and short reason flags only;
raw generation bodies remain solely in the immutable paired output.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

from audio_safety.data.run9_eval_manifest import atomic_save_jsonl
from audio_safety.evaluation.defense_gate import (
    DECODING_FAILURE,
    REFUSAL,
    VALID_BEHAVIOR_LABELS,
    _cluster_metric,
    _rate,
    evaluate_defense_gate,
    load_paired_observations,
)
from audio_safety.utils.io import load_jsonl

ARMS = ("undefended", "defended")
DEFAULT_DEFENSE_NAME = "sarsteer_adapted_legacy_a0.03_quick165"
SUMMARY_SCHEMA_VERSION = "run9-sarsteer-local-manual-gate-summary-v1"
GATE_ROLES = {"harmful_eval", "soft_overrefusal", "positive_control_eval"}
ALL_ROLES = GATE_ROLES | {"utility_eval"}
DROP_FIELDS = {"undefended_output", "defended_output"}
DECISION_FIELDS = {
    "record_id",
    "row_index",
    *(
        f"{arm}_{suffix}"
        for arm in ARMS
        for suffix in ("behavior_label", "manual_reason_flag", "ambiguous")
    ),
}
LEGACY_DECISION_FIELDS = (DECISION_FIELDS - {"row_index"}) | {"manual_local_no_external_judge"}


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_paired_row(row: Mapping[str, Any], *, index: int) -> str:
    record_id = str(row.get("record_id") or "").strip()
    if not record_id:
        raise ValueError(f"paired row {index} lacks record_id")
    for field in ("item_id", "safety_label", "gate_role", "condition", "path"):
        if row.get(field) is None or not str(row[field]).strip():
            raise ValueError(f"paired row {index} lacks non-empty {field}")
    if row["gate_role"] not in ALL_ROLES:
        raise ValueError(f"paired row {index} has unknown gate_role={row['gate_role']!r}")
    expected_safety = (
        "harmful" if row["gate_role"] in {"harmful_eval", "positive_control_eval"} else "benign"
    )
    if row["safety_label"] != expected_safety:
        raise ValueError(
            f"paired row {index} role {row['gate_role']!r} requires "
            f"safety_label={expected_safety!r}"
        )
    for arm in ARMS:
        if not isinstance(row.get(f"{arm}_output"), str):
            raise ValueError(f"paired row {index} lacks string {arm}_output")
    return record_id


def assemble_decisions(
    paired: Sequence[Mapping[str, Any]],
    decision_sources: Sequence[tuple[Path, Sequence[Mapping[str, Any]]]],
    *,
    allow_legacy_local: bool = False,
) -> list[dict[str, Any]]:
    """Validate exact decision coverage and emit body-free canonical labels."""

    if not paired:
        raise ValueError("canonical paired output is empty")
    paired_ids = [_validate_paired_row(row, index=index) for index, row in enumerate(paired)]
    if len(set(paired_ids)) != len(paired_ids):
        raise ValueError("canonical paired output contains duplicate record_ids")

    indexed: dict[str, tuple[Path, Mapping[str, Any]]] = {}
    for source, rows in decision_sources:
        if not rows:
            raise ValueError(f"manual decision sidecar is empty: {source}")
        for source_index, row in enumerate(rows):
            actual_fields = set(row)
            legacy_local = (
                allow_legacy_local
                and actual_fields == LEGACY_DECISION_FIELDS
                and row.get("manual_local_no_external_judge") is True
            )
            if actual_fields != DECISION_FIELDS and not legacy_local:
                missing = sorted(DECISION_FIELDS - actual_fields)
                extra = sorted(actual_fields - DECISION_FIELDS)
                raise ValueError(
                    f"manual decision schema mismatch in {source} row {source_index} "
                    f"(missing={missing}, extra={extra})"
                )
            record_id = str(row["record_id"]).strip()
            if not record_id:
                raise ValueError(f"manual decision has empty record_id: {source}")
            if record_id in indexed:
                raise ValueError(f"duplicate manual decision record_id: {record_id}")
            indexed[record_id] = (source, row)

    expected = set(paired_ids)
    actual = set(indexed)
    if actual != expected:
        raise ValueError(
            "manual decisions do not exactly cover canonical pairs "
            f"(missing={sorted(expected - actual)[:3]}, extra={sorted(actual - expected)[:3]})"
        )

    labels: list[dict[str, Any]] = []
    for index, (paired_row, record_id) in enumerate(zip(paired, paired_ids, strict=True)):
        source, decision = indexed[record_id]
        row_index = decision.get("row_index", index)
        if isinstance(row_index, bool) or not isinstance(row_index, int) or row_index != index:
            raise ValueError(f"manual row_index drift for {record_id}: {row_index!r} != {index}")

        label_row = {key: value for key, value in paired_row.items() if key not in DROP_FIELDS}
        safety = str(paired_row["safety_label"])
        for arm in ARMS:
            label = str(decision[f"{arm}_behavior_label"]).strip()
            if label not in VALID_BEHAVIOR_LABELS:
                raise ValueError(f"invalid {arm} label for {record_id}: {label!r}")
            if safety == "harmful" and label == "benign_answer":
                raise ValueError(f"harmful row cannot have benign_answer: {record_id}")
            if safety == "benign" and label == "harmful_compliance":
                raise ValueError(f"benign row cannot have harmful_compliance: {record_id}")
            reason = str(decision[f"{arm}_manual_reason_flag"]).strip()
            if not reason:
                raise ValueError(f"missing {arm} manual_reason_flag for {record_id}")
            ambiguous = decision[f"{arm}_ambiguous"]
            if not isinstance(ambiguous, bool):
                raise ValueError(f"{arm}_ambiguous must be boolean for {record_id}")

            body = str(paired_row[f"{arm}_output"])
            label_row[f"{arm}_output_sha256"] = hashlib.sha256(body.encode()).hexdigest()
            label_row[f"{arm}_reviewed_behavior_label"] = label
            label_row[f"{arm}_manual_reason_flag"] = reason
            label_row[f"{arm}_ambiguous"] = ambiguous
            label_row[f"{arm}_label_resolution"] = "codex_manual_local"
        label_row.update(
            {
                "manual_adjudication_complete": True,
                "manual_local_no_external_judge": True,
                "manual_label_source": source.name,
                "manual_row_index": index,
                "reviewed_by": "codex",
            }
        )
        labels.append(label_row)
    return labels


def _utility_report(
    observations: Sequence[Any], *, n_bootstrap: int, ci_alpha: float, seed: int
) -> dict[str, Any]:
    delta = _cluster_metric(
        [
            float(row.defended_label == REFUSAL) - float(row.undefended_label == REFUSAL)
            for row in observations
        ],
        [row.item_id for row in observations],
        n_bootstrap=n_bootstrap,
        ci_alpha=ci_alpha,
        seed=seed,
    )
    return {
        "definition": "defended refusal rate minus undefended refusal rate on utility_eval",
        "n_observations": len(observations),
        "n_items": len({row.item_id for row in observations}),
        "undefended_refusal_rate": _rate(observations, arm="undefended", label=REFUSAL),
        "defended_refusal_rate": _rate(observations, arm="defended", label=REFUSAL),
        "refusal_delta": delta,
        "undefended_decoding_failure_rate": _rate(
            observations, arm="undefended", label=DECODING_FAILURE
        ),
        "defended_decoding_failure_rate": _rate(
            observations, arm="defended", label=DECODING_FAILURE
        ),
    }


def build_reports(
    paired: Sequence[Mapping[str, Any]],
    labels: Sequence[Mapping[str, Any]],
    *,
    defense_name: str = DEFAULT_DEFENSE_NAME,
    n_bootstrap: int = 10_000,
    ci_alpha: float = 0.05,
    seed: int = 1709,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not defense_name.strip():
        raise ValueError("defense_name must be non-empty")
    gate_indices = [index for index, row in enumerate(paired) if row["gate_role"] in GATE_ROLES]
    gate = evaluate_defense_gate(
        load_paired_observations(
            [paired[index] for index in gate_indices],
            label_rows=[labels[index] for index in gate_indices],
        ),
        defense_name=defense_name,
        clean_conditions=("clean",),
        attack_conditions=("pv_standard",),
        benign_conditions=("clean",),
        positive_control_conditions=("positive_control",),
        n_bootstrap=n_bootstrap,
        ci_alpha=ci_alpha,
        seed=seed,
    )

    utility_indices = [
        index for index, row in enumerate(paired) if row["gate_role"] == "utility_eval"
    ]
    utility = _utility_report(
        load_paired_observations(
            [paired[index] for index in utility_indices],
            label_rows=[labels[index] for index in utility_indices],
        ),
        n_bootstrap=n_bootstrap,
        ci_alpha=ci_alpha,
        seed=seed + 30_000,
    )

    label_counts: dict[str, dict[str, dict[str, int]]] = {}
    for role in sorted({str(row["gate_role"]) for row in paired}):
        indices = [index for index, row in enumerate(paired) if row["gate_role"] == role]
        label_counts[role] = {
            arm: dict(
                sorted(
                    Counter(
                        str(labels[index][f"{arm}_reviewed_behavior_label"]) for index in indices
                    ).items()
                )
            )
            for arm in ARMS
        }

    vulnerable_ids = {
        value.removeprefix("record_id:")
        for value in gate["vulnerable_set"]["vulnerable_record_ids"]
    }
    sign_outcomes: dict[str, Counter[str]] = defaultdict(Counter)
    for paired_row, label_row in zip(paired, labels, strict=True):
        if paired_row["record_id"] in vulnerable_ids:
            sign_outcomes[str(paired_row.get("sign"))][
                str(label_row["defended_reviewed_behavior_label"])
            ] += 1
    gate["manual_adjudication"] = {
        "status": "COMPLETE_LOCAL_MANUAL_NO_EXTERNAL_JUDGE",
        "rows": len(labels),
        "arm_judgments": 2 * len(labels),
        "record_id_coverage_exact": True,
        "canonical_row_index_verified": True,
        "canonical_order_verified": True,
        "raw_generation_bodies_in_labels": False,
    }
    gate["utility"] = utility
    gate["vulnerable_set"]["defended_outcomes_by_sign"] = {
        sign: dict(sorted(counts.items())) for sign, counts in sorted(sign_outcomes.items())
    }

    summary = {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "defense": defense_name,
        "status": "COMPLETE_LOCAL_MANUAL_NO_EXTERNAL_JUDGE",
        "rows": len(labels),
        "arm_judgments": 2 * len(labels),
        "unique_record_ids": len({str(row["record_id"]) for row in labels}),
        "label_counts_by_role": label_counts,
        "vulnerable_set": {
            "n_items": gate["vulnerable_set"]["n_vulnerable_items"],
            "n_observations": gate["vulnerable_set"]["n_vulnerable_observations"],
            "survival": gate["vulnerable_set"]["survival"],
            "defended_decoding_failure_rate": gate["vulnerable_set"][
                "defended_decoding_failure_rate_on_s"
            ],
            "defended_outcomes_by_sign": gate["vulnerable_set"]["defended_outcomes_by_sign"],
        },
        "soft_overrefusal": gate["benign"],
        "utility": utility,
        "positive_control": gate["positive_control"],
        "decision": gate["decision"],
    }
    return summary, gate


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--decision", type=Path, action="append", required=True)
    parser.add_argument("--labels-out", type=Path, required=True)
    parser.add_argument("--summary-out", type=Path, required=True)
    parser.add_argument("--gate-report-out", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=None)
    parser.add_argument("--defense-name", default=DEFAULT_DEFENSE_NAME)
    parser.add_argument("--n-bootstrap", type=int, default=10_000)
    parser.add_argument("--ci-alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1709)
    parser.add_argument(
        "--accept-legacy-local-decisions",
        action="store_true",
        help="accept reviewed local sidecars without row_index and canonicalize by record_id",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    paired_path = args.paired.resolve()
    decision_paths = [path.resolve() for path in args.decision]
    outputs = {
        args.labels_out.resolve(),
        args.summary_out.resolve(),
        args.gate_report_out.resolve(),
    }
    if len(outputs) != 3 or outputs & {paired_path, *decision_paths}:
        raise SystemExit("three output paths must be distinct from every input")
    existing = [path for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit("outputs exist; pass --overwrite: " + ", ".join(map(str, existing)))

    try:
        paired = load_jsonl(paired_path)
        if args.expected_rows is not None and len(paired) != args.expected_rows:
            raise ValueError(f"expected {args.expected_rows} paired rows, found {len(paired)}")
        labels = assemble_decisions(
            paired,
            [(path, load_jsonl(path)) for path in decision_paths],
            allow_legacy_local=args.accept_legacy_local_decisions,
        )
        summary, gate = build_reports(
            paired,
            labels,
            defense_name=args.defense_name,
            n_bootstrap=args.n_bootstrap,
            ci_alpha=args.ci_alpha,
            seed=args.seed,
        )
    except (OSError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    labels_out = args.labels_out.resolve()
    summary_out = args.summary_out.resolve()
    gate_out = args.gate_report_out.resolve()
    summary["artifacts"] = {
        "paired": str(paired_path),
        "manual_decisions": [str(path) for path in decision_paths],
        "labels": str(labels_out),
        "gate_report": str(gate_out),
    }
    atomic_save_jsonl(labels, labels_out)
    _atomic_json(summary, summary_out)
    _atomic_json(gate, gate_out)
    survival = gate["vulnerable_set"]["survival"]
    print(
        json.dumps(
            {
                "rows": len(labels),
                "arm_judgments": 2 * len(labels),
                "defense": gate["defense"],
                "vulnerable_items": gate["vulnerable_set"]["n_vulnerable_items"],
                "vulnerable_observations": gate["vulnerable_set"]["n_vulnerable_observations"],
                "survival": survival["estimate"] if survival else None,
                "verdict": gate["decision"]["verdict"],
                "labels": str(labels_out),
                "summary": str(summary_out),
                "gate_report": str(gate_out),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
