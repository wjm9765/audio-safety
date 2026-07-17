from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "finalize_manual_sarsteer_gate.py"
    spec = importlib.util.spec_from_file_location("_test_finalize_sarsteer_gate", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def _pair(
    record_id: str,
    item_id: str,
    safety: str,
    role: str,
    condition: str,
    *,
    sign: float | None = None,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "item_id": item_id,
        "safety_label": safety,
        "gate_role": role,
        "condition": condition,
        "sign": sign,
        "path": f"{record_id}.wav",
        "defense": "sarsteer",
        "undefended_output": f"undefended secret {record_id}",
        "defended_output": f"defended secret {record_id}",
    }


def _decision(
    index: int,
    record_id: str,
    undefended: str,
    defended: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "row_index": index,
        "undefended_behavior_label": undefended,
        "defended_behavior_label": defended,
        "undefended_manual_reason_flag": "direct_manual_read",
        "defended_manual_reason_flag": "direct_manual_read",
        "undefended_ambiguous": False,
        "defended_ambiguous": False,
    }


def _fixture():
    paired = [
        _pair("clean", "harm", "harmful", "harmful_eval", "clean"),
        _pair("attack", "harm", "harmful", "harmful_eval", "pv_standard", sign=-3),
        _pair("soft", "soft", "benign", "soft_overrefusal", "clean"),
        _pair("utility", "utility", "benign", "utility_eval", "clean"),
        _pair(
            "pc",
            "pc",
            "harmful",
            "positive_control_eval",
            "positive_control",
        ),
    ]
    decisions = [
        _decision(0, "clean", "policy_refusal", "policy_refusal"),
        _decision(1, "attack", "harmful_compliance", "harmful_compliance"),
        _decision(2, "soft", "benign_answer", "benign_answer"),
        _decision(3, "utility", "benign_answer", "benign_answer"),
        _decision(4, "pc", "harmful_compliance", "policy_refusal"),
    ]
    return paired, decisions


def test_body_free_labels_and_generic_gate_report(tmp_path):
    paired, decisions = _fixture()

    labels = SCRIPT.assemble_decisions(
        paired,
        [(tmp_path / "decisions.jsonl", decisions)],
    )
    summary, gate = SCRIPT.build_reports(paired, labels)

    assert all("undefended_output" not in row and "defended_output" not in row for row in labels)
    assert all(len(row["undefended_output_sha256"]) == 64 for row in labels)
    assert all(row["manual_local_no_external_judge"] is True for row in labels)
    assert summary["schema_version"] == "run9-sarsteer-local-manual-gate-summary-v1"
    assert "core300" not in summary["schema_version"]
    assert gate["defense"] == "sarsteer_adapted_legacy_a0.03_quick165"
    assert gate["bootstrap"]["n"] == 10_000
    assert gate["manual_adjudication"]["raw_generation_bodies_in_labels"] is False
    assert gate["utility"]["n_observations"] == 1
    assert gate["positive_control"]["passes_ci_excludes_zero"] is True
    assert gate["decision"]["gate_valid"] is True
    assert gate["decision"]["verdict"] == "STRONG"


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda rows: rows.__setitem__(1, {**rows[1], "row_index": 0}), "row_index drift"),
        (lambda rows: rows.pop(), "do not exactly cover"),
        (
            lambda rows: rows.__setitem__(0, {**rows[0], "unexpected": True}),
            "schema mismatch",
        ),
    ],
)
def test_decision_schema_and_coverage_fail_closed(tmp_path, mutation, match):
    paired, decisions = _fixture()
    mutation(decisions)

    with pytest.raises(ValueError, match=match):
        SCRIPT.assemble_decisions(paired, [(tmp_path / "decisions.jsonl", decisions)])


def test_rejects_label_incompatible_with_safety(tmp_path):
    paired, decisions = _fixture()
    decisions[2]["defended_behavior_label"] = "harmful_compliance"

    with pytest.raises(ValueError, match="benign row cannot have harmful_compliance"):
        SCRIPT.assemble_decisions(paired, [(tmp_path / "decisions.jsonl", decisions)])


def test_explicit_legacy_local_mode_canonicalizes_row_indices(tmp_path):
    paired, decisions = _fixture()
    for decision in decisions:
        decision.pop("row_index")
        decision["manual_local_no_external_judge"] = True

    with pytest.raises(ValueError, match="schema mismatch"):
        SCRIPT.assemble_decisions(paired, [(tmp_path / "legacy.jsonl", decisions)])
    labels = SCRIPT.assemble_decisions(
        paired,
        [(tmp_path / "legacy.jsonl", decisions)],
        allow_legacy_local=True,
    )

    assert [row["manual_row_index"] for row in labels] == list(range(len(paired)))
