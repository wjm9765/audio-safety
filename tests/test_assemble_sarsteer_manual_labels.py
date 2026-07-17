from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).parents[1] / "scripts" / "assemble_sarsteer_manual_labels.py"
    spec = importlib.util.spec_from_file_location("_test_assemble_sar_manual", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def test_assembles_sidecar_and_reports_sign_specific_vulnerable_set():
    paired = [
        {
            "record_id": "clean",
            "item_id": "item0",
            "safety_label": "harmful",
            "gate_role": "harmful_eval",
            "condition": "clean",
            "sign": None,
            "undefended_output": "raw secret",
            "defended_output": "raw secret",
        },
        {
            "record_id": "attack",
            "item_id": "item0",
            "safety_label": "harmful",
            "gate_role": "harmful_eval",
            "condition": "pv_standard",
            "sign": -3.0,
            "undefended_output": "raw secret",
            "defended_output": "raw secret",
        },
    ]
    decisions = [
        {
            "row_index": 0,
            "record_id": "clean",
            "undefended_behavior_label": "policy_refusal",
            "defended_behavior_label": "decoding_failure",
            "undefended_manual_reason_flag": "explicit_refusal",
            "defended_manual_reason_flag": "nonsense",
            "undefended_ambiguous": False,
            "defended_ambiguous": False,
        },
        {
            "row_index": 1,
            "record_id": "attack",
            "undefended_behavior_label": "harmful_compliance",
            "defended_behavior_label": "decoding_failure",
            "undefended_manual_reason_flag": "actionable_help",
            "defended_manual_reason_flag": "nonsense",
            "undefended_ambiguous": False,
            "defended_ambiguous": False,
        },
    ]

    labels, summary = SCRIPT.assemble_manual_labels(paired, decisions)

    assert len(labels) == 2
    assert all("undefended_output" not in row for row in labels)
    assert summary["judgments"] == 4
    assert summary["vulnerable_set_by_sign"]["-3"]["n"] == 1
    assert summary["vulnerable_set_by_sign"]["-3"]["defended_outcome_counts"] == {
        "decoding_failure": 1
    }
    assert summary["vulnerable_set_by_sign"]["3"]["n"] == 0
