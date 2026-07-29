import hashlib
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from audio_safety.config import load_exp3_config
from audio_safety.config.schema import PathsConfig
from audio_safety.evaluation.refusal_instability import EXPLICIT_REFUSAL
from audio_safety.pipelines.exp3_qwen_mechanism import (
    _behavior_transport,
    _first_token_ids,
    _mechanism_cohort,
    _net_refusal_difference,
    _stratified_limit,
    behavior_contract_sha,
    build_pair_manifest,
    collect_behavior,
    refusal_matcher_sha,
    score_behavior_output,
)
from audio_safety.utils.paths import resolve_paths


def test_exp3_matcher_hash_is_order_sensitive_and_stable():
    patterns = ["I'm sorry", "I can't"]
    assert refusal_matcher_sha(patterns) == refusal_matcher_sha(patterns)
    assert refusal_matcher_sha(patterns) != refusal_matcher_sha(list(reversed(patterns)))


def test_exp3_behavior_has_marker_primary_and_failure_sensitivity():
    score = score_behavior_output(
        "I'm sorry, I can't assist.",
        reference_text=None,
        patterns=["I'm sorry"],
        declared_failure="repetition_loop",
    )
    assert score["marker_status"] == EXPLICIT_REFUSAL
    assert score["explicit_refusal"] is True
    assert score["refusal_status"] == "decoding_failure:repetition_loop"


def test_stratified_limit_is_deterministic_and_spans_buckets():
    rows = [{"item_id": f"h{i}", "role": "harmful", "category_name": "a"} for i in range(5)] + [
        {"item_id": f"b{i}", "role": "benign", "category_name": "b"} for i in range(5)
    ]
    first = _stratified_limit(rows, 4, seed=7)
    second = _stratified_limit(list(reversed(rows)), 4, seed=7)
    assert [row["item_id"] for row in first] == [row["item_id"] for row in second]
    assert {row["role"] for row in first} == {"harmful", "benign"}


def test_net_refusal_difference_keeps_direction_separate_from_disagreement():
    source = [
        {"item_id": "a", "refusal_status": "explicit_refusal"},
        {"item_id": "b", "refusal_status": "explicit_refusal_absent"},
    ]
    target = [
        {"item_id": "a", "refusal_status": "explicit_refusal_absent"},
        {"item_id": "b", "refusal_status": "explicit_refusal"},
    ]
    result = _net_refusal_difference(source, target, seed=0, n_bootstrap=100, ci_alpha=0.1)
    assert result["estimate"] == pytest.approx(0.0)


def test_bidirectional_patch_transport_does_not_cancel():
    cfg = load_exp3_config(
        "configs/experiments/exp3_qwen_refusal_mechanism.yaml",
        overrides=["exp3.behavior.n_bootstrap=100"],
    )
    rows = [
        {
            "pair_id": "a",
            "layer": 10,
            "condition": "real",
            "role": "harmful",
            "baseline_explicit_refusal": True,
            "donor_explicit_refusal": False,
            "explicit_refusal": False,
        },
        {
            "pair_id": "b",
            "layer": 10,
            "condition": "real",
            "role": "harmful",
            "baseline_explicit_refusal": False,
            "donor_explicit_refusal": True,
            "explicit_refusal": True,
        },
    ]
    result = _behavior_transport(cfg, rows, group_fields=["layer", "condition", "role"])
    group = next(iter(result.values()))
    assert group["host_to_patched"] == {
        "R_to_R": 0,
        "R_to_NR": 1,
        "NR_to_R": 1,
        "NR_to_NR": 0,
    }
    assert group["donorward_score_discordant_only"]["mean"] == pytest.approx(1.0)


def test_first_token_bank_filters_standalone_whitespace_token():
    class Tokenizer:
        all_special_ids = []

        def __call__(self, text, add_special_tokens=False):
            del add_special_tokens
            return SimpleNamespace(input_ids=[220] if text.startswith(" ") else [42])

        def decode(self, ids, **kwargs):
            del kwargs
            return " " if ids == [220] else "Sure"

    assert _first_token_ids(Tokenizer(), ["Sure"]) == [42]


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cpu_preflight_freezes_inputs_without_trusting_legacy_generations(tmp_path):
    cfg = load_exp3_config(
        "configs/experiments/exp3_qwen_refusal_mechanism.yaml",
        overrides=["exp3.behavior.n_bootstrap=100"],
    )
    cfg.paths = PathsConfig(workspace=tmp_path)
    paths = resolve_paths(cfg.paths)
    clean_rows = []
    arm_rows = []
    generations = []
    audio_hashes = {}
    for index, role in enumerate(("harmful", "benign")):
        item = f"i{index}"
        clean_relative = f"audio/clean/{item}.wav"
        clean_path = paths.data_dir / clean_relative
        clean_path.parent.mkdir(parents=True, exist_ok=True)
        clean_bytes = f"clean-{item}".encode()
        clean_path.write_bytes(clean_bytes)
        audio_hashes[(item, "clean")] = hashlib.sha256(clean_bytes).hexdigest()
        clean_rows.append(
            {
                "item_id": item,
                "safety_label": role,
                "path": clean_relative,
                "reference_text": f"request {item}",
                "category_name": "test",
            }
        )
        generations.append(
            {
                "item_id": item,
                "arm": "clean",
                "response": "I'm sorry, I cannot help." if index == 0 else "Here is an answer.",
            }
        )
        for arm in ("pv_locked", "pv_standard", "echo", "tone_p8", "tone_m8"):
            relative = f"audio/{arm}/{item}.wav"
            audio_path = paths.data_dir / relative
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            audio_bytes = f"{arm}-{item}".encode()
            audio_path.write_bytes(audio_bytes)
            audio_hashes[(item, arm)] = hashlib.sha256(audio_bytes).hexdigest()
            arm_rows.append(
                {
                    "item_id": item,
                    "arm": arm,
                    "path": relative,
                    "status": "rendered",
                }
            )
            response = "Here is an answer."
            if (item, arm) == ("i1", "pv_standard"):
                response = "I'm sorry, I cannot help."
            generations.append({"item_id": item, "arm": arm, "response": response})
    contract_sha = behavior_contract_sha(cfg)
    for row in generations:
        key = (row["item_id"], row["arm"])
        row["audio_sha256"] = audio_hashes[key]
        row["behavior_contract_sha256"] = contract_sha
    _write_jsonl(paths.data_dir / cfg.exp3.clean_manifest_file, clean_rows)
    _write_jsonl(paths.data_dir / cfg.exp3.arm_manifest_file, arm_rows)
    _write_jsonl(
        paths.output_dir / cfg.exp3.behavior.prior_generations_file,
        generations,
    )
    run_dir = paths.output_dir / "exp3_test"
    pairs = build_pair_manifest(cfg, paths, run_dir)
    assert len(pairs) == 2  # one mechanism contrast x two items
    behavior, pending = collect_behavior(cfg, paths, run_dir, generate_missing=False)
    assert pending == 4  # two physical arms for each of two phase pairs
    assert behavior == []
    metrics = json.loads((run_dir / cfg.exp3.artifacts.behavior_metrics_file).read_text())
    phase_harmful = metrics["contrasts"]["phase"]["roles"]["harmful"]
    assert phase_harmful["all_completed_outputs_primary"]["n_valid_pairs"] == 0

    # Audio bytes are part of the frozen input contract.
    (paths.data_dir / "audio/pv_standard/i0.wav").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="pair manifest changed"):
        build_pair_manifest(cfg, paths, run_dir)


def _cohort_pair(item, role, source_arm="pv_locked", target_arm="pv_standard"):
    return {
        "schema_version": "exp3.v1",
        "pair_id": f"phase:{item}",
        "item_id": item,
        "role": role,
        "category_name": "test",
        "reference_text": f"request {item}",
        "harmful_anchors": None,
        "contrast": "phase",
        "purpose": "mechanism",
        "source_arm": source_arm,
        "target_arm": target_arm,
        "source_path": f"/tmp/{item}-{source_arm}.wav",
        "target_path": f"/tmp/{item}-{target_arm}.wav",
        "source_sha256": f"sha-{item}-{source_arm}",
        "target_sha256": f"sha-{item}-{target_arm}",
    }


def _cohort_behavior(item, arm, refusing):
    status = "explicit_refusal" if refusing else "explicit_refusal_absent"
    return {
        "item_id": item,
        "arm": arm,
        "refusal_status": status,
        "marker_status": status,
        "explicit_refusal": refusing,
        "behavior_contract_sha256": "contract",
        "runtime_behavior_contract_sha256": "runtime",
    }


def test_mechanism_cohort_caps_discordant_pairs_per_role_and_direction(tmp_path):
    """Regression: the selector nests role under "stratum", not at the row top level."""

    cfg = load_exp3_config(
        "configs/experiments/exp3_qwen_refusal_mechanism.yaml",
        overrides=[
            "exp3.span_patch.cohort.max_discordant_each_direction=2",
            "exp3.span_patch.cohort.max_stable_refusal=0",
            "exp3.span_patch.cohort.max_stable_nonrefusal=0",
        ],
    )
    run_dir = tmp_path / "exp3_cohort"
    pairs, behavior = [], []
    # Three R->NR discordant pairs in each role, plus one NR->R harmful pair.
    for role in ("harmful", "benign"):
        for index in range(3):
            item = f"{role}{index}"
            pairs.append(_cohort_pair(item, role))
            behavior.append(_cohort_behavior(item, "pv_locked", True))
            behavior.append(_cohort_behavior(item, "pv_standard", False))
    pairs.append(_cohort_pair("harmful_up", "harmful"))
    behavior.append(_cohort_behavior("harmful_up", "pv_locked", False))
    behavior.append(_cohort_behavior("harmful_up", "pv_standard", True))
    _write_jsonl(run_dir / cfg.exp3.artifacts.pairs_file, pairs)
    _write_jsonl(run_dir / cfg.exp3.artifacts.behavior_file, behavior)

    cohort = _mechanism_cohort(cfg, run_dir, contrast_name="phase")

    assert all(row["selection_role"] == "discordant" for row in cohort)
    grouped = Counter((row["role"], row["transition"]) for row in cohort)
    # Each (role, transition) group is capped independently at 2.
    assert grouped[("harmful", "refusal_to_nonrefusal")] == 2
    assert grouped[("benign", "refusal_to_nonrefusal")] == 2
    assert grouped[("harmful", "nonrefusal_to_refusal")] == 1
    assert all("role" in row and "source_path" in row for row in cohort)
    # The frozen cohort is stable across calls.
    assert _mechanism_cohort(cfg, run_dir, contrast_name="phase") == cohort
