import json
from types import SimpleNamespace

import pytest
import yaml

from audio_safety.config import load_exp4_config
from audio_safety.pipelines.exp4_audio_kv_routing import (
    _expected_standard_checks,
    analyze_records,
    cache_max_abs_diff,
    clone_kv_cache,
    condition_code,
    condition_parts,
    freeze_source_inputs,
    greedy_logits_processors,
    manual_greedy_decode,
    merge_shard_records,
    shard_assignment,
    shard_records_path,
    total_expected_standard_checks,
    transplant_cache_positions,
)
from audio_safety.utils.paths import ResolvedPaths


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_cache_transplant_clones_and_reconstructs_host():
    torch = pytest.importorskip("torch")

    def cache(fill):
        layers = []
        for layer in range(3):
            keys = torch.full((1, 2, 6, 4), fill + layer, dtype=torch.float32)
            values = torch.full((1, 2, 6, 4), fill + 10 + layer, dtype=torch.float32)
            layers.append(SimpleNamespace(keys=keys, values=values))
        return SimpleNamespace(layers=layers)

    host = cache(0.0)
    injected = clone_kv_cache(host)
    for layer in injected.layers:
        layer.keys[..., [1, 2, 5], :] += 7.0
        layer.values[..., [1, 2, 5], :] -= 3.0

    audio_closed = transplant_cache_positions(injected, host, [1, 2])
    assert cache_max_abs_diff(audio_closed, host) == pytest.approx(7.0)
    fully_closed = transplant_cache_positions(audio_closed, host, [5])
    assert cache_max_abs_diff(fully_closed, host) == 0.0
    assert cache_max_abs_diff(injected, host) == pytest.approx(7.0)
    assert cache_max_abs_diff(clone_kv_cache(injected), injected) == 0.0


def test_cache_clone_preserves_transformers_dynamic_cache_behavior():
    torch = pytest.importorskip("torch")
    cache_utils = pytest.importorskip("transformers.cache_utils")
    dynamic_cache = cache_utils.DynamicCache(
        [(torch.zeros((1, 2, 3, 4)), torch.zeros((1, 2, 3, 4))) for _ in range(2)]
    )
    cloned = clone_kv_cache(dynamic_cache)
    cloned.update(torch.ones((1, 2, 1, 4)), torch.ones((1, 2, 1, 4)), 0)
    assert cloned.get_seq_length(0) == 4
    assert dynamic_cache.get_seq_length(0) == 3


def test_manual_decode_conditions_on_fixed_first_token():
    torch = pytest.importorskip("torch")
    cfg = load_exp4_config(
        "configs/experiments/exp4_audio_kv_routing.yaml",
        overrides=["exp4.max_new_tokens=4"],
    )

    class Model:
        generation_config = SimpleNamespace(eos_token_id=9, repetition_penalty=1.0)

        def __init__(self):
            self.calls = []

        def __call__(self, *, input_ids, attention_mask, past_key_values, **kwargs):
            del kwargs
            self.calls.append((input_ids.tolist(), attention_mask.shape[1]))
            next_id = 2 if len(self.calls) == 1 else 9
            logits = torch.zeros((1, 1, 12))
            logits[0, 0, next_id] = 1.0
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=9)

        @staticmethod
        def batch_decode(values, **kwargs):
            del kwargs
            return [" ".join(str(token) for token in values[0])]

    model = Model()
    result = manual_greedy_decode(
        cfg,
        model,
        Processor(),
        cache=object(),
        prompt_attention_mask=torch.ones((1, 5), dtype=torch.long),
        prompt_input_ids=torch.zeros((1, 5), dtype=torch.long),
        first_token_id=1,
    )
    assert result["generated_token_ids"] == [1, 2, 9]
    assert model.calls == [([[1]], 6), ([[2]], 7)]


def test_manual_decode_applies_the_generation_config_repetition_penalty():
    """Qwen2-Audio ships repetition_penalty=1.1, so greedy generate is not argmax.

    Exp3's whole frozen corpus came from `generate`, so a manual decoder that
    skipped the penalty would silently be a different decoder.
    """
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    cfg = load_exp4_config(
        "configs/experiments/exp4_audio_kv_routing.yaml",
        overrides=["exp4.max_new_tokens=2"],
    )

    class Model:
        def __init__(self, penalty):
            self.generation_config = SimpleNamespace(eos_token_id=99, repetition_penalty=penalty)

        def __call__(self, *, input_ids, attention_mask, past_key_values, **kwargs):
            del input_ids, attention_mask, kwargs
            logits = torch.zeros((1, 1, 12))
            # Token 7 already occurs in the prompt; token 3 does not. Without a
            # penalty 7 wins; with 1.1 the penalty divides it below 3.
            logits[0, 0, 7] = 2.0
            logits[0, 0, 3] = 1.9
            return SimpleNamespace(logits=logits, past_key_values=past_key_values)

    class Processor:
        tokenizer = SimpleNamespace(eos_token_id=99)

        @staticmethod
        def batch_decode(values, **kwargs):
            del kwargs
            return [" ".join(str(token) for token in values[0])]

    prompt = torch.tensor([[7, 7, 5]], dtype=torch.long)
    kwargs = {
        "cache": object(),
        "prompt_attention_mask": torch.ones((1, 3), dtype=torch.long),
        "prompt_input_ids": prompt,
        "first_token_id": 1,
    }
    unpenalised = manual_greedy_decode(cfg, Model(1.0), Processor(), **kwargs)
    penalised = manual_greedy_decode(cfg, Model(1.1), Processor(), **kwargs)
    assert unpenalised["generated_token_ids"] == [1, 7]
    assert penalised["generated_token_ids"] == [1, 3]


def test_greedy_logits_processors_rejects_unreplicated_generation_config():
    pytest.importorskip("transformers")
    model = SimpleNamespace(
        generation_config=SimpleNamespace(repetition_penalty=1.1, no_repeat_ngram_size=3)
    )
    with pytest.raises(NotImplementedError, match="no_repeat_ngram_size"):
        greedy_logits_processors(model)

    plain = SimpleNamespace(generation_config=SimpleNamespace(repetition_penalty=1.0))
    assert list(greedy_logits_processors(plain)) == []


def test_pair_clustered_factorial_identifies_audio_not_tab():
    cfg = load_exp4_config(
        "configs/experiments/exp4_audio_kv_routing.yaml",
        overrides=["exp4.n_bootstrap=100"],
    )
    outcomes = {
        "audio_injected__tab_injected": True,
        "audio_host__tab_injected": False,
        "audio_injected__tab_host": True,
        "audio_host__tab_host": False,
    }
    rows = []
    for pair_index in range(4):
        for direction in cfg.exp4.directions:
            for condition, outcome in outcomes.items():
                rows.append(
                    {
                        "pair_id": f"p{pair_index}",
                        "direction": direction,
                        "condition": condition,
                        "selection_role": "discordant",
                        "baseline_explicit_refusal": False,
                        "donor_explicit_refusal": True,
                        "explicit_refusal": outcome,
                        "injected_y1_changed_from_host": pair_index % 2 == 0,
                        "hook_counts_ok": True,
                        "cache_clone_max_abs_error": 0.0,
                        "full_host_cache_max_abs_error": (
                            0.0 if condition == "audio_host__tab_host" else None
                        ),
                        "standard_generate_checked": False,
                        "standard_generate_exact": None,
                        "host_reproduction_exact": None,
                    }
                )

    metrics = analyze_records(cfg, rows)
    assert metrics["cells"]["T_II"]["estimate"] == 1.0
    assert metrics["cells"]["T_HI"]["estimate"] == 0.0
    assert metrics["contrasts"]["D_audio"]["estimate"] == 1.0
    assert metrics["contrasts"]["D_audio"]["decision"] == ("material_conditional_contributor")
    assert metrics["contrasts"]["D_tAB"]["estimate"] == 0.0
    assert metrics["contrasts"]["D_tAB"]["decision"] == ("negligible_at_pilot_resolution")
    assert metrics["contrasts"]["D_joint"]["estimate"] == 1.0
    assert metrics["contrasts"]["factorial_interaction"]["estimate"] == 0.0


def _exp4_cfg(*overrides):
    return load_exp4_config(
        "configs/experiments/exp4_audio_kv_routing.yaml", overrides=list(overrides)
    )


def _shard_cohort(n_pairs):
    return [{"pair_id": f"phase:p{index}"} for index in range(n_pairs)]


def _cells(cfg, cohort):
    return [
        {"pair_id": pair["pair_id"], "direction": direction, "condition": condition}
        for pair in cohort
        for direction in cfg.exp4.directions
        for condition in cfg.exp4.conditions
    ]


def test_shard_assignment_partitions_cohort_and_keeps_global_indices():
    cohort = _shard_cohort(7)
    shards = [shard_assignment(cohort, shard_index=i, shard_count=3) for i in range(3)]

    assert [index for shard in shards for index, _pair in shard] == [0, 3, 6, 1, 4, 2, 5]
    covered = sorted(index for shard in shards for index, _pair in shard)
    assert covered == list(range(len(cohort)))
    for shard_index, shard in enumerate(shards):
        assert all(index % 3 == shard_index for index, _pair in shard)
    assert shard_assignment(cohort, shard_index=0, shard_count=1) == list(enumerate(cohort))

    with pytest.raises(ValueError, match="shard_index must lie"):
        shard_assignment(cohort, shard_index=3, shard_count=3)
    with pytest.raises(ValueError, match="shard_count must be"):
        shard_assignment(cohort, shard_index=0, shard_count=0)


def test_every_shard_owes_its_own_standard_generate_checks(tmp_path):
    """Each shard loads its own model replica, so each must prove its decoder."""
    cfg = _exp4_cfg()
    cohort = _shard_cohort(5)
    single = _expected_standard_checks(cfg, cohort, shard_index=0, shard_count=1)
    assert single == min(cfg.exp4.standard_generate_checks, len(cohort) * len(cfg.exp4.directions))

    for index in range(3):
        assert _expected_standard_checks(cfg, cohort, shard_index=index, shard_count=3) == single
    assert total_expected_standard_checks(cfg, cohort, shard_count=3) == 3 * single
    assert total_expected_standard_checks(cfg, cohort, shard_count=1) == single

    # A shard too small to supply the full count owes only what it can run.
    tiny = _shard_cohort(1)
    assert _expected_standard_checks(cfg, tiny, shard_index=1, shard_count=2) == 0


def test_shard_records_path_is_distinct_per_shard(tmp_path):
    cfg = _exp4_cfg()
    canonical = tmp_path / cfg.exp4.artifacts.records_file
    assert shard_records_path(cfg, tmp_path, shard_index=0, shard_count=1) == canonical
    paths = [
        shard_records_path(cfg, tmp_path, shard_index=index, shard_count=2) for index in range(2)
    ]
    assert len(set(paths)) == 2
    assert canonical not in paths
    assert paths[0].name == "records.shard00_of_02.jsonl"
    assert paths[0].parent == canonical.parent


def test_merge_shard_records_reconstructs_the_full_factorial(tmp_path):
    cfg = _exp4_cfg()
    cohort = _shard_cohort(5)
    _write_jsonl(tmp_path / cfg.exp4.artifacts.cohort_file, cohort)
    for shard_index in range(2):
        owned = [
            pair for _i, pair in shard_assignment(cohort, shard_index=shard_index, shard_count=2)
        ]
        _write_jsonl(
            shard_records_path(cfg, tmp_path, shard_index=shard_index, shard_count=2),
            _cells(cfg, owned),
        )

    rows = merge_shard_records(cfg, tmp_path, shard_count=2)
    assert len(rows) == len(cohort) * len(cfg.exp4.directions) * len(cfg.exp4.conditions)
    assert rows == sorted(
        rows, key=lambda row: (row["pair_id"], row["direction"], row["condition"])
    )
    canonical = tmp_path / cfg.exp4.artifacts.records_file
    assert json.loads(canonical.read_text().splitlines()[0])["pair_id"] == "phase:p0"
    # Re-merging is idempotent.
    assert merge_shard_records(cfg, tmp_path, shard_count=2) == rows


def test_merge_shard_records_rejects_missing_incomplete_and_foreign_rows(tmp_path):
    cfg = _exp4_cfg()
    cohort = _shard_cohort(4)
    _write_jsonl(tmp_path / cfg.exp4.artifacts.cohort_file, cohort)
    shard0 = shard_records_path(cfg, tmp_path, shard_index=0, shard_count=2)
    shard1 = shard_records_path(cfg, tmp_path, shard_index=1, shard_count=2)
    own0 = [pair for _i, pair in shard_assignment(cohort, shard_index=0, shard_count=2)]
    own1 = [pair for _i, pair in shard_assignment(cohort, shard_index=1, shard_count=2)]

    _write_jsonl(shard0, _cells(cfg, own0))
    with pytest.raises(FileNotFoundError, match="shard checkpoint is missing"):
        merge_shard_records(cfg, tmp_path, shard_count=2)

    # A shard that silently ran a pair it does not own must not be merged.
    _write_jsonl(shard1, _cells(cfg, own1) + _cells(cfg, own0[:1]))
    with pytest.raises(RuntimeError, match="outside its pair assignment"):
        merge_shard_records(cfg, tmp_path, shard_count=2)

    # A shard that stopped early must not produce a partial analysis.
    _write_jsonl(shard1, _cells(cfg, own1)[:-1])
    with pytest.raises(RuntimeError, match="do not cover the frozen cohort factorial"):
        merge_shard_records(cfg, tmp_path, shard_count=2)

    _write_jsonl(shard1, _cells(cfg, own1))
    rows = merge_shard_records(cfg, tmp_path, shard_count=2)
    assert len(rows) == 32

    # A canonical checkpoint that disagrees with the shards is never overwritten.
    _write_jsonl(shard1, list(reversed(_cells(cfg, own1))))
    _write_jsonl(tmp_path / cfg.exp4.artifacts.records_file, rows[:-1])
    with pytest.raises(RuntimeError, match="differ from the existing canonical"):
        merge_shard_records(cfg, tmp_path, shard_count=2)


def test_exp5_factorial_separates_y1_from_the_prompt_cache():
    """The y1 factor must be estimated only where the injection moved y1."""
    cfg = load_exp4_config(
        "configs/experiments/exp5_y1_cache_factorial.yaml",
        overrides=["exp4.n_bootstrap=200"],
    )
    assert len(cfg.exp4.conditions) == 8

    def row(pair, direction, condition, outcome, changed):
        return {
            "pair_id": pair,
            "direction": direction,
            "condition": condition,
            "selection_role": "discordant",
            "baseline_explicit_refusal": False,
            "donor_explicit_refusal": True,
            "explicit_refusal": outcome,
            "injected_y1_changed_from_host": changed,
            "hook_counts_ok": True,
            "cache_clone_max_abs_error": 0.0,
            "full_host_cache_max_abs_error": (0.0 if condition == "audio_host__tab_host" else None),
            "host_reproduction_exact": True if condition.endswith("y1_host") else None,
            "standard_generate_checked": False,
            "standard_generate_exact": None,
        }

    rows = []
    for index in range(6):
        for direction in cfg.exp4.directions:
            # Half the directions moved y1; only those may carry the y1 contrast.
            changed = direction == "source_to_target"
            for condition in cfg.exp4.conditions:
                y1_host = condition.endswith("__y1_host")
                # Ground truth: y1 alone drives the marker; the cache does not.
                outcome = (not y1_host) and changed
                rows.append(row(f"p{index}", direction, condition, outcome, changed))

    metrics = analyze_records(cfg, rows)
    assert metrics["design"] == "exp5.2x2x2"
    assert metrics["n_primary_directions"] == 6
    assert metrics["n_unchanged_y1_directions"] == 6
    assert set(metrics["cells"]) == {
        "T_III",
        "T_HII",
        "T_IHI",
        "T_HHI",
        "T_IIH",
        "T_HIH",
        "T_IHH",
        "T_HHH",
    }
    contrasts = metrics["contrasts"]
    # y1 carries everything; the prompt cache carries nothing.
    assert contrasts["D_y1_at_HH"]["estimate"] == 1.0
    assert contrasts["D_y1_at_HH"]["decision"] == "material_conditional_contributor"
    assert contrasts["D_joint_y1H"]["estimate"] == 0.0
    assert contrasts["D_joint_y1H"]["decision"] == "negligible_at_pilot_resolution"
    assert contrasts["D_audio_y1H"]["estimate"] == 0.0
    # Only the two pre-registered contrasts get a binary decision.
    decided = {name for name, v in contrasts.items() if "decision" in v}
    assert decided == {"D_joint_y1H", "D_y1_at_HH"}
    # The unchanged-y1 subpopulation is reported but kept out of the estimand.
    assert metrics["unchanged_y1_cells"]["T_III"]["estimate"] == 0.0


def test_exp4_condition_codes_are_unchanged_by_the_y1_extension():
    cfg = load_exp4_config("configs/experiments/exp4_audio_kv_routing.yaml")
    assert len(cfg.exp4.conditions) == 4
    assert condition_parts("audio_host__tab_injected") == ("host", "injected", "injected")
    assert condition_parts("audio_host__tab_injected__y1_host") == ("host", "injected", "host")
    assert condition_code("audio_injected__tab_injected") == "III"
    assert condition_code("audio_host__tab_host__y1_host") == "HHH"
    with pytest.raises(ValueError, match="unknown"):
        condition_parts("audio_nonsense__tab_host")


def test_source_preflight_freezes_hashes_and_detects_mutation(tmp_path):
    cfg = load_exp4_config(
        "configs/experiments/exp4_audio_kv_routing.yaml",
        overrides=["exp4.source_exp3_run=exp3_source"],
    )
    paths = ResolvedPaths(
        workspace=tmp_path,
        data_dir=tmp_path / "data",
        output_dir=tmp_path / "outputs",
        cache_dir=tmp_path / "cache",
    )
    source_dir = paths.output_dir / cfg.exp4.source_exp3_run
    source_dir.mkdir(parents=True)
    snapshot = {
        "git_commit": "source-commit",
        "config": {
            "model": cfg.model.model_dump(mode="json"),
            "paths": cfg.paths.model_dump(mode="json"),
            "exp3": cfg.exp3.model_dump(mode="json"),
        },
    }
    (source_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False), encoding="utf-8"
    )
    (source_dir / "provenance.json").write_text("{}\n", encoding="utf-8")

    cohort = [
        {
            "pair_id": "phase:p0",
            "item_id": "p0",
            "contrast": "phase",
            "selection_role": "discordant",
            "role": "harmful",
        }
    ]
    _write_jsonl(source_dir / cfg.exp3.artifacts.pairs_file, cohort)
    _write_jsonl(
        source_dir / cfg.exp3.artifacts.behavior_file,
        [
            {"item_id": "p0", "arm": "pv_locked", "explicit_refusal": False},
            {"item_id": "p0", "arm": "pv_standard", "explicit_refusal": True},
        ],
    )
    _write_jsonl(source_dir / cfg.exp4.source_cohort_file, cohort)
    span_rows, relay_rows = [], []
    for direction in cfg.exp4.directions:
        span_rows.append(
            {
                "pair_id": "phase:p0",
                "direction": direction,
                "layer": 10,
                "condition": "real",
                "full_generation": True,
            }
        )
        for condition in ("open", "relay_closed"):
            relay_rows.append(
                {
                    "pair_id": "phase:p0",
                    "direction": direction,
                    "condition": condition,
                }
            )
    _write_jsonl(source_dir / cfg.exp4.source_span_patch_file, span_rows)
    relay_path = source_dir / cfg.exp4.source_relay_file
    _write_jsonl(relay_path, relay_rows)

    run_dir = paths.output_dir / "exp4_test"
    assert freeze_source_inputs(cfg, paths, run_dir) == cohort
    assert (run_dir / cfg.exp4.artifacts.source_manifest_file).is_file()
    assert freeze_source_inputs(cfg, paths, run_dir) == cohort

    relay_path.write_text(relay_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after the Exp4 manifest was frozen"):
        freeze_source_inputs(cfg, paths, run_dir)
