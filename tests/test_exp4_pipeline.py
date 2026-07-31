import json
from types import SimpleNamespace

import pytest
import yaml

from audio_safety.config import load_exp4_config
from audio_safety.pipelines.exp4_audio_kv_routing import (
    analyze_records,
    cache_max_abs_diff,
    clone_kv_cache,
    freeze_source_inputs,
    manual_greedy_decode,
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
        generation_config = SimpleNamespace(eos_token_id=9)

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
        first_token_id=1,
    )
    assert result["generated_token_ids"] == [1, 2, 9]
    assert model.calls == [([[1]], 6), ([[2]], 7)]


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
