import json
import threading
import time
import urllib.error
from email.message import Message

import pytest
from pydantic import ValidationError

import audio_safety.data.openrouter_pairs as openrouter_pairs
from audio_safety.config.schema import (
    OpenRouterPairGenerationConfig,
    OpenRouterStyleVariantConfig,
)
from audio_safety.data.openrouter_pairs import (
    _extract_content,
    generate_benign_pair,
    generate_pair_manifest,
    generate_style_variant_manifest,
)


def test_extract_content_rejects_null_with_debug_summary():
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "native_finish_reason": None,
                "message": {"content": None, "refusal": None},
            }
        ]
    }
    with pytest.raises(ValueError, match="content is null"):
        _extract_content(response)


def test_extract_content_accepts_segmented_text_content():
    response = {"choices": [{"message": {"content": [{"text": '{"benign_text":"x"}'}]}}]}
    assert _extract_content(response) == '{"benign_text":"x"}'


def test_generate_pair_manifest_saves_incrementally_and_resumes(monkeypatch, tmp_path):
    rows = [
        {
            "item_id": "row_1",
            "category": "cat",
            "harmful_text": "harmful one",
            "source": "test",
        },
        {
            "item_id": "row_2",
            "category": "cat",
            "harmful_text": "harmful two",
            "source": "test",
        },
    ]
    calls = []

    def fake_generate(row, cfg):
        calls.append(row["item_id"])
        return {
            "benign_text": f"benign {row['item_id']}",
            "category": row["category"],
            "rationale": "test",
            "generation_model": "model",
            "generation_mode": "prompt_json",
        }

    monkeypatch.setattr(openrouter_pairs, "generate_benign_pair", fake_generate)
    output = tmp_path / "pairs.jsonl"
    cfg = OpenRouterPairGenerationConfig()

    pairs = generate_pair_manifest(rows, cfg, output, limit=2, show_progress=False)
    assert [pair.item_id for pair in pairs] == ["row_1", "row_2"]
    assert calls == ["row_1", "row_2"]
    assert len(output.read_text().splitlines()) == 2

    calls.clear()
    resumed = generate_pair_manifest(rows, cfg, output, limit=2, show_progress=False)
    assert [pair.item_id for pair in resumed] == ["row_1", "row_2"]
    assert calls == []


def test_generate_pair_manifest_records_failures_and_continues(monkeypatch, tmp_path):
    rows = [
        {
            "item_id": "row_1",
            "category": "cat",
            "harmful_text": "harmful one",
            "source": "test",
        },
        {
            "item_id": "row_2",
            "category": "cat",
            "harmful_text": "harmful two",
            "source": "test",
        },
    ]

    def fake_generate(row, cfg):
        if row["item_id"] == "row_1":
            raise RuntimeError("provider length stop")
        return {
            "benign_text": "benign row_2",
            "category": row["category"],
            "rationale": "test",
            "generation_model": "model",
            "generation_mode": "prompt_json",
        }

    monkeypatch.setattr(openrouter_pairs, "generate_benign_pair", fake_generate)
    output = tmp_path / "pairs.jsonl"
    cfg = OpenRouterPairGenerationConfig()

    pairs = generate_pair_manifest(rows, cfg, output, limit=2, show_progress=False)

    assert [pair.item_id for pair in pairs] == ["row_2"]
    assert len(output.read_text().splitlines()) == 1
    errors = (tmp_path / "pairs.jsonl.errors.jsonl").read_text().splitlines()
    assert len(errors) == 1
    assert "provider length stop" in errors[0]


def test_pair_manifest_uses_bounded_concurrency_and_preserves_order(monkeypatch, tmp_path):
    rows = [
        {
            "item_id": f"row_{index}",
            "category": "cat",
            "harmful_text": f"harmful {index}",
            "source": "test",
        }
        for index in range(5)
    ]
    lock = threading.Lock()
    release_first_wave = threading.Event()
    active = 0
    max_active = 0
    calls: list[str] = []

    def fake_generate(row, cfg):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            calls.append(row["item_id"])
            if active == cfg.max_concurrency:
                release_first_wave.set()
        assert release_first_wave.wait(timeout=2.0)
        time.sleep(0.005 * (5 - int(row["item_id"].split("_")[1])))
        with lock:
            active -= 1
        return {
            "benign_text": f"benign {row['item_id']}",
            "category": row["category"],
            "rationale": "test",
            "generation_model": "model",
            "generation_mode": "prompt_json",
        }

    monkeypatch.setattr(openrouter_pairs, "generate_benign_pair", fake_generate)
    output = tmp_path / "pairs.jsonl"
    cfg = OpenRouterPairGenerationConfig(max_concurrency=3)

    pairs = generate_pair_manifest(rows, cfg, output, show_progress=False)

    assert max_active == 3
    assert sorted(calls) == [f"row_{index}" for index in range(5)]
    assert [pair.item_id for pair in pairs] == [f"row_{index}" for index in range(5)]
    saved_ids = [json.loads(line)["item_id"] for line in output.read_text().splitlines()]
    assert saved_ids == [f"row_{index}" for index in range(5)]


def test_pair_manifest_checkpoints_fast_completion_before_slow_job(monkeypatch, tmp_path):
    rows = [
        {
            "item_id": "slow",
            "category": "cat",
            "harmful_text": "harmful slow",
            "source": "test",
        },
        {
            "item_id": "fast",
            "category": "cat",
            "harmful_text": "harmful fast",
            "source": "test",
        },
    ]
    slow_started = threading.Event()
    release_slow = threading.Event()
    fast_checkpointed = threading.Event()
    worker_errors: list[BaseException] = []
    output = tmp_path / "pairs.jsonl"

    def fake_generate(row, cfg):
        if row["item_id"] == "slow":
            slow_started.set()
            assert release_slow.wait(timeout=2.0)
        return {
            "benign_text": f"benign {row['item_id']}",
            "category": row["category"],
            "rationale": "test",
            "generation_model": "model",
            "generation_mode": "prompt_json",
        }

    original_save_jsonl = openrouter_pairs.save_jsonl

    def tracking_save_jsonl(records, path):
        materialized = list(records)
        original_save_jsonl(materialized, path)
        if path == output and [record["item_id"] for record in materialized] == ["fast"]:
            fast_checkpointed.set()

    def run_manifest():
        try:
            generate_pair_manifest(
                rows,
                OpenRouterPairGenerationConfig(max_concurrency=2),
                output,
                show_progress=False,
            )
        except BaseException as exc:  # pragma: no cover - surfaced by the assertion below
            worker_errors.append(exc)

    monkeypatch.setattr(openrouter_pairs, "generate_benign_pair", fake_generate)
    monkeypatch.setattr(openrouter_pairs, "save_jsonl", tracking_save_jsonl)
    thread = threading.Thread(target=run_manifest)
    thread.start()
    assert slow_started.wait(timeout=2.0)
    assert fast_checkpointed.wait(timeout=2.0)
    release_slow.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert worker_errors == []
    saved_ids = [json.loads(line)["item_id"] for line in output.read_text().splitlines()]
    assert saved_ids == ["slow", "fast"]


def test_style_manifest_concurrent_jobs_keep_composite_identity(monkeypatch, tmp_path):
    rows = [
        {
            "item_id": "row_1",
            "category": "cat",
            "safety_label": safety_label,
            "base_text": f"{safety_label} text",
            "source": "test",
        }
        for safety_label in ("harmful", "benign")
    ]

    def fake_generate(row, cfg, *, style):
        if row["safety_label"] == "harmful" and style == "sad":
            time.sleep(0.02)
        return {
            "styled_text": f"{row['safety_label']} {style}",
            "content_preservation": "high",
            "added_operational_detail": False,
            "refusal_or_warning": False,
            "rationale": "test",
            "generation_model": "model",
            "generation_mode": "prompt_json",
        }

    monkeypatch.setattr(openrouter_pairs, "generate_style_variant", fake_generate)
    output = tmp_path / "styles.jsonl"
    cfg = OpenRouterStyleVariantConfig(max_concurrency=4)

    records = generate_style_variant_manifest(
        rows,
        cfg,
        output,
        styles=["sad", "angry"],
        show_progress=False,
    )

    keys = [
        (record["item_id"], record["safety_label"], record["target_style"]) for record in records
    ]
    assert keys == [
        ("row_1", "harmful", "sad"),
        ("row_1", "harmful", "angry"),
        ("row_1", "benign", "sad"),
        ("row_1", "benign", "angry"),
    ]
    assert len(set(keys)) == 4


def test_rate_limit_retry_honors_retry_after_without_format_retry(monkeypatch):
    row = {
        "item_id": "row_1",
        "category": "cat",
        "harmful_text": "harmful",
        "source": "test",
    }
    headers = Message()
    headers["Retry-After"] = "3"
    calls: list[tuple[str, bool]] = []
    sleeps: list[float] = []

    def fake_call(row, cfg, *, model, api_key, structured_output):
        calls.append((model, structured_output))
        if len(calls) == 1:
            raise urllib.error.HTTPError(cfg.endpoint, 429, "rate limited", headers, None)
        return {"benign_text": "safe", "category": "cat", "rationale": "test"}

    monkeypatch.setattr(openrouter_pairs, "call_openrouter_pair_generator", fake_call)
    monkeypatch.setattr(openrouter_pairs.time, "sleep", sleeps.append)
    cfg = OpenRouterPairGenerationConfig(
        model="model",
        fallback_models=[],
        retries=1,
    )

    result = generate_benign_pair(row, cfg, api_key="test-key")

    assert result["benign_text"] == "safe"
    assert calls == [("model", True), ("model", True)]
    assert sleeps == [3.0]


def test_openrouter_concurrency_bounds_are_validated():
    with pytest.raises(ValidationError):
        OpenRouterPairGenerationConfig(max_concurrency=0)
    with pytest.raises(ValidationError):
        OpenRouterStyleVariantConfig(max_concurrency=65)


def test_pair_manifest_rejects_duplicate_item_ids_before_network(monkeypatch, tmp_path):
    row = {
        "item_id": "duplicate",
        "category": "cat",
        "harmful_text": "harmful",
        "source": "test",
    }
    called = False

    def fake_generate(row, cfg):
        nonlocal called
        called = True
        raise AssertionError("network should not be called")

    monkeypatch.setattr(openrouter_pairs, "generate_benign_pair", fake_generate)
    with pytest.raises(ValueError, match="duplicate item_id"):
        generate_pair_manifest(
            [row, row.copy()],
            OpenRouterPairGenerationConfig(max_concurrency=2),
            tmp_path / "pairs.jsonl",
            show_progress=False,
        )
    assert called is False
