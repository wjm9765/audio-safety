import pytest

import audio_safety.data.openrouter_pairs as openrouter_pairs
from audio_safety.config.schema import OpenRouterPairGenerationConfig
from audio_safety.data.openrouter_pairs import _extract_content, generate_pair_manifest


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
