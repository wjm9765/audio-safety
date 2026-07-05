from types import SimpleNamespace

from audio_safety.models import qwen2_audio
from audio_safety.models.qwen2_audio import (
    build_audio_analysis_conversation,
    prepare_qwen2_audio_inputs,
)


class FakeQwen2AudioProcessor:
    def __init__(self):
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.calls = []

    def apply_chat_template(self, conversation, *, add_generation_prompt, tokenize):
        assert add_generation_prompt is True
        assert tokenize is False
        return "chat prompt"

    def __call__(self, **kwargs):
        if "audios" in kwargs:
            raise AssertionError("Qwen2AudioProcessor expects audio=, not audios=")
        assert "audio" in kwargs
        self.calls.append(kwargs)
        return SimpleNamespace(input_ids=SimpleNamespace(shape=(1, 3)))


def test_prepare_qwen2_audio_inputs_uses_audio_keyword(monkeypatch):
    processor = FakeQwen2AudioProcessor()
    conversation = build_audio_analysis_conversation("clip.wav", "Answer the audio.")
    monkeypatch.setattr(qwen2_audio, "load_audio_array", lambda ref, sampling_rate: [0.0, 0.1])

    inputs = prepare_qwen2_audio_inputs(processor, conversation)

    assert inputs.input_ids.shape == (1, 3)
    assert processor.calls[0]["audio"] == [[0.0, 0.1]]
    assert processor.calls[0]["text"] == "chat prompt"
