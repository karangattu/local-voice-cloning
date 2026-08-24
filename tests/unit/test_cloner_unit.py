"""Fast behavioral tests for the Qwen3-TTS MLX voice-cloning engine."""

from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

import src.cloner as cloner_module


class FakeTTSModel:
    sample_rate = 24000

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        yield SimpleNamespace(
            audio=np.zeros(self.sample_rate, dtype=np.float32),
            sample_rate=self.sample_rate,
        )


class FakeSTTModel:
    def __init__(self, transcript: str = "The exact reference transcript."):
        self.transcript = transcript
        self.calls: list[tuple[str, dict]] = []

    def generate(self, audio_path, **kwargs):
        self.calls.append((str(audio_path), kwargs))
        return SimpleNamespace(text=self.transcript)


class ToneTTSModel(FakeTTSModel):
    def generate(self, **kwargs):
        self.calls.append(kwargs)
        tone = np.full(round(self.sample_rate * 0.1), 0.2, dtype=np.float32)
        yield SimpleNamespace(audio=tone, sample_rate=self.sample_rate)


@pytest.fixture
def sample_voice_file(tmp_path):
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    path = tmp_path / "ref.wav"
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
    return path


def test_model_id_for_quality_selects_high_fidelity_and_fast_checkpoints():
    assert cloner_module.model_id_for_quality("high") == (
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"
    )
    assert cloner_module.model_id_for_quality("fast") == (
        "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit"
    )
    with pytest.raises(ValueError, match="Unknown quality"):
        cloner_module.model_id_for_quality("ultra")


def test_cloner_initialization_is_lazy():
    loaded_models: list[str] = []

    def tts_loader(model_id):
        loaded_models.append(model_id)
        return FakeTTSModel()

    cloner = cloner_module.LocalVoiceCloner(quality="high", tts_loader=tts_loader)

    assert loaded_models == []
    assert cloner.model_loaded is False
    assert cloner.engine_name == "Qwen3-TTS 1.7B"


def test_clone_voice_uses_reference_transcript_and_reports_real_stages(sample_voice_file):
    tts_model = FakeTTSModel()
    stt_loads: list[str] = []
    stages: list[str] = []
    cloner = cloner_module.LocalVoiceCloner(
        quality="high",
        tts_loader=lambda _model_id: tts_model,
        stt_loader=lambda model_id: stt_loads.append(model_id),
    )

    result = cloner.clone_voice(
        sample_voice_file,
        text="Hello there",
        reference_text="Words from the recording.",
        speed=1.1,
        language="English",
        progress_callback=stages.append,
    )

    assert isinstance(result, cloner_module.SynthesisResult)
    assert result.sample_rate == 24000
    assert result.matched_voice == "Qwen3-TTS 1.7B · high fidelity"
    assert stages == ["prepare", "load", "voice", "finish"]
    assert stt_loads == []
    call = tts_model.calls[0]
    assert call["text"] == "Hello there."
    assert call["ref_text"] == "Words from the recording."
    assert call["speed"] == 1.1
    assert call["lang_code"] == "English"
    assert call["stream"] is False
    assert call["ref_audio"].endswith(".wav")


def test_clone_voice_auto_transcribes_when_reference_text_is_missing(sample_voice_file):
    tts_model = FakeTTSModel()
    stt_model = FakeSTTModel("Automatically transcribed reference.")
    stt_model_ids: list[str] = []

    def stt_loader(model_id):
        stt_model_ids.append(model_id)
        return stt_model

    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
        stt_loader=stt_loader,
    )
    cloner.clone_voice(sample_voice_file, text="Hello")

    assert stt_model_ids == ["mlx-community/whisper-large-v3-turbo-asr-fp16"]
    assert len(stt_model.calls) == 1
    assert tts_model.calls[0]["ref_text"] == "Automatically transcribed reference."


def test_clone_voice_rejects_empty_text_before_loading_models(sample_voice_file):
    loaded_models: list[str] = []
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda model_id: loaded_models.append(model_id),
    )

    with pytest.raises(ValueError, match="Input text cannot be empty"):
        cloner.clone_voice(sample_voice_file, text="")

    assert loaded_models == []


def test_prepare_gen_text_adds_final_stop():
    assert cloner_module.prepare_gen_text("Hello world") == "Hello world."
    assert cloner_module.prepare_gen_text("  Hello world  ") == "Hello world."
    assert cloner_module.prepare_gen_text("Is that so?") == "Is that so?"
    assert cloner_module.prepare_gen_text("") == ""


def test_clone_voice_inserts_an_audible_pause_at_each_newline(
    sample_voice_file, monkeypatch
):
    tts_model = ToneTTSModel()
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
    )
    monkeypatch.setattr(
        cloner_module,
        "enhance_audio",
        lambda audio, _sample_rate, **_kwargs: audio,
    )

    result = cloner.clone_voice(
        sample_voice_file,
        text="First line\nSecond line",
        reference_text="Reference words.",
    )

    assert [call["text"] for call in tts_model.calls] == [
        "First line.",
        "Second line.",
    ]
    line_samples = round(result.sample_rate * 0.1)
    pause_samples = round(result.sample_rate * 0.4)
    assert len(result.audio) == (line_samples * 2) + pause_samples
    assert np.all(result.audio[line_samples : line_samples + pause_samples] == 0)


def test_cloner_module_import_does_not_require_mlx(monkeypatch):
    import builtins
    import importlib
    import sys

    original_import = builtins.__import__

    def block_mlx_import(name, *args, **kwargs):
        if name == "mlx_audio" or name.startswith("mlx_audio."):
            raise ModuleNotFoundError("MLX is not available on this platform")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_mlx_import)
    sys.modules.pop("src.cloner", None)

    imported = importlib.import_module("src.cloner")

    assert imported.ENGINE_NAME == "Qwen3-TTS 1.7B"
