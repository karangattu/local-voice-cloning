"""Fast behavioral tests for the Qwen3-TTS MLX voice-cloning engine."""

import json
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


@pytest.mark.parametrize("engine", ["qwen", "chatterbox"])
def test_speaking_pace_changes_duration_without_changing_pitch(sample_voice_file, engine):
    sr = 24000
    tone = (0.2 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)).astype(np.float32)

    class Model:
        sample_rate = 24000
        sr = 24000

        def generate(self, *args, **kwargs):
            if engine == "chatterbox":
                return tone
            return iter([SimpleNamespace(audio=tone, sample_rate=sr)])

    cloner = cloner_module.LocalVoiceCloner(engine=engine, tts_loader=lambda _: Model())
    result = cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi", speed=1.25)
    assert result.duration_seconds == pytest.approx(0.8, abs=0.02)
    spectrum = np.abs(np.fft.rfft(result.audio))
    peak_hz = np.argmax(spectrum) * sr / len(result.audio)
    assert peak_hz == pytest.approx(440, abs=3)


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
    assert call["speed"] == 1.0
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


def test_transcribe_reference_audio(sample_voice_file):
    stt_model = FakeSTTModel("Transcribed words for user review.")
    stt_loads: list[str] = []
    cloner = cloner_module.LocalVoiceCloner(
        stt_loader=lambda model_id: stt_loads.append(model_id) or stt_model,
    )

    transcript = cloner.transcribe(sample_voice_file)

    assert transcript == "Transcribed words for user review."
    assert stt_loads == ["mlx-community/whisper-large-v3-turbo-asr-fp16"]
    assert len(stt_model.calls) == 1
    assert stt_model.calls[0][0].endswith(".wav")


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


def test_clone_voice_inserts_an_audible_pause_at_each_newline(sample_voice_file, monkeypatch):
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


def test_script_segments_with_inline_pause_and_breaks():
    text = "Hello there [pause 0.8s] welcome to Sona [break] enjoy your stay."
    segments = cloner_module._script_segments(text)
    assert len(segments) == 3
    assert segments[0] == ("Hello there.", 0.0)
    assert segments[1] == ("welcome to Sona.", 0.8)
    assert segments[2] == ("enjoy your stay.", 0.4)


def test_script_segments_multiline():
    text = "Line one\n\nLine two"
    segments = cloner_module._script_segments(text)
    assert len(segments) == 2
    assert segments[0] == ("Line one.", 0.0)
    assert segments[1] == ("Line two.", 0.8)


def test_unload_shared_cloners():
    import sys

    cloner = sys.modules.get("src.cloner") or cloner_module
    cloner._shared_cloners[("qwen", "test_high")] = cloner.LocalVoiceCloner(
        tts_loader=lambda _: FakeTTSModel()
    )
    cloner._shared_cloners[("qwen", "test_fast")] = cloner.LocalVoiceCloner(
        tts_loader=lambda _: FakeTTSModel()
    )

    count = cloner.unload_shared_cloners()
    assert count >= 2
    assert len(cloner._shared_cloners) == 0


def test_sidecar_transcript_reads_json_next_to_reference(tmp_path):
    ref_path = tmp_path / "ref.wav"
    ref_path.write_bytes(b"")
    assert cloner_module.sidecar_transcript(ref_path) == ""

    (tmp_path / "ref.json").write_text("not json", encoding="utf-8")
    assert cloner_module.sidecar_transcript(ref_path) == ""

    (tmp_path / "ref.json").write_text(json.dumps({"transcript": 3}), encoding="utf-8")
    assert cloner_module.sidecar_transcript(ref_path) == ""

    (tmp_path / "ref.json").write_text(
        json.dumps({"transcript": "  Padded words.  ", "reference_id": "x"}), encoding="utf-8"
    )
    assert cloner_module.sidecar_transcript(ref_path) == "Padded words."


def test_clone_voice_prefers_sidecar_transcript_over_transcription(sample_voice_file):
    sample_voice_file.with_suffix(".json").write_text(
        json.dumps({"transcript": "Words from the sidecar."}), encoding="utf-8"
    )
    tts_model = FakeTTSModel()
    stt_model = FakeSTTModel()
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
        stt_loader=lambda _model_id: stt_model,
    )

    cloner.clone_voice(sample_voice_file, text="Hello")

    assert tts_model.calls[0]["ref_text"] == "Words from the sidecar."
    assert stt_model.calls == []


def test_clone_voice_prefers_explicit_reference_text_over_sidecar(sample_voice_file):
    sample_voice_file.with_suffix(".json").write_text(
        json.dumps({"transcript": "Words from the sidecar."}), encoding="utf-8"
    )
    tts_model = FakeTTSModel()
    stt_model = FakeSTTModel()
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
        stt_loader=lambda _model_id: stt_model,
    )

    cloner.clone_voice(sample_voice_file, text="Hello", reference_text="Explicit words.")

    assert tts_model.calls[0]["ref_text"] == "Explicit words."
    assert stt_model.calls == []


def test_warmup_loads_models_and_reports_timing():
    loaded: list[str] = []
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda model_id: loaded.append(model_id) or FakeTTSModel(),
        stt_loader=lambda model_id: loaded.append(model_id) or FakeSTTModel(),
    )

    timings = cloner.warmup(include_transcriber=True)

    assert set(timings) == {"tts", "transcribe"}
    assert all(seconds >= 0 for seconds in timings.values())
    assert loaded == [cloner_module.MODEL_VARIANTS["high"], cloner_module.ASR_MODEL_ID]
    assert cloner.model_loaded is True

    cloner.warmup(include_transcriber=True)
    assert loaded == [cloner_module.MODEL_VARIANTS["high"], cloner_module.ASR_MODEL_ID]


def test_warmup_wraps_download_errors():
    def failing_loader(_model_id):
        raise OSError("[Errno 5] Input/output error")

    cloner = cloner_module.LocalVoiceCloner(tts_loader=failing_loader)

    with pytest.raises(RuntimeError, match="model download failed for") as excinfo:
        cloner.warmup()
    message = str(excinfo.value)
    assert cloner_module.MODEL_VARIANTS["high"] in message
    assert "check network/HF access and rerun warmup" in message
