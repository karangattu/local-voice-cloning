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


def test_reference_cache_skips_reload_and_retranscription(sample_voice_file, monkeypatch):
    tts_model = FakeTTSModel()
    stt_model = FakeSTTModel("Cached words.")
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
        stt_loader=lambda _model_id: stt_model,
    )
    loads: list[str] = []
    original_load = cloner._load_reference

    def counting_load(path):
        loads.append(str(path))
        return original_load(path)

    monkeypatch.setattr(cloner, "_load_reference", counting_load)

    cloner.clone_voice(sample_voice_file, text="One")
    cloner.clone_voice(sample_voice_file, text="Two")

    assert len(loads) == 1
    assert len(stt_model.calls) == 1
    assert [call["text"] for call in tts_model.calls] == ["One.", "Two."]
    assert [call["ref_text"] for call in tts_model.calls] == ["Cached words.", "Cached words."]


def test_reference_cache_reuses_transcribe_result(sample_voice_file):
    tts_model = FakeTTSModel()
    stt_model = FakeSTTModel("Shared words.")
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
        stt_loader=lambda _model_id: stt_model,
    )

    assert cloner.transcribe(sample_voice_file) == "Shared words."
    cloner.clone_voice(sample_voice_file, text="Hello")

    assert len(stt_model.calls) == 1
    assert tts_model.calls[0]["ref_text"] == "Shared words."


def test_reference_cache_evicts_beyond_four_entries(tmp_path):
    tts_model = FakeTTSModel()
    stt_model = FakeSTTModel()
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _model_id: tts_model,
        stt_loader=lambda _model_id: stt_model,
    )
    refs = []
    for index in range(5):
        path = tmp_path / f"ref{index}.wav"
        sf.write(str(path), np.full(24000, 0.1 * (index + 1), dtype=np.float32), 24000)
        refs.append(path)

    for path in refs:
        cloner.clone_voice(path, text="Hello")

    assert len(cloner._reference_cache) == cloner_module.REFERENCE_CACHE_SIZE == 4
    assert len(stt_model.calls) == 5

    cloner.clone_voice(refs[0], text="Hello")

    assert len(stt_model.calls) == 6


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


class NoiseTTSModel(FakeTTSModel):
    """Returns one random draw per call, so tests can see the sampler state."""

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        draw = float(np.random.rand())
        self.draws = [*getattr(self, "draws", []), draw]
        yield SimpleNamespace(
            audio=np.full(self.sample_rate // 10, 0.2, dtype=np.float32),
            sample_rate=self.sample_rate,
        )


def test_clone_voice_reseeds_each_line_with_one_job_seed(sample_voice_file, monkeypatch):
    seeds: list[int] = []
    real_seed = cloner_module.seed_generators
    monkeypatch.setattr(
        cloner_module, "seed_generators", lambda value: (seeds.append(value), real_seed(value))
    )
    model = NoiseTTSModel()
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model)

    cloner.clone_voice(sample_voice_file, "One.\nTwo.\nThree.", reference_text="Hi")

    assert len(seeds) == 3 and len(set(seeds)) == 1
    assert len(set(model.draws)) == 1


def test_clone_voice_explicit_seed_is_reproducible_and_default_varies(sample_voice_file):
    model = NoiseTTSModel()
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model)

    for _ in range(2):
        cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi", seed=7)
    for _ in range(2):
        cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi")

    assert model.draws[0] == model.draws[1]
    assert model.draws[2] != model.draws[3]


def test_script_segments_pause_follows_line_punctuation():
    segments = cloner_module._script_segments("Wait,\nthen go.\nWell...\nOkay.\n\nNext.")
    assert [pause for _, pause in segments] == [0.0, 0.2, 0.4, 0.6, 0.8]


def test_clone_voice_spells_out_numbers_for_english(sample_voice_file):
    model = FakeTTSModel()
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model)

    cloner.clone_voice(sample_voice_file, "It costs $5 [pause 0.5s] at 3:30", reference_text="Hi")
    cloner.clone_voice(
        sample_voice_file, "Il coûte 5 euros", reference_text="Hi", language="French"
    )
    cloner.clone_voice(sample_voice_file, "It costs $5", reference_text="Hi", normalize_text=False)

    assert [call["text"] for call in model.calls] == [
        "It costs five dollars.",
        "at three thirty.",
        "Il coûte 5 euros.",
        "It costs $5.",
    ]


def test_qwen_cloning_uses_configured_repetition_penalty(sample_voice_file):
    class ICLModel(FakeTTSModel):
        speech_tokenizer = SimpleNamespace(has_encoder=True)

        def _generate_icl(self, **kwargs):
            self.calls.append(kwargs)
            return self.generate_audio()

        def generate_audio(self):
            yield SimpleNamespace(audio=np.zeros(2400, dtype=np.float32), sample_rate=24000)

    icl_model = ICLModel()
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: icl_model)
    cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi")
    assert icl_model.calls[0]["repetition_penalty"] == cloner_module.QWEN_REPETITION_PENALTY
    assert icl_model.calls[0]["ref_text"] == "Hi"
    assert icl_model.calls[0]["ref_audio"].ndim == 1

    plain_model = FakeTTSModel()
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: plain_model)
    cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi", repetition_penalty=1.3)
    assert plain_model.calls[0]["repetition_penalty"] == 1.3


class TakeTTSModel(FakeTTSModel):
    """Each call returns a take filled with its own call number."""

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        take = len(self.calls)
        yield SimpleNamespace(
            audio=np.full(2400, 0.01 * take, dtype=np.float32), sample_rate=self.sample_rate
        )


class SequenceSTTModel:
    def __init__(self, transcripts):
        self.transcripts = list(transcripts)

    def generate(self, audio_path, **kwargs):
        return SimpleNamespace(text=self.transcripts.pop(0))


def _capture_pieces(monkeypatch):
    captured = []
    monkeypatch.setattr(
        cloner_module,
        "join_with_room_tone",
        lambda pieces, _sr: captured.extend(pieces) or np.concatenate([p for p, _ in pieces]),
    )
    return captured


def test_best_of_n_keeps_take_whose_transcript_matches(sample_voice_file, monkeypatch):
    pieces = _capture_pieces(monkeypatch)
    model = TakeTTSModel()
    stt = SequenceSTTModel(["Hello their friend", "Hello there.", "Hello."])
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model, stt_loader=lambda _: stt)

    cloner.clone_voice(sample_voice_file, "Hello there", reference_text="Hi", candidates=3)

    # Take 2 is a perfect match, so take 3 is never generated.
    assert len(model.calls) == 2
    assert pieces[0][0][0] == pytest.approx(0.02)


def test_best_of_n_seeds_takes_differently(sample_voice_file, monkeypatch):
    seeds: list[int] = []
    monkeypatch.setattr(cloner_module, "seed_generators", seeds.append)
    stt = SequenceSTTModel(["no", "no", "no", "no"])
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _: TakeTTSModel(), stt_loader=lambda _: stt
    )
    cloner.clone_voice(sample_voice_file, "One\nTwo", reference_text="Hi", candidates=2, seed=10)
    assert seeds == [10, 11, 10, 11]


def test_candidates_must_be_in_range(sample_voice_file):
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: FakeTTSModel())
    with pytest.raises(ValueError, match="candidates"):
        cloner.clone_voice(sample_voice_file, "Hello", reference_text="Hi", candidates=0)


def test_word_error_rate_and_tokens():
    assert cloner_module.word_error_rate(["a", "b", "c"], ["a", "x", "c"]) == pytest.approx(1 / 3)
    assert cloner_module.spoken_tokens("It's $5!", normalize=True) == ["its", "five", "dollars"]
    assert cloner_module.spoken_tokens("你好吗") == ["你", "好", "吗"]


@pytest.fixture
def long_voice_file(tmp_path):
    sr = 24000
    t = np.arange(sr * 20) / sr
    path = tmp_path / "long.wav"
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
    return path


def test_full_recording_transcript_is_replaced_when_clip_was_cut(long_voice_file):
    model = FakeTTSModel()
    stt = FakeSTTModel("Only the first few words.")
    cloner = cloner_module.LocalVoiceCloner(tts_loader=lambda _: model, stt_loader=lambda _: stt)
    full = " ".join(["word"] * 40)

    cloner.clone_voice(long_voice_file, "Hello", reference_text=full)
    cloner.clone_voice(long_voice_file, "Hello", reference_text="Only the first few words here.")

    assert model.calls[0]["ref_text"] == "Only the first few words."
    assert model.calls[1]["ref_text"] == "Only the first few words here."
    assert len(stt.calls) == 1


def test_given_transcript_is_kept_when_clip_was_not_cut(sample_voice_file):
    model = FakeTTSModel()
    cloner = cloner_module.LocalVoiceCloner(
        tts_loader=lambda _: model, stt_loader=lambda _: pytest.fail("no transcription needed")
    )
    full = " ".join(["word"] * 40)
    cloner.clone_voice(sample_voice_file, "Hello", reference_text=full)
    assert model.calls[0]["ref_text"] == full
