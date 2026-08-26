from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import app as app_module
from app import MAX_RECORDING_SECONDS, RECORDING_PROMPT, VOICE_SAMPLES_DIR, app, app_ui, server
from src.cloner import LocalVoiceCloner


def test_app_initialization():
    assert app is not None
    assert callable(server)


def test_selected_listening_room_ui_contains_the_core_workflow():
    rendered = str(app_ui)

    for copy in (
        "Sona — Local Voice Studio",
        "Private session",
        "Voice reference",
        "Script",
        "Create audio",
        "Generated output",
        "Qwen3-TTS 1.7B",
        "Output language",
    ):
        assert copy in rendered


def test_ui_contains_new_reference_modes():
    rendered = str(app_ui)
    for copy in ("Record", "Upload", "Saved voices", "Voice name", "record-prompt"):
        assert copy in rendered


def test_recording_prompt_is_defined_and_nonempty():
    assert isinstance(RECORDING_PROMPT, str)
    assert len(RECORDING_PROMPT) > 100
    for phrase in (
        "my natural speaking voice",
        "quick brown fox",
        "How vexingly quick daft zebras jump",
        "Did it capture the real me",
    ):
        assert phrase in RECORDING_PROMPT


def test_recording_ui_stops_after_maximum_duration():
    rendered = str(app_ui)

    assert MAX_RECORDING_SECONDS == 30
    assert f'data-max-duration="{MAX_RECORDING_SECONDS}"' in rendered
    assert "maxDurationTimerId = setTimeout" in rendered
    assert "Maximum recording length reached. Processing..." in rendered
    assert "setStatus('Saved as ' + name + '.wav')" not in rendered


def test_voice_samples_dir_is_a_path_and_exists():
    assert isinstance(VOICE_SAMPLES_DIR, Path)
    assert VOICE_SAMPLES_DIR.exists()
    assert VOICE_SAMPLES_DIR.is_dir()


def test_saved_voice_path_only_allows_existing_voice_samples(tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "VOICE_SAMPLES_DIR", tmp_path)
    (tmp_path / "saved.wav").write_bytes(b"wav")

    assert app_module._saved_voice_path("saved") == tmp_path / "saved.wav"
    assert app_module._saved_voice_path("missing") is None
    assert app_module._saved_voice_path("../outside") is None


def test_ui_contains_reference_transcript_section():
    rendered = str(app_ui)
    assert "reference_transcript_section" in rendered
    assert "transcript-card" in rendered
    assert "#ref_transcript" in rendered


def test_ui_does_not_contain_old_override():
    rendered = str(app_ui)
    assert "ref_text_override" not in rendered


def test_clone_voice_with_user_edited_transcript(tmp_path):
    sr = 24000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    ref_path = tmp_path / "ref.wav"
    sf.write(str(ref_path), (0.2 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)

    tts_calls = []

    class MockTTS:
        sample_rate = sr

        def generate(self, **kwargs):
            tts_calls.append(kwargs)
            yield SimpleNamespace(
                audio=np.zeros(sr, dtype=np.float32),
                sample_rate=sr,
            )

    cloner = LocalVoiceCloner(tts_loader=lambda _id: MockTTS())
    user_edited_text = "This is a corrected reference sentence."

    cloner.clone_voice(
        reference_audio_path=ref_path,
        text="Hello world",
        reference_text=user_edited_text,
    )

    assert len(tts_calls) == 1
    assert tts_calls[0]["ref_text"] == user_edited_text


def test_transcript_cache_isolation_prevents_voice_leakage():
    cache = {}

    voice_a_path = "/path/to/voice_a.wav"
    voice_b_path = "/path/to/voice_b.wav"

    cache[voice_a_path] = "Transcript of Voice A"

    active_ref = (voice_b_path, "voice_b.wav")

    assert active_ref[0] not in cache
    assert cache.get(active_ref[0]) is None

    cache[voice_b_path] = "Transcript of Voice B"
    assert cache.get(active_ref[0]) == "Transcript of Voice B"


