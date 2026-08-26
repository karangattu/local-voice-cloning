from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf

import app as app_module
from app import (
    MAX_RECORDING_SECONDS,
    RECORDING_PROMPT,
    RECORDING_TEMPLATES,
    VOICE_SAMPLES_DIR,
    app,
    app_ui,
    server,
)
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
    for copy in ("Record", "Upload", "Saved voices", "Voice name", "recording_prompt_display", "record_template"):
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


def test_recording_templates_defined_and_contain_conversational():
    assert "standard" in RECORDING_TEMPLATES
    assert "conversational" in RECORDING_TEMPLATES
    assert RECORDING_TEMPLATES["standard"] == RECORDING_PROMPT
    conversational = RECORDING_TEMPLATES["conversational"]
    assert "Hi, I’m [name]." in conversational
    assert "Today is a beautiful day, and I’m feeling pretty good." in conversational
    assert "Can you believe it? I have three things to finish, then I’m heading home." in conversational


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


def test_resolve_reference_transcript_states():
    assert app_module.resolve_reference_transcript("", {}, set()) == ("", "idle", False)
    assert app_module.resolve_reference_transcript("ref_a", {"ref_a": "Text A"}, set()) == ("Text A", "ready", False)
    assert app_module.resolve_reference_transcript("ref_b", {}, {"ref_b"}) == ("", "transcribing", False)
    assert app_module.resolve_reference_transcript("ref_c", {}, set()) == ("", "transcribing", True)


def test_user_corrections_persist_across_voice_switches():
    cache = {}
    pending = set()

    cache, pending, is_current, resolved, err = app_module.apply_transcription_result(
        ref_id="ref_a",
        transcript="I like bears",
        error=None,
        current_ref_id="ref_a",
        cache=cache,
        pending=pending,
    )
    assert is_current is True
    assert resolved == "I like bears"
    assert err is None

    cache = app_module.record_user_transcript_edit(
        ref_id="ref_a",
        edited_text="I like birds",
        cache=cache,
    )
    assert cache["ref_a"] == "I like birds"

    text_b, status_b, should_run_b = app_module.resolve_reference_transcript(
        ref_id="ref_b",
        cache=cache,
        pending=pending,
    )
    assert text_b == ""
    assert status_b == "transcribing"
    assert should_run_b is True

    text_a, status_a, should_run_a = app_module.resolve_reference_transcript(
        ref_id="ref_a",
        cache=cache,
        pending=pending,
    )
    assert text_a == "I like birds"
    assert status_a == "ready"
    assert should_run_a is False


def test_transcription_completion_does_not_leak_to_switched_reference():
    cache = {}
    pending = {"ref_a", "ref_b"}

    cache, pending, is_current, resolved, err = app_module.apply_transcription_result(
        ref_id="ref_a",
        transcript="Transcript of voice A",
        error=None,
        current_ref_id="ref_b",
        cache=cache,
        pending=pending,
    )

    assert is_current is False
    assert resolved == "Transcript of voice A"
    assert err is None
    assert "ref_a" not in pending
    assert "ref_b" in pending
    assert cache["ref_a"] == "Transcript of voice A"
    assert "ref_b" not in cache

    text_b, status_b, should_run_b = app_module.resolve_reference_transcript(
        ref_id="ref_b",
        cache=cache,
        pending=pending,
    )
    assert text_b == ""
    assert status_b == "transcribing"
    assert should_run_b is False


def test_transcription_failure_does_not_leak_or_trap_active_reference():
    cache = {}
    pending = {"ref_a", "ref_b"}

    cache, pending, is_current, resolved, err = app_module.apply_transcription_result(
        ref_id="ref_a",
        transcript=None,
        error="Whisper decoding error",
        current_ref_id="ref_b",
        cache=cache,
        pending=pending,
    )

    assert is_current is False
    assert resolved == ""
    assert err == "Whisper decoding error"
    assert "ref_a" not in pending
    assert "ref_b" in pending

    text_b, status_b, should_run_b = app_module.resolve_reference_transcript(
        ref_id="ref_b",
        cache=cache,
        pending=pending,
    )
    assert text_b == ""
    assert status_b == "transcribing"
    assert should_run_b is False

    text_a, status_a, should_run_a = app_module.resolve_reference_transcript(
        ref_id="ref_a",
        cache=cache,
        pending=pending,
    )
    assert text_a == ""
    assert status_a == "transcribing"
    assert should_run_a is True


def test_overwritten_audio_path_invalidates_old_transcript_cache(tmp_path):
    wav_file = tmp_path / "karan.wav"
    wav_file.write_bytes(b"initial audio recording")

    id_1 = app_module.get_reference_id(wav_file)
    cache = {id_1: "Transcript of first speech"}
    pending = set()

    text_1, status_1, should_run_1 = app_module.resolve_reference_transcript(
        ref_id=id_1,
        cache=cache,
        pending=pending,
    )
    assert text_1 == "Transcript of first speech"
    assert status_1 == "ready"
    assert should_run_1 is False

    import time
    time.sleep(0.01)
    wav_file.write_bytes(b"completely different speech content with different size")

    id_2 = app_module.get_reference_id(wav_file)
    assert id_1 != id_2

    text_2, status_2, should_run_2 = app_module.resolve_reference_transcript(
        ref_id=id_2,
        cache=cache,
        pending=pending,
    )
    assert text_2 == ""
    assert status_2 == "transcribing"
    assert should_run_2 is True


def test_estimate_speech_duration_seconds():
    assert app_module.estimate_speech_duration_seconds("") == 0.0
    assert app_module.estimate_speech_duration_seconds("   ") == 0.0
    words_24 = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four"
    assert round(app_module.estimate_speech_duration_seconds(words_24), 1) == 10.0


def test_ui_contains_vu_meter_and_progress_track():
    rendered = str(app_ui)
    assert "record-vu-meter" in rendered
    assert "vu-bar" in rendered
    assert "record-progress-track" in rendered
    assert "record-progress-fill" in rendered


def test_ui_contains_keyboard_shortcut_listener():
    rendered = str(app_ui)
    assert "sonaCopyTranscript" in rendered
    assert "sonaUseAsScript" in rendered
    assert "sonaSetSpeed" in rendered
    assert "kbd-shortcut" in rendered
    assert "⌘↵" in rendered
