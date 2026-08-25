from pathlib import Path

import app as app_module
from app import MAX_RECORDING_SECONDS, RECORDING_PROMPT, VOICE_SAMPLES_DIR, app, app_ui, server


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
