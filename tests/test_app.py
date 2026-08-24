from pathlib import Path

from app import RECORDING_PROMPT, VOICE_SAMPLES_DIR, app, app_ui, server


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


def test_voice_samples_dir_is_a_path_and_exists():
    assert isinstance(VOICE_SAMPLES_DIR, Path)
    assert VOICE_SAMPLES_DIR.exists()
    assert VOICE_SAMPLES_DIR.is_dir()
