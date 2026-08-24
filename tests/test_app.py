from app import app, app_ui, server


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
