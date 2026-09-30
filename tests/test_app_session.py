"""Exercise Shiny's actual reactive session with lightweight model output."""

import re
import threading
from types import SimpleNamespace

import numpy as np
import soundfile as sf
from starlette.testclient import TestClient

import app as module


class SessionMessages:
    def __init__(self, ws):
        self.ws = ws
        self.values = {}

    def until(self, output, expected):
        for _ in range(200):
            value = self.values.get(output)
            if value is not None and expected in value.get("html", ""):
                return value["html"]
            message = self.ws.receive_json()
            assert not message.get("errors"), message.get("errors")
            self.values.update(message.get("values", {}))
        raise AssertionError(f"Shiny did not settle on {output}: {expected}")


def test_generation_history_comparison_and_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(module, "VOICE_SAMPLES_DIR", tmp_path)
    sf.write(tmp_path / "alice.wav", np.full(24000, 0.1), 24000)
    sf.write(tmp_path / "bob.wav", np.full(24000, 0.2), 24000)
    module._save_voice_metadata("bob", {"transcript": "Bob words."}, tmp_path)
    release = threading.Event()
    requests = []

    class Cloner:
        def transcribe(self, path):
            return "Reference words."

        def clone_voice(self, **kwargs):
            requests.append(kwargs)
            assert release.wait(5), "Test did not release model generation"
            return SimpleNamespace(audio=np.full(24000, 0.1), sample_rate=24000)

    monkeypatch.setattr(module, "get_shared_cloner", lambda *a, **k: Cloner())
    initial = {
        "ref_mode": "library",
        "voice_library": "alice",
        "quality": "high",
        "engine": "qwen",
        "language": "auto",
        "speech_text": "First script.",
        "synthesis_speed": 1.0,
        "script_preset": "custom",
        "record_template": "standard",
        "ref_transcript": "Reference words.",
        "btn_generate:shiny.action": 0,
    }
    for name in ("session_history_ui", "ab_comparison_ui", "audio_result", "character_count"):
        initial[f".clientdata_output_{name}_hidden"] = False

    with TestClient(module.app) as client, client.websocket_connect("/websocket/") as ws:
        messages = SessionMessages(ws)
        ws.send_json({"method": "init", "data": initial})
        messages.until("session_history_ui", "<div>")
        ws.send_json({"method": "update", "data": {"btn_generate:shiny.action": 1}})
        ws.send_json({"method": "update", "data": {"speech_text": "Edited during generation."}})
        release.set()
        first = messages.until("session_history_ui", "Session Takes (1)")
        assert "First script." in first
        assert "Edited during generation." not in first
        first_id = re.search(r"take-audio-([a-f0-9]+)", first).group(1)
        download = re.search(r'href="([^"]*take-download-[^"]+)"', first).group(1)
        assert client.get("/" + download).content[:4] == b"RIFF"

        ws.send_json(
            {
                "method": "update",
                "data": {"speech_text": "Second script.", "btn_generate:shiny.action": 2},
            }
        )
        second = messages.until("session_history_ui", "Session Takes (2)")
        assert "First script." in second and "Second script." in second
        comparison = messages.until("ab_comparison_ui", "A/B Voice Comparison")
        assert 'id="ab_select_a"' in comparison and 'id="ab_select_b"' in comparison

        ws.send_json({"method": "update", "data": {"ab_select_a": first_id}})
        messages.values.pop("ab_comparison_ui", None)
        comparison = messages.until("ab_comparison_ui", "A/B Voice Comparison")
        assert re.search(rf'<option value="{first_id}" selected="">', comparison)

        # Editing after success must not append another take.
        ws.send_json(
            {
                "method": "update",
                "data": {"speech_text": "New draft.", "btn_generate:shiny.action": 3},
            }
        )
        third = messages.until("session_history_ui", "Session Takes (3)")
        assert third.count('class="history-item"') == 3

        # A newly selected voice must use its own transcript immediately,
        # even before the browser acknowledges the textarea update.
        ws.send_json(
            {"method": "update", "data": {"voice_library": "bob", "btn_generate:shiny.action": 4}}
        )
        messages.until("session_history_ui", "Session Takes (4)")
        assert requests[-1]["reference_text"] == "Bob words."
