"""Fast tests for API metadata and request validation."""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)


@pytest.fixture
def reference_wav(tmp_path):
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), tone, sr)
    return path


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_info():
    response = client.get("/info")
    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "Qwen3-TTS 1.7B"
    assert body["device"] == "mlx"
    assert body["default_quality"] == "high"
    assert body["quality_models"] == {
        "fast": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
        "high": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    }
    assert body["supported_languages"] == [
        "auto",
        "Chinese",
        "English",
        "French",
        "German",
        "Italian",
        "Japanese",
        "Korean",
        "Portuguese",
        "Russian",
        "Spanish",
    ]
    assert body["supported_output_formats"] == ["mp3", "wav"]


def test_synthesize_rejects_bad_format(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Hello", "output_format": "ogg"},
        )
    assert response.status_code == 422
    assert "Unsupported output format" in response.json()["detail"]


def test_synthesize_rejects_empty_text(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "   ", "output_format": "wav"},
        )
    assert response.status_code == 422


def test_synthesize_rejects_empty_upload():
    response = client.post(
        "/synthesize",
        files={"reference_audio": ("ref.wav", b"", "audio/wav")},
        data={"text": "Hello"},
    )
    assert response.status_code == 422


def test_synthesize_rejects_unknown_quality(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Hello", "quality": "ultra"},
        )
    assert response.status_code == 422
    assert "Unknown quality" in response.json()["detail"]


def test_synthesize_rejects_unknown_language(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Hello", "language": "Klingon"},
        )
    assert response.status_code == 422
    assert "Unsupported language" in response.json()["detail"]


def test_transcribe_endpoint(reference_wav, monkeypatch):
    import src.api as api_module

    class DummyCloner:
        def transcribe(self, _path):
            return "Transcribed words."

    monkeypatch.setattr(api_module, "get_shared_cloner", lambda _quality: DummyCloner())

    with open(reference_wav, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"quality": "high"},
        )
    assert response.status_code == 200
    assert response.json() == {"transcript": "Transcribed words."}


def test_transcribe_rejects_empty_upload():
    response = client.post(
        "/transcribe",
        files={"reference_audio": ("ref.wav", b"", "audio/wav")},
    )
    assert response.status_code == 422


def test_transcribe_rejects_unknown_quality(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"quality": "invalid"},
        )
    assert response.status_code == 422
    assert "Unknown quality" in response.json()["detail"]

