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
    assert body["engine"] == "F5-TTS"
    assert body["default_quality_steps"] in (32, 64)
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


def test_synthesize_wav(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Hello from the API test", "steps": "8", "output_format": "wav"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/wav"
    assert float(response.headers["x-duration-seconds"]) > 0
    assert len(response.content) > 1000


def test_synthesize_mp3(reference_wav):
    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Hello again", "steps": "8", "output_format": "mp3"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "audio/mpeg"
    assert len(response.content) > 500
