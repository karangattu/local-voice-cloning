"""Real-model tests for src/api.py /synthesize. Slow. Run with: pytest -m integration"""

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from src.api import app

pytestmark = pytest.mark.integration

client = TestClient(app)


@pytest.fixture
def reference_wav(tmp_path):
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    tone = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), tone, sr)
    return path


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
