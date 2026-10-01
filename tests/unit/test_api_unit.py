"""Fast tests for API metadata and request validation."""

import importlib.metadata
import json
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import src.api as api_module
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


class StubCloner:
    model_id = "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16"

    def __init__(self, error: Exception | None = None):
        self.error = error
        self.model_loaded = error is None
        self.warmup_calls: list[bool] = []

    def warmup(self, include_transcriber: bool = False):
        self.warmup_calls.append(include_transcriber)
        if self.error is not None:
            raise self.error
        return {"tts": 0.25, "transcribe": 0.1}


class RecordingCloner:
    def __init__(self):
        self.clone_kwargs: dict | None = None

    def clone_voice(self, **kwargs):
        self.clone_kwargs = kwargs
        return SimpleNamespace(
            audio=np.zeros(2400, dtype=np.float32),
            sample_rate=24000,
            duration_seconds=0.1,
        )


def test_health_reports_version_and_capabilities():
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model_loaded" in body
    assert body["version"]["package"] == importlib.metadata.version("local-voice-cloning")
    assert isinstance(body["version"]["git"], str)
    assert body["version"]["git"]
    capabilities = body["capabilities"]
    assert capabilities["speed_control"] == (
        "post-stretch (librosa) for qwen/chatterbox, native for omnivoice"
    )
    assert isinstance(capabilities["transcribe"], bool)
    assert capabilities["sidecar_transcripts"] is True


def test_git_version_prefers_version_file(tmp_path, monkeypatch):
    stamp = tmp_path / "GIT_VERSION"
    stamp.write_text("abc1234\n", encoding="utf-8")
    monkeypatch.setattr(api_module, "GIT_VERSION_FILE", stamp)
    assert api_module._git_version() == "abc1234"


def test_git_version_falls_back_to_git_then_unknown(tmp_path, monkeypatch):
    monkeypatch.setattr(api_module, "GIT_VERSION_FILE", tmp_path / "missing")
    monkeypatch.setattr(
        api_module.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="deadbee\n"),
    )
    assert api_module._git_version() == "deadbee"

    def no_git(*args, **kwargs):
        raise OSError("git not available")

    monkeypatch.setattr(api_module.subprocess, "run", no_git)
    assert api_module._git_version() == "unknown"


def test_warmup_loads_models_and_reports_timing(mocker):
    stub = StubCloner()
    mocker.patch("src.api.get_shared_cloner", return_value=stub)

    response = client.post("/warmup", data={"engine": "qwen", "quality": "high"})

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["engine"] == "qwen"
    assert body["quality"] == "high"
    assert body["model_id"] == stub.model_id
    assert body["model_loaded"] is True
    assert body["stages"] == {"tts": 0.25, "transcribe": 0.1}
    assert body["load_seconds"] == pytest.approx(0.35)
    assert stub.warmup_calls == [True]


def test_warmup_skips_transcriber_for_chatterbox(mocker):
    stub = StubCloner()
    mocker.patch("src.api.get_shared_cloner", return_value=stub)

    response = client.post("/warmup", data={"engine": "chatterbox", "quality": "fast"})

    assert response.status_code == 200
    assert stub.warmup_calls == [False]


def test_warmup_rejects_unknown_quality_and_engine():
    response = client.post("/warmup", data={"engine": "qwen", "quality": "ultra"})
    assert response.status_code == 422
    assert "Unknown quality" in response.json()["detail"]

    response = client.post("/warmup", data={"engine": "f5", "quality": "high"})
    assert response.status_code == 422
    assert "Unknown engine" in response.json()["detail"]


def test_warmup_failures_include_traceback(mocker):
    stub = StubCloner(
        error=RuntimeError(
            "model download failed for mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16; "
            "check network/HF access and rerun warmup"
        )
    )
    mocker.patch("src.api.get_shared_cloner", return_value=stub)

    response = client.post("/warmup", data={"engine": "qwen", "quality": "high"})

    assert response.status_code == 500
    body = response.json()
    assert body["detail"].startswith("Warmup failed:")
    assert "model download failed" in body["detail"]
    assert "Traceback (most recent call last)" in body["traceback"]


def test_synthesize_failures_include_traceback(reference_wav, mocker):
    class BrokenCloner:
        def clone_voice(self, **kwargs):
            raise OSError("[Errno 5] Input/output error")

    mocker.patch("src.api.get_shared_cloner", return_value=BrokenCloner())

    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
            data={"text": "Hello", "ref_text": "A steady synthetic tone."},
        )

    assert response.status_code == 500
    body = response.json()
    assert body["detail"] == "Synthesis failed: [Errno 5] Input/output error"
    assert "Traceback (most recent call last)" in body["traceback"]
    assert "OSError" in body["traceback"]


@pytest.mark.parametrize(
    "error",
    [
        OSError("[Errno 5] Input/output error"),
        RuntimeError("The reference recording could not be transcribed."),
        ImportError("No module named 'mlx_audio'"),
    ],
)
def test_transcribe_failures_return_422_with_guidance(reference_wav, mocker, error):
    class BrokenCloner:
        def transcribe(self, _path):
            raise error

    mocker.patch("src.api.get_shared_cloner", return_value=BrokenCloner())

    with open(reference_wav, "rb") as f:
        response = client.post(
            "/transcribe",
            files={"reference_audio": ("ref.wav", f, "audio/wav")},
        )

    assert response.status_code == 422
    body = response.json()
    assert body["detail"].startswith("Transcription failed:")
    assert "ref_text" in body["detail"]
    assert ".json sidecar" in body["detail"]
    assert "Traceback (most recent call last)" in body["traceback"]


def test_synthesize_uses_saved_voice_sidecar_transcript(reference_wav, tmp_path, mocker):
    voices = tmp_path / "voices"
    voices.mkdir()
    (voices / "karan.json").write_text(
        json.dumps({"transcript": "Words from the sidecar."}), encoding="utf-8"
    )
    mocker.patch.object(api_module, "SAVED_VOICES_DIR", voices)
    recorder = RecordingCloner()
    mocker.patch("src.api.get_shared_cloner", return_value=recorder)

    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("karan.wav", f, "audio/wav")},
            data={"text": "Hello", "output_format": "wav"},
        )

    assert response.status_code == 200
    assert recorder.clone_kwargs["reference_text"] == "Words from the sidecar."

    with open(reference_wav, "rb") as f:
        response = client.post(
            "/synthesize",
            files={"reference_audio": ("karan.wav", f, "audio/wav")},
            data={"text": "Hello", "ref_text": "Explicit words."},
        )

    assert response.status_code == 200
    assert recorder.clone_kwargs["reference_text"] == "Explicit words."

