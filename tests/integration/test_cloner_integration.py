"""Real-model tests for src/cloner.py. These download and run the actual
Qwen3-TTS MLX weights and are slow. Run with: pytest -m integration
See tests/unit/test_cloner_unit.py for the fast, mocked equivalents."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.cloner import LocalVoiceCloner, SynthesisResult

pytestmark = pytest.mark.integration


@pytest.fixture
def sample_voice_file():
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sr = 24000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        f0 = 180.0
        audio = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.25 * np.sin(2 * np.pi * 2 * f0 * t)
        sf.write(tmp.name, audio.astype(np.float32), sr)
        tmp_path = Path(tmp.name)
    yield tmp_path
    if tmp_path.exists():
        tmp_path.unlink()


def test_cloner_initialization():
    cloner = LocalVoiceCloner()
    assert cloner.model_loaded is False
    assert cloner.sample_rate == 24000
    assert cloner.device == "mlx"


def test_clone_voice_synthesis(sample_voice_file):
    cloner = LocalVoiceCloner()
    result = cloner.clone_voice(
        reference_audio_path=sample_voice_file,
        text="Testing zero-shot cloned voice output.",
        reference_text="A steady synthetic tone.",
        speed=1.0,
    )
    assert isinstance(result, SynthesisResult)
    assert result.sample_rate == 24000
    assert len(result.audio) > 0
    assert result.duration_seconds > 0.3
