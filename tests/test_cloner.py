import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from src.cloner import LocalVoiceCloner, SynthesisResult, prepare_gen_text, recommended_nfe_step


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
    assert cloner.f5 is not None
    assert cloner.sample_rate == 24000
    assert cloner.default_nfe_step == recommended_nfe_step(cloner.device)


def test_recommended_nfe_step():
    assert recommended_nfe_step("mps") == 64
    assert recommended_nfe_step("cuda") == 64
    assert recommended_nfe_step("cpu") == 32


def test_prepare_gen_text_adds_final_stop():
    assert prepare_gen_text("Hello world") == "Hello world."
    assert prepare_gen_text("  Hello world  ") == "Hello world."


def test_prepare_gen_text_keeps_existing_punctuation():
    assert prepare_gen_text("Hello world!") == "Hello world!"
    assert prepare_gen_text("Is that so?") == "Is that so?"
    assert prepare_gen_text("") == ""


def test_clone_voice_synthesis(sample_voice_file):
    cloner = LocalVoiceCloner()
    result = cloner.clone_voice(
        reference_audio_path=sample_voice_file,
        text="Testing zero-shot cloned voice output.",
        speed=1.0,
        nfe_step=8,
    )
    assert isinstance(result, SynthesisResult)
    assert result.sample_rate == 24000
    assert len(result.audio) > 0
    assert result.duration_seconds > 0.3


def test_clone_voice_empty_text(sample_voice_file):
    cloner = LocalVoiceCloner()
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        cloner.clone_voice(sample_voice_file, text="")
