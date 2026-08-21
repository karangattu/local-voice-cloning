"""Fast tests for src/cloner.py. F5TTS is mocked so these never download or
run the real model. See tests/integration/test_cloner_integration.py for the
real-model equivalents."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from src.cloner import (
    SynthesisResult,
    detect_device,
    prepare_gen_text,
    recommended_nfe_step,
)


def _mock_f5tts(sample_rate=24000, duration_seconds=1.0):
    instance = MagicMock()
    instance.infer.return_value = (
        np.zeros(int(sample_rate * duration_seconds), dtype=np.float32),
        sample_rate,
        None,
    )
    return instance


@pytest.fixture
def sample_voice_file(tmp_path):
    sr = 24000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    path = tmp_path / "ref.wav"
    sf.write(str(path), (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32), sr)
    return path


def test_detect_device_returns_a_known_backend():
    assert detect_device() in ("cuda", "mps", "cpu")


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


@patch("src.cloner.F5TTS")
def test_cloner_initialization_does_not_need_a_real_model(mock_f5tts_cls):
    from src.cloner import LocalVoiceCloner

    mock_f5tts_cls.return_value = _mock_f5tts()
    cloner = LocalVoiceCloner(device="cpu")
    assert cloner.device == "cpu"
    assert cloner.sample_rate == 24000
    assert cloner.default_nfe_step == recommended_nfe_step("cpu")
    mock_f5tts_cls.assert_called_once_with(device="cpu")


@patch("src.cloner.F5TTS")
def test_clone_voice_passes_prepared_text_and_settings(mock_f5tts_cls, sample_voice_file):
    from src.cloner import LocalVoiceCloner

    mock_instance = _mock_f5tts()
    mock_f5tts_cls.return_value = mock_instance

    cloner = LocalVoiceCloner(device="cpu")
    result = cloner.clone_voice(sample_voice_file, text="Hello there", nfe_step=8, cfg_strength=1.5)

    assert isinstance(result, SynthesisResult)
    call_kwargs = mock_instance.infer.call_args.kwargs
    assert call_kwargs["gen_text"] == "Hello there."
    assert call_kwargs["nfe_step"] == 8
    assert call_kwargs["cfg_strength"] == 1.5


@patch("src.cloner.F5TTS")
def test_clone_voice_empty_text_raises_before_calling_the_model(mock_f5tts_cls, sample_voice_file):
    from src.cloner import LocalVoiceCloner

    mock_instance = _mock_f5tts()
    mock_f5tts_cls.return_value = mock_instance

    cloner = LocalVoiceCloner(device="cpu")
    with pytest.raises(ValueError, match="Input text cannot be empty"):
        cloner.clone_voice(sample_voice_file, text="")
    mock_instance.infer.assert_not_called()
