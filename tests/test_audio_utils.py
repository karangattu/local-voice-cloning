import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from src.audio_utils import load_audio, normalize_audio, resample_audio, save_audio


def test_normalize_audio():
    audio = np.array([0.1, -0.2, 0.5, -0.8], dtype=np.float32)
    normalized = normalize_audio(audio)
    assert isinstance(normalized, np.ndarray)
    assert np.max(np.abs(normalized)) <= 1.0


def test_normalize_empty_audio():
    empty = np.array([], dtype=np.float32)
    res = normalize_audio(empty)
    assert len(res) == 0


def test_resample_audio():
    orig_sr = 16000
    target_sr = 24000
    audio_tensor = torch.randn(1, 16000)
    resampled = resample_audio(audio_tensor, orig_sr, target_sr)
    assert resampled.shape[1] == 24000
    assert resampled.shape[0] == 1


def test_save_and_load_audio():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.wav"
        sr = 24000
        t = np.linspace(0, 1.0, sr, endpoint=False)
        tone = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        saved_path = save_audio(file_path, tone, sample_rate=sr)
        assert saved_path.exists()

        loaded_tensor, loaded_sr = load_audio(saved_path, target_sr=sr, max_duration_seconds=30.0)
        assert loaded_sr == sr
        assert loaded_tensor.shape[0] == 1
        assert loaded_tensor.shape[1] == sr


def test_load_audio_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_audio("non_existent_audio_file.wav")
