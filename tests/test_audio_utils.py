import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from src.audio_utils import (
    analyze_reference_audio,
    apply_fades,
    enhance_audio,
    high_pass_filter,
    load_audio,
    normalize_audio,
    resample_audio,
    save_audio,
    trim_silence,
)


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


def _make_tone(sr: int = 24000, seconds: float = 1.0, freq: float = 440.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_save_audio_mp3():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.mp3"
        sr = 24000
        saved_path = save_audio(file_path, _make_tone(sr), sample_rate=sr)
        assert saved_path.exists()
        assert saved_path.stat().st_size > 0

        loaded_tensor, loaded_sr = load_audio(saved_path, target_sr=sr)
        assert loaded_sr == sr
        assert loaded_tensor.shape[1] > 0


def test_save_audio_format_override():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test.bin"
        saved_path = save_audio(file_path, _make_tone(), sample_rate=24000, output_format="mp3")
        assert saved_path.exists()


def test_save_audio_unsupported_format():
    with tempfile.TemporaryDirectory() as tmpdir, pytest.raises(ValueError, match="Unsupported output format"):
        save_audio(Path(tmpdir) / "test.xyz", _make_tone(), sample_rate=24000)


def test_normalize_audio_respects_peak_ceiling():
    audio = np.array([0.9, -0.95, 1.0, -1.0], dtype=np.float32)
    normalized = normalize_audio(audio, target_level_db=-6.0, peak_ceiling_db=-1.0)
    assert np.max(np.abs(normalized)) <= 10.0 ** (-1.0 / 20.0) + 1e-6


def test_high_pass_filter_removes_dc_offset():
    sr = 24000
    audio = _make_tone(sr) + 0.3
    filtered = high_pass_filter(audio, sr)
    assert abs(np.mean(filtered)) < 0.01


def test_trim_silence_removes_edges():
    sr = 24000
    silence = np.zeros(sr, dtype=np.float32)
    tone = _make_tone(sr)
    padded = np.concatenate([silence, tone, silence])
    trimmed = trim_silence(padded, sr)
    assert len(trimmed) < len(padded)
    assert len(trimmed) >= len(tone)


def test_trim_silence_all_quiet_returns_input():
    sr = 24000
    silence = np.zeros(sr, dtype=np.float32)
    assert len(trim_silence(silence, sr)) == len(silence)


def test_apply_fades_zeroes_endpoints():
    sr = 24000
    audio = np.ones(sr, dtype=np.float32)
    faded = apply_fades(audio, sr)
    assert faded[0] == 0.0
    assert faded[-1] == 0.0
    assert faded[sr // 2] == 1.0


def test_enhance_audio_pipeline():
    sr = 24000
    audio = np.concatenate([np.zeros(sr // 2, dtype=np.float32), _make_tone(sr), np.zeros(sr // 2, dtype=np.float32)])
    enhanced = enhance_audio(audio, sr)
    assert enhanced.dtype == np.float32
    assert len(enhanced) > 0
    assert np.max(np.abs(enhanced)) <= 1.0


def test_enhance_audio_empty():
    enhanced = enhance_audio(np.array([], dtype=np.float32), 24000)
    assert len(enhanced) == 0


def test_analyze_reference_audio_good_sample(tmp_path):
    sr = 24000
    path = tmp_path / "good.wav"
    tone = 0.3 * np.sin(2 * np.pi * 200 * np.linspace(0, 6.0, sr * 6, endpoint=False))
    sf.write(str(path), tone.astype(np.float32), sr)

    report = analyze_reference_audio(path)
    assert report["duration_seconds"] == pytest.approx(6.0)
    assert report["sample_rate"] == sr
    assert report["warnings"] == []


def test_analyze_reference_audio_short_sample(tmp_path):
    sr = 24000
    path = tmp_path / "short.wav"
    sf.write(str(path), _make_tone(sr, seconds=1.0), sr)

    report = analyze_reference_audio(path)
    assert any("shorter than 3 seconds" in w for w in report["warnings"])


def test_analyze_reference_audio_clipped_sample(tmp_path):
    sr = 24000
    path = tmp_path / "clipped.wav"
    tone = np.clip(3.0 * _make_tone(sr, seconds=6.0), -1.0, 1.0)
    sf.write(str(path), tone, sr)

    report = analyze_reference_audio(path)
    assert report["clipping_ratio"] > 0.001
    assert any("clipped" in w for w in report["warnings"])


def test_analyze_reference_audio_silent_sample(tmp_path):
    sr = 24000
    path = tmp_path / "silent.wav"
    tone = np.concatenate([_make_tone(sr, seconds=2.0), np.zeros(sr * 4, dtype=np.float32)])
    sf.write(str(path), tone, sr)

    report = analyze_reference_audio(path)
    assert report["silence_ratio"] > 0.4
    assert any("long silences" in w for w in report["warnings"])


def test_analyze_reference_audio_missing_file():
    with pytest.raises(FileNotFoundError):
        analyze_reference_audio("does_not_exist.wav")
