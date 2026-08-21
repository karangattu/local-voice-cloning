from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal


def load_audio(
    file_path: str | Path,
    target_sr: int = 24000,
    max_duration_seconds: float = 30.0,
) -> tuple[torch.Tensor, int]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        num_target_samples = round(len(data) * float(target_sr) / sr)
        data = signal.resample(data, num_target_samples).astype(np.float32)
        sr = target_sr

    max_samples = int(target_sr * max_duration_seconds)
    if len(data) > max_samples:
        data = data[:max_samples]

    data = normalize_audio(data)
    tensor_audio = torch.from_numpy(data).unsqueeze(0)
    return tensor_audio, sr


def normalize_audio(
    audio: np.ndarray,
    target_level_db: float = -20.0,
    peak_ceiling_db: float = -1.0,
) -> np.ndarray:
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    target_rms = 10.0 ** (target_level_db / 20.0)
    if rms > 0:
        audio = audio * (target_rms / rms)
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    max_val = np.max(np.abs(audio))
    if max_val > ceiling:
        audio = audio * (ceiling / max_val)
    return audio.astype(np.float32)


def high_pass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 50.0) -> np.ndarray:
    if len(audio) < 16:
        return audio
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    padding_seconds: float = 0.08,
) -> np.ndarray:
    if len(audio) == 0:
        return audio
    frame = max(1, int(sample_rate * 0.02))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    active = np.flatnonzero(frame_rms_db > threshold_db)
    if len(active) == 0:
        return audio
    pad = int(sample_rate * padding_seconds)
    start = max(0, active[0] * frame - pad)
    end = min(len(audio), (active[-1] + 1) * frame + pad)
    return audio[start:end]


def apply_fades(audio: np.ndarray, sample_rate: int, fade_seconds: float = 0.015) -> np.ndarray:
    n_fade = min(int(sample_rate * fade_seconds), len(audio) // 2)
    if n_fade <= 0:
        return audio
    audio = audio.copy()
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    audio[:n_fade] *= ramp
    audio[-n_fade:] *= ramp[::-1]
    return audio


def enhance_audio(audio: np.ndarray, sample_rate: int, target_level_db: float = -16.0) -> np.ndarray:
    """Post-processing chain for synthesized speech: rumble removal, edge
    silence trimming, click-free fades, and peak-safe loudness normalization."""
    if len(audio) == 0:
        return audio.astype(np.float32)
    audio = high_pass_filter(audio, sample_rate)
    audio = trim_silence(audio, sample_rate)
    audio = apply_fades(audio, sample_rate)
    return normalize_audio(audio, target_level_db=target_level_db)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    data_np = audio.squeeze().detach().cpu().numpy()
    num_target_samples = round(len(data_np) * float(target_sr) / orig_sr)
    resampled = signal.resample(data_np, num_target_samples).astype(np.float32)
    return torch.from_numpy(resampled).unsqueeze(0)


SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3"}


def save_audio(
    file_path: str | Path,
    audio: torch.Tensor | np.ndarray,
    sample_rate: int = 24000,
    output_format: str | None = None,
) -> Path:
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format is None:
        output_format = target_path.suffix.lstrip(".").lower() or "wav"
    output_format = output_format.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
        )

    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().detach().cpu().numpy()
    else:
        audio_np = audio.squeeze()

    audio_np = np.clip(audio_np, -1.0, 1.0)
    if output_format == "mp3":
        sf.write(str(target_path), audio_np, sample_rate, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(str(target_path), audio_np, sample_rate, format="WAV", subtype="PCM_24")
    return target_path
