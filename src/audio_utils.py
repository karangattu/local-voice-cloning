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


def normalize_audio(audio: np.ndarray, target_level_db: float = -20.0) -> np.ndarray:
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    target_rms = 10.0 ** (target_level_db / 20.0)
    if rms > 0:
        audio = audio * (target_rms / rms)
    max_val = np.max(np.abs(audio))
    if max_val > 1.0:
        audio = audio / max_val
    return audio.astype(np.float32)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    data_np = audio.squeeze().detach().cpu().numpy()
    num_target_samples = round(len(data_np) * float(target_sr) / orig_sr)
    resampled = signal.resample(data_np, num_target_samples).astype(np.float32)
    return torch.from_numpy(resampled).unsqueeze(0)


def save_audio(file_path: str | Path, audio: torch.Tensor | np.ndarray, sample_rate: int = 24000) -> Path:
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().detach().cpu().numpy()
    else:
        audio_np = audio.squeeze()

    audio_np = np.clip(audio_np, -1.0, 1.0)
    sf.write(str(target_path), audio_np, sample_rate, subtype="PCM_16")
    return target_path
