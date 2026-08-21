import os
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

os.environ["PYTHONHASHSEED"] = "0"
warnings.filterwarnings("ignore")

def _safe_torchaudio_load(filepath, *args, **kwargs):
    data, sr = sf.read(str(filepath), dtype="float32")
    tensor = torch.from_numpy(data).float()
    if tensor.ndim == 1:
        tensor = tensor.unsqueeze(0)
    else:
        tensor = tensor.transpose(0, 1)
    return tensor, sr

torchaudio.load = _safe_torchaudio_load

from f5_tts.api import F5TTS

from src.audio_utils import enhance_audio, load_audio


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    matched_voice: str


def recommended_nfe_step(device: str) -> int:
    """Highest diffusion step count the hardware can run at acceptable speed:
    GPU-accelerated devices get the full 64 steps, CPU falls back to 32."""
    return 64 if device in ("cuda", "mps") else 32


class LocalVoiceCloner:
    def __init__(self, device: str | None = None, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate

        if device is None:
            if torch.backends.mps.is_available():
                self.device = "mps"
            elif torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        try:
            self.f5 = F5TTS(device=self.device)
        except (RuntimeError, ValueError, OSError):
            self.device = "cpu"
            self.f5 = F5TTS(device="cpu")

        self.default_nfe_step = recommended_nfe_step(self.device)

    def clone_voice(
        self,
        reference_audio_path: str | Path,
        text: str,
        reference_text: str = "",
        speed: float = 1.0,
        nfe_step: int | None = None,
        cfg_strength: float = 2.0,
        sway_sampling_coef: float = -1.0,
    ) -> SynthesisResult:
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        if nfe_step is None:
            nfe_step = self.default_nfe_step

        tensor_audio, _ = load_audio(
            reference_audio_path,
            target_sr=self.sample_rate,
            max_duration_seconds=30.0,
        )
        audio_np = tensor_audio.squeeze().numpy()
        if len(audio_np) == 0:
            raise ValueError("Reference audio is empty.")

        safe_seed = random.randint(0, 2147483647)

        wav, sr_out, _ = self.f5.infer(
            ref_file=str(reference_audio_path),
            ref_text=reference_text.strip(),
            gen_text=text.strip(),
            nfe_step=nfe_step,
            cfg_strength=cfg_strength,
            sway_sampling_coef=sway_sampling_coef,
            speed=speed,
            remove_silence=False,
            seed=safe_seed,
        )

        enhanced = enhance_audio(np.asarray(wav), sr_out, target_level_db=-16.0)

        return SynthesisResult(
            audio=enhanced,
            sample_rate=sr_out,
            duration_seconds=float(len(enhanced) / sr_out),
            matched_voice="F5-TTS Direct Acoustic Conditioning",
        )
