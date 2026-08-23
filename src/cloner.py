import contextlib
import os
import random
import tempfile
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path

import f5_tts.infer.utils_infer as f5_utils
import numpy as np
import soundfile as sf
import torch
import torchaudio
from f5_tts.api import F5TTS
from transformers.utils import logging as tf_logging

from src.audio_utils import enhance_audio, load_audio

os.environ["PYTHONHASHSEED"] = "0"
tf_logging.disable_progress_bar()

try:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if getattr(f5_utils, "asr_pipe", None) is None:
            f5_utils.initialize_asr_pipeline(device=f5_utils.device)
except (RuntimeError, ValueError, OSError):
    pass

# F5-TTS internally clips the reference audio to its first 12 seconds
# (see f5_tts.infer.utils_infer.preprocess_ref_audio_text). We enforce the
# same limit ourselves so the truncation point is deterministic and matches
# what the UI and docs tell the user, instead of relying on F5-TTS's own
# silence-search heuristic to land in the same place.
MAX_REFERENCE_SECONDS = 12.0


def _shim_torchaudio_load(filepath, frame_offset=0, num_frames=-1, normalize=True, channels_first=True, **kwargs):
    if not normalize or not channels_first:
        raise NotImplementedError(
            "The local torchaudio.load shim only supports normalize=True and "
            "channels_first=True (F5-TTS does not request other combinations)."
        )
    data, sr = sf.read(str(filepath), dtype="float32")
    if frame_offset or num_frames != -1:
        end = None if num_frames == -1 else frame_offset + num_frames
        data = data[frame_offset:end]
    tensor = torch.from_numpy(data).float()
    tensor = tensor.unsqueeze(0) if tensor.ndim == 1 else tensor.transpose(0, 1)
    return tensor, sr


@contextlib.contextmanager
def _torchaudio_soundfile_shim():
    """Temporarily replace torchaudio.load with a soundfile-based equivalent,
    scoped to the single F5-TTS inference call that needs it, then restore
    the original attribute."""
    original_load = torchaudio.load
    torchaudio.load = _shim_torchaudio_load
    try:
        yield
    finally:
        torchaudio.load = original_load


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    matched_voice: str


_SENTENCE_END = (".", "!", "?", ",", ";", ":", "…", "。", "！", "？")


def prepare_gen_text(text: str) -> str:
    """F5-TTS derives pauses and intonation from punctuation; text without a
    final stop renders flat, so append one."""
    text = text.strip()
    if text and not text.endswith(_SENTENCE_END):
        text += "."
    return text


def detect_device() -> str:
    """Pick the best available compute device without loading any model
    weights, so callers can show device and quality information before
    paying the cost of a real model load."""
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def recommended_nfe_step(device: str) -> int:
    """Highest diffusion step count the hardware can run at acceptable speed:
    GPU-accelerated devices get the full 64 steps, CPU falls back to 32."""
    return 64 if device in ("cuda", "mps") else 32


class LocalVoiceCloner:
    def __init__(self, device: str | None = None, sample_rate: int = 24000) -> None:
        self.sample_rate = sample_rate
        self.device = device or detect_device()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
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

        tensor_audio, ref_sr = load_audio(
            reference_audio_path,
            target_sr=self.sample_rate,
            max_duration_seconds=MAX_REFERENCE_SECONDS,
        )
        audio_np = tensor_audio.squeeze().numpy()
        if len(audio_np) == 0:
            raise ValueError("Reference audio is empty.")

        safe_seed = random.randint(0, 2147483647)

        # Hand F5-TTS a canonical, already-resampled-and-trimmed reference
        # file instead of the raw upload, so our own preprocessing is the
        # one source of truth for what the model actually conditions on.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            canonical_ref_path = Path(tmp.name)
        try:
            sf.write(str(canonical_ref_path), audio_np, ref_sr, subtype="PCM_16")
            with _torchaudio_soundfile_shim(), warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                wav, sr_out, _ = self.f5.infer(
                    ref_file=str(canonical_ref_path),
                    ref_text=reference_text.strip(),
                    gen_text=prepare_gen_text(text),
                    nfe_step=nfe_step,
                    cfg_strength=cfg_strength,
                    sway_sampling_coef=sway_sampling_coef,
                    speed=speed,
                    remove_silence=False,
                    seed=safe_seed,
                )
        finally:
            canonical_ref_path.unlink(missing_ok=True)

        enhanced = enhance_audio(np.asarray(wav), sr_out, target_level_db=-16.0)

        return SynthesisResult(
            audio=enhanced,
            sample_rate=sr_out,
            duration_seconds=float(len(enhanced) / sr_out),
            matched_voice="F5-TTS Direct Acoustic Conditioning",
        )


_shared_cloner: LocalVoiceCloner | None = None
_shared_cloner_lock = threading.Lock()


def get_shared_cloner() -> LocalVoiceCloner:
    """Return a process-wide LocalVoiceCloner, constructing it on first use.
    The model is multiple gigabytes and slow to load, so callers (the web
    app, the API) share one instance instead of loading it per request."""
    global _shared_cloner
    if _shared_cloner is None:
        with _shared_cloner_lock:
            if _shared_cloner is None:
                _shared_cloner = LocalVoiceCloner()
    return _shared_cloner


def is_shared_cloner_loaded() -> bool:
    """Report whether get_shared_cloner() has already constructed the model,
    without triggering a load. Useful for status displays."""
    return _shared_cloner is not None
