import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from src.audio_utils import enhance_audio, load_audio

MAX_REFERENCE_SECONDS = 12.0
LINE_BREAK_PAUSE_SECONDS = 0.4
ENGINE_NAME = "Qwen3-TTS 1.7B"
ASR_MODEL_ID = "mlx-community/whisper-large-v3-turbo-asr-fp16"
MODEL_VARIANTS = {
    "high": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
    "fast": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit",
}
SUPPORTED_LANGUAGES = (
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
)
ProgressCallback = Callable[[str], None]


def _load_tts_model(model_id: str):
    from mlx_audio.tts.utils import load_model

    return load_model(model_id)


def _load_stt_model(model_id: str):
    from mlx_audio.stt.utils import load_model

    return load_model(model_id)


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    matched_voice: str


_SENTENCE_END = (".", "!", "?", ",", ";", ":", "…", "。", "！", "？")


def prepare_gen_text(text: str) -> str:
    text = text.strip()
    if text and not text.endswith(_SENTENCE_END):
        text += "."
    return text


def _script_segments(text: str) -> list[tuple[str, int]]:
    segments: list[tuple[str, int]] = []
    pending_newlines = 0
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for index, line in enumerate(normalized.split("\n")):
        if index:
            pending_newlines += 1
        prepared = prepare_gen_text(line)
        if not prepared:
            continue
        segments.append((prepared, pending_newlines if segments else 0))
        pending_newlines = 0
    return segments


def model_id_for_quality(quality: str) -> str:
    try:
        return MODEL_VARIANTS[quality]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_VARIANTS))
        raise ValueError(f"Unknown quality '{quality}'. Choose one of: {choices}.") from exc


def detect_device() -> str:
    """MLX uses Apple Silicon's unified-memory GPU backend."""
    return "mlx"


class LocalVoiceCloner:
    def __init__(
        self,
        quality: str = "high",
        sample_rate: int = 24000,
        tts_loader: Callable[[str], Any] | None = None,
        stt_loader: Callable[[str], Any] | None = None,
    ) -> None:
        self.quality = quality
        self.model_id = model_id_for_quality(quality)
        self.sample_rate = sample_rate
        self.device = detect_device()
        self.engine_name = ENGINE_NAME
        self._tts_loader = tts_loader or _load_tts_model
        self._stt_loader = stt_loader or _load_stt_model
        self._tts_model: Any | None = None
        self._stt_model: Any | None = None
        self._model_lock = threading.Lock()

    @property
    def model_loaded(self) -> bool:
        return self._tts_model is not None

    def _ensure_tts_model(self):
        if self._tts_model is None:
            with self._model_lock:
                if self._tts_model is None:
                    self._tts_model = self._tts_loader(self.model_id)
                    self.sample_rate = int(
                        getattr(self._tts_model, "sample_rate", self.sample_rate)
                    )
        return self._tts_model

    def _ensure_stt_model(self):
        if self._stt_model is None:
            with self._model_lock:
                if self._stt_model is None:
                    self._stt_model = self._stt_loader(ASR_MODEL_ID)
        return self._stt_model

    def clone_voice(
        self,
        reference_audio_path: str | Path,
        text: str,
        reference_text: str = "",
        speed: float = 1.0,
        language: str = "auto",
        progress_callback: ProgressCallback | None = None,
        **_legacy_options,
    ) -> SynthesisResult:
        if not text.strip():
            raise ValueError("Input text cannot be empty.")

        notify = progress_callback or (lambda _stage: None)
        notify("prepare")
        tensor_audio, ref_sr = load_audio(
            reference_audio_path,
            target_sr=self.sample_rate,
            max_duration_seconds=MAX_REFERENCE_SECONDS,
        )
        audio_np = tensor_audio.squeeze().numpy()
        if len(audio_np) == 0:
            raise ValueError("Reference audio is empty.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            canonical_ref_path = Path(tmp.name)

        try:
            sf.write(str(canonical_ref_path), audio_np, ref_sr, subtype="PCM_16")
            notify("load")
            tts_model = self._ensure_tts_model()

            transcript = reference_text.strip()
            if not transcript:
                stt_result = self._ensure_stt_model().generate(
                    str(canonical_ref_path),
                    verbose=False,
                )
                transcript = str(getattr(stt_result, "text", "")).strip()
                if not transcript:
                    raise RuntimeError("The reference recording could not be transcribed.")

            notify("voice")
            sample_rate = self.sample_rate
            pieces: list[np.ndarray] = []
            for segment, newline_count in _script_segments(text):
                generations = list(
                    tts_model.generate(
                        text=segment,
                        ref_audio=str(canonical_ref_path),
                        ref_text=transcript,
                        speed=speed,
                        lang_code=language,
                        stream=False,
                        verbose=False,
                    )
                )
                if not generations:
                    raise RuntimeError("The voice model returned no audio.")

                sample_rate = int(getattr(generations[0], "sample_rate", sample_rate))
                if pieces and newline_count:
                    pieces.append(
                        np.zeros(
                            round(sample_rate * LINE_BREAK_PAUSE_SECONDS * newline_count),
                            dtype=np.float32,
                        )
                    )
                for index, item in enumerate(generations):
                    if index:
                        pieces.append(np.zeros(round(sample_rate * 0.08), dtype=np.float32))
                    pieces.append(np.asarray(item.audio).squeeze().astype(np.float32))
            generated = np.concatenate(pieces)

            notify("finish")
            enhanced = enhance_audio(generated, sample_rate, target_level_db=-16.0)
        finally:
            canonical_ref_path.unlink(missing_ok=True)

        quality_label = "high fidelity" if self.quality == "high" else "fast"
        return SynthesisResult(
            audio=enhanced,
            sample_rate=sample_rate,
            duration_seconds=float(len(enhanced) / sample_rate),
            matched_voice=f"{ENGINE_NAME} · {quality_label}",
        )


_shared_cloners: dict[str, LocalVoiceCloner] = {}
_shared_cloner_lock = threading.Lock()


def get_shared_cloner(quality: str = "high") -> LocalVoiceCloner:
    if quality not in _shared_cloners:
        with _shared_cloner_lock:
            if quality not in _shared_cloners:
                _shared_cloners[quality] = LocalVoiceCloner(quality=quality)
    return _shared_cloners[quality]


def is_shared_cloner_loaded(quality: str | None = None) -> bool:
    if quality is not None:
        cloner = _shared_cloners.get(quality)
        return bool(cloner and cloner.model_loaded)
    return any(cloner.model_loaded for cloner in _shared_cloners.values())
