import json
import os
import re
import tempfile
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import soundfile as sf

from src.audio_utils import (
    enhance_audio,
    join_with_room_tone,
    load_audio,
    normalize_audio,
    prepare_reference_audio,
    trim_silence,
)

MAX_REFERENCE_SECONDS = 12.0
MAX_SOURCE_SECONDS = 120.0
LINE_BREAK_PAUSE_SECONDS = 0.4
ENGINE_NAME = "Qwen3-TTS 1.7B"
ENGINES = {"qwen": ENGINE_NAME, "omnivoice": "OmniVoice", "chatterbox": "Chatterbox"}
OMNIVOICE_MODEL_ID = "k2-fsa/OmniVoice"
CHATTERBOX_MODEL_IDS = {
    "high": "ResembleAI/chatterbox",
    "fast": "ResembleAI/chatterbox-turbo",
}
CHATTERBOX_LANGUAGES = ("auto", "English")
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


def sidecar_transcript(reference_audio_path: str | Path) -> str:
    """Transcript stored in a JSON sidecar next to a reference clip, if any.

    The sidecar shares the clip's stem and adds .json, e.g. voice_samples/karan.wav
    -> voice_samples/karan.json containing {"transcript": "...", "reference_id": "..."}.
    """
    sidecar_path = Path(reference_audio_path).with_suffix(".json")
    if not sidecar_path.is_file():
        return ""
    try:
        metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    transcript = metadata.get("transcript", "") if isinstance(metadata, dict) else ""
    if not isinstance(transcript, str):
        return ""
    return transcript.strip()


def _load_tts_model(model_id: str):
    from mlx_audio.tts.utils import load_model

    return load_model(model_id)


def _load_omnivoice_model(model_id: str):
    try:
        from omnivoice import OmniVoice
    except ImportError as exc:
        raise RuntimeError("OmniVoice is not installed. Run: uv sync --extra omnivoice") from exc
    import torch

    device = omnivoice_device()
    if device == "mps":
        # Parallel weight conversion can crash PyTorch Metal kernels (Transformers #48029).
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    return OmniVoice.from_pretrained(
        model_id,
        device_map=device,
        dtype=torch.float32 if device == "cpu" else torch.float16,
    )


def omnivoice_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def chatterbox_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda:0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_chatterbox_model(model_id: str):
    try:
        device = chatterbox_device()
        if model_id == CHATTERBOX_MODEL_IDS["fast"]:
            from chatterbox.tts_turbo import ChatterboxTurboTTS

            return ChatterboxTurboTTS.from_pretrained(device=device)
        from chatterbox.tts import ChatterboxTTS

        return ChatterboxTTS.from_pretrained(device=device)
    except ImportError as exc:
        raise RuntimeError(
            "Chatterbox is not installed. Install it with: pip install chatterbox-tts "
            "(note: its pinned dependencies conflict with mlx-audio, "
            "so a separate environment may be needed)"
        ) from exc


def validate_engine(engine: str) -> None:
    if engine not in ENGINES:
        raise ValueError(f"Unknown engine '{engine}'. Choose qwen or omnivoice.")


def _load_stt_model(model_id: str):
    from mlx_audio.stt.utils import load_model

    model = load_model(model_id)
    # Transformers skips this cleanup for Whisper's BPE tokenizer and logs a
    # warning on each load. Turn it off to get the same text without the warning.
    tokenizer = getattr(getattr(model, "_processor", None), "tokenizer", None)
    if tokenizer is not None:
        tokenizer.clean_up_tokenization_spaces = False
    return model


@dataclass
class SynthesisResult:
    audio: np.ndarray
    sample_rate: int
    duration_seconds: float
    matched_voice: str


_SENTENCE_END = (".", "!", "?", ",", ";", ":", "…", "。", "！", "？")
PAUSE_TAG_PATTERN = re.compile(r"\[(?:pause(?:\s*([\d.]+)\s*s?)?|break)\]", re.IGNORECASE)


def prepare_gen_text(text: str) -> str:
    text = text.strip()
    if text and not text.endswith(_SENTENCE_END):
        text += "."
    return text


def _script_segments(text: str) -> list[tuple[str, float]]:
    segments: list[tuple[str, float]] = []
    pending_pause = 0.0
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    for index, line in enumerate(normalized.split("\n")):
        if index:
            pending_pause += LINE_BREAK_PAUSE_SECONDS
        matches = list(PAUSE_TAG_PATTERN.finditer(line))
        if not matches:
            prepared = prepare_gen_text(line)
            if prepared:
                segments.append((prepared, pending_pause if segments else 0.0))
                pending_pause = 0.0
            continue

        last_pos = 0
        for match in matches:
            chunk = line[last_pos : match.start()]
            prepared = prepare_gen_text(chunk)
            if prepared:
                segments.append((prepared, pending_pause if segments else 0.0))
                pending_pause = 0.0
            sec_str = match.group(1)
            tag_pause = float(sec_str) if sec_str else LINE_BREAK_PAUSE_SECONDS
            pending_pause = max(pending_pause, tag_pause)
            last_pos = match.end()

        tail = line[last_pos:]
        prepared_tail = prepare_gen_text(tail)
        if prepared_tail:
            segments.append((prepared_tail, pending_pause if segments else 0.0))
            pending_pause = 0.0
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
        engine: str = "qwen",
    ) -> None:
        validate_engine(engine)
        self.engine = engine
        self.quality = quality
        self.model_id = model_id_for_quality(quality)
        if engine == "omnivoice":
            self.model_id = OMNIVOICE_MODEL_ID
        elif engine == "chatterbox":
            self.model_id = CHATTERBOX_MODEL_IDS[quality]
        self.sample_rate = sample_rate
        if engine == "qwen":
            self.device = detect_device()
        elif engine == "chatterbox":
            self.device = chatterbox_device()
        else:
            self.device = omnivoice_device()
        self.engine_name = ENGINES[engine]
        if tts_loader is not None:
            self._tts_loader = tts_loader
        elif engine == "qwen":
            self._tts_loader = _load_tts_model
        elif engine == "omnivoice":
            self._tts_loader = _load_omnivoice_model
        else:
            self._tts_loader = _load_chatterbox_model
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
                    if self.engine == "omnivoice":
                        attr = "sampling_rate"
                    elif self.engine == "chatterbox":
                        attr = "sr"
                    else:
                        attr = "sample_rate"
                    self.sample_rate = int(getattr(self._tts_model, attr, self.sample_rate))
        return self._tts_model

    def _ensure_stt_model(self):
        if self._stt_model is None:
            with self._model_lock:
                if self._stt_model is None:
                    self._stt_model = self._stt_loader(ASR_MODEL_ID)
        return self._stt_model

    def _transcribe_canonical(self, canonical_ref_path: Path | str) -> str:
        stt_result = self._ensure_stt_model().generate(
            str(canonical_ref_path),
            verbose=False,
        )
        transcript = str(getattr(stt_result, "text", "")).strip()
        if not transcript:
            raise RuntimeError("The reference recording could not be transcribed.")
        return transcript

    def _load_reference(self, reference_audio_path: str | Path) -> tuple[np.ndarray, int]:
        tensor_audio, ref_sr = load_audio(
            reference_audio_path,
            target_sr=self.sample_rate,
            max_duration_seconds=MAX_SOURCE_SECONDS,
        )
        audio_np = prepare_reference_audio(
            tensor_audio.squeeze().numpy(),
            ref_sr,
            max_duration_seconds=MAX_REFERENCE_SECONDS,
        )
        return normalize_audio(audio_np), ref_sr

    def transcribe(self, reference_audio_path: str | Path) -> str:
        audio_np, ref_sr = self._load_reference(reference_audio_path)
        if len(audio_np) == 0:
            raise ValueError("Reference audio is empty.")

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            canonical_ref_path = Path(tmp.name)

        try:
            sf.write(str(canonical_ref_path), audio_np, ref_sr, subtype="PCM_16")
            return self._transcribe_canonical(canonical_ref_path)
        finally:
            canonical_ref_path.unlink(missing_ok=True)

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
        if self.engine == "chatterbox" and language not in CHATTERBOX_LANGUAGES:
            raise ValueError(
                f"Unsupported Chatterbox language '{language}'. "
                "Chatterbox supports English only ('auto' or 'English')."
            )

        notify = progress_callback or (lambda _stage: None)
        notify("prepare")
        audio_np, ref_sr = self._load_reference(reference_audio_path)
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
                transcript = sidecar_transcript(reference_audio_path) or ""
            if not transcript and self.engine != "chatterbox":
                transcript = self._transcribe_canonical(canonical_ref_path)

            notify("voice")
            sample_rate = self.sample_rate
            pieces: list[tuple[np.ndarray, float]] = []
            for segment, segment_pause in _script_segments(text):
                if self.engine == "chatterbox":
                    wav = tts_model.generate(
                        segment,
                        audio_prompt_path=str(canonical_ref_path),
                    )
                    if hasattr(wav, "detach"):
                        wav = wav.detach().cpu().numpy()
                    generations = [
                        SimpleNamespace(
                            audio=np.asarray(wav),
                            sample_rate=int(getattr(tts_model, "sr", self.sample_rate)),
                        )
                    ]
                elif self.engine == "omnivoice":
                    audio = tts_model.generate(
                        text=segment,
                        ref_audio=str(canonical_ref_path),
                        ref_text=transcript,
                        speed=speed,
                        language=None if language == "auto" else language,
                        num_step=32 if self.quality == "high" else 16,
                    )
                    generations = [
                        SimpleNamespace(audio=piece, sample_rate=self.sample_rate)
                        for piece in audio
                    ]
                else:
                    generations = list(
                        tts_model.generate(
                            text=segment,
                            ref_audio=str(canonical_ref_path),
                            ref_text=transcript,
                            speed=1.0,
                            lang_code=language,
                            stream=False,
                            verbose=False,
                        )
                    )
                if not generations:
                    raise RuntimeError("The voice model returned no audio.")

                sample_rate = int(getattr(generations[0], "sample_rate", sample_rate))
                for index, item in enumerate(generations):
                    if index:
                        pause = 0.08
                    else:
                        pause = segment_pause if pieces else 0.0
                    piece = np.asarray(item.audio).squeeze().astype(np.float32)
                    # Remove the model's near-silent edges; room tone fills the joins instead.
                    piece = trim_silence(
                        piece, sample_rate, threshold_db=-70.0, padding_seconds=0.0
                    )
                    if self.engine != "omnivoice" and speed != 1.0 and len(piece):
                        from librosa.effects import time_stretch

                        piece = time_stretch(piece, rate=speed)
                    pieces.append((piece, pause))
            generated = join_with_room_tone(pieces, sample_rate)

            notify("finish")
            enhanced = enhance_audio(generated, sample_rate, target_level_db=-16.0)
        finally:
            canonical_ref_path.unlink(missing_ok=True)

        quality_label = "high fidelity" if self.quality == "high" else "fast"
        return SynthesisResult(
            audio=enhanced,
            sample_rate=sample_rate,
            duration_seconds=float(len(enhanced) / sample_rate),
            matched_voice=f"{self.engine_name} · {quality_label}",
        )


_shared_cloners: dict[tuple[str, str], LocalVoiceCloner] = {}
_shared_cloner_lock = threading.Lock()


def get_shared_cloner(quality: str = "high", engine: str = "qwen") -> LocalVoiceCloner:
    validate_engine(engine)
    model_id_for_quality(quality)
    key = (engine, quality)
    with _shared_cloner_lock:
        if key not in _shared_cloners:
            _shared_cloners[key] = LocalVoiceCloner(quality=quality, engine=engine)
        return _shared_cloners[key]


def is_shared_cloner_loaded(quality: str | None = None, engine: str = "qwen") -> bool:
    if quality is not None:
        cloner = _shared_cloners.get((engine, quality))
        return bool(cloner and cloner.model_loaded)
    return any(cloner.model_loaded for cloner in _shared_cloners.values())


def unload_shared_cloners() -> int:
    with _shared_cloner_lock:
        count = len(_shared_cloners)
        _shared_cloners.clear()
    import gc

    gc.collect()
    try:
        import mlx.core as mx

        if hasattr(mx, "clear_cache"):
            mx.clear_cache()
        elif hasattr(mx, "metal") and hasattr(mx.metal, "clear_cache"):
            mx.metal.clear_cache()
    except (ImportError, AttributeError):
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except (ImportError, AttributeError):
        pass
    return count
