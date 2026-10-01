"""REST API for local voice cloning.

Run with:
    uvicorn src.api:app --host 127.0.0.1 --port 8001

Example:
    curl -X POST http://127.0.0.1:8001/synthesize \
        -F "reference_audio=@voice_sample.wav" \
        -F "text=Hello from the API" \
        -F "output_format=mp3" \
        -o cloned.mp3
"""

import importlib.metadata
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, Response

from src.audio_utils import SUPPORTED_OUTPUT_FORMATS, save_audio
from src.cloner import (
    CHATTERBOX_LANGUAGES,
    CHATTERBOX_MODEL_IDS,
    ENGINE_NAME,
    ENGINES,
    MODEL_VARIANTS,
    SUPPORTED_LANGUAGES,
    detect_device,
    get_shared_cloner,
    is_shared_cloner_loaded,
    model_id_for_quality,
    sidecar_transcript,
    transcribe_backend_available,
    validate_engine,
)

MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024
SERVICE_DIR = Path(__file__).resolve().parent.parent
GIT_VERSION_FILE = SERVICE_DIR / "GIT_VERSION"
SAVED_VOICES_DIR = SERVICE_DIR / "voice_samples"

app = FastAPI(
    title="Local Voice Cloning API",
    description="Local voice cloning with Qwen3-TTS, optional OmniVoice, or optional "
    "Chatterbox. Upload a reference voice "
    "sample and text; receive synthesized speech as WAV or MP3.",
    version="2.1.0",
)


class APIError(HTTPException):
    """HTTP error that also carries the original traceback.

    The service binds to loopback only, so tracebacks in responses help clients
    diagnose failures (e.g. interrupted model downloads) without server logs.
    """

    def __init__(self, status_code: int, detail: str, traceback_text: str) -> None:
        super().__init__(status_code=status_code, detail=detail)
        self.traceback_text = traceback_text


@app.exception_handler(HTTPException)
async def http_exception_with_traceback(request: Request, exc: HTTPException) -> JSONResponse:
    body: dict[str, str] = {"detail": exc.detail}
    traceback_text = getattr(exc, "traceback_text", "")
    if traceback_text:
        body["traceback"] = traceback_text
    return JSONResponse(
        status_code=exc.status_code,
        content=body,
        headers=getattr(exc, "headers", None),
    )


def _package_version() -> str:
    try:
        return importlib.metadata.version("local-voice-cloning")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _git_version() -> str:
    """Short git SHA from a GIT_VERSION file or git itself; "unknown" when unavailable."""
    try:
        if GIT_VERSION_FILE.is_file():
            stamp = GIT_VERSION_FILE.read_text(encoding="utf-8").strip()
            if stamp:
                return stamp
    except OSError:
        pass
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
            cwd=SERVICE_DIR,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


PACKAGE_VERSION = _package_version()
GIT_VERSION = _git_version()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": is_shared_cloner_loaded(),
        "version": {"package": PACKAGE_VERSION, "git": GIT_VERSION},
        "capabilities": {
            "speed_control": "post-stretch (librosa) for qwen/chatterbox, native for omnivoice",
            "transcribe": transcribe_backend_available(),
            "sidecar_transcripts": True,
        },
    }


@app.get("/info")
def info():
    """Report engine defaults without forcing either model checkpoint to load."""
    return {
        "engine": ENGINE_NAME,
        "engines": ENGINES,
        "omnivoice": {"model": "k2-fsa/OmniVoice", "quality_steps": {"high": 32, "fast": 16},
                      "languages": "600+; use a language name, code, or auto"},
        "chatterbox": {"models": dict(sorted(CHATTERBOX_MODEL_IDS.items())),
                       "quality_models": {"high": "Chatterbox full (English)",
                                          "fast": "Chatterbox-Turbo (English)"},
                       "languages": list(CHATTERBOX_LANGUAGES)},
        "device": detect_device(),
        "sample_rate": 24000,
        "default_quality": "high",
        "quality_models": dict(sorted(MODEL_VARIANTS.items())),
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "model_loaded": is_shared_cloner_loaded(),
        "supported_output_formats": sorted(SUPPORTED_OUTPUT_FORMATS),
    }


@app.post("/warmup")
def warmup(
    engine: Annotated[str, Form(description="Voice engine: qwen, omnivoice, or chatterbox")] = "qwen",
    quality: Annotated[
        str,
        Form(description="Quality: Qwen BF16/8-bit; OmniVoice 32/16 steps; Chatterbox full/Turbo"),
    ] = "high",
):
    """Fetch and load model weights outside a synthesis or transcription request."""
    quality = quality.lower().strip()
    try:
        model_id_for_quality(quality)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    engine = engine.lower().strip()
    try:
        validate_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    cloner = get_shared_cloner(quality, engine=engine)
    try:
        timings = cloner.warmup(include_transcriber=engine != "chatterbox")
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    except Exception as e:
        raise APIError(
            status_code=500,
            detail=f"Warmup failed: {e}",
            traceback_text=traceback.format_exc(),
        ) from e

    return {
        "status": "ok",
        "engine": engine,
        "quality": quality,
        "model_id": cloner.model_id,
        "model_loaded": cloner.model_loaded,
        "stages": {stage: round(seconds, 3) for stage, seconds in timings.items()},
        "load_seconds": round(sum(timings.values()), 3),
    }


def _uploaded_sidecar_transcript(filename: str | None) -> str:
    """Sidecar transcript for an uploaded reference clip.

    Looks next to the client's file when the service shares its filesystem,
    then in the saved-voice library by file name (voice_samples/<name>.json).
    """
    if not filename:
        return ""
    name = Path(filename)
    return sidecar_transcript(name) or sidecar_transcript(SAVED_VOICES_DIR / name.name)


@app.post("/transcribe")
async def transcribe(
    reference_audio: Annotated[UploadFile, File(description="Voice sample to transcribe (wav/mp3/ogg/flac/m4a)")],
    quality: Annotated[
        str,
        Form(description="Quality: Qwen BF16/8-bit; OmniVoice 32/16 steps; Chatterbox full/Turbo"),
    ] = "high",
):
    quality = quality.lower().strip()
    try:
        model_id_for_quality(quality)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    ref_bytes = await reference_audio.read()
    if len(ref_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded reference audio is empty.")
    if len(ref_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Reference audio exceeds the 50 MB limit.")

    ref_suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / f"reference{ref_suffix}"
        ref_path.write_bytes(ref_bytes)
        try:
            transcript = get_shared_cloner(quality).transcribe(ref_path)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except (OSError, RuntimeError, ImportError) as e:
            raise APIError(
                status_code=422,
                detail=(
                    f"Transcription failed: {e}. "
                    "Pass ref_text directly to /synthesize, or place a transcript in the "
                    "reference's .json sidecar (same stem as the audio, key 'transcript')."
                ),
                traceback_text=traceback.format_exc(),
            ) from e
        except Exception as e:
            raise APIError(
                status_code=500,
                detail=f"Transcription failed: {e}",
                traceback_text=traceback.format_exc(),
            ) from e

    return {"transcript": transcript}


@app.post("/synthesize")
async def synthesize(
    reference_audio: Annotated[UploadFile, File(description="Voice sample to clone (wav/mp3/ogg/flac/m4a)")],
    text: Annotated[str, Form(description="Text for the cloned voice to speak")],
    ref_text: Annotated[
        str,
        Form(
            description="Transcript of the first 12 seconds of the reference audio only "
            "(auto-detected if empty; a longer transcript truncates the output)"
        ),
    ] = "",
    speed: Annotated[
        float,
        Form(ge=0.3, le=2.0, description="Compatibility option; accepted by the MLX backend"),
    ] = 1.0,
    quality: Annotated[
        str,
        Form(description="Quality: Qwen BF16/8-bit; OmniVoice 32/16 steps; Chatterbox full/Turbo"),
    ] = "high",
    language: Annotated[
        str,
        Form(description="Output language or auto"),
    ] = "auto",
    steps: Annotated[
        int | None,
        Form(ge=8, le=128, description="Deprecated F5-TTS option; accepted but ignored"),
    ] = None,
    cfg_strength: Annotated[
        float,
        Form(ge=1.0, le=4.0, description="Deprecated F5-TTS option; accepted but ignored"),
    ] = 2.0,
    output_format: Annotated[str, Form(description="Output audio format: wav or mp3")] = "wav",
    engine: Annotated[str, Form(description="Voice engine: qwen, omnivoice, or chatterbox")] = "qwen",
):
    output_format = output_format.lower().lstrip(".")
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported output format '{output_format}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}",
        )
    quality = quality.lower().strip()
    try:
        model_id_for_quality(quality)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    engine = engine.lower().strip()
    try:
        validate_engine(engine)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if engine == "qwen" and language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{language}'.",
        )
    if engine == "chatterbox" and language not in CHATTERBOX_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported Chatterbox language '{language}'. "
            "Chatterbox supports English only ('auto' or 'English').",
        )
    if not text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' cannot be empty.")

    ref_bytes = await reference_audio.read()
    if len(ref_bytes) == 0:
        raise HTTPException(status_code=422, detail="Uploaded reference audio is empty.")
    if len(ref_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Reference audio exceeds the 50 MB limit.")

    ref_suffix = Path(reference_audio.filename or "reference.wav").suffix or ".wav"
    reference_text = ref_text.strip() or _uploaded_sidecar_transcript(reference_audio.filename)

    with tempfile.TemporaryDirectory() as tmpdir:
        ref_path = Path(tmpdir) / f"reference{ref_suffix}"
        ref_path.write_bytes(ref_bytes)

        try:
            result = get_shared_cloner(quality, engine=engine).clone_voice(
                reference_audio_path=ref_path,
                text=text,
                reference_text=reference_text,
                speed=speed,
                language=language,
                nfe_step=steps,
                cfg_strength=cfg_strength,
            )
            out_path = Path(tmpdir) / f"output.{output_format}"
            save_audio(out_path, result.audio, sample_rate=result.sample_rate)
            audio_bytes = out_path.read_bytes()
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from e
        except Exception as e:
            raise APIError(
                status_code=500,
                detail=f"Synthesis failed: {e}",
                traceback_text=traceback.format_exc(),
            ) from e

    return Response(
        content=audio_bytes,
        media_type=MEDIA_TYPES[output_format],
        headers={
            "Content-Disposition": f'attachment; filename="cloned_voice.{output_format}"',
            "X-Duration-Seconds": f"{result.duration_seconds:.3f}",
            "X-Sample-Rate": str(result.sample_rate),
        },
    )
