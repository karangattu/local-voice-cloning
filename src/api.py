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

import tempfile
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from src.audio_utils import SUPPORTED_OUTPUT_FORMATS, save_audio
from src.cloner import (
    ENGINE_NAME,
    MODEL_VARIANTS,
    SUPPORTED_LANGUAGES,
    detect_device,
    get_shared_cloner,
    is_shared_cloner_loaded,
    model_id_for_quality,
)

MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

app = FastAPI(
    title="Local Voice Cloning API",
    description="Zero-shot voice cloning with Qwen3-TTS on Apple MLX. Upload a reference voice "
    "sample and text; receive synthesized speech as WAV or MP3.",
    version="2.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": is_shared_cloner_loaded(),
    }


@app.get("/info")
def info():
    """Report engine defaults without forcing either model checkpoint to load."""
    return {
        "engine": ENGINE_NAME,
        "device": detect_device(),
        "sample_rate": 24000,
        "default_quality": "high",
        "quality_models": dict(sorted(MODEL_VARIANTS.items())),
        "supported_languages": list(SUPPORTED_LANGUAGES),
        "model_loaded": is_shared_cloner_loaded(),
        "supported_output_formats": sorted(SUPPORTED_OUTPUT_FORMATS),
    }


@app.post("/transcribe")
async def transcribe(
    reference_audio: Annotated[UploadFile, File(description="Voice sample to transcribe (wav/mp3/ogg/flac/m4a)")],
    quality: Annotated[
        str,
        Form(description="Model quality: high (BF16) or fast (8-bit)"),
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
        except (OSError, RuntimeError) as e:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {e}") from e

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
        Form(description="Model quality: high (BF16) or fast (8-bit)"),
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
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported language '{language}'.",
        )
    if not text.strip():
        raise HTTPException(status_code=422, detail="Field 'text' cannot be empty.")

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
            result = get_shared_cloner(quality).clone_voice(
                reference_audio_path=ref_path,
                text=text,
                reference_text=ref_text,
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
        except (OSError, RuntimeError) as e:
            raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}") from e

    return Response(
        content=audio_bytes,
        media_type=MEDIA_TYPES[output_format],
        headers={
            "Content-Disposition": f'attachment; filename="cloned_voice.{output_format}"',
            "X-Duration-Seconds": f"{result.duration_seconds:.3f}",
            "X-Sample-Rate": str(result.sample_rate),
        },
    )
