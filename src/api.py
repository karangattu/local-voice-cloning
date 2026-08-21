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

MEDIA_TYPES = {"wav": "audio/wav", "mp3": "audio/mpeg"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

_cloner = None


def get_cloner():
    global _cloner
    if _cloner is None:
        from src.cloner import LocalVoiceCloner

        _cloner = LocalVoiceCloner()
    return _cloner


app = FastAPI(
    title="Local Voice Cloning API",
    description="Zero-shot voice cloning with F5-TTS. Upload a reference voice "
    "sample and text; receive synthesized speech as WAV or MP3.",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": _cloner is not None,
        "device": _cloner.device if _cloner is not None else None,
    }


@app.get("/info")
def info():
    cloner = get_cloner()
    return {
        "engine": "F5-TTS",
        "device": cloner.device,
        "sample_rate": cloner.sample_rate,
        "default_quality_steps": cloner.default_nfe_step,
        "supported_output_formats": sorted(SUPPORTED_OUTPUT_FORMATS),
    }


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
    speed: Annotated[float, Form(ge=0.3, le=2.0, description="Speech speed factor")] = 1.0,
    steps: Annotated[int | None, Form(ge=8, le=128, description="Diffusion steps (default: hardware maximum)")] = None,
    cfg_strength: Annotated[
        float,
        Form(ge=1.0, le=4.0, description="Voice adherence strength. Lower values sound more natural but less exact."),
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
            result = get_cloner().clone_voice(
                reference_audio_path=ref_path,
                text=text,
                reference_text=ref_text,
                speed=speed,
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
