import argparse
import sys
from pathlib import Path

from src.audio_utils import save_audio
from src.cloner import (
    CHATTERBOX_LANGUAGES,
    ENGINES,
    MAX_CANDIDATES,
    SUPPORTED_LANGUAGES,
    LocalVoiceCloner,
)

API_DOCS = """# Agent CLI and REST API reference

## CLI
uv run python -m src.cli --reference voice_sample.wav --text "Hello" --output output.wav
Flags: --quality (high|fast), --language (auto by default; English only for chatterbox),
--engine (qwen|omnivoice|chatterbox), --ref-text, --candidates (1-5), --seed, --warmup.

## REST API
Start on loopback: uv run uvicorn src.api:app --port 8001 (OpenAPI at /docs)

Endpoints:
- POST /synthesize: form fields reference_audio, text, ref_text, engine, quality,
  language, output_format (wav|mp3), candidates, seed, speed.
- POST /transcribe: form fields reference_audio, quality. Returns 422 if transcription
  fails; pass ref_text to /synthesize or use a sidecar JSON file instead.
- POST /warmup: form fields engine, quality. Preloads weights and returns model_id and
  stage timings; returns 500 on download failure.
- GET /health: returns status, model_loaded, version (package, git), capabilities, queue.
- GET /queue: returns in-flight requests and queued rows per model.
- GET /info: returns engines, models, languages, and formats without loading weights.
- Errors keep their HTTP status code and include a traceback field.

## Batching and seeds
Synthesis runs in worker threads so /health responds during requests. With Qwen and
candidates > 1, take 1 runs first; if its transcript matches, remaining takes are
skipped, otherwise takes 2..N run in one batch. Concurrent same-voice requests share a
batch. Take k uses seed + k - 1; takes 2..N in a batch can differ slightly from
unbatched takes with the same seed.
Env vars: VOICE_BATCHING=0 (disable), VOICE_BATCH_WINDOW_MS=50, VOICE_MAX_BATCH_SIZE=8.

## Sidecar transcripts
When ref_text is empty, a <stem>.json file next to the reference audio (or in
voice_samples/<name>.json for matching uploaded filenames) supplies {"transcript": "..."}.

## Optional engines
- OmniVoice (600+ languages): uv sync --extra omnivoice
- Chatterbox (English): install chatterbox-tts in a separate venv (pins torch==2.6.0,
  torchaudio==2.6.0, transformers==5.2.0). If loading fails with TypeError on NoneType,
  run pip install "setuptools<81". Restore the default env with uv sync.
"""


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Local Zero-Shot Neural Voice Cloning CLI")
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        default=None,
        help="Path to the reference audio file of the voice you want to clone "
        "(required unless --warmup).",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        default=None,
        help="Text to synthesize with the cloned voice (required unless --warmup).",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="",
        help="Transcript of the first 12 seconds of the reference audio only "
        "(optional, auto-transcribes if empty; a longer transcript truncates the output).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Path to save the generated audio file (.wav or .mp3).",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        choices=["wav", "mp3"],
        default=None,
        help="Output audio format (default: inferred from the output file extension).",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--engine", choices=ENGINES, default="qwen")
    parser.add_argument(
        "--quality",
        choices=["high", "fast"],
        default="high",
        help="Quality: Qwen BF16/8-bit; OmniVoice 32/16 diffusion steps; "
        "Chatterbox full/Turbo.",
    )
    parser.add_argument(
        "--language",
        default="auto",
        help="Output language (default: auto).",
    )
    parser.add_argument(
        "--candidates",
        type=int,
        choices=range(1, MAX_CANDIDATES + 1),
        default=1,
        metavar=f"1-{MAX_CANDIDATES}",
        help="Takes to generate per line. The take whose transcript best matches "
        "the line is kept. Slower, but catches skipped or garbled words (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. The same seed and inputs give the same audio (default: random).",
    )
    parser.add_argument(
        "--cfg-strength",
        type=float,
        default=2.0,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Download and load the models ahead of time, then exit. "
        "Prints the load time per model; --reference and --text are not needed.",
    )
    parser.add_argument(
        "--api-docs",
        action="store_true",
        help="Print CLI, REST API, batching, sidecar, and engine docs for agents, then exit.",
    )
    parsed = parser.parse_args(args)
    if not (parsed.warmup or parsed.api_docs) and (parsed.reference is None or parsed.text is None):
        parser.error("--reference and --text are required unless --warmup or --api-docs is given")
    if parsed.engine == "qwen" and parsed.language not in SUPPORTED_LANGUAGES:
        parser.error(f"Unsupported Qwen language: {parsed.language}")
    if parsed.engine == "chatterbox" and parsed.language not in CHATTERBOX_LANGUAGES:
        parser.error(
            f"Unsupported Chatterbox language: {parsed.language}. "
            "Chatterbox supports English only ('auto' or 'English')."
        )
    return parsed


def run_warmup(args) -> None:
    cloner = LocalVoiceCloner(quality=args.quality, engine=args.engine)
    print(f"Warming up {cloner.engine_name} ({args.quality}) on {cloner.device}...")
    try:
        timings = cloner.warmup(include_transcriber=args.engine != "chatterbox")
    except (OSError, RuntimeError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    for stage, seconds in timings.items():
        print(f"  {stage}: {seconds:.2f}s")
    print(f"Model {cloner.model_id} ready.")


def main():
    args = parse_args()
    if args.api_docs:
        print(API_DOCS.strip())
        return
    if args.warmup:
        run_warmup(args)
        return
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: Reference audio file not found at '{args.reference}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading reference voice from {ref_path}...")
    cloner = LocalVoiceCloner(quality=args.quality, engine=args.engine)
    print(f"Synthesizing with {cloner.engine_name} ({args.quality}) on {cloner.device}...")
    result = cloner.clone_voice(
        reference_audio_path=ref_path,
        text=args.text,
        reference_text=args.ref_text,
        speed=args.speed,
        language=args.language,
        candidates=args.candidates,
        seed=args.seed,
        nfe_step=args.steps,
        cfg_strength=args.cfg_strength,
    )

    try:
        out_path = save_audio(
            args.output,
            result.audio,
            sample_rate=result.sample_rate,
            output_format=args.format,
        )
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"Generated {result.duration_seconds:.2f}s high-definition audio saved to: {out_path}")


if __name__ == "__main__":
    main()
