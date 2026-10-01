import argparse
import sys
from pathlib import Path

from src.audio_utils import save_audio
from src.cloner import CHATTERBOX_LANGUAGES, ENGINES, SUPPORTED_LANGUAGES, LocalVoiceCloner


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
    parsed = parser.parse_args(args)
    if not parsed.warmup and (parsed.reference is None or parsed.text is None):
        parser.error("--reference and --text are required unless --warmup is given")
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
