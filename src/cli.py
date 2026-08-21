import argparse
import sys
from pathlib import Path

from src.audio_utils import save_audio
from src.cloner import LocalVoiceCloner


def parse_args(args=None):
    parser = argparse.ArgumentParser(description="Local Zero-Shot Neural Voice Cloning CLI")
    parser.add_argument(
        "-r",
        "--reference",
        type=str,
        required=True,
        help="Path to the reference audio file of the voice you want to clone.",
    )
    parser.add_argument(
        "-t",
        "--text",
        type=str,
        required=True,
        help="Text to synthesize with the cloned voice.",
    )
    parser.add_argument(
        "--ref-text",
        type=str,
        default="",
        help="Transcript of the reference audio clip (optional, auto-transcribes if empty).",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output.wav",
        help="Path to save the generated audio file.",
    )
    parser.add_argument(
        "-s",
        "--speed",
        type=float,
        default=1.0,
        help="Speech speed factor (default: 1.0).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=24,
        help="Diffusion quality steps (default: 24).",
    )
    return parser.parse_args(args)


def main():
    args = parse_args()
    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"Error: Reference audio file not found at '{args.reference}'", file=sys.stderr)
        sys.exit(1)

    print(f"Loading reference voice from {ref_path}...")
    cloner = LocalVoiceCloner()
    print("Synthesizing text conditioned directly on the uploaded reference voice...")
    result = cloner.clone_voice(
        reference_audio_path=ref_path,
        text=args.text,
        reference_text=args.ref_text,
        speed=args.speed,
        nfe_step=args.steps,
    )

    out_path = save_audio(args.output, result.audio, sample_rate=result.sample_rate)
    print(f"Generated {result.duration_seconds:.2f}s high-definition audio saved to: {out_path}")


if __name__ == "__main__":
    main()
