# Local Voice Cloning

Clone a voice locally with Qwen3-TTS 1.7B and Apple MLX. Reference audio and generated speech stay on your Mac.

Provide a short voice recording and a script. The app produces 24-bit WAV and compressed MP3 audio.

## Requirements

- An Apple Silicon Mac
- Python 3.10 or later
- About 8 GB of free disk space for the voice model and transcription model

The app downloads models from Hugging Face on first use. Later runs reuse the local cache.

## Installation

```bash
uv sync
```

For development (adds pytest, ruff, httpx):

```bash
uv sync --extra dev
```

## Optional OmniVoice engine

[OmniVoice](https://github.com/k2-fsa/OmniVoice) adds voice cloning in 600+ languages.
Qwen remains the default. Install and launch with the optional extra retained:

```bash
uv sync --extra dev --extra omnivoice
uv run --extra omnivoice shiny run app.py
```

Select **OmniVoice** under **Voice engine**. Enter a language name or code
(e.g. Hindi or ar), or leave `auto`. High fidelity uses 32 diffusion steps;
Fast draft uses 16 steps with the same checkpoint. The first generation downloads
`k2-fsa/OmniVoice` and its audio tokenizer, using additional disk space and memory.
On Apple Silicon the speech model uses PyTorch MPS; the upstream audio tokenizer
runs on CPU. On MPS, the app disables asynchronous Transformers weight loading
(`HF_DEACTIVATE_ASYNC_LOAD=1`) to avoid a native crash during weight conversion.
If an older app process stops at “Loading weights,” restart it with the updated
code; cached downloads are reused. A missing Hugging Face token warning does not
cause this crash. Mac performance has not been benchmarked against Qwen.

Use a 3–10 second reference for best results. The app retains its shared
12-second reference limit and MLX Whisper transcription/review workflow for both
engines, so transcripts continue to match when switching engines. This integration
still requires Apple Silicon. Transcription language coverage can be narrower than
OmniVoice's synthesis coverage; provide or correct the transcript when needed.

CLI example:

```bash
uv run --extra omnivoice python -m src.cli --engine omnivoice \
  --reference voice_sample.wav --ref-text "The words in the recording." \
  --text "Hello from OmniVoice." --language English --quality fast --output omni.wav
```

For the API, launch with `uv run --extra omnivoice uvicorn src.api:app`, then add
`-F "engine=omnivoice"` to the synthesis request below. `/transcribe` remains the
shared MLX Whisper service. Omit the engine field to keep using Qwen.

## Web app

Start Sona:

```bash
uv run shiny run app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), then:

1. Record yourself, upload a file, or pick a saved voice.
2. Review the auto-generated reference transcript and correct any wrong words.
3. Write the script you want the cloned voice to speak.
4. Select **High fidelity** or **Fast draft** quality.
5. Click **Create audio**.
6. Play or download the result as WAV or MP3.

## CLI

```bash
uv run python -m src.cli \
  --reference voice_sample.wav \
  --ref-text "The exact words spoken in the reference." \
  --text "Hello world" \
  --quality high \
  --language English \
  --output output.wav
```

| Option | Default | Description |
|---|---|---|
| `-r`, `--reference` | required | Reference recording |
| `-t`, `--text` | required | Text to synthesize |
| `--ref-text` | automatic | Exact transcript of the first 12 seconds |
| `--quality` | `high` | `high` uses BF16, `fast` uses the 8-bit checkpoint |
| `--language` | `auto` | Output language or automatic detection |
| `-o`, `--output` | `output.wav` | Output path |
| `-f`, `--format` | from extension | `wav` or `mp3` |

## REST API

Start the API:

```bash
uv run uvicorn src.api:app --host 127.0.0.1 --port 8001
```

Interactive documentation is available at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs).

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "reference_audio=@voice_sample.wav" \
  -F "ref_text=The exact words spoken in the reference." \
  -F "text=Hello from the API" \
  -F "quality=high" \
  -F "language=English" \
  -F "output_format=mp3" \
  -o cloned.mp3
```

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Service and model-load state |
| `/info` | GET | Engine, quality checkpoints, sample rate, and formats |
| `/transcribe` | POST | Transcribe a reference audio file |
| `/synthesize` | POST | Generate cloned speech |

## Getting a natural clone

1. Use 5 to 12 seconds of natural, conversational speech.
2. Record one speaker in a quiet room with no music.
3. Use full sentences with normal expression rather than a flat read.
4. Avoid clipping, aggressive noise removal, or heavy compression.
5. Review and correct the reference transcript before generating.
6. Use punctuation to control pauses and expression.

## Quality models

- **High fidelity:** `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`
- **Fast draft:** `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- **Automatic transcription:** `mlx-community/whisper-large-v3-turbo-asr-fp16`

## Tests

```bash
uv run pytest
```

Integration tests download model weights and are slow:

```bash
uv run pytest -m integration
```
