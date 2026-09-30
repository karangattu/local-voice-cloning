# Local Voice Cloning

Clone a voice locally on your Mac using Apple MLX. Audio never leaves your computer.

Provide a short audio sample and text, and the app generates speech as a WAV or MP3 file.

![Sona Voice Studio](docs/app-screenshot.png)

## Requirements

- Apple Silicon Mac (M-series)
- Python 3.10+
- ~8 GB free disk space (models download automatically on first run)

## Setup

```bash
uv sync
```

For development:

```bash
uv sync --extra dev
```

## Web App

Start the app:

```bash
uv run shiny run app.py
```

Open the URL shown in your terminal (e.g. `http://127.0.0.1:<port>`), then:
1. Record or upload a 5–12 second voice clip, or select / import a saved voice.
2. Check the automatic transcript and fix any mistakes.
3. Enter your script and click **Create audio**.

## CLI

```bash
uv run python -m src.cli \
  --reference voice_sample.wav \
  --text "Hello world" \
  --output output.wav
```

Options:
- `--quality`: `high` (default) or `fast`
- `--language`: Output language. The default value is `auto`. Chatterbox accepts English only.
- `--engine`: `qwen` (default), `omnivoice`, or `chatterbox`
- `--ref-text`: Transcript of reference audio (transcribed automatically if omitted)

## REST API

Start the API:

```bash
uv run uvicorn src.api:app --port 8001
```

Open `http://127.0.0.1:8001/docs` for API docs, or generate speech via curl:

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "reference_audio=@voice_sample.wav" \
  -F "text=Hello from the API" \
  -o output.wav
```

The `/synthesize` endpoint accepts an `engine` field:

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "reference_audio=@voice_sample.wav" \
  -F "text=Hello from the API" \
  -F "engine=chatterbox" \
  -o output.wav
```

## Voice engines

Qwen3-TTS is the default engine. It runs on Apple MLX and needs no other package. OmniVoice supports more than 600 languages. It needs the `omnivoice` extra. Chatterbox clones English voices. It needs the `chatterbox-tts` package.

## Optional: OmniVoice (600+ Languages)

To add voice cloning in 600+ languages:

```bash
uv sync --extra omnivoice
uv run --extra omnivoice shiny run app.py
```

Select **OmniVoice** under **Voice engine** in the app, or pass `--engine omnivoice` to the CLI or API.

## Optional: Chatterbox (English, experimental)

Chatterbox clones English voices. High fidelity uses the full model. Fast draft uses Chatterbox-Turbo.

To use Chatterbox in the app:

1. Install the package with `pip install chatterbox-tts`.
2. Select **Chatterbox** under **Voice engine**.
3. Write the script.
4. Click **Create audio**.

To use Chatterbox with the CLI or API instead, pass `--engine chatterbox`.

The first run downloads about 3 GB of model files. If downloads are slow, set a `HF_TOKEN` for higher rate limits.

`chatterbox-tts` pins `transformers==5.2.0`, `torch==2.6.0`, and `torchaudio==2.6.0`. These versions conflict with the default app environment. Use a separate environment for Chatterbox's pinned dependencies. Installing them into the default environment can break Whisper transcription and Qwen generation. There is no `uv` extra for Chatterbox; if the package is missing, the engine shows a setup error.

The default app pins matching PyTorch and torchaudio versions because their native libraries must match. If an optional install breaks transcription, restore the default environment with `uv sync` (or `uv sync --extra dev` for development), then restart Shiny. This also removes packages installed outside the project's dependency list.

If model loading stops with `TypeError: 'NoneType' object is not callable`, install an older setuptools:

```bash
pip install "setuptools<81"
```

Then start the app again.

## Tips for Best Results

- Use 5–12 seconds of clear, natural speech.
- Record in a quiet room with a single speaker.
- Check and correct the reference transcript before generating.
- Use punctuation (commas, periods) to control pauses and pacing.

## Testing

```bash
uv run pytest
```
