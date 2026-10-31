# Local voice cloning

This app clones a voice on your Mac with Apple MLX. Audio never leaves your computer. You provide a short audio sample and text, and the app generates speech as a WAV or MP3 file.

![Sona Voice Studio](docs/app-screenshot.png)

## Setup

You need an Apple Silicon Mac (M-series), Python 3.10 or newer, and about 8 GB of free disk space. The app downloads models automatically on the first run. Install the dependencies with `uv sync`, or use `uv sync --extra dev` for development. Run the test suite with `uv run pytest`:

```bash
uv sync
uv run pytest
```

## Web app

Start the Shiny web app:

```bash
uv run shiny run app.py
```

Open the URL from your terminal, for example `http://127.0.0.1:<port>`, and follow these steps:

1. Record or upload a 5 to 12 second voice clip, or select or import a saved voice.
2. Fix any mistakes in the automatic transcript.
3. Enter your script in the text box.
4. Click Create audio.

For best results, record clear, natural speech from one speaker in a quiet room. Use commas and periods in your script to control pauses and pacing.

## CLI

Generate speech from the command line:

```bash
uv run python -m src.cli \
  --reference voice_sample.wav \
  --text "Hello world" \
  --output output.wav
```

The CLI accepts these flags:

- Pass `--quality` with `high` (the default) or `fast`.
- Pass `--language` with the output language (`auto` by default, or English only for Chatterbox).
- Pass `--engine` with `qwen` (the default), `omnivoice`, or `chatterbox`.
- Pass `--ref-text` with the reference transcript, or omit it to transcribe the reference audio automatically.
- Pass `--warmup` without `--reference` or `--text` to download and load models, print the load time per model, and exit.

## REST API

Start the API server on loopback:

```bash
uv run uvicorn src.api:app --port 8001
```

Synthesis runs in worker threads, so `/health` responds while a request runs. When the Qwen engine runs with `candidates` above 1, take 1 runs first. If its transcript matches the line, the server skips the other takes. Otherwise, takes 2 to N run together in one batch. Concurrent requests with the same reference voice and language share a batch. `GET /queue` and `/health` report active requests and queued rows per model.

Take k uses `seed + k - 1`. A seed produces the exact same take only when the batch contents match. A single-take request and take 1 match unbatched generation unless a concurrent request joins the batch. Takes 2 to N run in a batch and can differ slightly from unbatched takes with the same seed. Transcript accuracy stays the same.

Set these environment variables to control batching:

- Set `VOICE_BATCHING` to `0` to generate one take at a time without batching.
- Set `VOICE_BATCH_WINDOW_MS` to the milliseconds that a new batch waits for other requests to join (default 50).
- Set `VOICE_MAX_BATCH_SIZE` to the maximum number of takes in one batch (default 8).

Open `http://127.0.0.1:8001/docs` for the API documentation, or call `/synthesize` with `curl` (add `-F "engine=chatterbox"` to select another engine):

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "reference_audio=@voice_sample.wav" \
  -F "text=Hello from the API" \
  -o output.wav
```

Call `/warmup` before the first request to download and load model weights outside a synthesis request:

```bash
curl -X POST http://127.0.0.1:8001/warmup -F "engine=qwen" -F "quality=high"
```

The `/warmup` response reports the loaded `model_id` and the load time per stage. If a model download fails, the server returns HTTP 500 with an actionable error message. `GET /health` reports `version` (package version and git revision) and `capabilities` (speed control, transcription, and sidecar transcripts). Clients can use these fields to detect stale servers. Failed requests keep their status codes and add a `traceback` field with the full Python traceback.

A sidecar file is a JSON transcript file. If `/transcribe` cannot run because the backend is missing or a model download fails, it returns HTTP 422. Pass `ref_text` to `/synthesize` or store the transcript in a sidecar file next to the reference audio. The sidecar file uses the audio file stem plus `.json`, for example `voice_samples/karan.json` for `voice_samples/karan.wav`:

```json
{ "transcript": "Words from the recording.", "reference_id": "..." }
```

When `ref_text` is empty, the cloner reads the `transcript` key from the sidecar file instead of running transcription. The JSON object must include the `transcript` key. When an uploaded file name matches a saved voice, `/synthesize` also reads `voice_samples/<name>.json`.

## Voice engines

Qwen3-TTS is the default engine. It runs on Apple MLX and needs no other package. OmniVoice clones voices in more than 600 languages. Install the `omnivoice` extra to run OmniVoice:

```bash
uv sync --extra omnivoice
uv run --extra omnivoice shiny run app.py
```

Select OmniVoice under Voice engine in the app, or pass `--engine omnivoice` to the CLI or API.

Chatterbox is an experimental English voice cloner. High fidelity uses the full model, and fast draft uses Chatterbox-Turbo. The first run downloads about 3 GB of model files. If downloads are slow, set `HF_TOKEN` for higher rate limits.

`chatterbox-tts` pins `transformers==5.2.0`, `torch==2.6.0`, and `torchaudio==2.6.0`, which conflict with the default app environment. Use a separate environment for Chatterbox. Install the package with `pip install chatterbox-tts`. Installing it into the default environment can break Whisper transcription and Qwen generation. Select Chatterbox under Voice engine in the app, or pass `--engine chatterbox` to the CLI or API. There is no `uv` extra for Chatterbox, so the engine shows a setup error when the package is missing.

The default app pins matching PyTorch and torchaudio versions because their native libraries must match. If an optional install breaks transcription, restore the default environment with `uv sync` (or `uv sync --extra dev`). Then restart Shiny. This command also removes packages that you installed outside the dependency list of the project.

If model loading stops with `TypeError: 'NoneType' object is not callable`, install an older setuptools before you start the app again:

```bash
pip install "setuptools<81"
```
