# Local Voice Cloning

Clone a voice locally with Qwen3-TTS 1.7B and Apple MLX. Reference audio and generated speech stay on your Mac.

Provide a short voice recording and a script. The app produces 24-bit WAV and compressed MP3 audio while showing each stage of the local generation process.

## Requirements

- An Apple Silicon Mac
- Python 3.10 or later
- About 8 GB of free disk space for the high-fidelity voice model and automatic transcription model

The models download from Hugging Face the first time they are needed. Later generations reuse the local model cache.

## Installation

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Web app

Start Sona:

```bash
shiny run app.py
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000), then:

1. Upload a clean reference recording.
2. Write the script you want the cloned voice to speak.
3. Keep **High fidelity · BF16** selected for the best output, or choose **Fast draft · 8-bit** when iteration speed matters more.
4. Click **Create audio**. The transport reports preparation, model loading, voice synthesis, and finalization as they happen.
5. Play or download the result as WAV or MP3.

Advanced settings include an exact reference transcript. Supplying it avoids loading the automatic transcription model and usually improves voice fidelity. If it is blank, the app transcribes the first 12 seconds locally with Whisper Large V3 Turbo.

## CLI

```bash
python -m src.cli \
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
| `--quality` | `high` | `high` uses BF16; `fast` uses the 8-bit checkpoint |
| `--language` | `auto` | Output language or automatic detection |
| `-o`, `--output` | `output.wav` | Output path |
| `-f`, `--format` | from extension | `wav` or `mp3` |

## REST API

Start the API on a separate port:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8001
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
| `/synthesize` | POST | Generate cloned speech |

### `/synthesize` fields

| Field | Required | Default | Description |
|---|---|---|---|
| `reference_audio` | yes | — | WAV, MP3, OGG, FLAC, or M4A; 50 MB maximum |
| `text` | yes | — | Text for the cloned voice |
| `ref_text` | no | automatic | Exact transcript of the reference window |
| `quality` | no | `high` | `high` or `fast` |
| `language` | no | `auto` | Output language or automatic detection |
| `output_format` | no | `wav` | `wav` or `mp3` |

For compatibility, the previous `speed`, `steps`, and `cfg_strength` fields are still accepted. The F5-specific `steps` and `cfg_strength` values are ignored by Qwen3-TTS.

The response body contains the audio bytes. It also includes `X-Duration-Seconds` and `X-Sample-Rate` headers.

## Getting a natural clone

Reference quality still matters, even with the stronger model:

1. Use 5–12 seconds of natural, conversational speech.
2. Record one speaker in a quiet room with no music.
3. Use full sentences with normal expression rather than a flat read.
4. Avoid clipping, aggressive noise removal, or heavy compression.
5. Enter the exact reference transcript when possible.
6. Punctuate the generated script as you want it delivered.

The web app checks duration, level, clipping, silence, and sample rate before generation.

## Quality models

- **High fidelity:** `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`
- **Fast draft:** `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit`
- **Automatic transcription:** `mlx-community/whisper-large-v3-turbo-asr-fp16`

Each running web app or API process caches the models it has used. Choosing both quality modes in one process loads both checkpoints into memory.

## Tests

Fast tests use small fake model boundaries and do not download weights:

```bash
python -m pytest
```

Real-model integration tests are intentionally separate and slow:

```bash
python -m pytest -m integration
```
