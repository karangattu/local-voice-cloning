# Local Voice Cloning

This project clones a voice on your computer. It uses the F5-TTS model. No audio leaves your machine.

You give a short voice sample and some text. The system speaks the text in that voice. You get the audio as WAV (24-bit) or MP3.

## Requirements

- Python 3.10 or later
- macOS, Linux, or Windows
- The first synthesis downloads the F5-TTS model. This download occurs one time.

## Installation

1. Create the virtual environment:
   ```bash
   uv venv .venv
   ```
2. Activate the environment:
   ```bash
   source .venv/bin/activate
   ```
3. Install the dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```

## Web App

1. Start the app:
   ```bash
   shiny run app.py
   ```
2. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. The page loads immediately; the F5-TTS model loads on your first generation, not on startup.
3. Upload a voice sample. The system uses the first 12 seconds and transcribes them automatically.
4. Type the text, pick a quality preset (Fast, Balanced, or Best), and click the generate button. The button disables while the audio generates; a Cancel button appears next to it.
5. Download the result as WAV or MP3.

Open "Advanced settings" for more control: an exact diffusion-step count (the quality preset sets this for you), the voice adherence strength (`cfg_strength`), and a reference transcript override for the first 12 seconds of your upload.

## CLI

Run one synthesis from the terminal:

```bash
python -m src.cli --reference voice_sample.wav --text "Hello world" --output output.wav
```

The file extension of `--output` sets the format (`.wav` or `.mp3`). You can also set the format with `-f wav` or `-f mp3`.

| Option | Default | Description |
|---|---|---|
| `-r`, `--reference` | required | The path to the voice sample |
| `-t`, `--text` | required | The text to speak |
| `-o`, `--output` | `output.wav` | The path for the output file (`.wav` or `.mp3`) |
| `-f`, `--format` | from extension | The output format (`wav` or `mp3`) |
| `--ref-text` | auto | The transcript of the first 12 seconds of the voice sample. Leave it empty for the best result. |
| `-s`, `--speed` | `1.0` | The speech speed factor |
| `--steps` | hardware maximum | The quality level (diffusion steps) |
| `--cfg-strength` | `2.0` | The voice adherence strength. Lower values sound more natural but less exact. |

## REST API

The API makes the voice cloner available to other tools. Any program that can send HTTP requests can use it.

### Start the server

Start the server on port 8001:

```bash
uvicorn src.api:app --host 127.0.0.1 --port 8001
```

NOTE: The web app uses port 8000. Use a different port for the API.

The server shows the API documentation at [http://127.0.0.1:8001/docs](http://127.0.0.1:8001/docs). You can send test requests from that page.

### Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Reports the server status. Tells you if the model is loaded. |
| `/info` | GET | Shows the device, the sample rate, the default quality, and the supported formats. |
| `/synthesize` | POST | Clones the voice and returns the audio bytes. |

### Generate audio

Send a POST request to `/synthesize` as a multipart form. Attach the voice sample as the `reference_audio` field.

Example (MP3 output):

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "reference_audio=@voice_sample.wav" \
  -F "text=Hello from the API" \
  -F "output_format=mp3" \
  -o cloned.mp3
```

Example (WAV output):

```bash
curl -X POST http://127.0.0.1:8001/synthesize \
  -F "reference_audio=@voice_sample.wav" \
  -F "text=Hello from the API" \
  -F "output_format=wav" \
  -o cloned.wav
```

### Form fields for `/synthesize`

| Field | Required | Default | Description |
|---|---|---|---|
| `reference_audio` | yes | — | The voice sample file (`wav`, `mp3`, `ogg`, `flac`, or `m4a`, 50 MB maximum) |
| `text` | yes | — | The text for the cloned voice to speak |
| `ref_text` | no | auto | The transcript of the first 12 seconds of the voice sample. Leave it empty for the best result. |
| `speed` | no | `1.0` | The speech speed factor (0.3 to 2.0) |
| `steps` | no | hardware maximum | The quality level (8 to 128 diffusion steps) |
| `cfg_strength` | no | `2.0` | The voice adherence strength (1.0 to 4.0) |
| `output_format` | no | `wav` | The output format (`wav` or `mp3`) |

### Response

The response body contains the audio bytes. The `Content-Type` header is `audio/wav` or `audio/mpeg`.

The response also contains these headers:

| Header | Description |
|---|---|
| `X-Duration-Seconds` | The length of the generated audio |
| `X-Sample-Rate` | The sample rate of the generated audio (24000) |

If the request is not valid, the server returns status 422 with an error message.

### Python example

```python
import httpx

with open("voice_sample.wav", "rb") as f:
    response = httpx.post(
        "http://127.0.0.1:8001/synthesize",
        files={"reference_audio": f},
        data={"text": "Hello from Python", "output_format": "mp3"},
        timeout=300,
    )

with open("cloned.mp3", "wb") as out:
    out.write(response.content)
```

## Get a Natural Voice

The voice sample controls the sound of the clone. A flat or damaged sample gives a robotic clone.

Follow these rules for the voice sample:

1. Record 5 to 12 seconds of natural speech.
2. Record one speaker, with no music and no background noise.
3. Speak full sentences with normal emotion. Do not read in a flat tone.
4. Do not let the recording clip (distort). Keep the input level moderate.
5. Remove long pauses from the sample. Long pauses make the output slow.

Follow these rules for the text:

1. Use normal punctuation. The model creates pauses and intonation from punctuation.
2. Write full sentences.

The web app examines each uploaded sample. It shows a warning when the sample has a problem.

If the voice still sounds flat, decrease `cfg_strength` to a value between 1.5 and 2.0 (in the web app, under "Advanced settings"; on the CLI, with `--cfg-strength`; in the API, with the `cfg_strength` field). Generate more than one time. Each run uses a different random seed, and some runs sound more natural than others.

## Quality

The system selects the highest quality that your hardware can run:

- Apple Silicon (MPS) or NVIDIA (CUDA): 64 diffusion steps
- CPU only: 32 diffusion steps

More steps give clearer audio and a longer generation time. The default gives the best quality. If you want faster drafts, set `steps` to a lower value, for example 16.

The web app exposes this choice as three quality presets: Fast (16 steps), Balanced (32 steps), and Best (the hardware maximum above). "Advanced settings" holds the exact diffusion-step count if you want a value in between.

NOTE: The model loads on the first synthesis request, in the web app, the CLI, and the API alike. The first request is slow because of this load. Later requests use the loaded model and are faster. Within one running web app or one running API server, every session and every request shares that same loaded model; the web app and the API are separate processes, so each loads its own copy if you run both.

## Tests

The test suite has two tiers. Fast tests mock the F5-TTS model and never download or run it. Integration tests use the real model and are slow.

Run the fast tests (this is the default):

```bash
python -m pytest
```

Run the integration tests:

```bash
python -m pytest -m integration
```
