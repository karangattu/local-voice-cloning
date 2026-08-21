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
2. Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
3. Upload a voice sample. The system uses the first 12 seconds and transcribes them automatically.
4. Type the text and click the generate button.
5. Download the result as WAV or MP3.

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

## Quality

The system selects the highest quality that your hardware can run:

- Apple Silicon (MPS) or NVIDIA (CUDA): 64 diffusion steps
- CPU only: 32 diffusion steps

More steps give clearer audio and a longer generation time. The default gives the best quality. If you want faster drafts, set `steps` to a lower value, for example 16.

NOTE: The model loads on the first synthesis request. The first request is slow because of this load. Later requests use the loaded model and are faster.

## Tests

Run the test suite:

```bash
python -m pytest tests/
```
