# Local Voice Cloning

Local zero-shot neural voice cloning system with Shiny web UI and CLI.

## Quickstart

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Run Web App
```bash
shiny run app.py
```
Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

### Run CLI
```bash
python -m src.cli --reference voice_sample.wav --text "Hello world" --output output.wav
```
