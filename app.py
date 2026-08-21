import base64
import tempfile
from pathlib import Path

from shiny import App, reactive, render, ui

from src.audio_utils import save_audio
from src.cloner import LocalVoiceCloner

cloner = LocalVoiceCloner()

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            body { background-color: #f1f5f9; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
            .main-container { max-width: 940px; margin: 30px auto; }
            .card-box { background: white; padding: 24px; border-radius: 14px; box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.08); margin-bottom: 20px; }
            .badge-custom { background: #e0e7ff; color: #3730a3; padding: 5px 12px; border-radius: 6px; font-weight: 600; font-size: 0.85rem; }
            """
        )
    ),
    ui.div(
        {"class": "main-container"},
        ui.div(
            {"class": "text-center mb-4"},
            ui.h2("🎙️ True Zero-Shot Voice Cloning Studio", class_="text-primary fw-bold"),
            ui.p("Directly learns and replicates the exact timbre and acoustics from your uploaded audio file using F5-TTS.", class_="text-muted"),
            ui.span("⚡ Neural Diffusion-Transformer Engine Active (24kHz HD)", class_="badge-custom"),
        ),
        ui.div(
            {"class": "card-box"},
            ui.h4("1. Reference Voice Sample (Your Target Voice)"),
            ui.p("Upload the audio clip of the person's voice you want to clone (up to 30s).", class_="text-muted small"),
            ui.input_file("audio_file", "Upload Voice Sample (.wav, .mp3, .ogg, .flac, .m4a)", accept=[".wav", ".mp3", ".ogg", ".flac", ".m4a"], multiple=False),
            ui.output_ui("reference_preview"),
            ui.input_text(
                "ref_text",
                "Reference Audio Transcript (Optional - auto-detected via Whisper if empty):",
                placeholder="What was spoken in the uploaded audio clip...",
            ),
        ),
        ui.div(
            {"class": "card-box"},
            ui.h4("2. Speech Text to Generate"),
            ui.input_text_area(
                "speech_text",
                "Text for the cloned voice to speak:",
                value="Hello! This audio was generated using your exact voice profile cloned locally.",
                placeholder="Enter what you want your cloned voice to say...",
                rows=3,
                width="100%",
            ),
            ui.layout_columns(
                ui.input_slider("speed", "Speaking Speed", min=0.6, max=1.6, value=1.0, step=0.05),
                ui.input_slider("nfe_step", "Quality / Diffusion Steps (Higher = clearer)", min=16, max=48, value=24, step=4),
            ),
            ui.input_action_button("btn_generate", "🚀 Clone Uploaded Voice & Generate Audio", class_="btn-primary w-100 btn-lg mt-3 fw-bold"),
        ),
        ui.div(
            {"class": "card-box"},
            ui.h4("3. Cloned Voice Audio Output"),
            ui.output_ui("generation_status"),
            ui.output_ui("audio_result"),
        ),
    ),
)


def server(input, output, session):
    output_audio_path = reactive.value(None)

    @render.ui
    def reference_preview():
        file_infos = input.audio_file()
        if not file_infos:
            return ui.p("No reference audio selected yet.", class_="text-muted small")
        file_info = file_infos[0]
        datapath = file_info["datapath"]
        with open(datapath, "rb") as f:
            b64_audio = base64.b64encode(f.read()).decode("utf-8")
        return ui.div(
            ui.p(f"Target Voice Loaded: {file_info['name']}", class_="fw-bold text-success mb-2"),
            ui.tags.audio(
                controls=True,
                src=f"data:audio/wav;base64,{b64_audio}",
                style="width: 100%;",
            ),
        )

    @reactive.effect
    @reactive.event(input.btn_generate)
    def handle_synthesis():
        file_infos = input.audio_file()
        text = input.speech_text()

        if not file_infos:
            ui.notification_show("Please upload a reference audio sample first!", type="warning")
            return

        if not text.strip():
            ui.notification_show("Please enter text to synthesize!", type="warning")
            return

        ref_path = file_infos[0]["datapath"]
        ref_text = input.ref_text() or ""
        try:
            with ui.Progress(min=1, max=3) as p:
                p.set(1, message="Extracting reference acoustic features...")
                speed = float(input.speed())
                nfe_step = int(input.nfe_step())

                p.set(2, message="Synthesizing audio conditioned on your reference voice...")
                result = cloner.clone_voice(
                    reference_audio_path=ref_path,
                    text=text,
                    reference_text=ref_text,
                    speed=speed,
                    nfe_step=nfe_step,
                )

                p.set(3, message="Saving generated 24kHz master...")
                temp_dir = Path(tempfile.gettempdir())
                out_file = temp_dir / "cloned_uploaded_voice.wav"
                save_audio(out_file, result.audio, sample_rate=result.sample_rate)

                output_audio_path.set(str(out_file))
                ui.notification_show("Voice cloned successfully with your uploaded audio!", type="message")

        except (ValueError, OSError, RuntimeError) as e:
            ui.notification_show(f"Synthesis failed: {e!s}", type="error")

    @render.ui
    def generation_status():
        path = output_audio_path()
        if not path:
            return ui.p("Click 'Clone Uploaded Voice & Generate Audio' when ready.", class_="text-muted")
        return ui.div(
            ui.p("✅ Voice clone audio ready! Conditioned directly on your uploaded reference sample.", class_="text-success fw-bold mb-1"),
        )

    @render.ui
    def audio_result():
        path = output_audio_path()
        if not path or not Path(path).exists():
            return ui.div()

        with open(path, "rb") as f:
            audio_bytes = f.read()
        b64_audio = base64.b64encode(audio_bytes).decode("utf-8")

        return ui.div(
            ui.tags.audio(
                controls=True,
                autoplay=True,
                src=f"data:audio/wav;base64,{b64_audio}",
                style="width: 100%; margin-bottom: 14px;",
            ),
            ui.tags.a(
                "⬇️ Download Cloned Audio (.wav)",
                href=f"data:audio/wav;base64,{b64_audio}",
                download="cloned_voice_output.wav",
                class_="btn btn-success fw-semibold",
            ),
        )


app = App(app_ui, server)
