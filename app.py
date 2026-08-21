import base64
import tempfile
from pathlib import Path

import shinyswatch
from faicons import icon_svg
from shiny import App, reactive, render, ui

from src.audio_utils import analyze_reference_audio, save_audio
from src.cloner import LocalVoiceCloner

cloner = LocalVoiceCloner()

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.style(
            """
            body { background: linear-gradient(160deg, #f4f1fb 0%, #eef4fd 100%); min-height: 100vh; }
            .main-container { max-width: 960px; margin: 0 auto 48px; }
            .hero {
                background: linear-gradient(135deg, #593196 0%, #7b4fc9 55%, #3d7dd8 100%);
                color: white; border-radius: 0 0 28px 28px;
                padding: 44px 32px 36px; margin-bottom: 32px;
                box-shadow: 0 12px 32px -12px rgba(89, 49, 150, 0.55);
            }
            .hero h1 { font-weight: 800; letter-spacing: -0.5px; }
            .hero .icon-badge {
                display: inline-flex; align-items: center; justify-content: center;
                width: 64px; height: 64px; border-radius: 18px;
                background: rgba(255, 255, 255, 0.15); margin-bottom: 14px;
            }
            .hero .icon-badge svg { width: 30px; height: 30px; fill: white; }
            .engine-badge {
                display: inline-flex; align-items: center; gap: 8px;
                background: rgba(255, 255, 255, 0.18); color: white;
                padding: 7px 16px; border-radius: 999px;
                font-weight: 600; font-size: 0.85rem; backdrop-filter: blur(4px);
            }
            .engine-badge svg { width: 14px; height: 14px; fill: #ffd95e; }
            .card-box {
                background: white; padding: 28px; border-radius: 18px;
                border: 1px solid rgba(89, 49, 150, 0.08);
                box-shadow: 0 8px 24px -14px rgba(30, 24, 60, 0.25);
                margin-bottom: 24px; transition: box-shadow 0.2s ease;
            }
            .card-box:hover { box-shadow: 0 12px 32px -14px rgba(89, 49, 150, 0.35); }
            .step-header { display: flex; align-items: center; gap: 14px; margin-bottom: 6px; }
            .step-number {
                display: inline-flex; align-items: center; justify-content: center;
                min-width: 40px; height: 40px; border-radius: 12px;
                background: linear-gradient(135deg, #593196, #7b4fc9);
                color: white; font-weight: 700; font-size: 1.1rem;
            }
            .step-header svg { width: 18px; height: 18px; fill: #593196; }
            .btn-generate {
                background: linear-gradient(135deg, #593196, #3d7dd8) !important;
                border: none !important; padding: 14px !important;
                border-radius: 12px !important; letter-spacing: 0.2px;
                box-shadow: 0 8px 20px -8px rgba(89, 49, 150, 0.6);
                transition: transform 0.15s ease, box-shadow 0.15s ease;
            }
            .btn-generate:hover { transform: translateY(-1px); box-shadow: 0 12px 24px -8px rgba(89, 49, 150, 0.7); }
            .btn-generate svg { width: 16px; height: 16px; fill: white; margin-right: 8px; }
            .btn-download svg { width: 14px; height: 14px; margin-right: 8px; }
            .icon-inline svg { width: 15px; height: 15px; margin-right: 6px; vertical-align: -2px; }
            audio { border-radius: 10px; }
            """
        )
    ),
    ui.div(
        {"class": "hero text-center"},
        ui.div({"class": "icon-badge"}, icon_svg("microphone-lines")),
        ui.h1("Voice Cloning Studio"),
        ui.p("Clone any voice locally. Your audio never leaves this machine.", class_="mb-3 opacity-75"),
        ui.span(
            {"class": "engine-badge"},
            icon_svg("bolt"),
            f"F5-TTS Engine | 24kHz HD | {cloner.device.upper()} accelerated | {cloner.default_nfe_step}-step max quality",
        ),
    ),
    ui.div(
        {"class": "main-container"},
        ui.div(
            {"class": "card-box"},
            ui.div(
                {"class": "step-header"},
                ui.span({"class": "step-number"}, "1"),
                ui.h4({"class": "mb-0"}, "Reference Voice Sample"),
            ),
            ui.p("Upload the audio clip of the person's voice you want to clone. The first 12 seconds are used; the transcript is auto-detected via Whisper.", class_="text-muted small"),
            ui.input_file("audio_file", "Upload Voice Sample (.wav, .mp3, .ogg, .flac, .m4a)", accept=[".wav", ".mp3", ".ogg", ".flac", ".m4a"], multiple=False),
            ui.output_ui("reference_preview"),
        ),
        ui.div(
            {"class": "card-box"},
            ui.div(
                {"class": "step-header"},
                ui.span({"class": "step-number"}, "2"),
                ui.h4({"class": "mb-0"}, "Speech Text to Generate"),
            ),
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
                ui.input_slider("nfe_step", "Quality / Diffusion Steps (Higher = clearer)", min=16, max=96, value=cloner.default_nfe_step, step=4),
            ),
            ui.input_action_button(
                "btn_generate",
                ui.TagList(icon_svg("wand-magic-sparkles"), "Clone Voice & Generate Audio"),
                class_="btn-primary btn-generate w-100 btn-lg mt-3 fw-bold",
            ),
        ),
        ui.div(
            {"class": "card-box"},
            ui.div(
                {"class": "step-header"},
                ui.span({"class": "step-number"}, "3"),
                ui.h4({"class": "mb-0"}, "Cloned Voice Output"),
            ),
            ui.output_ui("generation_status"),
            ui.output_ui("audio_result"),
        ),
    ),
    theme=shinyswatch.theme.pulse,
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

        try:
            report = analyze_reference_audio(datapath)
        except (ValueError, OSError, RuntimeError):
            report = None

        quality_feedback = ui.div()
        if report is not None:
            if report["warnings"]:
                quality_feedback = ui.div(
                    ui.p(
                        {"class": "fw-bold text-warning mb-1 mt-2 icon-inline"},
                        icon_svg("triangle-exclamation"),
                        "Sample quality warnings (these cause robotic-sounding clones):",
                    ),
                    ui.tags.ul(*[ui.tags.li(w, class_="small text-warning") for w in report["warnings"]]),
                )
            else:
                quality_feedback = ui.p(
                    {"class": "small text-success mt-2 icon-inline"},
                    icon_svg("circle-check"),
                    f"Good sample: {report['duration_seconds']:.1f}s, {report['sample_rate']} Hz, clean signal.",
                )

        return ui.div(
            ui.p(
                {"class": "fw-bold text-success mb-2 icon-inline"},
                icon_svg("file-audio"),
                f"Target Voice Loaded: {file_info['name']}",
            ),
            ui.tags.audio(
                controls=True,
                src=f"data:audio/wav;base64,{b64_audio}",
                style="width: 100%;",
            ),
            quality_feedback,
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
        try:
            with ui.Progress(min=1, max=3) as p:
                p.set(1, message="Transcribing and analyzing your reference voice...")
                speed = float(input.speed())
                nfe_step = int(input.nfe_step())

                p.set(2, message="Synthesizing audio conditioned on your reference voice...")
                result = cloner.clone_voice(
                    reference_audio_path=ref_path,
                    text=text,
                    speed=speed,
                    nfe_step=nfe_step,
                )

                p.set(3, message="Saving 24-bit WAV master and MP3 versions...")
                temp_dir = Path(tempfile.gettempdir())
                wav_file = temp_dir / "cloned_uploaded_voice.wav"
                mp3_file = temp_dir / "cloned_uploaded_voice.mp3"
                save_audio(wav_file, result.audio, sample_rate=result.sample_rate)
                save_audio(mp3_file, result.audio, sample_rate=result.sample_rate)

                output_audio_path.set(str(wav_file))
                ui.notification_show("Voice cloned successfully with your uploaded audio!", type="message")

        except (ValueError, OSError, RuntimeError) as e:
            ui.notification_show(f"Synthesis failed: {e!s}", type="error")

    @render.ui
    def generation_status():
        path = output_audio_path()
        if not path:
            return ui.p("Click 'Clone Voice & Generate Audio' when ready.", class_="text-muted")
        return ui.p(
            {"class": "text-success fw-bold mb-1 icon-inline"},
            icon_svg("circle-check"),
            "Voice clone audio ready! Conditioned directly on your uploaded reference sample.",
        )

    @render.ui
    def audio_result():
        path = output_audio_path()
        if not path or not Path(path).exists():
            return ui.div()

        wav_path = Path(path)
        mp3_path = wav_path.with_suffix(".mp3")

        with open(wav_path, "rb") as f:
            b64_wav = base64.b64encode(f.read()).decode("utf-8")

        download_buttons = [
            ui.tags.a(
                icon_svg("download"),
                "Download WAV (24-bit lossless)",
                href=f"data:audio/wav;base64,{b64_wav}",
                download="cloned_voice_output.wav",
                class_="btn btn-success btn-download fw-semibold me-2",
            ),
        ]
        if mp3_path.exists():
            with open(mp3_path, "rb") as f:
                b64_mp3 = base64.b64encode(f.read()).decode("utf-8")
            download_buttons.append(
                ui.tags.a(
                    icon_svg("download"),
                    "Download MP3 (compressed)",
                    href=f"data:audio/mpeg;base64,{b64_mp3}",
                    download="cloned_voice_output.mp3",
                    class_="btn btn-outline-success btn-download fw-semibold",
                )
            )

        return ui.div(
            ui.tags.audio(
                controls=True,
                autoplay=True,
                src=f"data:audio/wav;base64,{b64_wav}",
                style="width: 100%; margin-bottom: 14px;",
            ),
            ui.div(*download_buttons),
        )


app = App(app_ui, server)
