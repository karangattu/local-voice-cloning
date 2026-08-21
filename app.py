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
        ui.tags.link(
            rel="preconnect",
            href="https://fonts.googleapis.com",
        ),
        ui.tags.link(
            rel="preconnect",
            href="https://fonts.gstatic.com",
            crossorigin="anonymous",
        ),
        ui.tags.link(
            href="https://fonts.googleapis.com/css2?family=Mona+Sans:ital,wght@0,200..900;1,200..900&display=swap",
            rel="stylesheet",
        ),
        ui.tags.style(
            """
            :root {
                --minty-primary: #78c2ad;
                --minty-primary-dark: #56a590;
                --minty-primary-deep: #2f6b5b;
                --minty-teal: #20c997;
                --minty-soft-bg: #f2faf7;
                --minty-card-border: rgba(120, 194, 173, 0.22);
            }

            * {
                font-family: 'Mona Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
            }

            body {
                background: linear-gradient(165deg, #f0f9f6 0%, #e6f5ef 40%, #f4fbf8 100%);
                min-height: 100vh;
                color: #2c3e50;
            }

            .main-container {
                max-width: 980px;
                margin: 0 auto 56px;
                padding: 0 16px;
            }

            .hero {
                background: linear-gradient(135deg, #2d7a66 0%, #4ea890 50%, #78c2ad 100%);
                color: white;
                border-radius: 0 0 32px 32px;
                padding: 48px 32px 40px;
                margin-bottom: 36px;
                box-shadow: 0 16px 36px -12px rgba(45, 122, 102, 0.42);
                position: relative;
                overflow: hidden;
            }

            .hero::before {
                content: '';
                position: absolute;
                top: -60px;
                right: -40px;
                width: 240px;
                height: 240px;
                background: radial-gradient(circle, rgba(255, 255, 255, 0.15) 0%, transparent 70%);
                border-radius: 50%;
                pointer-events: none;
            }

            .hero h1 {
                font-weight: 850;
                letter-spacing: -0.8px;
                font-size: 2.35rem;
                margin-bottom: 8px;
            }

            .hero p {
                font-size: 1.05rem;
                letter-spacing: -0.1px;
            }

            .hero .icon-badge {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                width: 68px;
                height: 68px;
                border-radius: 20px;
                background: rgba(255, 255, 255, 0.2);
                margin-bottom: 16px;
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.25);
                box-shadow: 0 8px 20px rgba(0, 0, 0, 0.08);
            }

            .hero .icon-badge svg {
                width: 32px !important;
                height: 32px !important;
                margin: 0 !important;
                fill: white;
            }

            .engine-badge {
                display: inline-flex;
                align-items: center;
                gap: 8px;
                background: rgba(255, 255, 255, 0.22);
                color: white;
                padding: 8px 20px;
                border-radius: 999px;
                font-weight: 600;
                font-size: 0.88rem;
                backdrop-filter: blur(8px);
                border: 1px solid rgba(255, 255, 255, 0.28);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.06);
            }

            .engine-badge svg {
                width: 14px;
                height: 14px;
                fill: #fff07c;
            }

            .card-box {
                background: rgba(255, 255, 255, 0.92);
                backdrop-filter: blur(12px);
                padding: 30px;
                border-radius: 22px;
                border: 1px solid var(--minty-card-border);
                box-shadow: 0 10px 30px -12px rgba(45, 122, 102, 0.16);
                margin-bottom: 26px;
                transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
            }

            .card-box:hover {
                transform: translateY(-2px);
                box-shadow: 0 14px 38px -12px rgba(45, 122, 102, 0.25);
                border-color: rgba(120, 194, 173, 0.45);
            }

            .step-header {
                display: flex;
                align-items: center;
                gap: 14px;
                margin-bottom: 8px;
            }

            .step-number {
                display: inline-flex;
                align-items: center;
                justify-content: center;
                min-width: 42px;
                height: 42px;
                border-radius: 14px;
                background: linear-gradient(135deg, #4ea890, #78c2ad);
                color: white;
                font-weight: 800;
                font-size: 1.15rem;
                box-shadow: 0 4px 12px rgba(78, 168, 144, 0.35);
            }

            .step-header h4 {
                font-weight: 750;
                color: #213547;
                letter-spacing: -0.3px;
            }

            .btn-generate {
                background: linear-gradient(135deg, #2d7a66 0%, #4ea890 50%, #78c2ad 100%) !important;
                border: none !important;
                padding: 16px !important;
                border-radius: 14px !important;
                font-weight: 750 !important;
                font-size: 1.08rem !important;
                letter-spacing: 0.1px;
                color: white !important;
                box-shadow: 0 8px 24px -6px rgba(45, 122, 102, 0.55);
                transition: transform 0.18s ease, box-shadow 0.18s ease !important;
            }

            .btn-generate:hover {
                transform: translateY(-2px);
                box-shadow: 0 14px 30px -6px rgba(45, 122, 102, 0.65) !important;
                filter: brightness(1.04);
            }

            .btn-generate:active {
                transform: translateY(0);
            }

            .btn-generate svg {
                width: 18px;
                height: 18px;
                fill: white;
                margin-right: 10px;
                vertical-align: -2px;
            }

            .btn-download {
                border-radius: 12px !important;
                padding: 10px 18px !important;
                font-weight: 650 !important;
                transition: transform 0.15s ease, box-shadow 0.15s ease !important;
            }

            .btn-download:hover {
                transform: translateY(-1px);
            }

            .btn-download svg {
                width: 15px;
                height: 15px;
                margin-right: 8px;
                vertical-align: -2px;
            }

            .icon-inline svg {
                width: 16px;
                height: 16px;
                margin-right: 8px;
                vertical-align: -2px;
            }

            .quality-box {
                border-radius: 12px;
                padding: 12px 16px;
                margin-top: 12px;
            }

            .quality-good {
                background-color: #eafaf1;
                border: 1px solid #c2eed5;
                color: #276749;
            }

            .quality-warn {
                background-color: #fef9e7;
                border: 1px solid #f9e79f;
                color: #9a7d0a;
            }

            audio {
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }

            .form-control, .form-select {
                border-radius: 12px !important;
                border: 1px solid #d1e7dd !important;
                padding: 10px 14px !important;
                transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
            }

            .form-control:focus, .form-select:focus {
                border-color: #78c2ad !important;
                box-shadow: 0 0 0 0.25rem rgba(120, 194, 173, 0.25) !important;
            }
            """
        )
    ),
    ui.div(
        {"class": "hero text-center"},
        ui.div({"class": "icon-badge"}, icon_svg("microphone-lines")),
        ui.h1("Voice Cloning Studio"),
        ui.p("Clone any voice locally with zero cloud dependencies. Your audio never leaves this device.", class_="mb-3 opacity-90"),
        ui.span(
            {"class": "engine-badge"},
            icon_svg("bolt"),
            f"F5-TTS Diffusion Engine | 24kHz HD | {cloner.device.upper()} Accelerated | {cloner.default_nfe_step}-Step Quality",
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
            ui.p("Upload a clear voice recording (e.g. 5–30s). The first 12 seconds are conditioned; transcript is auto-detected via Whisper.", class_="text-muted small"),
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
                class_="btn-primary btn-generate w-100 btn-lg mt-3",
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
    theme=shinyswatch.theme.minty,
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
                    {"class": "quality-box quality-warn"},
                    ui.p(
                        {"class": "fw-bold mb-1 icon-inline"},
                        icon_svg("triangle-exclamation"),
                        "Sample quality warnings (these cause robotic-sounding clones):",
                    ),
                    ui.tags.ul(*[ui.tags.li(w, class_="small") for w in report["warnings"]]),
                )
            else:
                quality_feedback = ui.div(
                    {"class": "quality-box quality-good"},
                    ui.p(
                        {"class": "small fw-semibold mb-0 icon-inline"},
                        icon_svg("circle-check"),
                        f"Optimal Voice Sample: {report['duration_seconds']:.1f}s, {report['sample_rate']} Hz, clean signal dynamics.",
                    ),
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
                "Download WAV (24-bit Lossless)",
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
                    "Download MP3 (Compressed)",
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
