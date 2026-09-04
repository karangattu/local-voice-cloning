import asyncio
import base64
import mimetypes
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
import shinyswatch
import soundfile as sf
from faicons import icon_svg
from shiny import App, reactive, render, ui

from src.audio_utils import (
    analyze_reference_audio,
    apply_fades,
    load_audio,
    save_audio,
    trim_silence,
)
from src.cloner import ENGINE_NAME, get_shared_cloner
from src.progress import progress_snapshot, run_with_progress

VOICE_SAMPLES_DIR = Path(__file__).parent / "voice_samples"
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

RECORDING_TEMPLATES = {
    "standard": """Hi, I'm [your name], and this is my natural speaking voice. The quick brown fox jumps over the lazy dog. How vexingly quick daft zebras jump! Did it capture the real me?""",
    "conversational": """Hi, I’m [name]. Today is a beautiful day, and I’m feeling pretty good. Can you believe it? I have three things to finish, then I’m heading home.""",
}
RECORDING_PROMPT = RECORDING_TEMPLATES["standard"]
MAX_RECORDING_SECONDS = 30

app_ui = ui.page_fluid(
    ui.tags.head(
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
        ui.tags.meta(name="theme-color", content="#141617"),
        ui.tags.meta(name="color-scheme", content="dark"),
        ui.tags.title("Sona — Local Voice Studio"),
        ui.tags.link(rel="stylesheet", href="sona.css"),
        ui.tags.style("#ref_transcript, #speech_text { }"),
        ui.tags.script(
            """
            (function() {
                let mediaRecorder = null;
                let audioChunks = [];
                let audioContext = null;
                let analyserNode = null;
                let animationFrameId = null;
                let mediaStream = null;
                let isRecording = false;
                let timerId = null;
                let maxDurationTimerId = null;
                let startTime = 0;

                function setButtonState(recording) {
                    const btn = document.getElementById('btn-record');
                    if (!btn) return;
                    if (recording) {
                        btn.textContent = ' Stop recording';
                        btn.classList.add('recording');
                    } else {
                        btn.textContent = ' Start recording';
                        btn.classList.remove('recording');
                    }
                }

                function setTimer(seconds) {
                    const el = document.getElementById('record-timer');
                    if (el) el.textContent = Math.floor(seconds / 60) + ':' + String(Math.floor(seconds % 60)).padStart(2, '0');
                    const maxDuration = Number(document.getElementById('btn-record')?.dataset.maxDuration) || 30;
                    const pct = Math.min(100, (seconds / maxDuration) * 100);
                    const fill = document.getElementById('record-progress-fill');
                    if (fill) {
                        fill.style.width = pct + '%';
                        if (pct > 80) fill.classList.add('danger');
                        else fill.classList.remove('danger');
                    }
                }

                function updateVU() {
                    if (!analyserNode || !isRecording) return;
                    const dataArray = new Uint8Array(analyserNode.frequencyBinCount);
                    analyserNode.getByteFrequencyData(dataArray);
                    let sum = 0;
                    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
                    const avg = sum / dataArray.length;
                    const normalized = Math.min(1, avg / 70);
                    const vuBars = document.querySelectorAll('#record-vu-meter .vu-bar');
                    vuBars.forEach(function(bar, index) {
                        const threshold = (index + 1) / (vuBars.length + 1);
                        if (normalized >= threshold) {
                            bar.classList.add('active');
                            bar.style.height = (7 + (index + 1) * 3) + 'px';
                        } else {
                            bar.classList.remove('active');
                            bar.style.height = '6px';
                        }
                    });
                    animationFrameId = requestAnimationFrame(updateVU);
                }

                function resetVU() {
                    const vuBars = document.querySelectorAll('#record-vu-meter .vu-bar');
                    vuBars.forEach(function(bar) {
                        bar.classList.remove('active');
                        bar.style.height = '6px';
                    });
                    const fill = document.getElementById('record-progress-fill');
                    if (fill) {
                        fill.style.width = '0%';
                        fill.classList.remove('danger');
                    }
                }

                function setStatus(msg) {
                    const el = document.getElementById('record-status');
                    if (el) el.textContent = msg;
                }

                function sanitizeName(name) {
                    return name.trim().toLowerCase()
                        .replace(/\\s+/g, '-')
                        .replace(/[^a-z0-9_-]/g, '')
                        .replace(/^-+|-+$/g, '');
                }

                function encodeWav(audioBuffer) {
                    const numChannels = 1;
                    const sampleRate = audioBuffer.sampleRate;
                    const source = audioBuffer.getChannelData(0);
                    let samples = source;
                    if (audioBuffer.numberOfChannels > 1) {
                        samples = new Float32Array(source.length);
                        for (let ch = 0; ch < audioBuffer.numberOfChannels; ch++) {
                            const data = audioBuffer.getChannelData(ch);
                            for (let i = 0; i < data.length; i++) samples[i] += data[i] / audioBuffer.numberOfChannels;
                        }
                    }
                    const bytesPerSample = 2;
                    const blockAlign = numChannels * bytesPerSample;
                    const dataSize = samples.length * bytesPerSample;
                    const buffer = new ArrayBuffer(44 + dataSize);
                    const view = new DataView(buffer);
                    function writeStr(off, str) { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); }
                    writeStr(0, 'RIFF');
                    view.setUint32(4, 36 + dataSize, true);
                    writeStr(8, 'WAVE');
                    writeStr(12, 'fmt ');
                    view.setUint32(16, 16, true);
                    view.setUint16(20, 1, true);
                    view.setUint16(22, numChannels, true);
                    view.setUint32(24, sampleRate, true);
                    view.setUint32(28, sampleRate * blockAlign, true);
                    view.setUint16(32, blockAlign, true);
                    view.setUint16(34, 16, true);
                    writeStr(36, 'data');
                    view.setUint32(40, dataSize, true);
                    let offset = 44;
                    for (let i = 0; i < samples.length; i++, offset += 2) {
                        const s = Math.max(-1, Math.min(1, samples[i]));
                        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
                    }
                    return new Blob([buffer], { type: 'audio/wav' });
                }

                window.sonaToggleRecording = function() {
                    if (isRecording) {
                        sonaStopRecording();
                    } else {
                        sonaStartRecording();
                    }
                };

                function sonaStartRecording() {
                    const nameInput = document.getElementById('voice_name');
                    if (!nameInput || !nameInput.value.trim()) {
                        setStatus('Enter a voice name before recording.');
                        if (typeof Shiny !== 'undefined' && Shiny.notifications) {
                            Shiny.notifications.show({ html: 'Enter a voice name before recording.', type: 'warning' });
                        }
                        return;
                    }
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        setStatus('Recording is not supported in this browser.');
                        return;
                    }
                    navigator.mediaDevices.getUserMedia({ audio: true })
                        .then(function(stream) {
                            mediaStream = stream;
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            const source = audioContext.createMediaStreamSource(stream);
                            analyserNode = audioContext.createAnalyser();
                            analyserNode.fftSize = 64;
                            source.connect(analyserNode);

                            mediaRecorder = new MediaRecorder(stream);
                            audioChunks = [];
                            mediaRecorder.ondataavailable = function(e) {
                                if (e.data.size > 0) audioChunks.push(e.data);
                            };
                            mediaRecorder.onstop = function() {
                                var blob = new Blob(audioChunks, { type: 'audio/webm' });
                                blob.arrayBuffer().then(function(buf) {
                                    return audioContext.decodeAudioData(buf);
                                }).then(function(audioBuffer) {
                                    var wavBlob = encodeWav(audioBuffer);
                                    var reader = new FileReader();
                                    reader.onloadend = function() {
                                        var name = sanitizeName(document.getElementById('voice_name').value);
                                        if (typeof Shiny !== 'undefined') {
                                            Shiny.setInputValue('recorded_audio_data', { data: reader.result, name: name });
                                        }
                                        setStatus('Processing...');
                                    };
                                    reader.readAsDataURL(wavBlob);
                                }).catch(function(err) {
                                    setStatus('Could not process recording: ' + err.message);
                                }).finally(function() {
                                    if (mediaStream) mediaStream.getTracks().forEach(function(t) { t.stop(); });
                                    if (audioContext) { audioContext.close(); audioContext = null; }
                                    analyserNode = null;
                                });
                            };
                            mediaRecorder.start();
                            isRecording = true;
                            startTime = Date.now();
                            setButtonState(true);
                            setStatus('Recording...');
                            updateVU();
                            timerId = setInterval(function() {
                                setTimer((Date.now() - startTime) / 1000);
                            }, 150);
                            const maxDuration = Number(document.getElementById('btn-record').dataset.maxDuration) || 30;
                            maxDurationTimerId = setTimeout(function() {
                                setTimer(maxDuration);
                                sonaStopRecording('Maximum recording length reached. Processing...');
                            }, maxDuration * 1000);
                        })
                        .catch(function(err) {
                            setStatus('Microphone access denied: ' + err.message);
                            setButtonState(false);
                        });
                }

                function sonaStopRecording(statusMessage) {
                    if (!isRecording) return;
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }
                    isRecording = false;
                    setButtonState(false);
                    if (timerId) { clearInterval(timerId); timerId = null; }
                    if (maxDurationTimerId) { clearTimeout(maxDurationTimerId); maxDurationTimerId = null; }
                    if (animationFrameId) { cancelAnimationFrame(animationFrameId); animationFrameId = null; }
                    resetVU();
                    setStatus(statusMessage || 'Processing...');
                }

                window.sonaCopyTranscript = function(btn) {
                    const text = document.getElementById('ref_transcript')?.value || '';
                    if (!text) return;
                    navigator.clipboard.writeText(text).then(function() {
                        const orig = btn.innerHTML;
                        btn.innerHTML = '<svg width="11" height="11" viewBox="0 0 448 512" fill="currentColor"><path d="M438.6 105.4c12.5 12.5 12.5 32.8 0 45.3l-256 256c-12.5 12.5-32.8 12.5-45.3 0l-128-128c-12.5-12.5-12.5-32.8 0-45.3s32.8-12.5 45.3 0L160 338.7 393.4 105.4c12.5-12.5 32.8-12.5 45.3 0z"/></svg> Copied';
                        setTimeout(function() { btn.innerHTML = orig; }, 1800);
                    });
                };

                window.sonaUseAsScript = function() {
                    const text = document.getElementById('ref_transcript')?.value || '';
                    if (!text) return;
                    const speechArea = document.getElementById('speech_text');
                    if (speechArea) {
                        speechArea.value = text;
                        speechArea.dispatchEvent(new Event('input', { bubbles: true }));
                        if (typeof Shiny !== 'undefined') {
                            Shiny.setInputValue('speech_text', text);
                        }
                        speechArea.focus();
                    }
                };

                window.sonaSetSpeed = function(btn, rate) {
                    const player = btn.closest('.output-surface');
                    const audio = player ? player.querySelector('audio') : null;
                    if (audio) audio.playbackRate = rate;
                    const group = btn.closest('.speed-control-group');
                    if (group) {
                        group.querySelectorAll('.btn-speed').forEach(function(b) { b.classList.remove('active'); });
                    }
                    btn.classList.add('active');
                };

                document.addEventListener('DOMContentLoaded', function() {
                    const preview = document.getElementById('reference_preview');
                    if (!preview) return;
                    let previousSource = null;
                    new MutationObserver(function() {
                        const source = preview.querySelector('audio')?.getAttribute('src') || null;
                        if (source !== previousSource) {
                            document.getElementById('voice-setup').open = !source;
                            previousSource = source;
                        }
                    }).observe(preview, { childList: true, subtree: true });
                });

                document.addEventListener('keydown', function(e) {
                    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') {
                        const btn = document.getElementById('btn_generate');
                        if (btn && !btn.disabled) {
                            e.preventDefault();
                            btn.click();
                        }
                    }
                });
            })();
            """
        ),
    ),
    ui.div(
        {"class": "app-shell"},
        ui.tags.a("Skip to script", href="#script-heading", class_="skip-link"),
        ui.tags.header(
            {"class": "app-header"},
            ui.div(
                {"class": "brand"},
                ui.span(
                    {"class": "brand-mark", "aria-hidden": "true"},
                    ui.span({"class": "eq-bar"}),
                    ui.span({"class": "eq-bar"}),
                    ui.span({"class": "eq-bar"}),
                    ui.span({"class": "eq-bar"}),
                    ui.span({"class": "eq-bar"}),
                ),
                ui.span({"class": "brand-text"}, "Sona — Local Voice Studio"),
            ),
            ui.div(
                {"class": "privacy-note"},
                ui.span({"class": "live-dot", "aria-hidden": "true"}),
                icon_svg("shield-halved"),
                "Private session · nothing leaves your Mac",
            ),
            ui.output_ui("engine_badge"),
        ),
        ui.tags.main(
            {"class": "stage"},
            ui.div(
                {"class": "workspace"},
                ui.tags.aside(
                    {"class": "reference-pane", "aria-labelledby": "reference-heading"},
                    ui.h2(
                        {"class": "section-title", "id": "reference-heading"},
                        "Voice reference",
                    ),
                    ui.p(
                        "Record yourself, upload a file, or pick a saved voice. "
                        "The first 12 seconds create the voice profile.",
                        class_="section-copy",
                    ),
                    ui.output_ui("reference_preview"),
                    ui.tags.details(
                        ui.tags.summary("Change voice", class_="voice-setup-summary"),
                        ui.div(
                            {"class": "ref-mode-selector"},
                            ui.input_radio_buttons(
                                "ref_mode",
                                None,
                                choices={
                                    "record": "Record",
                                    "upload": "Upload",
                                    "library": "Saved voices",
                                },
                                selected="record",
                                inline=True,
                            ),
                        ),
                        ui.panel_conditional(
                            "input.ref_mode === 'record'",
                            ui.div(
                                {"class": "record-panel"},
                                ui.div(
                                    {"class": "voice-name-field"},
                                    ui.input_text(
                                        "voice_name",
                                        "Voice name",
                                        placeholder="e.g. my-voice",
                                        width="100%",
                                    ),
                                ),
                                ui.div(
                                    {"class": "template-options"},
                                    ui.input_radio_buttons(
                                        "record_template",
                                        "Recording template",
                                        choices={
                                            "standard": "Standard",
                                            "conversational": "Conversational",
                                        },
                                        selected="standard",
                                        inline=True,
                                    ),
                                ),
                                ui.div(
                                    {"class": "record-prompt-caption"},
                                    "Read this aloud at a natural pace:",
                                ),
                                ui.output_ui("recording_prompt_display"),
                                ui.div(
                                    {"class": "record-controls"},
                                    ui.tags.button(
                                        {"id": "btn-record", "type": "button", "class": "btn-record", "data-max-duration": str(MAX_RECORDING_SECONDS), "onclick": "sonaToggleRecording()"},
                                        " Start recording",
                                    ),
                                    ui.div(
                                        {"id": "record-vu-meter", "class": "record-vu-meter", "aria-label": "Audio level"},
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                    ),
                                    ui.tags.span({"id": "record-timer", "class": "record-timer"}, "0:00"),
                                ),
                                ui.div(
                                    {"class": "record-progress-track"},
                                    ui.div({"id": "record-progress-fill", "class": "record-progress-fill"}),
                                ),
                                ui.tags.div({"id": "record-status", "class": "record-status"}, "Ready"),
                            ),
                        ),
                        ui.panel_conditional(
                            "input.ref_mode === 'upload'",
                            ui.div(
                                {"class": "reference-upload"},
                                ui.input_file(
                                    "audio_file",
                                    "Choose a WAV, MP3, OGG, FLAC, or M4A file",
                                    accept=[".wav", ".mp3", ".ogg", ".flac", ".m4a"],
                                    multiple=False,
                                ),
                            ),
                        ),
                        ui.panel_conditional(
                            "input.ref_mode === 'library'",
                            ui.div(
                                {"class": "library-panel"},
                                ui.output_ui("library_selector"),
                            ),
                        ),
                        id="voice-setup", open=True,
                    ),
                    ui.output_ui("reference_transcript_section"),
                ),
                ui.tags.section(
                    {"class": "script-pane", "aria-labelledby": "script-heading"},
                    ui.h2(
                        {"class": "section-title", "id": "script-heading"},
                            "Script",
                    ),
                    ui.p(
                        "Enter the text you want to synthesize with your cloned voice.",
                        class_="section-copy",
                    ),
                    ui.input_text_area(
                        "speech_text",
                        None,
                        value="Hello! If you're hearing this, it means the voice clone worked. Every word you're hearing was spoken by a computer, in my voice, running entirely on this Mac. Pretty wild, right?",
                        placeholder="Write the words you want the cloned voice to speak…",
                        rows=9,
                        width="100%",
                    ),
                    ui.div(
                        {"class": "field-footer"},
                        ui.span("Natural punctuation helps shape the delivery."),
                        ui.output_text("character_count", inline=True),
                    ),
                    ui.div(
                        {"class": "delivery-controls"},
                        ui.div(
                            {"class": "quality-options", "title": "Qwen3-TTS 1.7B: High fidelity uses BF16 precision. Fast draft uses the smaller 8-bit model."},
                            ui.input_radio_buttons(
                                "quality",
                                "Model quality",
                                choices={
                                    "high": "High fidelity",
                                    "fast": "Fast draft",
                                },
                                selected="high",
                                inline=True,
                            ),
                        ),
                        ui.input_select(
                            "language",
                            "Output language",
                            choices={
                                "auto": "Auto detect",
                                "English": "English",
                                "Spanish": "Spanish",
                                "French": "French",
                                "German": "German",
                                "Italian": "Italian",
                                "Portuguese": "Portuguese",
                                "Chinese": "Chinese",
                                "Japanese": "Japanese",
                                "Korean": "Korean",
                                "Russian": "Russian",
                            },
                            selected="auto",
                        ),
                    ),
                    ui.tags.section(
                        {"class": "transport", "aria-label": "Audio generation progress"},
                        ui.div(
                            {"class": "transport-actions"},
                            ui.input_action_button(
                                "btn_generate",
                                ui.TagList(
                                    icon_svg("wave-square"),
                                    "Create audio",
                                    ui.span("⌘↵", class_="kbd-shortcut"),
                                ),
                                class_="btn-create w-100",
                            ),
                            ui.input_action_button(
                                "btn_cancel",
                                "Cancel generation",
                                class_="btn btn-outline-secondary btn-cancel w-100",
                                disabled=True,
                            ),
                            ui.output_text("speech_duration", inline=True),
                        ),
                        ui.output_ui("generation_progress"),
                    ),
                ),
            ),
            ui.tags.section(
                {"class": "output-pane", "aria-labelledby": "output-heading"},
                ui.div(
                    {"class": "output-header"},
                    ui.h2(
                        {"class": "output-heading", "id": "output-heading"},
                        icon_svg("volume-high"),
                        "Generated output",
                    ),
                    ui.output_ui("output_status"),
                ),
                ui.output_ui("audio_result"),
            ),
        ),
    ),
    theme=shinyswatch.theme.darkly,
)


def estimate_speech_duration_seconds(text: str) -> float:
    words = len(text.strip().split()) if text.strip() else 0
    return words / 2.4


def _guess_mime_type(filename: str) -> str:
    mime_type, _ = mimetypes.guess_type(filename)
    return mime_type or "audio/wav"


def _sanitize_voice_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"\s+", "-", name)
    name = re.sub(r"[^a-z0-9_-]", "", name)
    return name.strip("-")


def _saved_voice_path(selected: str) -> Path | None:
    if not isinstance(selected, str):
        return None
    saved_names = {
        path.stem for path in VOICE_SAMPLES_DIR.glob("*.wav") if path.is_file()
    }
    if selected not in saved_names:
        return None

    path = VOICE_SAMPLES_DIR / f"{selected}.wav"
    try:
        path.resolve().relative_to(VOICE_SAMPLES_DIR.resolve())
    except ValueError:
        return None
    return path


def get_reference_id(audio_path: str | Path | None) -> str:
    if not audio_path:
        return ""
    path = Path(audio_path)
    if not path.exists():
        return str(audio_path)
    try:
        stat = path.stat()
        return f"{path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        return str(path)


def resolve_reference_transcript(
    ref_id: str,
    cache: dict[str, str],
    pending: set[str],
) -> tuple[str, str, bool]:
    if not ref_id:
        return "", "idle", False
    if ref_id in cache:
        return cache[ref_id], "ready", False
    if ref_id in pending:
        return "", "transcribing", False
    return "", "transcribing", True


def apply_transcription_result(
    ref_id: str,
    transcript: str | None,
    error: str | None,
    current_ref_id: str,
    cache: dict[str, str],
    pending: set[str],
) -> tuple[dict[str, str], set[str], bool, str, str | None]:
    new_cache = dict(cache)
    new_pending = set(pending)
    new_pending.discard(ref_id)
    is_current = bool(current_ref_id and current_ref_id == ref_id)

    if error is None and transcript is not None:
        if ref_id not in new_cache or not new_cache[ref_id].strip():
            new_cache[ref_id] = transcript
        resolved_text = new_cache[ref_id]
        return new_cache, new_pending, is_current, resolved_text, None

    resolved_text = new_cache.get(ref_id, "")
    return new_cache, new_pending, is_current, resolved_text, error


def record_user_transcript_edit(
    ref_id: str,
    edited_text: str | None,
    cache: dict[str, str],
) -> dict[str, str]:
    if not ref_id or edited_text is None:
        return cache
    new_cache = dict(cache)
    new_cache[ref_id] = edited_text
    return new_cache


def server(input, output, session):
    session_dir = Path(tempfile.gettempdir()) / "local-voice-cloning" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    session.on_ended(lambda: shutil.rmtree(session_dir, ignore_errors=True))

    output_audio_path = reactive.value(None)
    generation_stage = reactive.value(None)
    generation_started_at = reactive.value(None)
    last_recorded_path = reactive.value(None)
    last_recorded_name = reactive.value(None)
    library_refresh = reactive.value(0)
    transcript_cache = reactive.value({})
    pending_transcriptions = reactive.value(set())
    ref_transcript_value = reactive.value("")
    transcription_status = reactive.value("idle")

    @reactive.calc
    def active_reference():
        mode = input.ref_mode() if input.ref_mode() else "record"
        if mode == "upload":
            file_infos = input.audio_file()
            if file_infos:
                return (file_infos[0]["datapath"], file_infos[0]["name"])
        elif mode == "record":
            path = last_recorded_path()
            if path:
                name = last_recorded_name() or "recording"
                return (str(path), f"{name}.wav")
        elif mode == "library":
            selected = input.voice_library() or ""
            if selected:
                path = _saved_voice_path(selected)
                if path:
                    return (str(path), f"{selected}.wav")
        return None

    @reactive.extended_task
    async def run_transcription(audio_path: str, ref_id: str, quality: str):
        def work():
            try:
                transcript = get_shared_cloner(quality).transcribe(audio_path)
                return ref_id, transcript, None
            except Exception as exc:  # noqa: BLE001
                return ref_id, None, str(exc)

        return await asyncio.to_thread(work)

    @reactive.effect
    @reactive.event(run_transcription.result)
    def _handle_transcription_result():
        res = run_transcription.result()
        if not res:
            return

        ref_id, transcript, error = res
        current_ref = active_reference()
        current_ref_id = get_reference_id(current_ref[0]) if current_ref else ""

        new_cache, new_pending, is_current, resolved_text, err = apply_transcription_result(
            ref_id,
            transcript,
            error,
            current_ref_id,
            transcript_cache(),
            pending_transcriptions(),
        )
        transcript_cache.set(new_cache)
        pending_transcriptions.set(new_pending)

        if is_current:
            if err is not None:
                transcription_status.set("error")
                ui.notification_show(f"Transcription failed: {err}", type="warning")
            else:
                ref_transcript_value.set(resolved_text)
                ui.update_text_area("ref_transcript", value=resolved_text)
                transcription_status.set("ready")
                ui.notification_show("Reference transcript ready for review.", type="message")

    @reactive.effect
    @reactive.event(active_reference, ignore_init=False, ignore_none=False)
    def _sync_reference_transcript():
        ref = active_reference()
        audio_path = ref[0] if ref else None
        ref_id = get_reference_id(audio_path)
        text, status, should_run = resolve_reference_transcript(
            ref_id,
            transcript_cache(),
            pending_transcriptions(),
        )
        ref_transcript_value.set(text)
        ui.update_text_area("ref_transcript", value=text)
        transcription_status.set(status)

        if should_run and audio_path:
            new_pending = set(pending_transcriptions())
            new_pending.add(ref_id)
            pending_transcriptions.set(new_pending)
            run_transcription(audio_path, ref_id, input.quality() or "high")

    @reactive.effect
    @reactive.event(input.ref_transcript)
    def _save_user_transcript_edit():
        ref = active_reference()
        if not ref:
            return
        ref_id = get_reference_id(ref[0])
        text = input.ref_transcript()
        new_cache = record_user_transcript_edit(ref_id, text, transcript_cache())
        transcript_cache.set(new_cache)

    @render.ui
    def engine_badge():
        quality = input.quality() if input.quality() else "high"
        label = "BF16" if quality == "high" else "8-bit"
        return ui.div(
            {"class": "engine-pill", "title": f"{ENGINE_NAME} · {label} · Apple MLX"},
            icon_svg("microchip"),
            "Local engine",
        )

    @render.text
    def speech_duration():
        seconds = estimate_speech_duration_seconds(input.speech_text() or "")
        return f"~{seconds:.0f}s estimated audio" if seconds else "Add a script to begin"

    @render.text
    def character_count():
        text = input.speech_text() or ""
        chars = len(text)
        est = estimate_speech_duration_seconds(text)
        if est > 0:
            return f"{chars:,} / 5,000 characters"
        return f"{chars:,} / 5,000 chars"

    @render.ui
    def recording_prompt_display():
        choice = input.record_template() if input.record_template() else "standard"
        prompt_text = RECORDING_TEMPLATES.get(choice, RECORDING_PROMPT)
        return ui.tags.pre(
            {"class": "record-prompt"},
            prompt_text,
        )

    @reactive.calc
    def library_choices():
        library_refresh()
        return sorted(p.stem for p in VOICE_SAMPLES_DIR.glob("*.wav"))

    @render.ui
    def library_selector():
        voices = library_choices()
        if not voices:
            return ui.div(
                {"class": "library-empty"},
                "No saved voices yet. Record one in the Record tab.",
            )
        return ui.div(
            {"class": "library-controls"},
            ui.input_select(
                "voice_library",
                "Saved voices",
                choices={v: v for v in voices},
                selected=voices[0],
            ),
            ui.input_action_button(
                "btn_refresh_voices",
                icon_svg("rotate-right"),
                class_="btn btn-outline-secondary btn-sm btn-refresh-voices",
                title="Rescan directory",
            ),
            ui.input_action_button(
                "btn_delete_voice",
                icon_svg("trash"),
                class_="btn btn-outline-danger btn-sm btn-delete-voice",
                title="Delete selected voice",
            ),
        )

    @reactive.effect
    @reactive.event(input.recorded_audio_data)
    def _save_recording():
        payload = input.recorded_audio_data()
        if not payload:
            return
        data_uri = payload["data"]
        raw_name = payload.get("name", "")
        name = _sanitize_voice_name(raw_name)
        if not name:
            ui.notification_show("Voice name is empty or invalid.", type="warning")
            return
        if "," not in data_uri:
            ui.notification_show("Recording data is malformed.", type="error")
            return
        _header, b64_content = data_uri.split(",", 1)
        wav_bytes = base64.b64decode(b64_content)
        save_path = VOICE_SAMPLES_DIR / f"{name}.wav"

        # Post-process the raw recording: resample to 24 kHz, strip leading
        # and trailing silence, apply short fades, then save. Falls back to
        # writing the raw bytes if any step fails.
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(wav_bytes)
            tmp_path = Path(tmp.name)
        try:
            tensor_audio, sr = load_audio(tmp_path)
            audio_np = tensor_audio.squeeze(0).numpy()
            audio_np = trim_silence(audio_np, sr)
            audio_np = apply_fades(audio_np, sr)
            save_audio(save_path, audio_np, sample_rate=sr)
        except (OSError, RuntimeError, ValueError):
            save_path.write_bytes(wav_bytes)
        finally:
            tmp_path.unlink(missing_ok=True)

        last_recorded_path.set(str(save_path))
        last_recorded_name.set(name)
        library_refresh.set(library_refresh() + 1)
        ui.notification_show(f"Saved voice profile '{name}'.", type="message")

    @reactive.effect
    @reactive.event(input.btn_refresh_voices)
    def _refresh_voices():
        library_refresh.set(library_refresh() + 1)

    @reactive.effect
    @reactive.event(input.btn_delete_voice)
    def _delete_voice():
        selected = input.voice_library() or ""
        if not selected:
            return
        path = _saved_voice_path(selected)
        if path:
            path.unlink()
            library_refresh.set(library_refresh() + 1)
            ui.notification_show(f"Deleted voice profile '{selected}'.", type="message")

    @reactive.effect
    @reactive.event(input.btn_retranscribe)
    def _handle_retranscribe():
        ref = active_reference()
        if not ref:
            ui.notification_show("No reference audio loaded to transcribe.", type="warning")
            return
        audio_path = ref[0]
        ref_id = get_reference_id(audio_path)
        new_pending = set(pending_transcriptions())
        new_pending.add(ref_id)
        pending_transcriptions.set(new_pending)
        transcription_status.set("transcribing")
        new_cache = dict(transcript_cache())
        new_cache.pop(ref_id, None)
        transcript_cache.set(new_cache)
        run_transcription(audio_path, ref_id, input.quality() or "high")

    @render.ui
    def reference_preview():
        ref = active_reference()
        if not ref:
            return ui.div(
                {"class": "reference-empty"},
                icon_svg("file-audio"),
                ui.strong("No reference loaded"),
                ui.span("Record, upload, or select a saved voice to begin."),
            )

        datapath, display_name = ref
        mime_type = _guess_mime_type(display_name)
        with open(datapath, "rb") as audio_file:
            b64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

        try:
            report = analyze_reference_audio(datapath)
        except (ValueError, OSError, RuntimeError):
            report = None

        duration = report["duration_seconds"] if report else 0.0
        sample_rate = report["sample_rate"] if report else 0
        used = min(duration, 12.0)

        quality_pills = []
        if report is not None:
            if not report["warnings"]:
                quality_pills.append(ui.span({"class": "quality-pill good"}, icon_svg("circle-check"), "Clean levels"))
            for w in report["warnings"]:
                quality_pills.append(ui.span({"class": "quality-pill warn"}, icon_svg("triangle-exclamation"), w))
        quality_feedback = ui.div({"class": "quality-pill-group"}, *quality_pills) if quality_pills else ui.div()

        return ui.div(
            {"class": "reference-file"},
            ui.div({"class": "file-name"}, display_name),
            ui.div(
                {"class": "file-caption"},
                f"{duration:.1f}s recording · first {used:.1f}s used",
            ),
            ui.tags.audio(
                controls=True,
                preload="metadata",
                src=f"data:{mime_type};base64,{b64_audio}",
            ),
            ui.div(
                {"class": "meta-list"},
                ui.div({"class": "meta-row"}, ui.span("Format"), ui.span(Path(display_name).suffix.lstrip(".").upper() or "Audio")),
                ui.div({"class": "meta-row"}, ui.span("Sample rate"), ui.span(f"{sample_rate:,} Hz")),
                ui.div({"class": "meta-row"}, ui.span("Profile window"), ui.span(f"00:00 – 00:{used:04.1f}")),
            ),
            quality_feedback,
        )

    @render.ui
    def reference_transcript_section():
        ref = active_reference()
        if not ref:
            return ui.div()

        status = transcription_status()
        if status == "transcribing":
            status_badge = ui.span({"class": "transcript-status transcribing"}, icon_svg("spinner"), "Transcribing...")
            content = ui.div(
                {"class": "transcript-shimmer"},
                ui.div({"class": "shimmer-line"}),
                ui.div({"class": "shimmer-line short"}),
            )
            action_buttons = []
        elif status == "ready":
            status_badge = ui.span({"class": "transcript-status ready"}, icon_svg("circle-check"), "Ready for review")
            content = ui.input_text_area(
                "ref_transcript",
                None,
                value=ref_transcript_value(),
                placeholder="Exact words spoken in the reference recording...",
                rows=3,
                width="100%",
            )
            action_buttons = [
                ui.tags.button(
                    ui.TagList(icon_svg("copy"), " Copy"),
                    type="button",
                    class_="btn-transcript-action",
                    onclick="sonaCopyTranscript(this)",
                    title="Copy transcript to clipboard",
                ),
                ui.tags.button(
                    ui.TagList(icon_svg("pen"), " Use as script"),
                    type="button",
                    class_="btn-transcript-action",
                    onclick="sonaUseAsScript()",
                    title="Paste transcript into script area",
                ),
            ]
        elif status == "error":
            status_badge = ui.span({"class": "transcript-status"}, icon_svg("triangle-exclamation"), "Transcription failed")
            content = ui.input_text_area(
                "ref_transcript",
                None,
                value=ref_transcript_value(),
                placeholder="Exact words spoken in the reference recording...",
                rows=3,
                width="100%",
            )
            action_buttons = []
        else:
            status_badge = ui.span({"class": "transcript-status"}, "Editable")
            content = ui.input_text_area(
                "ref_transcript",
                None,
                value=ref_transcript_value(),
                placeholder="Exact words spoken in the reference recording...",
                rows=3,
                width="100%",
            )
            action_buttons = []

        return ui.tags.details(
            {"class": "transcript-card", "open": status == "error"},
            ui.tags.summary("Review reference transcript"),
            ui.div(
                {"class": "transcript-card-header"},
                ui.div(
                    {"class": "transcript-title"},
                    icon_svg("file-lines"),
                    "Reference transcript",
                ),
                ui.div(
                    {"class": "transcript-actions"},
                    *action_buttons,
                    ui.input_action_button(
                        "btn_retranscribe",
                        ui.TagList(icon_svg("rotate-right"), " Re-transcribe"),
                        class_="btn-transcript-action",
                    ),
                ),
            ),
            ui.div(
                "Review words detected in your reference audio. You can edit any incorrect words before cloning.",
                class_="transcript-card-caption",
            ),
            content,
            ui.div(
                {"class": "transcript-footer"},
                status_badge,
                ui.span("Passes to voice cloner", class_="transcript-status"),
            ),
        )

    @reactive.extended_task
    async def run_synthesis(
        ref_path: str,
        text: str,
        ref_text: str,
        quality: str,
        language: str,
    ):
        def work(report):
            return get_shared_cloner(quality).clone_voice(
                reference_audio_path=ref_path,
                text=text,
                reference_text=ref_text,
                language=language,
                progress_callback=report,
            )

        return await run_with_progress(work, generation_stage.set)

    @reactive.effect
    @reactive.event(input.btn_generate)
    def handle_synthesis():
        ref = active_reference()
        text = input.speech_text()
        if not ref:
            ui.notification_show("Add a reference recording before creating audio.", type="warning")
            return
        if not text.strip():
            ui.notification_show("Write a script before creating audio.", type="warning")
            return

        ref_text = (input.ref_transcript() or "").strip()

        output_audio_path.set(None)
        generation_stage.set("prepare")
        generation_started_at.set(time.monotonic())
        run_synthesis(
            ref[0],
            text,
            ref_text,
            input.quality() or "high",
            input.language() or "auto",
        )

    @reactive.effect
    @reactive.event(input.btn_cancel)
    def handle_cancel():
        run_synthesis.cancel()
        ui.notification_show("Generation cancelled.", type="warning")

    @reactive.effect
    def _toggle_buttons():
        running = run_synthesis.status() == "running"
        ui.update_action_button("btn_generate", disabled=running)
        ui.update_action_button("btn_cancel", disabled=not running)

    @reactive.effect
    def _save_result():
        if run_synthesis.status() != "success":
            return
        result = run_synthesis.result()
        generation_id = uuid.uuid4().hex[:8]
        wav_file = session_dir / f"clone_{generation_id}.wav"
        mp3_file = session_dir / f"clone_{generation_id}.mp3"
        save_audio(wav_file, result.audio, sample_rate=result.sample_rate)
        save_audio(mp3_file, result.audio, sample_rate=result.sample_rate)
        output_audio_path.set(str(wav_file))
        ui.notification_show("Your cloned voice is ready.", type="message")

    @reactive.effect
    def _report_error():
        if run_synthesis.status() != "error":
            return
        ui.notification_show(f"Synthesis failed: {run_synthesis.error()!s}", type="error")

    @render.ui
    def generation_progress():
        status = run_synthesis.status()
        if status not in {"running", "error"}:
            return None
        stage = generation_stage()
        snapshot = progress_snapshot(stage, status)
        active_index = next((index for index, item in enumerate(snapshot) if item.state in {"active", "error"}), 0)
        if status == "success":
            fill_width = 80
            message, detail = "Audio ready", "The cloned voice is ready to play and download."
        elif status == "error":
            fill_width = (active_index / 3) * 80
            message, detail = "Generation stopped", "Review the error notification and try again."
        elif status == "running":
            fill_width = (active_index / 3) * 80
            message, detail = "Synthesizing natural speech", snapshot[active_index].detail
        else:
            fill_width = 0
            message, detail = "Ready to create", "Progress will update here during synthesis."

        started = generation_started_at()
        elapsed = 0
        if started is not None:
            if status == "running":
                reactive.invalidate_later(1)
            elapsed = max(0, int(time.monotonic() - started))

        stage_nodes = []
        for step in snapshot:
            if step.state == "complete":
                marker = icon_svg("check")
            elif step.state == "error":
                marker = icon_svg("exclamation")
            else:
                marker = icon_svg("circle")
            stage_nodes.append(
                ui.div(
                    {"class": f"stage-step {step.state}"},
                    ui.div({"class": "stage-dot"}, marker),
                    ui.div({"class": "stage-label"}, step.label),
                    ui.div({"class": "stage-detail"}, step.detail),
                )
            )

        return ui.div(
            {"class": "progress-region"},
            ui.div(
                {"class": "progress-top"},
                ui.div(
                    ui.div({"class": "progress-message"}, message),
                    ui.div({"class": "progress-detail", "aria-live": "polite"}, detail),
                ),
                ui.div(
                    ui.div({"class": "progress-detail"}, "Elapsed"),
                    ui.div({"class": "elapsed"}, f"{elapsed // 60:02d}:{elapsed % 60:02d}"),
                ),
            ),
            ui.div(
                {"class": "stage-track"},
                ui.div({"class": "stage-fill", "style": f"width: {fill_width:.1f}%"}),
                *stage_nodes,
            ),
        )

    @render.ui
    def output_status():
        if run_synthesis.status() == "success" and output_audio_path():
            started = generation_started_at()
            elapsed_str = ""
            if started is not None:
                elapsed = max(0.1, time.monotonic() - started)
                elapsed_str = f" in {elapsed:.1f}s"
            return ui.div(
                {"class": "status-chip ready"},
                ui.span({"class": "status-dot"}),
                f"Ready to play{elapsed_str}",
            )
        return ui.div(
            {"class": "status-chip"},
            ui.span({"class": "status-dot"}),
            "Waiting for audio",
        )

    @render.ui
    def audio_result():
        path = output_audio_path()
        if run_synthesis.status() != "success" or not path or not Path(path).exists():
            return ui.div(
                {"class": "output-surface output-empty"},
                icon_svg("music"),
                ui.div(
                    ui.strong("Your synthesized audio will appear here"),
                    ui.div("Playback and downloads unlock when generation finishes.", class_="file-caption"),
                ),
            )

        wav_path = Path(path)
        mp3_path = wav_path.with_suffix(".mp3")
        with open(wav_path, "rb") as wav_file:
            b64_wav = base64.b64encode(wav_file.read()).decode("utf-8")

        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        amplitudes = np.max(np.abs(samples), axis=1)
        peaks = [float(chunk.max()) if len(chunk) else 0.0 for chunk in np.array_split(amplitudes, 100)]
        maximum = max(max(peaks), 1e-9)
        bars = "".join(
            f'<rect x="{index * 6}" y="{32 - max(1, peak / maximum * 30):.2f}" '
            f'width="3" height="{max(2, peak / maximum * 60):.2f}" rx="1.5" />'
            for index, peak in enumerate(peaks)
        )
        waveform = ui.HTML(f'<svg viewBox="0 0 600 64" preserveAspectRatio="none" aria-hidden="true">{bars}</svg>')
        duration = len(samples) / sample_rate

        buttons = [
            ui.tags.a(
                icon_svg("download"),
                "Download WAV",
                href=f"data:audio/wav;base64,{b64_wav}",
                download="cloned_voice_output.wav",
                class_="btn btn-download",
            )
        ]
        if mp3_path.exists():
            with open(mp3_path, "rb") as mp3_file:
                b64_mp3 = base64.b64encode(mp3_file.read()).decode("utf-8")
            buttons.append(
                ui.tags.a(
                    icon_svg("download"),
                    "Download MP3",
                    href=f"data:audio/mpeg;base64,{b64_mp3}",
                    download="cloned_voice_output.mp3",
                    class_="btn btn-download",
                )
            )

        return ui.div(
            {"class": "output-surface"},
            ui.div(ui.strong("Your voice, rendered"), ui.span(f"{duration:.1f}s", class_="audio-duration"), class_="result-title"),
            ui.div(waveform, class_="audio-waveform"),
            ui.div(
                {"class": "result-player"},
                ui.tags.audio(
                    controls=True,
                    autoplay=True,
                    src=f"data:audio/wav;base64,{b64_wav}",
                ),
                ui.div({"class": "download-group"}, *buttons),
            ),
            ui.div(
                {"class": "speed-control-group"},
                ui.span("Playback speed:", class_="speed-label"),
                ui.tags.button("0.8×", type="button", class_="btn-speed", onclick="sonaSetSpeed(this, 0.8)"),
                ui.tags.button("1.0×", type="button", class_="btn-speed active", onclick="sonaSetSpeed(this, 1.0)"),
                ui.tags.button("1.25×", type="button", class_="btn-speed", onclick="sonaSetSpeed(this, 1.25)"),
                ui.tags.button("1.5×", type="button", class_="btn-speed", onclick="sonaSetSpeed(this, 1.5)"),
            ),
        )


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")
