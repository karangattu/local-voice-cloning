import asyncio
import base64
import inspect
import json
import mimetypes
import os
import re
import shutil
import sys
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import numpy as np
import shinyswatch
import soundfile as sf
from faicons import icon_svg
from shiny import App, _utils, reactive, render, ui
from starlette.responses import FileResponse, Response


def _configure_shiny_port(
    config: object | None = None,
    argv: list[str] | None = None,
    environ: dict[str, str] | None = None,
) -> int | None:
    try:
        args = sys.argv if argv is None else argv
        has_explicit_port = any(
            arg in ("-p", "--port") or arg.startswith(("--port=", "-p=")) for arg in args
        )
        if has_explicit_port:
            return None

        env = os.environ if environ is None else environ
        env_port = env.get("SHINY_PORT") or env.get("PORT")

        cfg = config
        if cfg is None:
            for frame_info in inspect.stack():
                locals_dict = frame_info.frame.f_locals
                if "config" in locals_dict:
                    candidate = locals_dict["config"]
                    if getattr(candidate, "port", None) == 8000:
                        cfg = candidate
                        break

        if cfg is not None and getattr(cfg, "port", None) == 8000:
            if env_port and env_port.isdigit() and int(env_port) > 0:
                cfg.port = int(env_port)
            else:
                cfg.port = _utils.random_port(host=getattr(cfg, "host", "127.0.0.1"))
            return cfg.port
    except (AttributeError, LookupError, OSError, RuntimeError, ValueError):
        return None
    return None


_configure_shiny_port()

from src.audio_utils import (
    analyze_reference_audio,
    apply_fades,
    load_audio,
    save_audio,
    slice_audio,
    trim_silence,
)
from src.cloner import ENGINE_NAME, ENGINES, get_shared_cloner, unload_shared_cloners
from src.progress import progress_snapshot, run_with_progress

VOICE_SAMPLES_DIR = Path(__file__).parent / "voice_samples"
VOICE_SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_REF_MODE = "library" if any(VOICE_SAMPLES_DIR.glob("*.wav")) else "record"

RECORDING_TEMPLATES = {
    "standard": """Hi, I'm [your name], and this is my natural speaking voice. The quick brown fox jumps over the lazy dog. How vexingly quick daft zebras jump! Did it capture the real me?""",
    "conversational": """Hi, I’m [name]. Today is a beautiful day, and I’m feeling pretty good. Can you believe it? I have three things to finish, then I’m heading home.""",
}
RECORDING_PROMPT = RECORDING_TEMPLATES["standard"]
MAX_RECORDING_SECONDS = 30
MAX_SCRIPT_CHARACTERS = 5000

SCRIPT_PRESETS = {
    "custom": "Choose a preset…",
    "default": "Default Studio Demo",
    "narration": "Audiobook Narration",
    "podcast": "Podcast Intro",
    "voicemail": "Voicemail Greeting",
    "technical": "Technical Walkthrough",
    "multilingual": "Multilingual Greeting",
}

SCRIPT_PRESET_TEXTS = {
    "default": "Hello! If you're hearing this, it means the voice clone worked. Every word you're hearing was spoken by a computer, in my voice, running entirely on this Mac. Pretty wild, right?",
    "narration": "The ship drifted silently through the rings of Saturn. Below them, ribbons of ice and dust reflected the pale light of a distant Sun. Captain Miller checked the navigational array one last time.",
    "podcast": "Welcome back to the show. Today, we're diving deep into the world of local artificial intelligence, open-source audio models, and running generative voice models right on your laptop.",
    "voicemail": "Hi, you've reached my voicemail. I can't take your call right now, but please leave your name, number, and a brief message after the tone. I'll get back to you as soon as I can.",
    "technical": "To start the application, open your terminal and run 'uv run app.py'. The model will compile local Metal kernels on your Apple Silicon chip, enabling low-latency neural synthesis without an internet connection.",
    "multilingual": "Bonjour! Hallo! Ciao! This is a test of multilingual synthesis running completely locally on this device.",
}

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
                let isProcessing = false;
                let vuData = null;
                let vuBars = [];
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
                    const dataArray = vuData;
                    analyserNode.getByteFrequencyData(dataArray);
                    let sum = 0;
                    for (let i = 0; i < dataArray.length; i++) sum += dataArray[i];
                    const avg = sum / dataArray.length;
                    const normalized = Math.min(1, avg / 70);
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

                function finishRecording(message) {
                    isProcessing = false;
                    document.getElementById('btn-record').disabled = false;
                    setStatus(message);
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

                let pendingTake = null;
                window.sonaAcceptTake = function() {
                    if (!pendingTake) return;
                    setStatus('Processing...');
                    if (typeof Shiny !== 'undefined') {
                        Shiny.setInputValue('recorded_audio_data', pendingTake, {priority: 'event'});
                    }
                    const previewEl = document.getElementById('record-take-preview');
                    if (previewEl) previewEl.style.display = 'none';
                    pendingTake = null;
                };

                window.sonaDiscardTake = function() {
                    pendingTake = null;
                    const previewEl = document.getElementById('record-take-preview');
                    if (previewEl) previewEl.style.display = 'none';
                    const audioEl = document.getElementById('record-preview-audio');
                    if (audioEl) { audioEl.pause(); audioEl.src = ''; }
                    finishRecording('Ready for a new take');
                };

                function populateMicDevices() {
                    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
                    navigator.mediaDevices.enumerateDevices().then(function(devices) {
                        const select = document.getElementById('mic-device-select');
                        if (!select) return;
                        const audioInputs = devices.filter(function(d) { return d.kind === 'audioinput'; });
                        if (audioInputs.length === 0) return;
                        const currentVal = select.value;
                        select.innerHTML = '<option value="">Default microphone</option>';
                        audioInputs.forEach(function(dev, idx) {
                            const opt = document.createElement('option');
                            opt.value = dev.deviceId;
                            opt.text = dev.label || ('Microphone ' + (idx + 1));
                            if (dev.deviceId === currentVal) opt.selected = true;
                            select.appendChild(opt);
                        });
                    }).catch(function() {});
                }

                window.sonaToggleRecording = function() {
                    if (isProcessing) return;
                    if (isRecording) {
                        sonaStopRecording();
                    } else {
                        sonaStartRecording();
                    }
                };

                function sonaStartRecording() {
                    const nameInput = document.getElementById('voice_name');
                    if (!nameInput) return;
                    if (!sanitizeName(nameInput.value)) {
                        const now = new Date();
                        const pad = function(n) { return String(n).padStart(2, '0'); };
                        nameInput.value = 'voice-' + pad(now.getMonth() + 1) + pad(now.getDate())
                            + '-' + pad(now.getHours()) + pad(now.getMinutes());
                        nameInput.dispatchEvent(new Event('change', { bubbles: true }));
                    }
                    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                        setStatus('Recording is not supported in this browser.');
                        return;
                    }
                    document.getElementById("record-script").open = true;
                    isProcessing = true;
                    document.getElementById("btn-record").disabled = true;
                    const micSelect = document.getElementById('mic-device-select');
                    const audioConstraint = {
                        echoCancellation: false,
                        noiseSuppression: false,
                        autoGainControl: false,
                        channelCount: 1
                    };
                    if (micSelect && micSelect.value) {
                        audioConstraint.deviceId = { exact: micSelect.value };
                    }
                    // Browser voice processing gates quiet speech; the clone copies those gaps.
                    navigator.mediaDevices.getUserMedia({ audio: audioConstraint })
                        .then(function(stream) {
                            mediaStream = stream;
                            audioContext = new (window.AudioContext || window.webkitAudioContext)();
                            const source = audioContext.createMediaStreamSource(stream);
                            analyserNode = audioContext.createAnalyser();
                            analyserNode.fftSize = 64;
                            vuData = new Uint8Array(analyserNode.frequencyBinCount);
                            vuBars = document.querySelectorAll("#record-vu-meter .vu-bar");
                            source.connect(analyserNode);

                            mediaRecorder = new MediaRecorder(stream);
                            audioChunks = [];
                            mediaRecorder.ondataavailable = function(e) {
                                if (e.data.size > 0) audioChunks.push(e.data);
                            };
                            mediaRecorder.onstop = function() {
                                let awaitingSave = false;
                                var blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
                                blob.arrayBuffer().then(function(buf) {
                                    return audioContext.decodeAudioData(buf);
                                }).then(function(audioBuffer) {
                                    var wavBlob = encodeWav(audioBuffer);
                                    var reader = new FileReader();
                                    reader.onload = function() {
                                        var name = sanitizeName(document.getElementById('voice_name').value);
                                        pendingTake = { data: reader.result, name: name };
                                        const audioEl = document.getElementById('record-preview-audio');
                                        if (audioEl) audioEl.src = URL.createObjectURL(wavBlob);
                                        const previewEl = document.getElementById('record-take-preview');
                                        if (previewEl) previewEl.style.display = 'block';
                                        finishRecording('Audition your take below, then save or discard.');
                                    };
                                    reader.onerror = function() {
                                        finishRecording('Could not read recording. Please try again.');
                                    };
                                    awaitingSave = true;
                                    reader.readAsDataURL(wavBlob);
                                }).catch(function(err) {
                                    awaitingSave = false;
                                    setStatus('Could not process recording: ' + err.message);
                                }).finally(function() {
                                    if (mediaStream) mediaStream.getTracks().forEach(function(t) { t.stop(); });
                                    if (audioContext) { audioContext.close(); audioContext = null; }
                                    analyserNode = null;
                                    if (!awaitingSave) {
                                        isProcessing = false;
                                        document.getElementById("btn-record").disabled = false;
                                    }
                                });
                            };
                            mediaRecorder.start();
                            isRecording = true;
                            isProcessing = false;
                            document.getElementById("btn-record").disabled = false;
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
                            isProcessing = false;
                            document.getElementById("btn-record").disabled = false;
                            if (mediaStream) mediaStream.getTracks().forEach(function(t) { t.stop(); });
                            if (audioContext) { audioContext.close(); audioContext = null; }
                            setStatus("Could not start recording: " + err.message);
                            setButtonState(false);
                        });
                }

                function sonaStopRecording(statusMessage) {
                    if (!isRecording) return;
                    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
                        mediaRecorder.stop();
                    }
                    isRecording = false;
                    isProcessing = true;
                    document.getElementById("btn-record").disabled = true;
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
                    document.addEventListener('input', function(e) {
                        const el = e.target;
                        if (!['ref_transcript', 'ref_trim_start', 'ref_trim_end'].includes(el.id)) return;
                        const value = el.id === 'ref_transcript' ? el.value : Number(el.value);
                        Shiny.setInputValue(el.id, value, {priority: 'event'});
                    });
                    populateMicDevices();
                    if (navigator.mediaDevices && navigator.mediaDevices.addEventListener) {
                        navigator.mediaDevices.addEventListener('devicechange', populateMicDevices);
                    }
                    document.addEventListener('click', function(e) {
                        const waveform = e.target.closest('.audio-waveform');
                        if (!waveform) return;
                        const rect = waveform.getBoundingClientRect();
                        const clickX = e.clientX - rect.left;
                        const pct = Math.max(0, Math.min(1, clickX / rect.width));
                        const player = waveform.closest('.output-surface')?.querySelector('audio');
                        if (player && player.duration) {
                            player.currentTime = pct * player.duration;
                            player.play();
                        }
                    });
                    Shiny.addCustomMessageHandler('recording-status', function(message) {
                        finishRecording(message.text);
                    });
                    const recordingScript = document.getElementById('record-script');
                    if (recordingScript && window.matchMedia('(max-width: 620px)').matches) {
                        recordingScript.open = false;
                    }
                    const result = document.getElementById('audio_result');
                    if (result) {
                        let previousResult = null;
                        new MutationObserver(function() {
                            const source = result.querySelector('audio')?.getAttribute('src') || null;
                            if (source && source !== previousResult) {
                                result.closest('.output-pane')?.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                            }
                            previousResult = source;
                        }).observe(result, { childList: true, subtree: true });
                    }
                    const script = document.getElementById('speech_text');
                    if (script) {
                        script.setAttribute('aria-label', 'Script to synthesize');
                        script.setAttribute('aria-describedby', 'character_count');
                        script.addEventListener('input', function() {
                            const invalid = script.value.length > 5000;
                            script.setAttribute('aria-invalid', String(invalid));
                            Shiny.setInputValue('speech_text', script.value, {priority: 'event'});
                        });
                    }
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
            ui.div(
                {"class": "header-actions"},
                ui.output_ui("engine_badge"),
                ui.input_action_button(
                    "btn_unload_models",
                    ui.TagList(icon_svg("trash-can"), " Free RAM"),
                    class_="btn btn-sm btn-outline-secondary btn-memory-unload",
                    title="Unload models from unified memory / GPU cache",
                ),
            ),
        ),
        ui.tags.main(
            {"class": "stage"},
            ui.div(
                {"class": "workspace"},
                ui.tags.aside(
                    {"class": "reference-pane", "aria-labelledby": "reference-heading"},
                    ui.h2(
                        {"class": "section-title", "id": "reference-heading"},
                        ui.span("1", class_="step-num"),
                        "Choose a voice",
                    ),
                    ui.p(
                        "Pick a saved voice, record a new one, or upload a file. "
                        "Use 5 to 12 seconds of clear speech.",
                        class_="section-copy",
                    ),
                    ui.div(
                        {"class": "voice-setup"},
                        ui.div(
                            {"class": "ref-mode-selector"},
                            ui.input_radio_buttons(
                                "ref_mode",
                                None,
                                choices={
                                    "library": "Saved voices",
                                    "record": "Record",
                                    "upload": "Upload",
                                },
                                selected=DEFAULT_REF_MODE,
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
                                        placeholder="Optional, e.g. my-voice",
                                        width="100%",
                                    ),
                                ),
                                ui.div(
                                    {"class": "mic-device-field"},
                                    ui.tags.label(
                                        "Microphone",
                                        {"for": "mic-device-select", "class": "control-label"},
                                    ),
                                    ui.tags.select(
                                        {
                                            "id": "mic-device-select",
                                            "class": "form-select mic-select",
                                        },
                                        ui.tags.option("Default microphone", value=""),
                                    ),
                                ),
                                ui.tags.details(
                                    ui.tags.summary("Read-aloud script"),
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
                                    id="record-script",
                                    class_="record-script",
                                    open=True,
                                ),
                                ui.div(
                                    {"class": "record-controls"},
                                    ui.tags.button(
                                        {
                                            "id": "btn-record",
                                            "type": "button",
                                            "class": "btn-record",
                                            "data-max-duration": str(MAX_RECORDING_SECONDS),
                                            "onclick": "sonaToggleRecording()",
                                        },
                                        " Start recording",
                                    ),
                                    ui.div(
                                        {
                                            "id": "record-vu-meter",
                                            "class": "record-vu-meter",
                                            "aria-label": "Audio level",
                                        },
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                        ui.tags.span({"class": "vu-bar"}),
                                    ),
                                    ui.tags.span(
                                        {"id": "record-timer", "class": "record-timer"}, "0:00"
                                    ),
                                ),
                                ui.div(
                                    {"class": "record-progress-track"},
                                    ui.div(
                                        {
                                            "id": "record-progress-fill",
                                            "class": "record-progress-fill",
                                        }
                                    ),
                                ),
                                ui.tags.div(
                                    {
                                        "id": "record-status",
                                        "class": "record-status",
                                        "role": "status",
                                    },
                                    "Ready",
                                ),
                                ui.div(
                                    {
                                        "id": "record-take-preview",
                                        "class": "record-take-preview",
                                        "style": "display: none;",
                                    },
                                    ui.div(
                                        "Audition your recording before saving:",
                                        class_="preview-caption",
                                    ),
                                    ui.tags.audio(
                                        {
                                            "id": "record-preview-audio",
                                            "controls": "controls",
                                            "class": "w-100",
                                        }
                                    ),
                                    ui.div(
                                        {"class": "preview-actions"},
                                        ui.tags.button(
                                            "✓ Accept & Save Take",
                                            {
                                                "id": "btn-accept-take",
                                                "type": "button",
                                                "class": "btn btn-sm btn-success",
                                                "onclick": "sonaAcceptTake()",
                                            },
                                        ),
                                        ui.tags.button(
                                            "✕ Discard Take",
                                            {
                                                "id": "btn-discard-take",
                                                "type": "button",
                                                "class": "btn btn-sm btn-outline-danger",
                                                "onclick": "sonaDiscardTake()",
                                            },
                                        ),
                                    ),
                                ),
                                ui.output_ui("recorded_take_actions"),
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
                                ui.div(
                                    {"class": "library-import-section"},
                                    ui.input_file(
                                        "import_voice_files",
                                        "Import voices (audio or .zip)",
                                        accept=[".wav", ".mp3", ".ogg", ".flac", ".m4a", ".zip"],
                                        multiple=True,
                                        button_label="Import...",
                                        placeholder="Choose audio or zip archive",
                                    ),
                                ),
                            ),
                        ),
                    ),
                    ui.output_ui("reference_preview"),
                    ui.output_ui("reference_transcript_section"),
                ),
                ui.div(
                    {"class": "main-column"},
                    ui.tags.section(
                        {"class": "script-pane", "aria-labelledby": "script-heading"},
                        ui.div(
                            {"class": "script-header-row"},
                            ui.div(
                                ui.h2(
                                    {"class": "section-title", "id": "script-heading"},
                                    ui.span("2", class_="step-num"),
                                    "Write your script",
                                ),
                                ui.p(
                                    "Type the words for the cloned voice to say. Use [pause 0.5s] or newlines to shape pauses.",
                                    class_="section-copy",
                                ),
                            ),
                            ui.div(
                                {"class": "script-preset-field"},
                                ui.input_select(
                                    "script_preset",
                                    "Preset template",
                                    choices=SCRIPT_PRESETS,
                                    selected="custom",
                                ),
                            ),
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
                            ui.input_select(
                                "engine", "Voice engine", choices=ENGINES, selected="qwen"
                            ),
                            ui.div(
                                {
                                    "class": "quality-options",
                                    "title": "High fidelity prioritizes quality. Fast draft uses a smaller Qwen model, fewer OmniVoice steps, or Chatterbox-Turbo.",
                                },
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
                            ui.input_slider(
                                "synthesis_speed",
                                "Speaking pace",
                                min=0.6,
                                max=1.5,
                                value=1.0,
                                step=0.05,
                            ),
                            ui.panel_conditional(
                                "input.engine === 'qwen'",
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
                            ui.panel_conditional(
                                "input.engine === 'omnivoice'",
                                ui.input_text(
                                    "omni_language",
                                    "Output language",
                                    value="auto",
                                    placeholder="auto, Hindi, ar, …",
                                ),
                            ),
                        ),
                        ui.panel_conditional(
                            "input.engine === 'omnivoice'",
                            ui.p(
                                "OmniVoice supports more than 600 languages. The first use downloads its model. "
                                "A reference of 3 to 10 seconds gives the best result.",
                                class_="engine-note",
                            ),
                        ),
                        ui.panel_conditional(
                            "input.engine === 'chatterbox'",
                            ui.p(
                                "Chatterbox clones English voices; High fidelity uses the full model "
                                "and Fast draft uses Chatterbox-Turbo. The first use downloads its model. "
                                "A reference of 5 to 20 seconds gives the best result.",
                                class_="engine-note",
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
                                    aria_describedby="generation_hint",
                                ),
                                ui.input_action_button(
                                    "btn_cancel",
                                    "Cancel generation",
                                    class_="btn btn-outline-secondary btn-cancel w-100",
                                    disabled=True,
                                ),
                                ui.div(
                                    ui.output_text("generation_hint", inline=True),
                                    ui.output_text("speech_duration", inline=True),
                                    class_="generation-guidance",
                                ),
                            ),
                            ui.output_ui("generation_progress"),
                        ),
                    ),
                    ui.tags.section(
                        {"class": "output-pane", "aria-labelledby": "output-heading"},
                        ui.div(
                            {"class": "output-header"},
                            ui.h2(
                                {"class": "output-heading", "id": "output-heading"},
                                ui.span("3", class_="step-num"),
                                "Listen and download",
                            ),
                            ui.output_ui("output_status"),
                        ),
                        ui.output_ui("audio_result"),
                        ui.output_ui("session_history_ui"),
                        ui.output_ui("ab_comparison_ui"),
                    ),
                ),
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


def _saved_voice_path(selected: str, voices_dir: Path | None = None) -> Path | None:
    if not isinstance(selected, str):
        return None
    dir_path = VOICE_SAMPLES_DIR if voices_dir is None else Path(voices_dir)
    saved_names = {path.stem for path in dir_path.glob("*.wav") if path.is_file()}
    if selected not in saved_names:
        return None

    path = dir_path / f"{selected}.wav"
    try:
        path.resolve().relative_to(dir_path.resolve())
    except ValueError:
        return None
    return path


def create_voices_zip(
    output_path: Path,
    voice_names: list[str] | None = None,
    voices_dir: Path | None = None,
) -> Path:
    dir_path = VOICE_SAMPLES_DIR if voices_dir is None else Path(voices_dir)
    if voice_names is None:
        voice_files = sorted(dir_path.glob("*.wav"))
    else:
        name_set = set(voice_names)
        voice_files = sorted(p for p in dir_path.glob("*.wav") if p.stem in name_set)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for vf in voice_files:
            if vf.is_file():
                zf.write(vf, arcname=vf.name)
    return output_path


def _load_voice_metadata(name: str, voices_dir: Path | None = None) -> dict:
    if not isinstance(name, str):
        return {}
    dir_path = VOICE_SAMPLES_DIR if voices_dir is None else Path(voices_dir)
    safe_name = _sanitize_voice_name(name)
    if not safe_name:
        return {}
    json_path = dir_path / f"{safe_name}.json"
    if json_path.is_file():
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
            return metadata if isinstance(metadata, dict) else {}
        except (OSError, ValueError):
            return {}
    return {}


def _save_voice_metadata(name: str, metadata: dict, voices_dir: Path | None = None) -> None:
    if not isinstance(name, str):
        return
    dir_path = VOICE_SAMPLES_DIR if voices_dir is None else Path(voices_dir)
    safe_name = _sanitize_voice_name(name)
    if not safe_name:
        return
    existing = _load_voice_metadata(safe_name, dir_path)
    existing.update(metadata)
    if "transcript" in metadata:
        existing["reference_id"] = get_reference_id(dir_path / f"{safe_name}.wav")
    json_path = dir_path / f"{safe_name}.json"
    try:
        json_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    except OSError:
        pass


def _rename_voice_profile(old_name: str, new_name: str, voices_dir: Path | None = None) -> str:
    dir_path = VOICE_SAMPLES_DIR if voices_dir is None else Path(voices_dir)
    clean_old = _sanitize_voice_name(old_name)
    clean_new = _sanitize_voice_name(new_name)
    if not clean_old or not clean_new:
        raise ValueError("Invalid voice name for rename.")
    old_wav = dir_path / f"{clean_old}.wav"
    if not old_wav.is_file():
        raise FileNotFoundError(f"Voice '{clean_old}' not found.")
    new_wav = dir_path / f"{clean_new}.wav"
    if new_wav.exists() and clean_old != clean_new:
        raise FileExistsError(f"A voice named '{clean_new}' already exists.")

    old_wav.rename(new_wav)
    old_json = dir_path / f"{clean_old}.json"
    new_json = dir_path / f"{clean_new}.json"
    if old_json.is_file():
        old_json.rename(new_json)
    return clean_new


def _process_and_save_voice_audio(
    source_path: Path,
    voice_name: str,
    voices_dir: Path,
) -> str:
    name = _sanitize_voice_name(voice_name)
    if not name:
        raise ValueError(f"Invalid voice name '{voice_name}'.")
    target_path = voices_dir / f"{name}.wav"
    try:
        target_path.resolve().relative_to(voices_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Invalid target path for '{name}'.") from exc

    tensor_audio, sr = load_audio(source_path)
    audio_np = tensor_audio.squeeze(0).numpy()
    audio_np = trim_silence(audio_np, sr)
    audio_np = apply_fades(audio_np, sr)
    save_audio(target_path, audio_np, sample_rate=sr)
    dur = float(len(audio_np) / sr) if sr > 0 else 0.0
    _save_voice_metadata(
        name, {"duration_seconds": dur, "sample_rate": sr, "transcript": ""}, voices_dir
    )
    return name


def import_voice_file(
    source_path: Path,
    filename: str,
    voices_dir: Path | None = None,
) -> list[str]:
    dir_path = VOICE_SAMPLES_DIR if voices_dir is None else Path(voices_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    imported_names: list[str] = []
    suffix = Path(filename).suffix.lower()

    if suffix == ".zip":
        if not zipfile.is_zipfile(source_path):
            raise ValueError(f"'{filename}' is not a valid zip archive.")
        with zipfile.ZipFile(source_path, "r") as zf:
            allowed_suffixes = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
            with tempfile.TemporaryDirectory() as extract_dir:
                ext_path = Path(extract_dir)
                for idx, member in enumerate(zf.namelist()):
                    if (
                        member.endswith("/")
                        or member.startswith("__MACOSX/")
                        or Path(member).name.startswith(".")
                    ):
                        continue
                    member_suffix = Path(member).suffix.lower()
                    if member_suffix not in allowed_suffixes:
                        continue
                    safe_stem = _sanitize_voice_name(Path(member).stem)
                    if not safe_stem:
                        continue
                    extracted_file = ext_path / f"tmp_{idx}{member_suffix}"
                    with zf.open(member) as src_f, open(extracted_file, "wb") as dst_f:
                        shutil.copyfileobj(src_f, dst_f)
                    try:
                        saved = _process_and_save_voice_audio(extracted_file, safe_stem, dir_path)
                        imported_names.append(saved)
                    except (OSError, RuntimeError, ValueError):
                        continue
                    finally:
                        extracted_file.unlink(missing_ok=True)
        if not imported_names:
            raise ValueError(f"No valid voice audio files found in '{filename}'.")
    else:
        stem = Path(filename).stem
        saved = _process_and_save_voice_audio(source_path, stem, dir_path)
        imported_names.append(saved)

    return imported_names


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


def saved_reference_name(ref: tuple[str, str] | None) -> str | None:
    if not ref:
        return None
    path = Path(ref[0]).resolve()
    if path.parent == VOICE_SAMPLES_DIR.resolve() and path.suffix.lower() == ".wav":
        return path.stem
    return None


def resolve_reference_transcript(
    ref_id: str,
    cache: dict[str, str],
    pending: set[str],
    voice_name: str | None = None,
    voices_dir: Path | None = None,
) -> tuple[str, str, bool]:
    if not ref_id:
        return "", "idle", False
    if ref_id in cache:
        return cache[ref_id], "ready", False
    if voice_name:
        meta = _load_voice_metadata(voice_name, voices_dir)
        transcript = meta.get("transcript", "")
        transcript = transcript.strip() if isinstance(transcript, str) else ""
        if transcript and meta.get("reference_id", ref_id) == ref_id:
            return transcript, "ready", False
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
    voice_name: str | None = None,
    voices_dir: Path | None = None,
) -> tuple[dict[str, str], set[str], bool, str, str | None]:
    new_cache = dict(cache)
    new_pending = set(pending)
    new_pending.discard(ref_id)
    is_current = bool(current_ref_id and current_ref_id == ref_id)

    if error is None and transcript is not None:
        if ref_id not in new_cache or not new_cache[ref_id].strip():
            new_cache[ref_id] = transcript
        resolved_text = new_cache[ref_id]
        if voice_name and is_current:
            _save_voice_metadata(voice_name, {"transcript": resolved_text}, voices_dir)
        return new_cache, new_pending, is_current, resolved_text, None

    resolved_text = new_cache.get(ref_id, "")
    return new_cache, new_pending, is_current, resolved_text, error


def record_user_transcript_edit(
    ref_id: str,
    edited_text: str | None,
    cache: dict[str, str],
    voice_name: str | None = None,
    voices_dir: Path | None = None,
) -> dict[str, str]:
    if not ref_id or edited_text is None:
        return cache
    new_cache = dict(cache)
    new_cache[ref_id] = edited_text
    if voice_name:
        _save_voice_metadata(voice_name, {"transcript": edited_text}, voices_dir)
    return new_cache


def audio_waveform(path: Path, bins: int = 100) -> tuple[list[float], float]:
    """Summarize audio with bounded reads, including the final partial block."""
    if bins < 1:
        raise ValueError("bins must be positive")
    peaks = []
    with sf.SoundFile(path) as audio:
        duration = audio.frames / audio.samplerate
        size, remainder = divmod(audio.frames, bins)
        for index in range(bins):
            remaining = size + (index < remainder)
            peak = 0.0
            while remaining:
                block = audio.read(min(remaining, 65536), dtype="float32", always_2d=True)
                peak = max(peak, float(np.max(np.abs(block))))
                remaining -= len(block)
            peaks.append(peak)
    return peaks, duration


def audio_file_response(
    path: str | Path, filename: str | None = None, *, media_type: str | None = None
):
    """Serve audio without copying it into a reactive UI message."""
    return FileResponse(
        path,
        media_type=media_type or _guess_mime_type(str(path)),
        filename=filename,
        content_disposition_type="attachment" if filename else "inline",
        headers={"Cache-Control": "private, no-store"},
    )


def script_validation_message(text: str) -> str:
    if not text.strip():
        return "Write a script before creating audio."
    if len(text) > MAX_SCRIPT_CHARACTERS:
        excess = len(text) - MAX_SCRIPT_CHARACTERS
        unit = "character" if excess == 1 else "characters"
        return f"Shorten your script by {excess:,} {unit}."
    return ""


def server(input, output, session):
    session_dir = Path(tempfile.gettempdir()) / "local-voice-cloning" / session.id
    session_dir.mkdir(parents=True, exist_ok=True)
    session.on_ended(lambda: shutil.rmtree(session_dir, ignore_errors=True))

    output_audio_path = reactive.value(None)
    output_waveform = reactive.value(None)
    generation_stage = reactive.value(None)
    generation_started_at = reactive.value(None)
    generation_request = reactive.value({})
    last_recorded_path = reactive.value(None)
    last_recorded_name = reactive.value(None)
    library_refresh = reactive.value(0)
    transcript_cache = reactive.value({})
    pending_transcriptions = reactive.value(set())
    ref_transcript_value = reactive.value("")
    transcription_status = reactive.value("idle")
    transcription_error = reactive.value("")
    trimmed_reference_path = reactive.value(None)
    session_takes = reactive.value([])

    @reactive.calc
    def active_reference():
        mode = input.ref_mode() if input.ref_mode() else "record"
        if mode == "upload":
            file_infos = input.audio_file()
            if file_infos:
                return (file_infos[0]["datapath"], file_infos[0]["name"])
        elif mode == "record":
            library_refresh()
            path = last_recorded_path()
            if path and Path(path).exists():
                name = last_recorded_name() or "recording"
                return (str(path), f"{name}.wav")
        elif mode == "library":
            library_refresh()
            selected = input.voice_library() or ""
            if selected:
                path = _saved_voice_path(selected)
                if path:
                    return (str(path), f"{selected}.wav")
        return None

    @reactive.calc
    def effective_reference():
        trimmed = trimmed_reference_path()
        if trimmed and Path(trimmed).exists():
            orig = active_reference()
            display = f"trimmed_{orig[1]}" if orig else "trimmed_sample.wav"
            return (trimmed, display)
        return active_reference()

    @reactive.effect
    @reactive.event(active_reference)
    def _clear_trim_on_voice_change():
        trimmed_reference_path.set(None)

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
        current_ref = effective_reference()
        current_ref_id = get_reference_id(current_ref[0]) if current_ref else ""
        voice_name = saved_reference_name(current_ref)

        new_cache, new_pending, is_current, resolved_text, err = apply_transcription_result(
            ref_id,
            transcript,
            error,
            current_ref_id,
            transcript_cache(),
            pending_transcriptions(),
            voice_name=voice_name,
            voices_dir=VOICE_SAMPLES_DIR,
        )
        transcript_cache.set(new_cache)
        pending_transcriptions.set(new_pending)

        if is_current:
            if err is not None:
                transcription_error.set(err)
                transcription_status.set("error")
                ui.notification_show(f"Transcription failed: {err}", type="warning")
            else:
                transcription_error.set("")
                ref_transcript_value.set(resolved_text)
                ui.update_text_area("ref_transcript", value=resolved_text)
                transcription_status.set("ready")
                ui.notification_show("Reference transcript ready for review.", type="message")

    @reactive.effect
    @reactive.event(effective_reference, ignore_init=False, ignore_none=False)
    def _sync_reference_transcript():
        ref = effective_reference()
        audio_path = ref[0] if ref else None
        ref_id = get_reference_id(audio_path)
        voice_name = saved_reference_name(ref)
        text, status, should_run = resolve_reference_transcript(
            ref_id,
            transcript_cache(),
            pending_transcriptions(),
            voice_name=voice_name,
            voices_dir=VOICE_SAMPLES_DIR,
        )
        ref_transcript_value.set(text)
        ui.update_text_area("ref_transcript", value=text)
        transcription_status.set(status)
        transcription_error.set("")

        if should_run and audio_path:
            new_pending = set(pending_transcriptions())
            new_pending.add(ref_id)
            pending_transcriptions.set(new_pending)
            run_transcription(audio_path, ref_id, input.quality() or "high")

    @reactive.effect
    @reactive.event(input.ref_transcript)
    def _save_user_transcript_edit():
        ref = effective_reference()
        if not ref:
            return
        ref_id = get_reference_id(ref[0])
        voice_name = saved_reference_name(ref)
        text = input.ref_transcript()
        new_cache = record_user_transcript_edit(
            ref_id,
            text,
            transcript_cache(),
            voice_name=voice_name,
            voices_dir=VOICE_SAMPLES_DIR,
        )
        transcript_cache.set(new_cache)
        ref_transcript_value.set(text or "")

    @render.ui
    def engine_badge():
        quality = input.quality() if input.quality() else "high"
        engine = input.engine() or "qwen"
        label = "BF16" if quality == "high" else "8-bit"
        title = f"{ENGINE_NAME} · {label} · Apple MLX"
        if engine == "omnivoice":
            title = f"OmniVoice · {32 if quality == 'high' else 16} steps · PyTorch"
        elif engine == "chatterbox":
            title = f"Chatterbox · {'full' if quality == 'high' else 'Turbo'} · PyTorch"
        return ui.div(
            {"class": "engine-pill", "title": title},
            icon_svg("microchip"),
            ui.span("Local", class_="engine-label"),
        )

    @render.text
    def generation_hint():
        if run_synthesis.status() == "running":
            return "Creating your audio…"
        if not effective_reference():
            return "Add a voice reference to continue."
        return script_validation_message(input.speech_text() or "") or "Ready when you are."

    @render.text
    def speech_duration():
        seconds = estimate_speech_duration_seconds(input.speech_text() or "")
        return f"~{seconds:.0f}s estimated audio" if seconds else "Add a script to begin"

    @render.text
    def character_count():
        text = input.speech_text() or ""
        chars = len(text)
        if chars > MAX_SCRIPT_CHARACTERS:
            return script_validation_message(text)
        if text.strip():
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

    def serve_selected_voice(request):
        selected = input.voice_library() or ""
        path = _saved_voice_path(selected)
        if not path:
            return Response(content="Voice not found", status_code=404)
        return audio_file_response(path, f"{selected}.wav")

    def serve_all_voices(request):
        zip_path = session_dir / "saved_voices.zip"
        create_voices_zip(output_path=zip_path, voices_dir=VOICE_SAMPLES_DIR)
        return FileResponse(
            zip_path,
            media_type="application/zip",
            filename="saved_voices.zip",
            content_disposition_type="attachment",
            headers={"Cache-Control": "private, no-store"},
        )

    @render.ui
    def library_selector():
        voices = library_choices()
        if not voices:
            return ui.div(
                {"class": "library-empty"},
                "No saved voices yet. Record one in the Record tab or import below.",
            )

        export_voice_url = session.dynamic_route("export-voice", serve_selected_voice)
        export_all_url = session.dynamic_route("export-all-voices", serve_all_voices)
        with reactive.isolate():
            selected = input.voice_library() if input.voice_library.is_set() else None

        return ui.div(
            {"class": "library-controls"},
            ui.input_select(
                "voice_library",
                "Saved voices",
                choices={v: v for v in voices},
                selected=selected if selected in voices else voices[0],
            ),
            ui.tags.a(
                icon_svg("download"),
                href=export_voice_url,
                download="",
                class_="btn btn-outline-secondary btn-sm btn-export-voice",
                title="Export selected voice (.wav)",
            ),
            ui.tags.a(
                icon_svg("file-zipper"),
                href=export_all_url,
                download="saved_voices.zip",
                class_="btn btn-outline-secondary btn-sm btn-export-all-voices",
                title="Export all voices (.zip)",
            ),
            ui.input_action_button(
                "btn_refresh_voices",
                icon_svg("rotate-right"),
                class_="btn btn-outline-secondary btn-sm btn-refresh-voices",
                title="Rescan directory",
            ),
            ui.input_action_button(
                "btn_rename_voice",
                icon_svg("pen-to-square"),
                class_="btn btn-outline-secondary btn-sm btn-rename-voice",
                title="Rename selected voice",
            ),
            ui.input_action_button(
                "btn_delete_voice",
                icon_svg("trash"),
                class_="btn btn-outline-danger btn-sm btn-delete-voice",
                title="Delete selected voice",
            ),
        )

    @reactive.effect
    @reactive.event(input.import_voice_files)
    def _import_voices():
        file_infos = input.import_voice_files()
        if not file_infos:
            return
        total_imported = []
        errors = []
        for info in file_infos:
            file_path = Path(info["datapath"])
            orig_name = info["name"]
            try:
                imported = import_voice_file(file_path, orig_name, VOICE_SAMPLES_DIR)
                total_imported.extend(imported)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{orig_name}: {exc}")

        if total_imported:
            library_refresh.set(library_refresh() + 1)
            last_name = total_imported[-1]
            ui.update_select("voice_library", selected=last_name)
            if len(total_imported) == 1:
                ui.notification_show(
                    f"Imported voice profile '{total_imported[0]}'.", type="message"
                )
            else:
                summary = ", ".join(total_imported[:5])
                if len(total_imported) > 5:
                    summary += f" and {len(total_imported) - 5} more"
                ui.notification_show(
                    f"Imported {len(total_imported)} voice profiles: {summary}.", type="message"
                )

        if errors:
            ui.notification_show(f"Import error: {'; '.join(errors)}", type="error")

    @reactive.effect
    @reactive.event(input.recorded_audio_data)
    async def _save_recording():
        payload = input.recorded_audio_data()
        if not payload:
            return
        try:
            data_uri = payload["data"]
            raw_name = payload.get("name", "")
            name = _sanitize_voice_name(raw_name)
            if not name:
                raise ValueError("Voice name is empty or invalid.")
            if "," not in data_uri:
                raise ValueError("Recording data is malformed.")
            _header, b64_content = data_uri.split(",", 1)
            wav_bytes = base64.b64decode(b64_content, validate=True)
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

            _save_voice_metadata(name, {"transcript": ""}, VOICE_SAMPLES_DIR)

            last_recorded_path.set(str(save_path))
            last_recorded_name.set(name)
            library_refresh.set(library_refresh() + 1)
            ui.notification_show(f"Saved voice profile '{name}'.", type="message")
        except (OSError, RuntimeError, ValueError, KeyError, TypeError) as exc:
            message = f"Could not save recording: {exc}"
            ui.notification_show(message, type="error")
        else:
            message = f"Voice saved · {name}"
        await session.send_custom_message("recording-status", {"text": message})

    @render.ui
    def recorded_take_actions():
        library_refresh()
        path = last_recorded_path()
        if not path or not Path(path).exists():
            return ui.div()
        name = last_recorded_name() or "recording"
        return ui.div(
            {"class": "record-take"},
            ui.span(
                {"class": "record-take-name"}, icon_svg("file-audio"), f" Current take: {name}.wav"
            ),
            ui.input_action_button(
                "btn_delete_take",
                ui.TagList(icon_svg("trash"), " Delete take"),
                class_="btn btn-outline-danger btn-sm btn-delete-voice",
                title="Delete this take so you can record it again",
            ),
        )

    @reactive.effect
    @reactive.event(input.btn_delete_take)
    async def _delete_take():
        path = last_recorded_path()
        if not path:
            return
        name = last_recorded_name() or "recording"
        Path(path).unlink(missing_ok=True)
        last_recorded_path.set(None)
        last_recorded_name.set(None)
        library_refresh.set(library_refresh() + 1)
        ui.notification_show(f"Deleted take '{name}'.", type="message")
        await session.send_custom_message("recording-status", {"text": "Ready for a new take"})

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
            path.with_suffix(".json").unlink(missing_ok=True)
            library_refresh.set(library_refresh() + 1)
            ui.notification_show(f"Deleted voice profile '{selected}'.", type="message")

    @reactive.effect
    @reactive.event(input.btn_rename_voice)
    def _show_rename_modal():
        selected = input.voice_library() or ""
        if not selected:
            return
        m = ui.modal(
            ui.input_text("new_voice_name", "New voice name", value=selected),
            ui.input_action_button("btn_confirm_rename", "Rename", class_="btn btn-primary"),
            title=f"Rename voice '{selected}'",
            easy_close=True,
            footer=None,
        )
        ui.modal_show(m)

    @reactive.effect
    @reactive.event(input.btn_confirm_rename)
    def _confirm_rename():
        selected = input.voice_library() or ""
        new_name = (input.new_voice_name() or "").strip()
        if not selected or not new_name:
            return
        try:
            clean_new = _rename_voice_profile(selected, new_name, VOICE_SAMPLES_DIR)
            ui.modal_remove()
            library_refresh.set(library_refresh() + 1)
            ui.update_select("voice_library", selected=clean_new)
            ui.notification_show(f"Renamed voice to '{clean_new}'.", type="message")
        except Exception as exc:  # noqa: BLE001
            ui.notification_show(f"Rename failed: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.script_preset)
    def _apply_script_preset():
        choice = input.script_preset()
        if choice and choice in SCRIPT_PRESET_TEXTS:
            ui.update_text_area("speech_text", value=SCRIPT_PRESET_TEXTS[choice])

    @reactive.effect
    @reactive.event(input.btn_unload_models)
    def _unload_models():
        count = unload_shared_cloners()
        if count:
            ui.notification_show(f"Freed {count} model(s) from memory.", type="message")
        else:
            ui.notification_show("No models currently in memory.", type="message")

    @reactive.effect
    @reactive.event(input.btn_apply_trim)
    def _apply_trim():
        ref = active_reference()
        if not ref:
            return
        datapath = ref[0]
        start = float(input.ref_trim_start() or 0.0)
        end = float(input.ref_trim_end() or 12.0)
        if start >= end:
            ui.notification_show("Start time must be less than end time.", type="warning")
            return
        target = session_dir / f"trimmed_{uuid.uuid4().hex}.wav"
        try:
            slice_audio(datapath, start_sec=start, end_sec=end, output_path=target)
            trimmed_reference_path.set(str(target))
            ui.notification_show(f"Trimmed reference to {end - start:.1f}s.", type="message")
        except Exception as exc:  # noqa: BLE001
            ui.notification_show(f"Trim failed: {exc}", type="error")

    @reactive.effect
    @reactive.event(input.btn_reset_trim)
    def _reset_trim():
        trimmed_reference_path.set(None)
        ui.notification_show("Reset to original reference sample.", type="message")

    @reactive.effect
    @reactive.event(input.btn_retranscribe)
    def _handle_retranscribe():
        ref = effective_reference()
        if not ref:
            ui.notification_show("No reference audio loaded to transcribe.", type="warning")
            return
        audio_path = ref[0]
        ref_id = get_reference_id(audio_path)
        if ref_id in pending_transcriptions():
            return
        new_pending = set(pending_transcriptions())
        new_pending.add(ref_id)
        pending_transcriptions.set(new_pending)
        transcription_status.set("transcribing")
        transcription_error.set("")
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

        eff_ref = effective_reference()
        datapath, display_name = eff_ref
        audio_url = session.dynamic_route(
            "reference-audio",
            lambda request: audio_file_response(
                datapath, media_type=_guess_mime_type(display_name)
            ),
        )

        try:
            report = analyze_reference_audio(datapath)
        except (ValueError, OSError, RuntimeError):
            report = None

        duration = report["duration_seconds"] if report else 0.0
        caption = f"{duration:.1f}s sample"
        if trimmed_reference_path():
            caption += " (trimmed)"

        quality_pills = []
        if trimmed_reference_path():
            quality_pills.append(
                ui.span({"class": "quality-pill good"}, icon_svg("scissors"), "Trim active")
            )
        if report is not None:
            if not report["warnings"]:
                quality_pills.append(
                    ui.span(
                        {"class": "quality-pill good"}, icon_svg("circle-check"), "Clean levels"
                    )
                )
            for w in report["warnings"]:
                quality_pills.append(
                    ui.span({"class": "quality-pill warn"}, icon_svg("triangle-exclamation"), w)
                )
        quality_feedback = (
            ui.div({"class": "quality-pill-group"}, *quality_pills) if quality_pills else ui.div()
        )

        trim_card = ui.tags.details(
            ui.tags.summary("Trim reference sample (optional)"),
            ui.div(
                {"class": "ref-trim-box"},
                ui.div(
                    {"class": "trim-inputs"},
                    ui.input_numeric("ref_trim_start", "Start (s)", value=0.0, min=0.0, step=0.5),
                    ui.input_numeric(
                        "ref_trim_end",
                        "End (s)",
                        value=round(min(12.0, max(0.5, duration)), 1),
                        min=0.5,
                        step=0.5,
                    ),
                ),
                ui.div(
                    {"class": "trim-btn-row"},
                    ui.input_action_button(
                        "btn_apply_trim", "Apply Trim", class_="btn btn-sm btn-outline-secondary"
                    ),
                    ui.input_action_button(
                        "btn_reset_trim", "Reset", class_="btn btn-sm btn-outline-secondary"
                    ),
                ),
            ),
            class_="trim-details",
        )

        return ui.div(
            {"class": "reference-file"},
            ui.div({"class": "reference-label"}, "Selected voice"),
            ui.div({"class": "file-name"}, icon_svg("microphone-lines"), Path(display_name).stem),
            ui.div({"class": "file-caption"}, caption),
            ui.tags.audio(
                controls=True,
                preload="metadata",
                src=audio_url,
            ),
            quality_feedback,
            trim_card,
        )

    @render.ui
    def reference_transcript_section():
        ref = effective_reference()
        if not ref:
            return ui.div()

        status = transcription_status()
        with reactive.isolate():
            transcript_text = ref_transcript_value()
        if status == "transcribing":
            status_badge = ui.span(
                {"class": "transcript-status transcribing"}, icon_svg("spinner"), "Transcribing..."
            )
            content = ui.div(
                {"class": "transcript-shimmer"},
                ui.div({"class": "shimmer-line"}),
                ui.div({"class": "shimmer-line short"}),
            )
            action_buttons = []
        elif status == "ready":
            status_badge = ui.span(
                {"class": "transcript-status ready"}, icon_svg("circle-check"), "Ready for review"
            )
            content = ui.input_text_area(
                "ref_transcript",
                None,
                value=transcript_text,
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
            status_badge = ui.span(
                {"class": "transcript-status"},
                icon_svg("triangle-exclamation"),
                "Transcription failed",
            )
            content = ui.input_text_area(
                "ref_transcript",
                None,
                value=transcript_text,
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
                value=transcript_text,
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
            ui.div(transcription_error(), class_="transcript-card-caption")
            if status == "error"
            else None,
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
        engine: str,
        speed: float = 1.0,
    ):
        def work(report):
            result = get_shared_cloner(quality, engine=engine).clone_voice(
                reference_audio_path=ref_path,
                text=text,
                reference_text=ref_text,
                speed=speed,
                language=language,
                progress_callback=report,
            )
            report("finish")
            wav_file = session_dir / f"clone_{uuid.uuid4().hex}.wav"
            save_audio(wav_file, result.audio, sample_rate=result.sample_rate)
            save_audio(wav_file.with_suffix(".mp3"), result.audio, sample_rate=result.sample_rate)
            return str(wav_file), audio_waveform(wav_file)

        return await run_with_progress(work, generation_stage.set)

    @reactive.effect
    @reactive.event(input.btn_generate)
    def handle_synthesis():
        if run_synthesis.status() == "running":
            return
        ref = effective_reference()
        text = input.speech_text() or ""
        if not ref:
            ui.notification_show("Add a reference recording before creating audio.", type="warning")
            return
        validation = script_validation_message(text)
        if validation:
            ui.notification_show(validation, type="warning")
            return

        ref_text = ref_transcript_value().strip()
        speed = float(input.synthesis_speed() or 1.0)
        generation_request.set(
            {
                "voice": Path(ref[1]).stem,
                "engine": input.engine() or "qwen",
                "quality": input.quality() or "high",
                "speed": speed,
                "snippet": (text[:60] + "…") if len(text) > 60 else text,
            }
        )

        output_audio_path.set(None)
        generation_stage.set("prepare")
        generation_started_at.set(time.monotonic())
        run_synthesis(
            ref[0],
            text,
            ref_text,
            input.quality() or "high",
            (input.omni_language() if input.engine() == "omnivoice" else input.language())
            or "auto",
            input.engine() or "qwen",
            speed,
        )

    @reactive.effect
    @reactive.event(input.btn_cancel)
    def handle_cancel():
        run_synthesis.cancel()
        ui.notification_show("Generation cancelled.", type="warning")

    @reactive.effect
    def _toggle_buttons():
        running = run_synthesis.status() == "running"
        invalid = bool(script_validation_message(input.speech_text() or ""))
        ui.update_action_button(
            "btn_generate", disabled=running or invalid or not effective_reference()
        )
        ui.update_action_button("btn_cancel", disabled=not running)

    @reactive.effect
    @reactive.event(run_synthesis.status)
    def _save_result():
        if run_synthesis.status() != "success":
            return
        path, waveform = run_synthesis.result()
        output_waveform.set(waveform)
        output_audio_path.set(path)
        ui.notification_show("Your cloned voice is ready.", type="message")

        _, duration = waveform
        take_info = {
            **generation_request(),
            "id": uuid.uuid4().hex[:6],
            "time": time.strftime("%H:%M:%S"),
            "duration": duration,
            "path": path,
        }
        session_takes.set([take_info] + session_takes()[:9])

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
        active_index = next(
            (index for index, item in enumerate(snapshot) if item.state in {"active", "error"}), 0
        )
        if status == "success":
            fill_width = 80
            message, detail = "Audio ready", "The cloned voice is ready to play and download."
        elif status == "error":
            fill_width = (active_index / 3) * 80
            message, detail = "Generation stopped", str(run_synthesis.error())
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
                ui.div(
                    *(
                        ui.span(style=f"height: {height}px")
                        for height in (8, 14, 22, 12, 30, 38, 20, 32, 16, 26, 12, 8)
                    ),
                    class_="empty-waveform",
                    aria_hidden="true",
                ),
                ui.div(
                    ui.strong("Your audio will appear here"),
                    ui.div(
                        "Playback and downloads unlock when generation finishes.",
                        class_="file-caption",
                    ),
                ),
            )

        wav_path = Path(path)
        mp3_path = wav_path.with_suffix(".mp3")
        wav_url = session.dynamic_route("output-wav", lambda request: audio_file_response(wav_path))
        wav_download = session.dynamic_route(
            "download-wav",
            lambda request: audio_file_response(wav_path, "cloned_voice_output.wav"),
        )
        peaks, duration = output_waveform()
        maximum = max(max(peaks), 1e-9)
        bars = "".join(
            f'<rect x="{index * 6}" y="{32 - max(1, peak / maximum * 30):.2f}" '
            f'width="3" height="{max(2, peak / maximum * 60):.2f}" rx="1.5" />'
            for index, peak in enumerate(peaks)
        )
        waveform = ui.HTML(
            f'<svg viewBox="0 0 600 64" preserveAspectRatio="none" aria-hidden="true">{bars}</svg>'
        )

        buttons = [
            ui.tags.a(
                icon_svg("download"),
                "Download WAV",
                href=wav_download,
                download="cloned_voice_output.wav",
                class_="btn btn-download",
            )
        ]
        if mp3_path.exists():
            mp3_download = session.dynamic_route(
                "download-mp3",
                lambda request: audio_file_response(mp3_path, "cloned_voice_output.mp3"),
            )
            buttons.append(
                ui.tags.a(
                    icon_svg("download"),
                    "Download MP3",
                    href=mp3_download,
                    download="cloned_voice_output.mp3",
                    class_="btn btn-download",
                )
            )

        return ui.div(
            {"class": "output-surface"},
            ui.div(
                ui.strong("Your voice, rendered"),
                ui.span(f"{duration:.1f}s", class_="audio-duration"),
                class_="result-title",
            ),
            ui.div(waveform, class_="audio-waveform"),
            ui.div(
                {"class": "result-player"},
                ui.tags.audio(
                    controls=True,
                    preload="metadata",
                    aria_label="Generated speech",
                    src=wav_url,
                ),
                ui.div({"class": "download-group"}, *buttons),
            ),
            ui.div(
                {"class": "speed-control-group"},
                ui.span("Playback speed:", class_="speed-label"),
                ui.tags.button(
                    "0.8×", type="button", class_="btn-speed", onclick="sonaSetSpeed(this, 0.8)"
                ),
                ui.tags.button(
                    "1.0×",
                    type="button",
                    class_="btn-speed active",
                    onclick="sonaSetSpeed(this, 1.0)",
                ),
                ui.tags.button(
                    "1.25×", type="button", class_="btn-speed", onclick="sonaSetSpeed(this, 1.25)"
                ),
                ui.tags.button(
                    "1.5×", type="button", class_="btn-speed", onclick="sonaSetSpeed(this, 1.5)"
                ),
            ),
        )

    @render.ui
    def session_history_ui():
        takes = session_takes()
        if not takes:
            return ui.div()

        take_nodes = []
        for index, take in enumerate(takes):
            take_path = Path(take["path"])
            take_url = session.dynamic_route(
                f"take-audio-{take['id']}",
                lambda request, p=take_path: audio_file_response(p),
            )
            take_id = take["id"]
            take_voice = take["voice"]
            take_download = session.dynamic_route(
                f"take-download-{take_id}",
                lambda request, p=take_path, name=take_voice, tid=take_id: audio_file_response(
                    p, f"take_{name}_{tid}.wav"
                ),
            )
            take_nodes.append(
                ui.div(
                    {"class": "history-item"},
                    ui.div(
                        {"class": "history-item-header"},
                        ui.span(f"Take {len(takes) - index}", class_="history-badge"),
                        ui.span(take["voice"], class_="history-voice"),
                        ui.span(f"{take['engine']} · {take['speed']:.2f}×", class_="history-meta"),
                        ui.span(f"{take['duration']:.1f}s", class_="history-dur"),
                        ui.span(take["time"], class_="history-time"),
                    ),
                    ui.div(take["snippet"], class_="history-snippet"),
                    ui.div(
                        {"class": "history-playback"},
                        ui.tags.audio(controls=True, preload="none", src=take_url),
                        ui.tags.a(
                            icon_svg("download"),
                            href=take_download,
                            download=f"take_{take['voice']}_{take['id']}.wav",
                            class_="btn btn-sm btn-outline-secondary btn-download-take",
                            title="Download WAV",
                        ),
                    ),
                )
            )

        return ui.tags.details(
            ui.tags.summary(f"Session Takes ({len(takes)})"),
            ui.div({"class": "history-list"}, *take_nodes),
            class_="session-history-card",
            open=True,
        )

    @render.ui
    def ab_comparison_ui():
        takes = session_takes()
        if len(takes) < 2:
            return ui.div()

        choices = {
            t["id"]: f"Take {t['id']} ({t['voice']} · {t['engine']}) - {t['snippet'][:30]}"
            for t in takes
        }
        sel_a = takes[0]["id"]
        sel_b = takes[1]["id"]

        selected_a = input.ab_select_a() if input.ab_select_a.is_set() else sel_a
        selected_b = input.ab_select_b() if input.ab_select_b.is_set() else sel_b
        take_a_obj = next((t for t in takes if t["id"] == selected_a), takes[0])
        take_b_obj = next((t for t in takes if t["id"] == selected_b), takes[1])

        url_a = session.dynamic_route(
            f"ab-audio-a-{take_a_obj['id']}",
            lambda request, p=Path(take_a_obj["path"]): audio_file_response(p),
        )
        url_b = session.dynamic_route(
            f"ab-audio-b-{take_b_obj['id']}",
            lambda request, p=Path(take_b_obj["path"]): audio_file_response(p),
        )

        return ui.tags.details(
            ui.tags.summary("A/B Voice Comparison"),
            ui.div(
                {"class": "ab-comparison-box"},
                ui.div(
                    {"class": "ab-track-column"},
                    ui.h4("Track A", class_="ab-track-title"),
                    ui.input_select(
                        "ab_select_a", None, choices=choices, selected=take_a_obj["id"]
                    ),
                    ui.tags.audio(controls=True, preload="none", src=url_a, class_="w-100"),
                ),
                ui.div(
                    {"class": "ab-track-column"},
                    ui.h4("Track B", class_="ab-track-title"),
                    ui.input_select(
                        "ab_select_b", None, choices=choices, selected=take_b_obj["id"]
                    ),
                    ui.tags.audio(controls=True, preload="none", src=url_b, class_="w-100"),
                ),
            ),
            class_="ab-comparison-card",
            open=True,
        )


app = App(app_ui, server, static_assets=Path(__file__).parent / "www")


def main() -> None:
    import argparse

    from shiny import run_app

    parser = argparse.ArgumentParser(
        description="Start Sona Shiny app on a random or specified port"
    )
    parser.add_argument(
        "--port", "-p", type=int, default=0, help="Port to listen on (default: 0 for random port)"
    )
    parser.add_argument(
        "--host", "-H", type=str, default="127.0.0.1", help="Host address (default: 127.0.0.1)"
    )
    parser.add_argument("--reload", "-r", action="store_true", help="Enable auto-reload")
    parser.add_argument(
        "--launch-browser", "-b", action="store_true", help="Launch browser on start"
    )
    args, _ = parser.parse_known_args()

    run_app(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        launch_browser=args.launch_browser,
    )


if __name__ == "__main__":
    main()
