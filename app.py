import asyncio
import base64
import mimetypes
import re
import shutil
import tempfile
import time
import uuid
from pathlib import Path

import shinyswatch
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

RECORDING_PROMPT = """Hi, I'm [your name], and this is my natural speaking voice. The quick brown fox jumps over the lazy dog. How vexingly quick daft zebras jump! Did it capture the real me?"""
MAX_RECORDING_SECONDS = 30

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
            rel="stylesheet",
            href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Mona+Sans:ital,wght@0,200..900;1,200..900&display=swap",
        ),
        ui.tags.style(
            """
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Mona+Sans:ital,wght@0,200..900;1,200..900&display=swap');

            :root {
                --font-sans: 'Mona Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                --font-text: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                --bg: #101017;
                --surface: #15151e;
                --surface-raised: #1b1a25;
                --surface-soft: #201e2a;
                --border: #35323f;
                --border-strong: #484352;
                --text: #f5f2eb;
                --muted: #aaa5b0;
                --subtle: #77717f;
                --mint: #72d8aa;
                --mint-soft: #b6edd3;
                --amber: #d7922f;
                --amber-hover: #e7a243;
                --danger: #ef7d79;
                --focus: #f3b860;
            }

            * { box-sizing: border-box; }

            html, body {
                background: var(--bg) !important;
                color: var(--text) !important;
                min-height: 100%;
            }

            body, button, select {
                margin: 0;
                font-family: var(--font-sans) !important;
                font-size: 15px;
                line-height: 1.45;
            }

            .container-fluid { padding: 0 !important; }

            .app-shell { min-height: 100vh; background: var(--bg); }

            .app-header {
                height: 76px;
                padding: 0 30px;
                display: grid;
                grid-template-columns: 1fr auto 1fr;
                align-items: center;
                border-bottom: 1px solid var(--border);
                background: #111118;
            }

            .brand, .privacy-note, .engine-pill,
            .section-title, .status-chip, .meta-row,
            .transport-actions, .output-heading, .button-label {
                display: flex;
                align-items: center;
            }

            .brand { gap: 14px; font-size: 20px; font-weight: 730; letter-spacing: -0.35px; }
            .brand-mark {
                width: 34px;
                height: 34px;
                display: inline-flex;
                align-items: center;
                justify-content: center;
                border: 1px solid var(--text);
                border-radius: 10px;
            }
            .brand-mark svg { width: 18px; height: 18px; fill: var(--text); }

            .privacy-note { justify-self: center; gap: 9px; color: var(--muted); }
            .privacy-note svg { width: 15px; height: 15px; fill: var(--mint); }

            .engine-pill {
                justify-self: end;
                gap: 9px;
                min-height: 38px;
                padding: 8px 13px;
                border: 1px solid var(--border);
                border-radius: 9px;
                background: var(--surface);
                color: #d8d3dd;
                font-size: 13px;
            }
            .engine-pill svg { width: 14px; height: 14px; fill: var(--mint); }

            .workspace {
                display: grid;
                grid-template-columns: minmax(0, 2.05fr) minmax(320px, 0.95fr);
                min-height: 555px;
                border-bottom: 1px solid var(--border);
            }

            .script-pane, .reference-pane { padding: 26px 29px 22px; }
            .reference-pane { border-left: 1px solid var(--border); background: #121219; }

            .section-title { gap: 10px; margin: 0 0 5px; font-size: 16px; font-weight: 720; }
            .section-title svg, .output-heading svg { width: 15px; height: 15px; fill: var(--text); }
            .section-copy { margin: 0 0 15px; color: var(--muted); font-size: 13px; }

            .form-group { margin-bottom: 0; }
            label.control-label { color: var(--text); font-weight: 650; margin-bottom: 9px; }

            #speech_text {
                min-height: 260px;
                resize: vertical;
                background: #14131c !important;
                border: 1px solid var(--border-strong) !important;
                border-radius: 8px !important;
                color: #f7f5fa !important;
                padding: 18px 20px !important;
                font-family: var(--font-text) !important;
                font-size: 16px !important;
                line-height: 1.7 !important;
                letter-spacing: -0.01em;
                box-shadow: none !important;
            }
            #speech_text::placeholder, .form-control::placeholder { color: var(--subtle) !important; }

            .field-footer {
                display: flex;
                justify-content: space-between;
                margin-top: 8px;
                color: var(--subtle);
                font-size: 12px;
            }

            .delivery-controls {
                display: grid;
                grid-template-columns: 1.45fr .75fr;
                gap: 26px;
                align-items: end;
                margin-top: 20px;
            }
            .delivery-controls .form-group { margin: 0; }
            .settings-disclosure { margin-top: 14px; }
            .accordion, .accordion-item { background: transparent !important; border: 0 !important; }
            .accordion-item { border: 1px solid var(--border) !important; border-radius: 8px !important; overflow: hidden; }
            .accordion-button {
                background: var(--surface) !important;
                color: #d5d0da !important;
                min-height: 46px;
                font-size: 13px;
                box-shadow: none !important;
            }
            .accordion-button::after { filter: invert(1); opacity: .65; }
            .accordion-body { background: var(--surface) !important; padding: 18px !important; }

            .form-check-input { background-color: #13121a; border-color: #77717f; }
            .form-check-input:checked { background-color: var(--mint); border-color: var(--mint); }

            .form-control, .form-select {
                background: #13121a !important;
                border: 1px solid var(--border-strong) !important;
                color: var(--text) !important;
                border-radius: 7px !important;
                font-family: var(--font-text) !important;
                font-size: 14.5px !important;
                box-shadow: none !important;
            }
            .form-control:focus, .form-select:focus, button:focus-visible, a:focus-visible {
                outline: 2px solid var(--focus) !important;
                outline-offset: 2px;
            }

            .reference-upload .form-group { margin: 0; }
            .reference-upload .control-label { color: var(--muted); font-size: 12px; font-weight: 560; }
            .reference-upload .input-group { border: 1px dashed var(--border-strong); border-radius: 8px; padding: 8px; }
            .reference-upload .btn-file { background: var(--surface-soft); border: 1px solid var(--border-strong); color: var(--text); }

            .reference-empty {
                min-height: 205px;
                margin-top: 18px;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: 9px;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--surface);
                color: var(--muted);
                text-align: center;
            }
            .reference-empty svg { width: 24px; height: 24px; fill: var(--mint); }

            .reference-file {
                margin-top: 18px;
                padding: 15px;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--surface);
            }
            .file-name { color: var(--text); font-weight: 650; overflow-wrap: anywhere; }
            .file-caption { color: var(--muted); font-size: 12px; }
            .reference-file audio { width: 100%; margin: 14px 0 6px; }

            .meta-list { margin-top: 13px; border-top: 1px solid var(--border); padding-top: 9px; }
            .meta-row { justify-content: space-between; gap: 20px; padding: 5px 0; color: var(--muted); font-size: 12px; }
            .meta-row span:last-child { color: #d6d1da; font-variant-numeric: tabular-nums; }

            .quality-box { margin-top: 12px; padding: 11px 12px; border-radius: 7px; font-size: 12px; }
            .quality-good { border: 1px solid rgba(114,216,170,.35); color: var(--mint-soft); background: rgba(114,216,170,.07); }
            .quality-warn { border: 1px solid rgba(215,146,47,.42); color: #f0c477; background: rgba(215,146,47,.08); }
            .quality-box ul { margin: 7px 0 0; padding-left: 18px; }

            .transport {
                display: grid;
                grid-template-columns: 230px minmax(0, 1fr);
                min-height: 146px;
                border-bottom: 1px solid var(--border);
                background: #111118;
            }
            .transport-actions { padding: 22px 28px; border-right: 1px solid var(--border); align-items: stretch; flex-direction: column; justify-content: center; }
            .btn-create {
                min-height: 52px;
                border: 1px solid #f0ae50 !important;
                border-radius: 8px !important;
                background: var(--amber) !important;
                color: #171118 !important;
                font-size: 15px !important;
                font-weight: 760 !important;
                box-shadow: 0 7px 18px rgba(215,146,47,.17) !important;
            }
            .btn-create:hover { background: var(--amber-hover) !important; }
            .btn-create svg { width: 16px; height: 16px; margin-right: 9px; vertical-align: -2px; fill: currentColor; }
            .btn-cancel { margin-top: 8px; color: var(--muted) !important; border-color: var(--border) !important; }
            .action-caption { margin: 8px 0 0; color: var(--subtle); font-size: 11px; }

            .progress-region { padding: 22px 29px 18px; min-width: 0; }
            .progress-top { display: flex; justify-content: space-between; gap: 18px; margin-bottom: 19px; }
            .progress-message { color: var(--text); font-weight: 650; }
            .progress-detail { color: var(--muted); font-size: 12px; margin-top: 2px; }
            .elapsed { color: var(--mint); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-variant-numeric: tabular-nums; }

            .stage-track { position: relative; display: grid; grid-template-columns: repeat(4, 1fr); gap: 0; }
            .stage-track::before {
                content: "";
                position: absolute;
                left: 10%; right: 10%; top: 13px;
                height: 2px;
                background: var(--border-strong);
            }
            .stage-fill {
                position: absolute;
                left: 10%; top: 13px;
                height: 2px;
                background: var(--mint);
                transition: width .3s ease;
            }
            .stage { position: relative; z-index: 1; text-align: center; color: var(--subtle); }
            .stage-dot {
                width: 27px; height: 27px; margin: 0 auto 7px;
                display: flex; align-items: center; justify-content: center;
                border-radius: 50%; border: 1px solid var(--border-strong); background: #111118;
            }
            .stage-dot svg { width: 11px; height: 11px; fill: currentColor; }
            .stage-label { color: inherit; font-size: 12px; font-weight: 700; }
            .stage-detail { color: var(--subtle); font-size: 10px; margin-top: 2px; }
            .stage.complete, .stage.active { color: var(--mint); }
            .stage.complete .stage-dot { background: var(--mint); border-color: var(--mint); color: #0d2018; }
            .stage.active .stage-dot { border: 3px solid var(--mint); box-shadow: 0 0 0 5px rgba(114,216,170,.12); }
            .stage.error { color: var(--danger); }
            .stage.error .stage-dot { border-color: var(--danger); }

            @media (prefers-reduced-motion: no-preference) {
                .stage.active .stage-dot { animation: pulse 1.45s ease-in-out infinite; }
                @keyframes pulse { 50% { box-shadow: 0 0 0 9px rgba(114,216,170,.03); } }
            }

            .output-pane { padding: 19px 28px 28px; min-height: 192px; background: var(--bg); }
            .output-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px; }
            .output-heading { gap: 9px; font-size: 15px; font-weight: 720; }
            .status-chip { gap: 7px; color: var(--muted); font-size: 12px; }
            .status-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--border-strong); }
            .status-chip.ready { color: var(--mint); }
            .status-chip.ready .status-dot { background: var(--mint); }
            .output-surface { border: 1px solid var(--border); border-radius: 8px; background: #14131b; padding: 20px; }
            .output-empty { display: flex; align-items: center; gap: 12px; color: var(--muted); min-height: 76px; }
            .output-empty svg { width: 20px; height: 20px; fill: var(--subtle); }
            .result-player { display: grid; grid-template-columns: minmax(260px, 1fr) auto; align-items: center; gap: 20px; }
            .result-player audio { width: 100%; }
            .download-group { display: flex; gap: 9px; flex-wrap: wrap; justify-content: flex-end; }
            .btn-download { border: 1px solid var(--border-strong) !important; color: var(--text) !important; background: var(--surface) !important; border-radius: 7px !important; font-size: 12px !important; }
            .btn-download:hover { border-color: var(--mint) !important; color: var(--mint-soft) !important; }
            .btn-download svg { width: 13px; height: 13px; fill: currentColor; margin-right: 7px; vertical-align: -2px; }

            /* Segmented Controls (iOS / macOS DAW Style) */
            .ref-mode-selector { margin-bottom: 18px; margin-top: 4px; width: 100%; }
            .ref-mode-selector .shiny-input-container,
            .quality-options .shiny-input-container {
                width: 100% !important;
                margin-bottom: 0 !important;
            }
            .ref-mode-selector .shiny-options-group,
            .quality-options .shiny-options-group {
                display: flex !important;
                background: #0d0c14 !important;
                padding: 4px !important;
                border-radius: 9px !important;
                border: 1px solid var(--border-strong) !important;
                gap: 4px !important;
                width: 100% !important;
                box-sizing: border-box !important;
            }
            .ref-mode-selector .radio-inline,
            .quality-options .radio-inline {
                flex: 1 1 0 !important;
                margin: 0 !important;
                padding: 8px 12px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                position: relative !important;
                border-radius: 6px !important;
                font-size: 13px !important;
                font-weight: 580 !important;
                color: var(--muted) !important;
                cursor: pointer !important;
                transition: all .18s cubic-bezier(0.16, 1, 0.3, 1) !important;
                white-space: nowrap !important;
                user-select: none !important;
                text-align: center !important;
            }
            .ref-mode-selector .radio-inline:hover,
            .quality-options .radio-inline:hover {
                color: var(--text) !important;
                background: rgba(255, 255, 255, 0.04) !important;
            }
            .ref-mode-selector .radio-inline input[type="radio"],
            .quality-options .radio-inline input[type="radio"] {
                position: absolute !important;
                opacity: 0 !important;
                width: 0 !important;
                height: 0 !important;
                margin: 0 !important;
                padding: 0 !important;
                pointer-events: none !important;
                appearance: none !important;
                -webkit-appearance: none !important;
            }
            .ref-mode-selector .radio-inline:has(input:checked),
            .quality-options .radio-inline:has(input:checked) {
                background: var(--surface-soft) !important;
                color: var(--mint) !important;
                font-weight: 680 !important;
                box-shadow: 0 2px 6px rgba(0, 0, 0, .45), inset 0 0 0 1px rgba(114, 216, 170, .25) !important;
            }
            .ref-mode-selector .radio-inline span,
            .quality-options .radio-inline span {
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
                color: inherit !important;
                font-weight: inherit !important;
                margin: 0 !important;
                padding: 0 !important;
            }

            /* Record panel & VU Meter */
            .record-panel { margin-top: 14px; }
            .voice-name-field { margin-bottom: 16px; }
            .voice-name-field .control-label { color: var(--muted); font-size: 12px; font-weight: 560; margin-bottom: 7px; }

            .record-prompt {
                margin-bottom: 18px;
                padding: 18px 20px;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: #14131c;
                color: #f7f5fa;
                font-family: var(--font-text) !important;
                font-size: 16.5px;
                font-weight: 480;
                letter-spacing: -0.01em;
                line-height: 1.75;
                white-space: pre-wrap;
                max-height: 240px;
                overflow-y: auto;
            }
            .record-prompt-caption { color: var(--subtle); font-size: 12px; margin-bottom: 7px; }

            .record-controls { display: flex; align-items: center; gap: 14px; }
            .btn-record {
                display: inline-flex; align-items: center; gap: 9px;
                min-height: 46px; padding: 0 20px;
                border: 1px solid var(--mint) !important;
                border-radius: 8px !important;
                background: var(--surface-soft) !important;
                color: var(--mint) !important;
                font-size: 14px !important; font-weight: 700 !important;
                cursor: pointer;
                transition: all .18s ease;
            }
            .btn-record:hover { background: rgba(114,216,170,.12) !important; }
            .btn-record.recording {
                border-color: var(--danger) !important;
                color: var(--danger) !important;
                background: rgba(239,125,121,.1) !important;
            }
            .btn-record.recording::before {
                content: ""; display: inline-block; width: 10px; height: 10px;
                border-radius: 50%; background: var(--danger);
                animation: rec-pulse 1.1s ease-in-out infinite;
            }
            .btn-record:not(.recording)::before {
                content: ""; display: inline-block; width: 10px; height: 10px;
                border-radius: 50%; background: var(--mint);
            }
            @keyframes rec-pulse { 50% { opacity: .35; } }
            .record-timer {
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                font-variant-numeric: tabular-nums;
                font-size: 18px; color: var(--text);
            }
            .record-vu-meter {
                display: inline-flex;
                align-items: center;
                gap: 3px;
                height: 22px;
                padding: 0 6px;
                background: rgba(0,0,0,0.25);
                border-radius: 4px;
                border: 1px solid var(--border);
            }
            .vu-bar {
                width: 3px;
                height: 6px;
                background: var(--border-strong);
                border-radius: 2px;
                transition: height 0.08s ease, background-color 0.08s ease;
            }
            .vu-bar.active {
                background: var(--mint);
                box-shadow: 0 0 5px rgba(114,216,170,.5);
            }
            .record-progress-track {
                width: 100%;
                height: 4px;
                background: rgba(255,255,255,0.06);
                border-radius: 3px;
                margin-top: 10px;
                overflow: hidden;
            }
            .record-progress-fill {
                height: 100%;
                width: 0%;
                background: var(--mint);
                transition: width 0.2s linear;
                border-radius: 3px;
            }
            .record-progress-fill.danger {
                background: var(--danger);
            }
            .record-status { color: var(--subtle); font-size: 12px; margin-top: 6px; }

            /* Library panel */
            .library-panel { margin-top: 14px; }
            .library-controls { display: flex; gap: 9px; align-items: end; }
            .library-controls .form-group { flex: 1; margin: 0; }
            .library-controls .control-label { color: var(--muted); font-size: 12px; font-weight: 560; margin-bottom: 7px; }
            .btn-refresh-voices, .btn-delete-voice {
                min-height: 38px; font-size: 12px !important; white-space: nowrap;
            }
            .btn-refresh-voices { border: 1px solid var(--border-strong) !important; color: var(--text) !important; background: var(--surface) !important; }
            .btn-refresh-voices:hover { border-color: var(--mint) !important; color: var(--mint-soft) !important; }
            .btn-delete-voice { border: 1px solid rgba(239,125,121,.35) !important; color: var(--danger) !important; background: var(--surface) !important; }
            .btn-delete-voice:hover { background: rgba(239,125,121,.1) !important; }
            .library-empty { color: var(--subtle); font-size: 13px; margin-top: 12px; }

            /* Transcript card */
            .transcript-card {
                margin-top: 14px;
                padding: 14px;
                border: 1px solid var(--border);
                border-radius: 8px;
                background: var(--surface);
            }
            .transcript-card-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 6px;
                gap: 8px;
            }
            .transcript-title {
                font-size: 13px;
                font-weight: 650;
                color: var(--text);
                display: flex;
                align-items: center;
                gap: 7px;
            }
            .transcript-title svg {
                width: 14px;
                height: 14px;
                fill: var(--mint);
            }
            .transcript-actions {
                display: flex;
                gap: 6px;
                align-items: center;
            }
            .btn-transcript-action {
                display: inline-flex;
                align-items: center;
                gap: 4px;
                padding: 3px 8px !important;
                font-size: 11px !important;
                font-weight: 560 !important;
                border: 1px solid var(--border) !important;
                background: var(--surface-soft) !important;
                color: var(--muted) !important;
                border-radius: 5px !important;
                cursor: pointer;
                transition: all .15s ease;
                text-decoration: none;
            }
            .btn-transcript-action:hover {
                color: var(--text) !important;
                border-color: var(--border-strong) !important;
                background: rgba(255, 255, 255, 0.08) !important;
            }
            .btn-transcript-action svg {
                width: 11px;
                height: 11px;
                fill: currentColor;
            }
            .transcript-card-caption {
                color: var(--muted);
                font-size: 12px;
                margin-bottom: 8px;
                line-height: 1.4;
            }
            #ref_transcript {
                min-height: 84px;
                resize: vertical;
                background: #14131c !important;
                border: 1px solid var(--border-strong) !important;
                border-radius: 6px !important;
                color: #f7f5fa !important;
                padding: 12px 14px !important;
                font-family: var(--font-text) !important;
                font-size: 15px !important;
                line-height: 1.65 !important;
                letter-spacing: -0.01em;
            }
            .transcript-shimmer {
                display: flex;
                flex-direction: column;
                gap: 8px;
                padding: 14px 0 10px;
            }
            .shimmer-line {
                height: 12px;
                border-radius: 4px;
                background: linear-gradient(90deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.09) 50%, rgba(255,255,255,0.03) 100%);
                background-size: 200% 100%;
                animation: shimmer-anim 1.5s infinite linear;
            }
            .shimmer-line.short { width: 60%; }
            @keyframes shimmer-anim {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            .transcript-footer {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-top: 8px;
            }
            .transcript-status {
                font-size: 11px;
                color: var(--subtle);
                display: flex;
                align-items: center;
                gap: 5px;
            }
            .transcript-status.transcribing {
                color: var(--amber);
            }
            .transcript-status.ready {
                color: var(--mint);
            }

            /* Quality badges, speed controls & keyboard shortcut */
            .quality-pill-group {
                display: flex;
                flex-wrap: wrap;
                gap: 6px;
                margin-top: 10px;
            }
            .quality-pill {
                display: inline-flex;
                align-items: center;
                gap: 5px;
                padding: 3px 8px;
                border-radius: 5px;
                font-size: 11px;
                font-weight: 560;
            }
            .quality-pill.good {
                background: rgba(114,216,170,.1);
                color: var(--mint);
                border: 1px solid rgba(114,216,170,.25);
            }
            .quality-pill.warn {
                background: rgba(215,146,47,.12);
                color: #f0c477;
                border: 1px solid rgba(215,146,47,.3);
            }
            .speed-control-group {
                display: flex;
                align-items: center;
                gap: 5px;
                margin-top: 12px;
            }
            .speed-label {
                font-size: 11px;
                color: var(--subtle);
                margin-right: 4px;
            }
            .btn-speed {
                padding: 2px 7px;
                font-size: 11px;
                font-weight: 600;
                border-radius: 4px;
                border: 1px solid var(--border);
                background: var(--surface-soft);
                color: var(--muted);
                cursor: pointer;
                transition: all .15s ease;
            }
            .btn-speed:hover {
                color: var(--text);
                border-color: var(--border-strong);
            }
            .btn-speed.active {
                background: var(--mint);
                color: #0d2018;
                border-color: var(--mint);
            }
            .kbd-shortcut {
                display: inline-block;
                padding: 1px 5px;
                font-size: 10px;
                font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                border-radius: 4px;
                background: rgba(0,0,0,0.3);
                border: 1px solid rgba(255,255,255,0.15);
                color: inherit;
                margin-left: 6px;
                vertical-align: middle;
            }

            @media (max-width: 940px) {
                .app-header { grid-template-columns: 1fr auto; }
                .privacy-note { display: none; }
                .workspace { grid-template-columns: 1fr; }
                .reference-pane { border-left: 0; border-top: 1px solid var(--border); }
                .transport { grid-template-columns: 1fr; }
                .transport-actions { border-right: 0; border-bottom: 1px solid var(--border); }
                .result-player { grid-template-columns: 1fr; }
                .download-group { justify-content: flex-start; }
            }

            @media (max-width: 620px) {
                .app-header { height: auto; min-height: 72px; padding: 14px 18px; gap: 10px; }
                .brand { font-size: 17px; }
                .engine-pill { padding: 7px 9px; font-size: 0; }
                .engine-pill svg { margin: 0; }
                .script-pane, .reference-pane, .progress-region, .output-pane { padding-left: 18px; padding-right: 18px; }
                .delivery-controls { grid-template-columns: minmax(0, 1fr); gap: 14px; }
                .delivery-controls .shiny-input-container { width: 100% !important; }
                .stage-detail { display: none; }
                .stage-label { font-size: 10px; }
            }
            """
        ),
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
                    const player = btn.closest('.result-player');
                    const audio = player ? player.querySelector('audio') : null;
                    if (audio) audio.playbackRate = rate;
                    const group = btn.closest('.speed-control-group');
                    if (group) {
                        group.querySelectorAll('.btn-speed').forEach(function(b) { b.classList.remove('active'); });
                    }
                    btn.classList.add('active');
                };

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
        ui.tags.header(
            {"class": "app-header"},
            ui.div(
                {"class": "brand"},
                ui.span({"class": "brand-mark"}, icon_svg("wave-square")),
                "Sona — Local Voice Studio",
            ),
            ui.div(
                {"class": "privacy-note"},
                icon_svg("shield-halved"),
                "Private session · nothing leaves your Mac",
            ),
            ui.output_ui("engine_badge"),
        ),
        ui.tags.main(
            ui.div(
                {"class": "workspace"},
                ui.tags.section(
                    {"class": "script-pane", "aria-labelledby": "script-heading"},
                    ui.h2(
                        {"class": "section-title", "id": "script-heading"},
                        icon_svg("pen"),
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
                            {"class": "quality-options"},
                            ui.input_radio_buttons(
                                "quality",
                                "Model quality",
                                choices={
                                    "high": "High fidelity · BF16",
                                    "fast": "Fast draft · 8-bit",
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
                ),
                ui.tags.aside(
                    {"class": "reference-pane", "aria-labelledby": "reference-heading"},
                    ui.h2(
                        {"class": "section-title", "id": "reference-heading"},
                        icon_svg("microphone"),
                        "Voice reference",
                    ),
                    ui.p(
                        "Record yourself, upload a file, or pick a saved voice. "
                        "The first 12 seconds create the voice profile.",
                        class_="section-copy",
                    ),
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
                                {"class": "record-prompt-caption"},
                                "Read this aloud at a natural pace:",
                            ),
                            ui.tags.pre(
                                {"class": "record-prompt"},
                                RECORDING_PROMPT,
                            ),
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
                    ui.output_ui("reference_preview"),
                    ui.output_ui("reference_transcript_section"),
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
                    ui.p("Runs locally with Qwen3-TTS 1.7B and Apple MLX.", class_="action-caption"),
                ),
                ui.output_ui("generation_progress"),
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
            {"class": "engine-pill"},
            icon_svg("microchip"),
            f"{ENGINE_NAME} · {label} · Apple MLX",
        )

    @render.text
    def character_count():
        text = input.speech_text() or ""
        chars = len(text)
        est = estimate_speech_duration_seconds(text)
        if est > 0:
            return f"{chars:,} / 5,000 chars · ~{est:.0f}s speech"
        return f"{chars:,} / 5,000 chars"

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

        return ui.div(
            {"class": "transcript-card"},
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
                    {"class": f"stage {step.state}"},
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


app = App(app_ui, server)
