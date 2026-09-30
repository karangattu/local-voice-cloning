from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy import signal
from scipy.ndimage import maximum_filter1d


def load_audio(
    file_path: str | Path,
    target_sr: int = 24000,
    max_duration_seconds: float = 30.0,
) -> tuple[torch.Tensor, int]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    if sr != target_sr:
        num_target_samples = round(len(data) * float(target_sr) / sr)
        data = signal.resample(data, num_target_samples).astype(np.float32)
        sr = target_sr

    max_samples = int(target_sr * max_duration_seconds)
    if len(data) > max_samples:
        data = data[:max_samples]

    data = normalize_audio(data)
    tensor_audio = torch.from_numpy(data).unsqueeze(0)
    return tensor_audio, sr


def normalize_audio(
    audio: np.ndarray,
    target_level_db: float = -20.0,
    peak_ceiling_db: float = -1.0,
) -> np.ndarray:
    if len(audio) == 0:
        return audio
    rms = np.sqrt(np.mean(audio**2) + 1e-9)
    target_rms = 10.0 ** (target_level_db / 20.0)
    if rms > 0:
        audio = audio * (target_rms / rms)
    ceiling = 10.0 ** (peak_ceiling_db / 20.0)
    max_val = np.max(np.abs(audio))
    if max_val > ceiling:
        audio = audio * (ceiling / max_val)
    return audio.astype(np.float32)


def analyze_reference_audio(file_path: str | Path) -> dict:
    """Score an uploaded voice sample for cloning suitability. Returns metrics
    plus human-readable warnings; a flat or damaged sample is the main cause
    of robotic-sounding clones."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    duration = len(data) / sr if sr > 0 else 0.0
    warnings: list[str] = []

    if duration == 0.0:
        return {
            "duration_seconds": 0.0,
            "sample_rate": sr,
            "rms_db": float("-inf"),
            "clipping_ratio": 0.0,
            "silence_ratio": 1.0,
            "warnings": ["The sample is empty."],
        }

    rms_db = float(20.0 * np.log10(np.sqrt(np.mean(data**2)) + 1e-9))
    clipping_ratio = float(np.mean(np.abs(data) >= 0.999))

    frame = max(1, int(sr * 0.02))
    n_frames = len(data) // frame
    if n_frames > 0:
        frames = data[: n_frames * frame].reshape(n_frames, frame)
        frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
        silence_ratio = float(np.mean(frame_rms_db < -45.0))
    else:
        silence_ratio = 0.0

    if duration < 3.0:
        warnings.append("The sample is shorter than 3 seconds. Use 5 to 12 seconds of speech.")
    elif duration > 12.0:
        warnings.append(
            "The sample is longer than 12 seconds. The app uses about 12 seconds and cuts at a pause."
        )
    if clipping_ratio > 0.001:
        warnings.append("The sample is clipped (distorted). Record again at a lower input level.")
    if rms_db < -35.0:
        warnings.append("The sample is very quiet. Record closer to the microphone.")
    if silence_ratio > 0.4:
        warnings.append(
            "The sample contains long silences. They make the output slow and unnatural."
        )
    if sr < 16000:
        warnings.append(f"The sample rate is low ({sr} Hz). Use a recording of 24000 Hz or more.")

    return {
        "duration_seconds": float(duration),
        "sample_rate": int(sr),
        "rms_db": rms_db,
        "clipping_ratio": clipping_ratio,
        "silence_ratio": silence_ratio,
        "warnings": warnings,
    }


def high_pass_filter(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 50.0) -> np.ndarray:
    if len(audio) < 16:
        return audio
    sos = signal.butter(2, cutoff_hz, btype="highpass", fs=sample_rate, output="sos")
    return signal.sosfiltfilt(sos, audio).astype(np.float32)


def trim_silence(
    audio: np.ndarray,
    sample_rate: int,
    threshold_db: float = -45.0,
    padding_seconds: float = 0.08,
) -> np.ndarray:
    if len(audio) == 0:
        return audio
    frame = max(1, int(sample_rate * 0.02))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_rms_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    active = np.flatnonzero(frame_rms_db > threshold_db)
    if len(active) == 0:
        return audio
    pad = int(sample_rate * padding_seconds)
    start = max(0, active[0] * frame - pad)
    end = min(len(audio), (active[-1] + 1) * frame + pad)
    return audio[start:end]


def apply_fades(audio: np.ndarray, sample_rate: int, fade_seconds: float = 0.015) -> np.ndarray:
    n_fade = min(int(sample_rate * fade_seconds), len(audio) // 2)
    if n_fade <= 0:
        return audio
    audio = audio.copy()
    ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    audio[:n_fade] *= ramp
    audio[-n_fade:] *= ramp[::-1]
    return audio


def prepare_reference_audio(
    audio: np.ndarray,
    sample_rate: int,
    max_duration_seconds: float = 12.0,
    max_pause_seconds: float = 0.4,
    tail_silence_seconds: float = 0.3,
) -> np.ndarray:
    """Clean a voice sample before it is used as a cloning prompt.

    The model continues speech directly from the end of the reference. A
    reference that is cut mid-word leaks a burst of that word into the start
    of every generated line, and long pauses teach the model to pause. This
    function trims the edges, shortens long pauses, cuts at a pause instead of
    mid-word, and ends the sample with a short silence."""
    frame = max(1, int(sample_rate * 0.02))
    n_frames = len(audio) // frame
    if n_frames == 0:
        return audio.astype(np.float32)
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    threshold_db = max(-50.0, float(np.percentile(frame_db, 95)) - 35.0)
    active = frame_db > threshold_db
    speech = np.flatnonzero(active)
    if len(speech) == 0:
        return audio.astype(np.float32)

    edge_pad = int(0.1 / 0.02)
    first = max(0, speech[0] - edge_pad)
    last = min(n_frames - 1, speech[-1] + edge_pad)
    keep = np.zeros(n_frames, dtype=bool)
    keep[first : last + 1] = True

    max_pause_frames = int(max_pause_seconds / 0.02)
    half = max_pause_frames // 2
    index = speech[0]
    while index <= speech[-1]:
        if active[index]:
            index += 1
            continue
        end = index
        while end <= speech[-1] and not active[end]:
            end += 1
        if end - index > max_pause_frames:
            keep[index + half : end - half] = False
        index = end

    kept = np.flatnonzero(keep)
    kept_active = active[kept]
    audio = frames[kept].reshape(-1)

    max_frames = int(max_duration_seconds / 0.02)
    if len(kept) > max_frames:
        window_start = int(max_frames * 0.6)
        pauses = np.flatnonzero(~kept_active[window_start:max_frames])
        if len(pauses):
            cut = window_start + int(pauses[-1])
        else:
            cut = window_start + int(np.argmin(frame_db[kept][window_start:max_frames]))
        audio = audio[: (cut + 1) * frame]

    audio = audio.astype(np.float32).copy()
    n_fade = min(int(sample_rate * 0.02), len(audio) // 2)
    audio[:n_fade] *= np.linspace(0.0, 1.0, n_fade, dtype=np.float32)
    # End on room tone, not digital silence, so the model keeps the background noise.
    tail = room_tone(frames.reshape(-1), sample_rate, int(sample_rate * tail_silence_seconds))
    return join_with_room_tone([(audio, 0.0)], sample_rate, tail=tail)


def room_tone(audio: np.ndarray, sample_rate: int, num_samples: int, seed: int = 0) -> np.ndarray:
    """Synthesize background noise with the spectrum and level of the quietest
    parts of audio. Returns digital silence when audio has no background noise."""
    silence = np.zeros(max(0, num_samples), dtype=np.float32)
    frame = max(1, int(sample_rate * 0.02))
    n_frames = len(audio) // frame
    if num_samples <= 0 or n_frames == 0:
        return silence
    frames = audio[: n_frames * frame].reshape(n_frames, frame)
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    valid = frame_db > -90.0
    if not np.any(valid):
        return silence
    quiet_db = np.percentile(frame_db[valid], 15)
    # Without real pauses, the quietest frames are speech, not background noise.
    if quiet_db > np.percentile(frame_db[valid], 90) - 20.0:
        return silence
    quiet = frames[valid & (frame_db <= quiet_db)].reshape(-1)
    if len(quiet) < 256:
        return silence
    freqs, psd = signal.welch(quiet, fs=sample_rate, nperseg=min(512, len(quiet)))
    rng = np.random.default_rng(seed)
    spectrum = np.fft.rfft(rng.standard_normal(num_samples))
    shape = np.sqrt(np.interp(np.fft.rfftfreq(num_samples, 1.0 / sample_rate), freqs, psd))
    noise = np.fft.irfft(spectrum * shape, n=num_samples)
    noise *= np.sqrt(np.mean(quiet**2)) / (np.sqrt(np.mean(noise**2)) + 1e-12)
    return noise.astype(np.float32)


def join_with_room_tone(
    pieces: list[tuple[np.ndarray, float]],
    sample_rate: int,
    crossfade_seconds: float = 0.03,
    tail: np.ndarray | None = None,
) -> np.ndarray:
    """Join speech pieces. Each piece is (audio, pause_before_seconds). Pauses
    are filled with matching room tone and every edge is crossfaded, so the
    background noise stays constant instead of dropping out between pieces.
    An optional tail of room tone is crossfaded onto the end."""
    if not pieces:
        return np.zeros(0, dtype=np.float32)
    source = np.concatenate([np.asarray(audio, dtype=np.float32) for audio, _ in pieces])
    tail_len = 0 if tail is None else len(tail)
    total = sum(len(audio) + round(sample_rate * pause) for audio, pause in pieces) + tail_len
    bed = (
        room_tone(source, sample_rate, total)
        if tail is None
        else np.concatenate([room_tone(source, sample_rate, total - tail_len), tail])
    )
    speech = np.zeros(total, dtype=np.float32)
    gain = np.zeros(total, dtype=np.float32)
    position = 0
    for audio, pause in pieces:
        position += round(sample_rate * pause)
        length = len(audio)
        speech[position : position + length] = audio
        piece_gain = np.ones(length, dtype=np.float32)
        n_fade = min(int(sample_rate * crossfade_seconds), length // 2)
        if n_fade:
            ramp = np.sin(np.linspace(0.0, np.pi / 2, n_fade, dtype=np.float32))
            if position > 0:
                piece_gain[:n_fade] = ramp
            if position + length < total or tail is not None:
                piece_gain[-n_fade:] = ramp[::-1]
        gain[position : position + length] = piece_gain
        position += length
    # Equal-power crossfade: speech and room tone are uncorrelated noise at the joins.
    return (speech * gain + bed * np.sqrt(1.0 - gain**2)).astype(np.float32)


def reduce_background_noise(
    audio: np.ndarray,
    sample_rate: int,
    reduction_db: float = 15.0,
    lookahead_seconds: float = 0.03,
    release_seconds: float = 0.12,
) -> np.ndarray:
    """Lower steady background hiss in pauses with a soft downward expander.

    Speech passes at full level. The gain opens before each word so onsets are
    not cut, and closes slowly so the background fades instead of pumping."""
    hop = max(1, int(sample_rate * 0.01))
    n_frames = len(audio) // hop
    if n_frames < 2:
        return audio.astype(np.float32)
    frames = audio[: n_frames * hop].reshape(n_frames, hop)
    frame_db = 20.0 * np.log10(np.sqrt(np.mean(frames**2, axis=1)) + 1e-9)
    valid = frame_db > -90.0
    if not np.any(valid):
        return audio.astype(np.float32)
    floor_db = float(np.percentile(frame_db[valid], 15))
    if floor_db > np.percentile(frame_db[valid], 90) - 20.0:
        return audio.astype(np.float32)

    # Full reduction at the noise floor, no reduction 12 dB above it.
    knee = np.clip((frame_db - (floor_db + 3.0)) / 12.0, 0.0, 1.0)
    gain_db = -reduction_db * (1.0 - knee)
    lookahead = max(1, int(lookahead_seconds / 0.01))
    gain_db = maximum_filter1d(gain_db, size=2 * lookahead + 1)
    release = np.exp(-0.01 / release_seconds)
    smoothed = np.empty_like(gain_db)
    current = gain_db[0]
    for index, target in enumerate(gain_db):
        current = target if target > current else release * current + (1.0 - release) * target
        smoothed[index] = current

    positions = (np.arange(n_frames) + 0.5) * hop
    gain = 10.0 ** (np.interp(np.arange(len(audio)), positions, smoothed) / 20.0)
    return (audio * gain).astype(np.float32)


def enhance_audio(
    audio: np.ndarray, sample_rate: int, target_level_db: float = -16.0
) -> np.ndarray:
    """Post-processing chain for synthesized speech: rumble removal, background
    hiss reduction, edge silence trimming, click-free fades, and peak-safe
    loudness normalization."""
    if len(audio) == 0:
        return audio.astype(np.float32)
    audio = high_pass_filter(audio, sample_rate)
    audio = reduce_background_noise(audio, sample_rate)
    audio = trim_silence(audio, sample_rate)
    audio = apply_fades(audio, sample_rate)
    return normalize_audio(audio, target_level_db=target_level_db)


def resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    data_np = audio.squeeze().detach().cpu().numpy()
    num_target_samples = round(len(data_np) * float(target_sr) / orig_sr)
    resampled = signal.resample(data_np, num_target_samples).astype(np.float32)
    return torch.from_numpy(resampled).unsqueeze(0)


SUPPORTED_OUTPUT_FORMATS = {"wav", "mp3"}


def save_audio(
    file_path: str | Path,
    audio: torch.Tensor | np.ndarray,
    sample_rate: int = 24000,
    output_format: str | None = None,
) -> Path:
    target_path = Path(file_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if output_format is None:
        output_format = target_path.suffix.lstrip(".").lower() or "wav"
    output_format = output_format.lower()
    if output_format not in SUPPORTED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output format '{output_format}'. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_OUTPUT_FORMATS))}"
        )

    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().detach().cpu().numpy()
    else:
        audio_np = audio.squeeze()

    audio_np = np.clip(audio_np, -1.0, 1.0)
    if output_format == "mp3":
        sf.write(str(target_path), audio_np, sample_rate, format="MP3", subtype="MPEG_LAYER_III")
    else:
        sf.write(str(target_path), audio_np, sample_rate, format="WAV", subtype="PCM_24")
    return target_path


def slice_audio(
    source_path: str | Path,
    start_sec: float = 0.0,
    end_sec: float | None = None,
    output_path: str | Path | None = None,
) -> tuple[np.ndarray, int]:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")
    data, sr = sf.read(str(path), dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    start_frame = max(0, int(start_sec * sr))
    end_frame = len(data) if end_sec is None else min(len(data), int(end_sec * sr))
    if start_frame >= end_frame:
        start_frame = 0
    sliced = data[start_frame:end_frame]
    if output_path is not None:
        save_audio(output_path, sliced, sample_rate=sr)
    return sliced, sr
