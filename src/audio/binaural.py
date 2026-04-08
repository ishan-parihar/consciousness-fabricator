"""
Binaural beat generator — creates stereo WAV files using NumPy oscillators.

Binaural beats work by playing slightly different frequencies in each ear.
The brain perceives the difference as a beat frequency.

Left ear:  sin(2π * carrier_freq * t)
Right ear: sin(2π * (carrier_freq + beat_freq) * t)

Typical carrier: 100-200 Hz. Beat frequencies map to brainwave states.
"""

from __future__ import annotations

import wave
from pathlib import Path
from typing import Literal

import numpy as np

# ---------------------------------------------------------------------------
# Brainwave presets
# ---------------------------------------------------------------------------

BRAINWAVE_PRESETS: dict[str, dict[str, float | str]] = {
    "delta": {"beat_freq": 2.0, "description": "Deep sleep and healing"},
    "theta": {"beat_freq": 6.0, "description": "Deep meditation and creativity"},
    "alpha": {"beat_freq": 10.0, "description": "Relaxed awareness and calm"},
    "beta": {"beat_freq": 20.0, "description": "Active thinking and focus"},
    "gamma": {"beat_freq": 40.0, "description": "Heightened perception"},
}

# ---------------------------------------------------------------------------
# Core generation
# ---------------------------------------------------------------------------


def generate_binaural(
    carrier_freq_hz: float,
    beat_freq_hz: float,
    duration_s: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate a binaural beat as a stereo NumPy array.

    Args:
        carrier_freq_hz: Base frequency (left ear), typically 100-200 Hz.
        beat_freq_hz: Frequency difference between ears — the perceived beat.
        duration_s: Duration in seconds.
        sample_rate: Samples per second.

    Returns:
        Stereo audio array with shape (n_samples, 2), dtype float32,
        values in [-0.5, 0.5].
    """
    n_samples = int(duration_s * sample_rate)
    t = np.arange(n_samples, dtype=np.float32) / sample_rate

    amplitude = 0.5  # prevent clipping when mixed with voice
    left = amplitude * np.sin(2.0 * np.pi * carrier_freq_hz * t)
    right = amplitude * np.sin(2.0 * np.pi * (carrier_freq_hz + beat_freq_hz) * t)

    audio: np.ndarray = np.column_stack((left, right))

    # Apply fade-in (first 2s) and fade-out (last 2s)
    audio = _apply_fades(audio, sample_rate, fade_in_s=2.0, fade_out_s=2.0)

    return audio


def generate_binaural_with_preset(
    preset_name: Literal["delta", "theta", "alpha", "beta", "gamma"],
    carrier_freq_hz: float,
    duration_s: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """Generate a binaural beat using a named brainwave preset.

    Args:
        preset_name: One of delta, theta, alpha, beta, gamma.
        carrier_freq_hz: Base frequency (left ear), typically 100-200 Hz.
        duration_s: Duration in seconds.
        sample_rate: Samples per second.

    Returns:
        Stereo audio array with shape (n_samples, 2).

    Raises:
        KeyError: If preset_name is not a valid preset.
    """
    preset = BRAINWAVE_PRESETS[preset_name]
    beat_freq = preset["beat_freq"]
    return generate_binaural(
        carrier_freq_hz=carrier_freq_hz,
        beat_freq_hz=float(beat_freq),
        duration_s=duration_s,
        sample_rate=sample_rate,
    )


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_binaural(
    audio: np.ndarray,
    output_path: str | Path,
    sample_rate: int = 44100,
) -> None:
    """Save a stereo NumPy array as a 16-bit PCM WAV file.

    Tries ``soundfile`` first for a one-liner, falls back to stdlib ``wave``.

    Args:
        audio: Stereo array with shape (n_samples, 2), values in [-1, 1].
        output_path: Destination file path.
        sample_rate: Samples per second.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Attempt soundfile (preferred — handles float-to-PCM conversion)
    try:
        import soundfile as sf

        sf.write(str(path), audio, sample_rate, subtype="PCM_16")
        return
    except ImportError:
        pass

    # Fallback: stdlib wave module — manual 16-bit PCM encoding
    _save_wav_fallback(path, audio, sample_rate)


def generate_and_save(
    preset_name: Literal["delta", "theta", "alpha", "beta", "gamma"],
    carrier_freq_hz: float,
    duration_s: float,
    output_path: str | Path,
    sample_rate: int = 44100,
) -> None:
    """Full pipeline: generate a binaural beat from a preset and save as WAV.

    Args:
        preset_name: One of delta, theta, alpha, beta, gamma.
        carrier_freq_hz: Base frequency (left ear), typically 100-200 Hz.
        duration_s: Duration in seconds.
        output_path: Destination WAV file path.
        sample_rate: Samples per second.
    """
    audio = generate_binaural_with_preset(
        preset_name=preset_name,
        carrier_freq_hz=carrier_freq_hz,
        duration_s=duration_s,
        sample_rate=sample_rate,
    )
    save_binaural(audio, output_path, sample_rate)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _apply_fades(
    audio: np.ndarray,
    sample_rate: int,
    fade_in_s: float = 2.0,
    fade_out_s: float = 2.0,
) -> np.ndarray:
    """Apply linear fade-in at the start and fade-out at the end.

    Args:
        audio: Stereo array (n_samples, 2).
        sample_rate: Samples per second.
        fade_in_s: Fade-in duration in seconds.
        fade_out_s: Fade-out duration in seconds.

    Returns:
        Audio array with fades applied.
    """
    n_samples = audio.shape[0]
    fade_in_len = min(int(fade_in_s * sample_rate), n_samples)
    fade_out_len = min(int(fade_out_s * sample_rate), n_samples)

    if fade_in_len > 0:
        fade_in_curve = np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
        audio[:fade_in_len] *= fade_in_curve[:, np.newaxis]

    if fade_out_len > 0 and fade_out_len < n_samples:
        fade_out_curve = np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)
        audio[-fade_out_len:] *= fade_out_curve[:, np.newaxis]

    return audio


def _save_wav_fallback(
    path: Path,
    audio: np.ndarray,
    sample_rate: int,
) -> None:
    """Write stereo 16-bit PCM WAV using only the stdlib wave module.

    Args:
        path: Output file path.
        audio: Stereo array (n_samples, 2), values in [-1, 1].
        sample_rate: Samples per second.
    """
    # Clamp and convert to 16-bit signed integers
    audio_clipped = np.clip(audio, -1.0, 1.0)
    audio_int16 = (audio_clipped * 32767).astype(np.int16)

    # Interleave channels: L R L R ...
    interleaved = audio_int16.reshape(-1)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(interleaved.tobytes())
