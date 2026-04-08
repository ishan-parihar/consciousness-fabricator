"""
FFmpeg filter graph builder for audio mixing.

Builds FFmpeg `-filter_complex` strings for meditation audio production:
voiceover + ambient music + binaural beats + ducking.

Ported from openscript/crates/openscript-ffmpeg/src/filter_graph.rs (audio-only).
"""

from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Event dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MusicEvent:
    """Ambient music file with volume multiplier."""

    path: str
    volume: float = 0.3


@dataclass
class VoiceoverEvent:
    """TTS voiceover file."""

    path: str
    start_ms: int = 0
    gain_db: float = 0.0


@dataclass
class BinauralEvent:
    """Binaural beat parameters.

    Generates two sine oscillators at slightly different frequencies,
    one per ear, producing a beat frequency equal to their difference.
    """

    carrier_freq_hz: float = 120.0
    beat_freq_hz: float = 10.0
    duration_s: float = 600.0


@dataclass
class DuckingEvent:
    """Sidechain compression parameters for music ducking during voiceover."""

    reduction_db: float = -20.0
    attack_ms: int = 50
    release_ms: int = 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _audio_format_ext(path: str) -> str:
    """Return the file extension for format detection, or empty string."""
    ext = Path(path).suffix.lower().lstrip(".")
    if ext in ("mp3", "wav", "ogg", "flac", "aac", "m4a"):
        return ext
    return ""


def _amovie_filter(path: str, stream: str = "a") -> str:
    """Build an amovie filter string for loading an external audio file."""
    escaped = path.replace("\\", "/")
    fmt = _audio_format_ext(path)
    if fmt:
        return f"amovie='{escaped}':f={fmt}:s={stream}"
    return f"amovie='{escaped}':s={stream}"


def _db_to_linear(db: float) -> float:
    """Convert decibels to linear amplitude multiplier."""
    return 10 ** (db / 20.0)


# ---------------------------------------------------------------------------
# FilterGraphBuilder
# ---------------------------------------------------------------------------


class FilterGraphBuilder:
    """Builder for FFmpeg -filter_complex audio mixing graphs.

    The mixing chain order is:
        loudnorm -> voiceover mixing -> music mixing (with optional ducking)
        -> binaural overlay -> fade out

    Usage:
        builder = (
            FilterGraphBuilder(duration_s=600.0)
            .with_loudnorm()
            .with_voiceover([VoiceoverEvent(path="voice.wav")])
            .with_music([MusicEvent(path="ambient.mp3", volume=0.3)])
            .with_ducking(DuckingEvent(reduction_db=-20.0))
            .with_binaural(BinauralEvent(carrier_freq_hz=120.0, beat_freq_hz=10.0))
            .with_fade_out(start_s=590.0, duration_s=10.0)
        )
        filter_complex = builder.build()
    """

    def __init__(self, duration_s: float = 600.0):
        self.duration_s = duration_s

        # Configured events / flags
        self.loudnorm_enabled: bool = False
        self.voiceover_events: list[VoiceoverEvent] = []
        self.music_events: list[MusicEvent] = []
        self.binaural_event: Optional[BinauralEvent] = None
        self.ducking_event: Optional[DuckingEvent] = None
        self.fade_out_start: Optional[float] = None
        self.fade_out_duration: Optional[float] = None

    # -- Builder methods ----------------------------------------------------

    def with_loudnorm(self) -> "FilterGraphBuilder":
        """Enable EBU R128 loudness normalization (I=-16, TP=-1.5, LRA=11)."""
        self.loudnorm_enabled = True
        return self

    def with_voiceover(self, events: list[VoiceoverEvent]) -> "FilterGraphBuilder":
        """Add voiceover events to be mixed into the main audio."""
        self.voiceover_events = list(events)
        return self

    def with_music(self, events: list[MusicEvent]) -> "FilterGraphBuilder":
        """Add ambient music events to be mixed into the main audio."""
        self.music_events = list(events)
        return self

    def with_binaural(
        self, event: Optional[BinauralEvent] = None
    ) -> "FilterGraphBuilder":
        """Add a binaural beat generator.

        If *event* is None, a default 120 Hz carrier / 10 Hz beat is used.
        """
        self.binaural_event = event or BinauralEvent()
        return self

    def with_ducking(
        self, event: Optional[DuckingEvent] = None
    ) -> "FilterGraphBuilder":
        """Enable sidechain ducking of music during voiceover.

        The sidechain source is the voiceover track.
        """
        self.ducking_event = event or DuckingEvent()
        return self

    def with_fade_out(
        self, start_s: float, duration_s: float = 10.0
    ) -> "FilterGraphBuilder":
        """Add a fade-out at the end of the output."""
        self.fade_out_start = start_s
        self.fade_out_duration = duration_s
        return self

    # -- Build --------------------------------------------------------------

    def build(self) -> str:
        """Return the complete -filter_complex string.

        Returns an empty string if no processing is configured (passthrough).
        """
        parts: list[str] = []
        current_audio = "[0:a]"

        # 1. Loudnorm
        if self.loudnorm_enabled:
            parts.append(f"[0:a]loudnorm=I=-16:TP=-1.5:LRA=11[loud]")
            current_audio = "[loud]"

        # 2. Voiceover mixing
        if self.voiceover_events:
            current_audio = self._mix_voiceovers(parts, current_audio)

        # 3. Music mixing (with optional ducking)
        has_ducking = self.ducking_event is not None and self.music_events
        sidechain_label: Optional[str] = None

        if has_ducking:
            prev_label = current_audio.strip("[]")
            parts.append(f"[{prev_label}]asplit=2[music_in][sidechain_src]")
            current_audio = "[music_in]"
            sidechain_label = "[sidechain_src]"

        if self.music_events:
            current_audio = self._mix_music(parts, current_audio, sidechain_label)

        # 4. Binaural beat overlay
        if self.binaural_event:
            current_audio = self._add_binaural(parts, current_audio)

        # 5. Fade out
        if self.fade_out_start is not None and self.fade_out_duration is not None:
            prev_label = current_audio.strip("[]")
            out_label = "aout"
            parts.append(
                f"[{prev_label}]afade=t=out:st={self.fade_out_start}"
                f":d={self.fade_out_duration}[{out_label}]"
            )
            current_audio = f"[{out_label}]"

        # If nothing was added, just label the input as aout
        if not parts:
            parts.append("[0:a][aout]")
            return ""

        final_label = current_audio.strip("[]")
        if final_label != "aout" and parts:
            last = parts[-1]
            if last.endswith(current_audio):
                parts[-1] = last[: -len(current_audio)] + "[aout]"
            else:
                parts.append(f"[{final_label}]anull[aout]")
        elif not parts:
            parts.append("[0:a][aout]")

        return ",".join(parts)

    # -- Internal mixing helpers --------------------------------------------

    def _mix_voiceovers(self, parts: list[str], input_label: str) -> str:
        """Mix voiceover events into the main audio stream."""
        current = input_label

        for i, vo in enumerate(self.voiceover_events):
            gain = _db_to_linear(vo.gain_db)

            parts.append(f"{_amovie_filter(vo.path, 'a')}[vo_{i}]")
            parts.append(f"[vo_{i}]volume={gain}[vo_vol_{i}]")
            parts.append(
                f"[vo_vol_{i}]adelay={vo.start_ms}|{vo.start_ms}[vo_delayed_{i}]"
            )

            out_label = (
                "[amix_voiceover]"
                if i == len(self.voiceover_events) - 1
                else f"[vo_mix_{i}]"
            )
            parts.append(
                f"{current}[vo_delayed_{i}]"
                f"amix=inputs=2:duration=first:dropout_transition=1:normalize=0"
                f"{out_label}"
            )
            current = out_label

        return current

    def _mix_music(
        self,
        parts: list[str],
        input_label: str,
        sidechain_label: Optional[str],
    ) -> str:
        """Mix music events, optionally with sidechain ducking."""
        ducking = self.ducking_event
        current = input_label

        for i, music in enumerate(self.music_events):
            parts.append(f"{_amovie_filter(music.path, 'a')}[music_{i}]")
            parts.append(f"[music_{i}]volume={music.volume}[music_vol_{i}]")

            if ducking and sidechain_label:
                attack = ducking.attack_ms
                release = ducking.release_ms
                threshold = _db_to_linear(ducking.reduction_db)

                ducked_label = f"[music_ducked_{i}]"
                parts.append(
                    f"[music_vol_{i}]{sidechain_label}"
                    f"sidechaincompress="
                    f"threshold={threshold:.4f}"
                    f":ratio=4"
                    f":attack={attack}"
                    f":release={release}"
                    f":makeup=1"
                    f":level_sc=1"
                    f"{ducked_label}"
                )
                music_out = ducked_label
            else:
                music_out = f"[music_vol_{i}]"

            if i == 0:
                out_label = f"[amix_{i}]"
                parts.append(
                    f"{current}{music_out}"
                    f"amix=inputs=2:duration=first:dropout_transition=2:normalize=0"
                    f"{out_label}"
                )
            else:
                prev = f"[amix_{i - 1}]"
                out_label = f"[amix_{i}]"
                parts.append(
                    f"{prev}{music_out}"
                    f"amix=inputs=2:duration=first:dropout_transition=2:normalize=0"
                    f"{out_label}"
                )

            current = out_label

        return current

    def _add_binaural(self, parts: list[str], input_label: str) -> str:
        """Generate binaural beats and mix them with the main audio."""
        evt = self.binaural_event
        assert evt is not None

        carrier = evt.carrier_freq_hz
        beat = evt.beat_freq_hz
        duration = evt.duration_s

        binaural_label = "[binaural]"
        parts.append(
            f"aevalsrc="
            f"'sin(2*PI*{carrier}*t)"
            f"|sin(2*PI*{carrier + beat}*t)'"
            f":d={duration}"
            f"{binaural_label}"
        )

        out_label = "[binaural_mixed]"
        parts.append(
            f"{input_label}{binaural_label}"
            f"amix=inputs=2:duration=first:dropout_transition=2:normalize=0"
            f"{out_label}"
        )
        return out_label


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------


def render(
    filter_complex: str,
    inputs: list[str],
    output_path: str,
    duration_s: float,
    progress_callback=None,
) -> tuple[int, float]:
    """Spawn FFmpeg subprocess with the given filter_complex and wait for completion.

    Args:
        filter_complex: The -filter_complex string from FilterGraphBuilder.build().
        inputs: List of input file paths (used as -i arguments).
        output_path: Path for the output file.
        duration_s: Expected duration in seconds (for progress reporting).
        progress_callback: Optional callable(progress: float, elapsed: float, eta: float)
            called periodically with progress 0.0-1.0.

    Returns:
        (exit_code, elapsed_seconds)
    """
    # Build input arguments
    input_args: list[str] = []
    for inp in inputs:
        input_args.extend(["-i", inp])

    # Build the full FFmpeg command
    cmd: list[str] = [
        "ffmpeg",
        *input_args,
        "-filter_complex",
        filter_complex,
        "-map",
        "[aout]",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-y",
        output_path,
    ]

    start_time = time.monotonic()

    proc = subprocess.Popen(
        cmd,
        stderr=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        universal_newlines=True,
    )

    time_pattern = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")
    last_progress = 0.0

    assert proc.stderr is not None
    try:
        for line in proc.stderr:
            match = time_pattern.search(line)
            if match:
                hours = int(match.group(1))
                minutes = int(match.group(2))
                seconds = float(match.group(3))
                elapsed_render = hours * 3600 + minutes * 60 + seconds

                if duration_s > 0:
                    progress = min(elapsed_render / duration_s, 1.0)
                else:
                    progress = 0.0

                if progress != last_progress:
                    last_progress = progress
                    wall_elapsed = time.monotonic() - start_time
                    if progress > 0:
                        eta = wall_elapsed / progress - wall_elapsed
                    else:
                        eta = 0.0

                    if progress_callback:
                        progress_callback(progress, wall_elapsed, eta)

    except Exception:
        pass

    exit_code = proc.wait()
    total_elapsed = time.monotonic() - start_time

    return exit_code, total_elapsed
