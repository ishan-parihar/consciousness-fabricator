"""Tests for FFmpeg FilterGraphBuilder in src/audio/mixer.py."""

import pytest

from src.audio.mixer import (
    FilterGraphBuilder,
    MusicEvent,
    VoiceoverEvent,
    BinauralEvent,
    DuckingEvent,
    _db_to_linear,
    _amovie_filter,
)


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderEmpty
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderEmpty:
    def test_no_processing_returns_empty_string(self):
        """Builder with no processing configured returns empty string (passthrough)."""
        result = FilterGraphBuilder().build()
        assert result == ""


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderLoudnorm
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderLoudnorm:
    def test_loudnorm_enabled(self):
        """Loudnorm filter includes EBU R128 parameters."""
        result = FilterGraphBuilder().with_loudnorm().build()
        assert "loudnorm" in result
        assert "I=-16" in result
        assert "TP=-1.5" in result


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderVoiceover
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderVoiceover:
    def test_single_voiceover(self):
        """Single voiceover includes amovie filter and file path."""
        path = "/tmp/voice.wav"
        result = (
            FilterGraphBuilder().with_voiceover([VoiceoverEvent(path=path)]).build()
        )
        assert "amovie" in result
        assert path in result

    def test_multiple_voiceovers(self):
        """Multiple voiceovers include all file paths."""
        path1 = "/tmp/voice1.wav"
        path2 = "/tmp/voice2.wav"
        result = (
            FilterGraphBuilder()
            .with_voiceover(
                [
                    VoiceoverEvent(path=path1),
                    VoiceoverEvent(path=path2),
                ]
            )
            .build()
        )
        assert path1 in result
        assert path2 in result


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderMusic
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderMusic:
    def test_single_music_track(self):
        """Single music track includes file path and volume multiplier."""
        path = "/tmp/ambient.mp3"
        result = (
            FilterGraphBuilder().with_music([MusicEvent(path=path, volume=0.3)]).build()
        )
        assert path in result
        assert "volume=0.3" in result


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderBinaural
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderBinaural:
    def test_default_binaural(self):
        """Default binaural uses 120 Hz carrier."""
        result = FilterGraphBuilder().with_binaural().build()
        assert "aevalsrc" in result
        assert "120.0" in result

    def test_custom_binaural(self):
        """Custom binaural uses specified carrier and carrier+beat frequencies."""
        result = (
            FilterGraphBuilder()
            .with_binaural(BinauralEvent(carrier_freq_hz=150.0, beat_freq_hz=6.0))
            .build()
        )
        assert "150.0" in result
        assert "156.0" in result


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderDucking
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderDucking:
    def test_ducking_with_music(self):
        """Ducking with music includes sidechaincompress filter."""
        result = (
            FilterGraphBuilder()
            .with_music([MusicEvent(path="/tmp/ambient.mp3")])
            .with_voiceover([VoiceoverEvent(path="/tmp/voice.wav")])
            .with_ducking(DuckingEvent(reduction_db=-20.0))
            .build()
        )
        assert "sidechaincompress" in result


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderFadeOut
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderFadeOut:
    def test_fade_out_added(self):
        """Fade-out includes afade filter with t=out."""
        result = (
            FilterGraphBuilder()
            .with_loudnorm()
            .with_fade_out(start_s=590.0, duration_s=10.0)
            .build()
        )
        assert "afade" in result
        assert "t=out" in result


# ---------------------------------------------------------------------------
# TestFilterGraphBuilderFullChain
# ---------------------------------------------------------------------------


class TestFilterGraphBuilderFullChain:
    def test_complete_mixing_chain(self):
        """Full chain includes loudnorm, amix, aevalsrc, and afade."""
        result = (
            FilterGraphBuilder(duration_s=600.0)
            .with_loudnorm()
            .with_voiceover([VoiceoverEvent(path="/tmp/voice.wav")])
            .with_music([MusicEvent(path="/tmp/ambient.mp3", volume=0.3)])
            .with_binaural(BinauralEvent(carrier_freq_hz=120.0, beat_freq_hz=10.0))
            .with_fade_out(start_s=590.0, duration_s=10.0)
            .build()
        )
        assert "loudnorm" in result
        assert "amix" in result
        assert "aevalsrc" in result
        assert "afade" in result


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    def test_db_to_linear_zero_db(self):
        """0 dB equals linear amplitude of 1.0."""
        assert _db_to_linear(0.0) == pytest.approx(1.0)

    def test_db_to_linear_minus_20_db(self):
        """-20 dB equals linear amplitude of approximately 0.1."""
        assert _db_to_linear(-20.0) == pytest.approx(0.1)

    def test_amovie_filter_with_format(self):
        """MP3 extension produces f=mp3 in amovie filter."""
        result = _amovie_filter("/path/to/track.mp3")
        assert "f=mp3" in result

    def test_amovie_filter_without_format(self):
        """Unknown extension does NOT include f= parameter."""
        result = _amovie_filter("/path/to/track.unknown")
        assert "f=" not in result
