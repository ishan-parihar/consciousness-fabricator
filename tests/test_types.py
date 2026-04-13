"""Tests for enums and dataclasses in src/types.py."""

import pytest

from src.types import (
    MeditationStyle,
    Brainwave,
    BinauralConfig,
    PacingConfig,
    ToneConfig,
    LanguageConfig,
    TrajectoryConfig,
    MeditationReference,
    SessionRequest,
    SessionState,
)


# ---------------------------------------------------------------------------
# MeditationStyle
# ---------------------------------------------------------------------------


class TestMeditationStyle:
    def test_silva_method_value(self):
        assert MeditationStyle.SILVA_METHOD.value == "silva-method"

    def test_shadow_realm_value(self):
        assert MeditationStyle.SHADOW_REALM.value == "shadow-realm"

    def test_invalid_style_raises_value_error(self):
        with pytest.raises(ValueError):
            MeditationStyle("invalid-style")

    def test_all_styles_have_values(self):
        for style in MeditationStyle:
            assert style.value
            assert isinstance(style.value, str)


# ---------------------------------------------------------------------------
# Brainwave
# ---------------------------------------------------------------------------


class TestBrainwave:
    def test_alpha_value(self):
        assert Brainwave.ALPHA.value == "alpha"

    def test_theta_value(self):
        assert Brainwave.THETA.value == "theta"

    def test_delta_value(self):
        assert Brainwave.DELTA.value == "delta"

    def test_beta_value(self):
        assert Brainwave.BETA.value == "beta"

    def test_gamma_value(self):
        assert Brainwave.GAMMA.value == "gamma"

    def test_all_five_brainwaves(self):
        assert len(list(Brainwave)) == 5


# ---------------------------------------------------------------------------
# BinauralConfig
# ---------------------------------------------------------------------------


class TestBinauralConfig:
    def test_alpha_config(self):
        config = BinauralConfig(
            brainwave=Brainwave.ALPHA,
            carrier_freq_hz=120.0,
            beat_freq_hz=10.0,
        )
        assert config.brainwave == Brainwave.ALPHA
        assert config.carrier_freq_hz == 120.0
        assert config.beat_freq_hz == 10.0

    def test_theta_config(self):
        config = BinauralConfig(
            brainwave=Brainwave.THETA,
            carrier_freq_hz=140.0,
            beat_freq_hz=6.0,
        )
        assert config.brainwave == Brainwave.THETA
        assert config.carrier_freq_hz == 140.0
        assert config.beat_freq_hz == 6.0

    def test_defaults(self):
        config = BinauralConfig(brainwave=Brainwave.ALPHA)
        assert config.carrier_freq_hz == 120.0
        assert config.beat_freq_hz == 10.0


# ---------------------------------------------------------------------------
# PacingConfig
# ---------------------------------------------------------------------------


class TestPacingConfig:
    def test_with_act_boundaries(self):
        config = PacingConfig(
            act_boundaries=[120, 300, 480],
        )
        assert config.act_boundaries == [120, 300, 480]

    def test_defaults(self):
        config = PacingConfig()
        assert config.avg_speaking_rate_wpm == 98.0
        assert config.instruction_pause_seconds == 4.2
        assert config.body_scan_pause_seconds == 8.5
        assert config.countdown_pause_seconds == 2.1
        assert config.act_boundaries == []


# ---------------------------------------------------------------------------
# ToneConfig
# ---------------------------------------------------------------------------


class TestToneConfig:
    def test_construction(self):
        config = ToneConfig(
            energy="low",
            warmth="warm",
            formality="instructor",
            description="Calm guidance",
        )
        assert config.energy == "low"
        assert config.warmth == "warm"
        assert config.formality == "instructor"
        assert config.description == "Calm guidance"


# ---------------------------------------------------------------------------
# TrajectoryConfig
# ---------------------------------------------------------------------------


class TestTrajectoryConfig:
    def test_construction(self):
        config = TrajectoryConfig(
            opening="Gentle induction",
            deepening="Progressive relaxation",
            transitions="Smooth flow",
            deviation_handling="Acknowledge and return",
        )
        assert config.opening == "Gentle induction"
        assert config.deepening == "Progressive relaxation"
        assert config.transitions == "Smooth flow"
        assert config.deviation_handling == "Acknowledge and return"


# ---------------------------------------------------------------------------
# MeditationReference
# ---------------------------------------------------------------------------


class TestMeditationReference:
    def test_full_construction(self, sample_reference):
        assert sample_reference.id == "test-meditation"
        assert sample_reference.name == "Test Meditation"
        assert sample_reference.collection == "test-collection"
        assert sample_reference.category == "relaxation"
        assert sample_reference.total_duration_seconds == 600
        assert sample_reference.total_phrases == 100
        assert isinstance(sample_reference.tone, ToneConfig)
        assert isinstance(sample_reference.pacing, PacingConfig)
        assert isinstance(sample_reference.language, LanguageConfig)
        assert isinstance(sample_reference.trajectory, TrajectoryConfig)
        assert isinstance(sample_reference.binaural, BinauralConfig)


# ---------------------------------------------------------------------------
# SessionRequest
# ---------------------------------------------------------------------------


class TestSessionRequest:
    def test_with_voice_profile(self, sample_session_request):
        assert sample_session_request.style == MeditationStyle.SILVA_METHOD
        assert sample_session_request.duration_minutes == 10
        assert sample_session_request.user_request == "relaxation"
        assert sample_session_request.voice_profile_id == "calm_instructor"
        assert sample_session_request.deviation is False

    def test_without_voice_profile(self):
        request = SessionRequest(
            style=MeditationStyle.SHADOW_REALM,
            duration_minutes=15,
            user_request="inner journey",
        )
        assert request.voice_profile_id is None
        assert request.deviation is False


# ---------------------------------------------------------------------------
# SessionState
# ---------------------------------------------------------------------------


class TestSessionState:
    def test_initial_state(self, sample_session_state):
        assert sample_session_state.playhead == 0
        assert sample_session_state.total_chunks == 0
        assert sample_session_state.is_playing is False
        assert sample_session_state.is_deviation is False

    def test_playing_state_toggle(self, sample_session_state):
        # Start not playing
        assert sample_session_state.is_playing is False

        # Toggle to playing
        sample_session_state.is_playing = True
        assert sample_session_state.is_playing is True

        # Toggle back
        sample_session_state.is_playing = False
        assert sample_session_state.is_playing is False
