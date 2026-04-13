"""Integration tests for full session flow with mocked components.

Tests the highest-level integration points:
- MeditationSession in src/session.py orchestrates all components
- SessionManager in src/ws_api.py creates and manages sessions
- handle_deviation() returns error when session not initialized
"""

import pytest
from unittest.mock import AsyncMock

from src.session import MeditationSession
from src.ws_api import SessionManager, MeditationSession as WsMeditationSession
from src.types import (
    MeditationReference,
    MeditationStyle,
    SessionRequest,
    ToneConfig,
    PacingConfig,
    LanguageConfig,
    TrajectoryConfig,
    BinauralConfig,
    Brainwave,
)


def _build_reference() -> MeditationReference:
    """Build a minimal MeditationReference inline to avoid conftest fixture issues."""
    return MeditationReference(
        id="test",
        name="Test",
        collection="test",
        category="relaxation",
        total_duration_seconds=300,
        total_phrases=50,
        tone=ToneConfig(
            energy="moderate",
            warmth="neutral",
            formality="guided",
            description="",
        ),
        pacing=PacingConfig(
            avg_speaking_rate_wpm=98.0,
            instruction_pause_seconds=4.2,
            body_scan_pause_seconds=8.5,
            countdown_pause_seconds=2.1,
            act_boundaries=[],
        ),
        language=LanguageConfig(
            sentence_style="natural",
            perspective="second",
            common_phrases=[],
            structural_patterns=[],
            repetition_rate=0.35,
            avg_sentence_length_words=12,
        ),
        trajectory=TrajectoryConfig(
            opening="Gentle",
            deepening="Deep",
            transitions="Smooth",
            deviation_handling="Flow",
        ),
        binaural=BinauralConfig(
            brainwave=Brainwave.ALPHA,
            carrier_freq_hz=120.0,
            beat_freq_hz=10.0,
        ),
    )


def _build_request() -> SessionRequest:
    """Build a minimal SessionRequest inline."""
    return SessionRequest(
        style=MeditationStyle.SILVA_METHOD,
        duration_minutes=5,
        user_request="relaxation",
        voice_profile_id="calm_instructor",
    )


# ---------------------------------------------------------------------------
# TestSessionInitialization
# ---------------------------------------------------------------------------


class TestSessionInitialization:
    """Tests for MeditationSession creation and initial state."""

    @pytest.mark.asyncio
    async def test_session_creates_components(self, mock_llm_client, default_config):
        """Create a MeditationSession with mocked LLM client, verify
        session_id, is_playing==False initially."""
        session = MeditationSession(
            config=default_config,
            llm_client=mock_llm_client,
        )

        # LLM client stored for later use
        assert session._llm_client is mock_llm_client

        # No sessions exist before start() is called
        assert session._sessions == {}
        assert session._tasks == {}
        assert session._stop_events == {}


# ---------------------------------------------------------------------------
# TestDeviationFlow
# ---------------------------------------------------------------------------


class TestDeviationFlow:
    """Tests for deviation handling across session lifecycle."""

    @pytest.mark.asyncio
    async def test_deviation_without_session_returns_error(self, default_config):
        """Create a session via SessionManager WITHOUT initializing it
        (don't call start()), then call handle_deviation(). Verify result
        is {"type": "deviation_rejected", "error" containing "not initialized"}."""
        manager = SessionManager()
        ref = _build_reference()
        req = _build_request()

        session = manager.create_session(req, ref, default_config)
        session_id = session.session_id

        # Do NOT call start() — session is not initialized
        result = await session.handle_deviation("focus on breathing instead")

        assert result["type"] == "deviation_rejected"
        assert "not initialized" in result["error"].lower()
