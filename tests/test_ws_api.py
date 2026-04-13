"""Tests for WebSocket API module (src/ws_api.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.ws_api import SessionManager, _parse_reference, MeditationSession
from src.types import Brainwave, MeditationReference, SessionRequest, MeditationStyle
from src.config import EngineConfig


# ---------------------------------------------------------------------------
# TestParseReference
# ---------------------------------------------------------------------------


class TestParseReference:
    """Tests for _parse_reference function."""

    def test_parse_minimal_reference(self):
        """Parse a minimal valid reference dict into MeditationReference."""
        data = {
            "id": "minimal-test",
            "name": "Minimal Test",
            "collection": "test",
            "category": "general",
            "total_duration_seconds": 300,
            "total_phrases": 50,
            "tone": {
                "energy": "low",
                "warmth": "warm",
                "formality": "guided",
                "description": "test",
            },
            "pacing": {
                "avg_speaking_rate_wpm": 98.0,
                "instruction_pause_seconds": 4.2,
                "body_scan_pause_seconds": 8.5,
                "countdown_pause_seconds": 2.1,
                "act_boundaries": [],
            },
            "language": {
                "sentence_style": "natural",
                "perspective": "second",
                "common_phrases": [],
                "structural_patterns": [],
                "repetition_rate": 0.35,
                "avg_sentence_length_words": 12,
            },
            "trajectory": {
                "opening": "induction",
                "deepening": "relaxation",
                "transitions": "smooth",
                "deviation_handling": "natural",
            },
            "binaural": {
                "brainwave": "alpha",
                "carrier_freq_hz": 150.0,
                "beat_freq_hz": 10.0,
            },
        }

        result = _parse_reference(data)

        assert result.id == "minimal-test"
        assert result.binaural.carrier_freq_hz == 150.0

    def test_parse_invalid_brainwave_defaults_to_alpha(self):
        """When binaural.brainwave is invalid, defaults to Brainwave.ALPHA."""
        data = {
            "id": "test-invalid",
            "name": "Test",
            "collection": "test",
            "category": "general",
            "total_duration_seconds": 600,
            "total_phrases": 100,
            "tone": {
                "energy": "moderate",
                "warmth": "neutral",
                "formality": "guided",
                "description": "",
            },
            "pacing": {
                "avg_speaking_rate_wpm": 98.0,
                "instruction_pause_seconds": 4.2,
                "body_scan_pause_seconds": 8.5,
                "countdown_pause_seconds": 2.1,
                "act_boundaries": [],
            },
            "language": {
                "sentence_style": "natural",
                "perspective": "second",
                "common_phrases": [],
                "structural_patterns": [],
                "repetition_rate": 0.35,
                "avg_sentence_length_words": 12,
            },
            "trajectory": {
                "opening": "induction",
                "deepening": "relaxation",
                "transitions": "smooth",
                "deviation_handling": "natural",
            },
            "binaural": {
                "brainwave": "invalid-brainwave",
                "carrier_freq_hz": 120.0,
                "beat_freq_hz": 10.0,
            },
        }

        result = _parse_reference(data)

        assert result.binaural.brainwave == Brainwave.ALPHA


# ---------------------------------------------------------------------------
# TestSessionManager
# ---------------------------------------------------------------------------


class TestSessionManager:
    """Tests for SessionManager class."""

    def _make_reference(self) -> MeditationReference:
        """Build a minimal MeditationReference inline."""
        from src.types import (
            ToneConfig,
            PacingConfig,
            LanguageConfig,
            TrajectoryConfig,
            BinauralConfig,
        )

        return MeditationReference(
            id="test-ref",
            name="Test",
            collection="test",
            category="general",
            total_duration_seconds=600,
            total_phrases=100,
            tone=ToneConfig(
                energy="moderate",
                warmth="neutral",
                formality="guided",
                description="",
            ),
            pacing=PacingConfig(),
            language=LanguageConfig(
                sentence_style="natural",
                perspective="second",
                common_phrases=[],
                structural_patterns=[],
            ),
            trajectory=TrajectoryConfig(
                opening="",
                deepening="",
                transitions="",
                deviation_handling="",
            ),
            binaural=BinauralConfig(
                brainwave=Brainwave.ALPHA,
                carrier_freq_hz=120.0,
                beat_freq_hz=10.0,
            ),
        )

    def test_create_session(self):
        """create_session returns a session, session_id is in active_sessions."""
        manager = SessionManager()
        request = SessionRequest(
            style=MeditationStyle.SILVA_METHOD,
            duration_minutes=10,
            user_request="relaxation",
        )
        reference = self._make_reference()
        config = EngineConfig.default()

        session = manager.create_session(request, reference, config)

        assert session is not None
        assert session.session_id in manager.active_sessions

    def test_get_session(self):
        """Creating then getting returns the same session object."""
        manager = SessionManager()
        request = SessionRequest(
            style=MeditationStyle.SILVA_METHOD,
            duration_minutes=10,
            user_request="relaxation",
        )
        reference = self._make_reference()
        config = EngineConfig.default()

        created = manager.create_session(request, reference, config)
        retrieved = manager.get_session(created.session_id)

        assert retrieved is created

    def test_get_missing_session(self):
        """get_session for nonexistent ID returns None."""
        manager = SessionManager()
        result = manager.get_session("nonexistent")
        assert result is None


# ---------------------------------------------------------------------------
# TestSessionLifecycle
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    """Async tests for MeditationSession lifecycle."""

    def _make_reference(self) -> MeditationReference:
        """Build a minimal MeditationReference inline."""
        from src.types import (
            ToneConfig,
            PacingConfig,
            LanguageConfig,
            TrajectoryConfig,
            BinauralConfig,
        )

        return MeditationReference(
            id="test-ref",
            name="Test",
            collection="test",
            category="general",
            total_duration_seconds=600,
            total_phrases=100,
            tone=ToneConfig(
                energy="moderate",
                warmth="neutral",
                formality="guided",
                description="",
            ),
            pacing=PacingConfig(),
            language=LanguageConfig(
                sentence_style="natural",
                perspective="second",
                common_phrases=[],
                structural_patterns=[],
            ),
            trajectory=TrajectoryConfig(
                opening="",
                deepening="",
                transitions="",
                deviation_handling="",
            ),
            binaural=BinauralConfig(
                brainwave=Brainwave.ALPHA,
                carrier_freq_hz=120.0,
                beat_freq_hz=10.0,
            ),
        )

    def _make_request(self) -> SessionRequest:
        return SessionRequest(
            style=MeditationStyle.SILVA_METHOD,
            duration_minutes=10,
            user_request="relaxation",
        )

    @pytest.mark.asyncio
    async def test_session_start(self, mock_llm_client):
        """start() returns session_id in info and calls generate on LLM."""
        from src.ws_api import SessionComponents

        reference = self._make_reference()
        request = self._make_request()
        config = EngineConfig.default()

        session = MeditationSession(
            session_id="test-session-1",
            request=request,
            reference=reference,
            config=config,
        )

        session._llm_client = mock_llm_client

        mock_text_buffer = MagicMock()
        mock_text_buffer.chunks = []
        mock_text_buffer.is_complete.return_value = True
        mock_text_buffer.spoken_chunks = []

        mock_components = SessionComponents(
            config=config,
            reference=reference,
            text_buffer=mock_text_buffer,
            context_window=MagicMock(),
            tts_client=MagicMock(),
            llm_client=mock_llm_client,
            deviation_handler=MagicMock(),
        )

        async def fake_initialize():
            mock_llm_client.generate.return_value = "test script"
            await mock_llm_client.generate(messages=[{"role": "system", "content": ""}])
            return mock_components

        session._initialize = fake_initialize

        info = await session.start()

        assert info["session_id"] == "test-session-1"
        mock_llm_client.generate.assert_called()

    @pytest.mark.asyncio
    async def test_session_stop(self, mock_llm_client):
        """start() then stop() sets is_playing to False."""
        from src.ws_api import SessionComponents

        reference = self._make_reference()
        request = self._make_request()
        config = EngineConfig.default()

        session = MeditationSession(
            session_id="test-session-2",
            request=request,
            reference=reference,
            config=config,
        )

        session._llm_client = mock_llm_client

        mock_text_buffer = MagicMock()
        mock_text_buffer.chunks = []
        mock_text_buffer.is_complete.return_value = True
        mock_text_buffer.spoken_chunks = []

        mock_components = SessionComponents(
            config=config,
            reference=reference,
            text_buffer=mock_text_buffer,
            context_window=MagicMock(),
            tts_client=MagicMock(),
            llm_client=mock_llm_client,
            deviation_handler=MagicMock(),
        )
        session._initialize = AsyncMock(return_value=mock_components)

        # Start and stop
        await session.start()
        await session.stop()

        assert session.is_playing is False

    @pytest.mark.asyncio
    async def test_session_status(self, mock_llm_client):
        """After start(), playhead==0 and total_chunks==0."""
        from src.ws_api import SessionComponents

        reference = self._make_reference()
        request = self._make_request()
        config = EngineConfig.default()

        session = MeditationSession(
            session_id="test-session-3",
            request=request,
            reference=reference,
            config=config,
        )

        session._llm_client = mock_llm_client

        mock_text_buffer = MagicMock()
        mock_text_buffer.chunks = []
        mock_text_buffer.is_complete.return_value = True
        mock_text_buffer.spoken_chunks = []
        mock_text_buffer.playhead = 0

        mock_components = SessionComponents(
            config=config,
            reference=reference,
            text_buffer=mock_text_buffer,
            context_window=MagicMock(),
            tts_client=MagicMock(),
            llm_client=mock_llm_client,
            deviation_handler=MagicMock(),
        )
        session._initialize = AsyncMock(return_value=mock_components)

        await session.start()

        assert session.playhead == 0
        assert session.total_chunks == 0
