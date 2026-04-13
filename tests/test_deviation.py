"""Tests for DeviationHandler (src/agent/deviation.py)."""

import pytest
from unittest.mock import AsyncMock

from src.agent.deviation import DeviationHandler, DeviationResult
from src.agent.text_buffer import TextBuffer
from src.agent.context_window import ContextWindow
from src.types import PacingConfig, BinauralConfig, Brainwave


@pytest.fixture
def sample_binaural():
    """Override broken conftest fixture (description is a property, not a field)."""
    return BinauralConfig(
        brainwave=Brainwave.ALPHA,
        carrier_freq_hz=120.0,
        beat_freq_hz=10.0,
    )


# ---------------------------------------------------------------------------
# Local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def test_pacing():
    """PacingConfig for deviation tests."""
    return PacingConfig(
        avg_speaking_rate_wpm=98.0,
        instruction_pause_seconds=4.2,
        body_scan_pause_seconds=8.5,
        countdown_pause_seconds=2.1,
        act_boundaries=[],
    )


@pytest.fixture
def handler(sample_reference, mock_llm_client, test_pacing):
    """DeviationHandler with a populated text buffer and context window."""
    context_window = ContextWindow()
    context_window.set_system_prompt("You are a meditation assistant.")

    initial_script = (
        "Welcome to this meditation session. Relax your body and mind. "
        "Take a deep breath and let go of any tension. "
        "Feel yourself going deeper into a state of calm awareness."
    )
    text_buffer = TextBuffer(initial_script, test_pacing)

    return DeviationHandler(
        reference=sample_reference,
        context_window=context_window,
        text_buffer=text_buffer,
        llm_client=mock_llm_client,
    )


# ---------------------------------------------------------------------------
# TestDeviationHandlerSuccess
# ---------------------------------------------------------------------------


class TestDeviationHandlerSuccess:
    """Tests for successful deviation handling."""

    @pytest.mark.asyncio
    async def test_successful_deviation(self, handler, mock_llm_client):
        """handler.handle returns DeviationResult with success=True,
        deviation_request matches input, new_chunks > 0, and
        mock_llm_client.generate was called once."""
        result = await handler.handle("Now do a body scan")

        assert result.success is True
        assert result.deviation_request == "Now do a body scan"
        assert result.new_chunks > 0
        mock_llm_client.generate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_text_buffer_replaced(self, handler):
        """Captures old_playhead before deviation, after deviation
        success=True and buffer has new content."""
        old_playhead = handler._text_buffer.playhead
        result = await handler.handle("Now do a body scan")

        assert result.success is True
        assert result.old_playhead == old_playhead
        assert len(handler._text_buffer.chunks) > 0
        # Playhead reset to 0 after replace
        assert handler._text_buffer.playhead == 0

    @pytest.mark.asyncio
    async def test_context_window_has_deviation_prompt(self, handler):
        """After handling, last message in context_window is a user message
        containing the deviation request text."""
        deviation_request = "Now do a body scan"
        await handler.handle(deviation_request)

        messages = handler._context_window.messages
        last_message = messages[-1]

        assert last_message["role"] == "user"
        assert deviation_request in last_message["content"]


# ---------------------------------------------------------------------------
# TestDeviationHandlerFailure
# ---------------------------------------------------------------------------


class TestDeviationHandlerFailure:
    """Tests for failed deviation handling."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_failure(self, handler, mock_llm_client):
        """When mock_llm.generate raises Exception("API timeout"), handle
        returns DeviationResult with success=False, error_message=='API timeout',
        deviation_request preserved."""
        mock_llm_client.generate.side_effect = Exception("API timeout")

        result = await handler.handle("Now do a body scan")

        assert result.success is False
        assert result.error_message == "API timeout"
        assert result.deviation_request == "Now do a body scan"
        assert result.new_chunks == 0
