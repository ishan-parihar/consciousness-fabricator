"""Tests for TextBuffer: chunk splitting, playhead, spoken chunks, reset/replace."""

import pytest

from src.agent.text_buffer import TextBuffer


# ---------------------------------------------------------------------------
# Script fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def short_script():
    """Script under 500 chars — should produce exactly 1 chunk."""
    return "Relax your body and mind. Take a deep breath and let go of tension."


@pytest.fixture
def multi_paragraph_script():
    """Script with 3 paragraphs containing key phrases."""
    return (
        "Relax your body and let go of all tension. "
        "Feel the weight melting away with each breath.\n\n"
        "Now focus on your breathing. In through the nose, "
        "out through the mouth. Let each breath deepen your relaxation.\n\n"
        "A calm awareness spreads through your entire being. "
        "You are present, grounded, and at peace."
    )


# ---------------------------------------------------------------------------
# TestTextBufferShortScript
# ---------------------------------------------------------------------------


class TestTextBufferShortScript:
    """Tests for scripts shorter than MAX_CHUNK_SIZE (500 chars)."""

    def test_single_chunk(self, short_script, sample_pacing):
        """Script < 500 chars produces exactly 1 chunk."""
        buf = TextBuffer(short_script, sample_pacing)
        assert len(buf.chunks) == 1

    def test_get_next_chunk_returns_content(self, short_script, sample_pacing):
        """get_next_chunk returns the script content."""
        buf = TextBuffer(short_script, sample_pacing)
        result = buf.get_next_chunk()
        assert result == short_script.strip()

    def test_playhead_advances(self, short_script, sample_pacing):
        """playhead advances from 0 to 1 after get_next_chunk."""
        buf = TextBuffer(short_script, sample_pacing)
        assert buf.playhead == 0
        buf.get_next_chunk()
        assert buf.playhead == 1

    def test_none_when_exhausted(self, short_script, sample_pacing):
        """Second get_next_chunk returns None."""
        buf = TextBuffer(short_script, sample_pacing)
        buf.get_next_chunk()
        assert buf.get_next_chunk() is None

    def test_is_complete(self, short_script, sample_pacing):
        """False initially, True after consuming all chunks."""
        buf = TextBuffer(short_script, sample_pacing)
        assert buf.is_complete() is False
        buf.get_next_chunk()
        assert buf.is_complete() is True

    def test_remaining_chunks(self, short_script, sample_pacing):
        """1 initially, 0 after consuming."""
        buf = TextBuffer(short_script, sample_pacing)
        assert buf.remaining_chunks() == 1
        buf.get_next_chunk()
        assert buf.remaining_chunks() == 0


# ---------------------------------------------------------------------------
# TestTextBufferMultiParagraph
# ---------------------------------------------------------------------------


class TestTextBufferMultiParagraph:
    """Tests for multi-paragraph scripts that split into multiple chunks."""

    def test_splits_into_multiple_chunks(self, multi_paragraph_script, sample_pacing):
        """Script with 3 paragraphs splits into multiple chunks (>= 2)."""
        buf = TextBuffer(multi_paragraph_script, sample_pacing)
        assert len(buf.chunks) >= 2

    def test_chunks_preserve_content(self, multi_paragraph_script, sample_pacing):
        """Chunks preserve all content: key phrases all present."""
        buf = TextBuffer(multi_paragraph_script, sample_pacing)
        all_text = " ".join(buf.chunks)
        assert "Relax your body" in all_text
        assert "breathing" in all_text
        assert "calm awareness" in all_text

    def test_consume_all_chunks_sequentially(
        self, multi_paragraph_script, sample_pacing
    ):
        """Can consume all chunks sequentially until None."""
        buf = TextBuffer(multi_paragraph_script, sample_pacing)
        chunks_consumed = []
        while True:
            chunk = buf.get_next_chunk()
            if chunk is None:
                break
            chunks_consumed.append(chunk)
        assert len(chunks_consumed) == len(buf.chunks)
        assert buf.is_complete() is True


# ---------------------------------------------------------------------------
# TestTextBufferSpokenChunks
# ---------------------------------------------------------------------------


class TestTextBufferSpokenChunks:
    """Tests for mark_spoken and get_context_messages."""

    def test_mark_spoken_after_get_next_chunk(self, short_script, sample_pacing):
        """mark_spoken records the chunk after get_next_chunk."""
        buf = TextBuffer(short_script, sample_pacing)
        buf.get_next_chunk()
        buf.mark_spoken()
        assert len(buf.spoken_chunks) == 1
        assert buf.spoken_chunks[0] == short_script.strip()

    def test_mark_spoken_no_effect_before_get_next_chunk(
        self, short_script, sample_pacing
    ):
        """mark_spoken has no effect before get_next_chunk."""
        buf = TextBuffer(short_script, sample_pacing)
        buf.mark_spoken()
        assert buf.spoken_chunks == []

    def test_get_context_messages_format(self, multi_paragraph_script, sample_pacing):
        """get_context_messages returns list of {'role': 'assistant', 'content': text}."""
        buf = TextBuffer(multi_paragraph_script, sample_pacing)
        # Consume and mark two chunks
        for _ in range(2):
            buf.get_next_chunk()
            buf.mark_spoken()
        messages = buf.get_context_messages()
        assert isinstance(messages, list)
        assert all(m["role"] == "assistant" for m in messages)
        assert all("content" in m for m in messages)
        assert len(messages) == 2

    def test_context_respects_max_chunks(self, multi_paragraph_script, sample_pacing):
        """get_context_messages respects max_chunks limit."""
        buf = TextBuffer(multi_paragraph_script, sample_pacing)
        # Consume and mark all available chunks
        while not buf.is_complete():
            buf.get_next_chunk()
            buf.mark_spoken()
        # Limit to 2
        messages = buf.get_context_messages(max_chunks=2)
        assert len(messages) == 2


# ---------------------------------------------------------------------------
# TestTextBufferResetAndReplace
# ---------------------------------------------------------------------------


class TestTextBufferResetAndReplace:
    """Tests for reset and replace methods."""

    def test_reset_clears_state(self, short_script, sample_pacing):
        """reset clears chunks, playhead, and spoken_chunks."""
        buf = TextBuffer(short_script, sample_pacing)
        buf.get_next_chunk()
        buf.mark_spoken()
        buf.reset()
        assert buf.chunks == []
        assert buf.playhead == 0
        assert buf.spoken_chunks == []

    def test_replace_resets_state(
        self, short_script, multi_paragraph_script, sample_pacing
    ):
        """replace with new script resets playhead to 0 and clears spoken_chunks."""
        buf = TextBuffer(short_script, sample_pacing)
        buf.get_next_chunk()
        buf.mark_spoken()
        buf.replace(multi_paragraph_script, sample_pacing)
        assert buf.playhead == 0
        assert buf.spoken_chunks == []
        assert len(buf.chunks) >= 2

    def test_estimated_duration_decreases(self, multi_paragraph_script, sample_pacing):
        """estimated_duration_remaining decreases as chunks are consumed."""
        buf = TextBuffer(multi_paragraph_script, sample_pacing)
        initial_duration = buf.estimated_duration_remaining()
        assert initial_duration > 0
        buf.get_next_chunk()
        after_first = buf.estimated_duration_remaining()
        assert after_first < initial_duration
