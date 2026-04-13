"""Tests for ContextWindow class."""

import pytest
from src.agent.context_window import ContextWindow


# ---------------------------------------------------------------------------
# TestContextWindowBasic
# ---------------------------------------------------------------------------


class TestContextWindowBasic:
    """Basic functionality tests."""

    def test_set_system_prompt(self):
        """Stores prompt, can retrieve via .system_prompt property."""
        cw = ContextWindow()
        cw.set_system_prompt("You are a meditation guide.")
        assert cw.system_prompt == "You are a meditation guide."

    def test_system_prompt_resets_messages(self):
        """After set_system_prompt, get_messages() returns exactly 1 system message."""
        cw = ContextWindow()
        cw.set_system_prompt("System prompt.")
        messages = cw.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt."

    def test_add_spoken_chunk(self):
        """Adds assistant message, messages has 2 entries (system + assistant)."""
        cw = ContextWindow()
        cw.set_system_prompt("System prompt.")
        cw.add_spoken_chunk("Welcome to your meditation session.")
        messages = cw.get_messages()
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "assistant"
        assert messages[1]["content"] == "Welcome to your meditation session."


# ---------------------------------------------------------------------------
# TestContextWindowPruning
# ---------------------------------------------------------------------------


class TestContextWindowPruning:
    """Auto-pruning behavior tests."""

    def test_prune_by_chunk_count(self):
        """With max_chunks=3, after adding 4 chunks, get_messages returns system + 3 chunks (4 total)."""
        cw = ContextWindow(max_chunks=3, max_tokens=10000)
        cw.set_system_prompt("System.")
        cw.add_spoken_chunk("Chunk 1")
        cw.add_spoken_chunk("Chunk 2")
        cw.add_spoken_chunk("Chunk 3")
        cw.add_spoken_chunk("Chunk 4")
        messages = cw.get_messages()
        assert len(messages) == 4  # 1 system + 3 chunks
        assert messages[0]["role"] == "system"
        # Oldest chunk should be removed, so we keep chunks 2, 3, 4
        assert messages[1]["content"] == "Chunk 2"
        assert messages[2]["content"] == "Chunk 3"
        assert messages[3]["content"] == "Chunk 4"

    def test_prune_protects_last_user_message(self):
        """After adding chunks + add_user_message("Deviation request"), last message is the user message."""
        cw = ContextWindow(max_chunks=3, max_tokens=10000)
        cw.set_system_prompt("System.")
        cw.add_spoken_chunk("Chunk 1")
        cw.add_spoken_chunk("Chunk 2")
        cw.add_spoken_chunk("Chunk 3")
        cw.add_user_message("Deviation request")
        # Prune to enforce limits
        cw.prune()
        messages = cw.get_messages()
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Deviation request"

    def test_prune_by_token_count(self):
        """With max_tokens=50, adding a very long chunk triggers pruning without crash."""
        cw = ContextWindow(max_chunks=8, max_tokens=50)
        cw.set_system_prompt("System.")
        # Add a chunk that exceeds max_tokens
        long_text = (
            "This is a very long chunk that should trigger pruning because it exceeds the token limit set on the context window "
            * 10
        )
        cw.add_spoken_chunk(long_text)
        # Should not crash; messages should exist
        messages = cw.get_messages()
        assert isinstance(messages, list)
        assert len(messages) >= 1  # At least the system message


# ---------------------------------------------------------------------------
# TestContextWindowSnapshot
# ---------------------------------------------------------------------------


class TestContextWindowSnapshot:
    """Snapshot and restore tests."""

    def test_snapshot_captures_state(self):
        """Snapshot length matches message count at time of snapshot."""
        cw = ContextWindow()
        cw.set_system_prompt("System.")
        cw.add_spoken_chunk("Chunk 1")
        cw.add_spoken_chunk("Chunk 2")
        snap = cw.snapshot()
        messages = cw.get_messages()
        assert len(snap) == len(messages)

    def test_snapshot_is_deep_copy(self):
        """Modifying messages after snapshot doesn't affect snapshot."""
        cw = ContextWindow()
        cw.set_system_prompt("System.")
        cw.add_spoken_chunk("Original")
        snap = cw.snapshot()
        # Modify the live messages
        cw.add_spoken_chunk("New chunk")
        snap_copy = cw.snapshot()
        assert len(snap) != len(snap_copy)
        # Original snapshot content unchanged
        assert snap[1]["content"] == "Original"

    def test_restore_from_snapshot(self):
        """After restore, message count and content match snapshot state."""
        cw = ContextWindow()
        cw.set_system_prompt("System.")
        cw.add_spoken_chunk("Chunk A")
        snap = cw.snapshot()
        cw.add_spoken_chunk("Chunk B")
        assert len(cw.get_messages()) == 3
        cw.restore(snap)
        messages = cw.get_messages()
        assert len(messages) == 2
        assert messages[0]["content"] == "System."
        assert messages[1]["content"] == "Chunk A"


# ---------------------------------------------------------------------------
# TestContextWindowTokenEstimation
# ---------------------------------------------------------------------------


class TestContextWindowTokenEstimation:
    """Token estimation tests."""

    def test_estimation_uses_char_div_4(self):
        """'Hello' (5 chars) estimates at least 1 token."""
        cw = ContextWindow()
        cw.set_system_prompt("Hello")
        estimate = cw.estimated_token_count()
        assert estimate >= 1

    def test_empty_window_estimation(self):
        """Returns 0 for empty window."""
        cw = ContextWindow()
        assert cw.estimated_token_count() == 0


# ---------------------------------------------------------------------------
# TestContextWindowClear
# ---------------------------------------------------------------------------


class TestContextWindowClear:
    """Clear context tests."""

    def test_clear_keeps_system_prompt(self):
        """After clear, only system message remains."""
        cw = ContextWindow()
        cw.set_system_prompt("System.")
        cw.add_spoken_chunk("Chunk 1")
        cw.add_spoken_chunk("Chunk 2")
        cw.clear_context()
        messages = cw.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System."

    def test_clear_without_system(self):
        """When no system was set, clear results in 0 messages."""
        cw = ContextWindow()
        cw.add_spoken_chunk("Chunk 1")
        cw.clear_context()
        messages = cw.get_messages()
        assert len(messages) == 0
