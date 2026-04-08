"""Text buffer for managing pre-generated meditation scripts.

Handles chunk splitting, playhead tracking, spoken-text recording,
and buffer replacement during session deviations.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.types import PacingConfig


class TextBuffer:
    """Manages pre-generated meditation text with playhead tracking.

    Splits a full LLM-generated script into TTS-ready chunks, tracks which
    chunk is next to speak, and records spoken text for the context window.
    """

    # Maximum characters per chunk before splitting
    MAX_CHUNK_SIZE = 500
    # Maximum number of spoken chunks to keep (default for context window)
    DEFAULT_CONTEXT_WINDOW = 8

    def __init__(self, full_script: str, pacing: PacingConfig) -> None:
        """Split script into chunks and initialize state.

        Args:
            full_script: Complete LLM-generated meditation script.
            pacing: Pacing configuration (used for duration estimation).
        """
        self.chunks: list[str] = self._split_into_chunks(full_script)
        self.playhead: int = 0
        self.spoken_chunks: list[str] = []
        self._pacing: PacingConfig = pacing

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_next_chunk(self) -> str | None:
        """Return the next chunk to send to TTS and advance playhead.

        Returns:
            The next text chunk, or None if all chunks have been consumed.
        """
        if self.playhead >= len(self.chunks):
            return None
        chunk = self.chunks[self.playhead]
        self.playhead += 1
        return chunk

    def mark_spoken(self) -> None:
        """Mark the chunk at the previous playhead position as spoken.

        Must be called AFTER TTS completes for the chunk returned by
        ``get_next_chunk()``. Moves the chunk into ``spoken_chunks``.
        """
        if self.playhead <= 0:
            return
        spoken = self.chunks[self.playhead - 1]
        self.spoken_chunks.append(spoken)

    def get_context_messages(
        self, max_chunks: int = DEFAULT_CONTEXT_WINDOW
    ) -> list[dict]:
        """Return the last N spoken chunks as chat messages for the context window.

        Args:
            max_chunks: Number of most-recent spoken chunks to include.

        Returns:
            List of dicts in chat-message format:
            ``[{"role": "assistant", "content": text}, ...]``
        """
        window = self.spoken_chunks[-max_chunks:] if max_chunks > 0 else []
        return [{"role": "assistant", "content": text} for text in window]

    def is_complete(self) -> bool:
        """Return True if the playhead has reached or exceeded the chunk count."""
        return self.playhead >= len(self.chunks)

    def remaining_chunks(self) -> int:
        """Return the number of chunks not yet spoken."""
        return max(0, len(self.chunks) - self.playhead)

    def reset(self) -> None:
        """Clear all chunks and spoken text.

        Use this when the session needs a fresh buffer (e.g., deviation).
        """
        self.chunks = []
        self.playhead = 0
        self.spoken_chunks = []

    def replace(self, new_script: str, pacing: PacingConfig) -> None:
        """Replace the buffer content with a new script and reset playhead.

        Args:
            new_script: New LLM-generated meditation script.
            pacing: Updated pacing configuration.
        """
        self.chunks = self._split_into_chunks(new_script)
        self.playhead = 0
        self.spoken_chunks = []
        self._pacing = pacing

    def estimated_duration_remaining(self, wpm: float | None = None) -> float:
        """Estimate seconds remaining based on word count and WPM.

        Args:
            wpm: Words per minute. If None, uses ``pacing.avg_speaking_rate_wpm``.

        Returns:
            Estimated duration in seconds.
        """
        rate = wpm if wpm is not None else self._pacing.avg_speaking_rate_wpm
        if rate <= 0:
            return 0.0

        remaining_text = " ".join(self.chunks[self.playhead :])
        word_count = len(remaining_text.split())
        return (word_count / rate) * 60.0

    # ------------------------------------------------------------------
    # Internal: chunk splitting
    # ------------------------------------------------------------------

    def _split_into_chunks(self, text: str) -> list[str]:
        """Split text into TTS-ready chunks.

        Strategy (in order):
        1. Split on paragraph boundaries (double newlines).
        2. If a paragraph exceeds MAX_CHUNK_SIZE, split on sentence
           boundaries (.!?).
        3. If still too long, hard-split at MAX_CHUNK_SIZE.

        Returns:
            List of non-empty, stripped text chunks.
        """
        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []

        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue

            if len(paragraph) <= self.MAX_CHUNK_SIZE:
                chunks.append(paragraph)
            else:
                chunks.extend(self._split_paragraph(paragraph))

        return chunks

    def _split_paragraph(self, text: str) -> list[str]:
        """Split an oversized paragraph into smaller chunks.

        First attempts sentence-boundary splitting, then falls back to
        hard-splitting at MAX_CHUNK_SIZE.
        """
        # Split on sentence boundaries (.!?) while keeping delimiters
        sentences = re.split(r"(?<=[.!?])\s+", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # If every sentence fits individually, recombine up to MAX_CHUNK_SIZE
        if all(len(s) <= self.MAX_CHUNK_SIZE for s in sentences):
            return self._recombine_sentences(sentences)

        # Fallback: some sentences are still too long, hard-split those
        chunks: list[str] = []
        for sentence in sentences:
            if len(sentence) <= self.MAX_CHUNK_SIZE:
                chunks.append(sentence)
            else:
                chunks.extend(self._hard_split(sentence))
        return chunks

    def _recombine_sentences(self, sentences: list[str]) -> list[str]:
        """Combine sentences into chunks that stay under MAX_CHUNK_SIZE."""
        chunks: list[str] = []
        current: list[str] = []
        current_len = 0

        for sentence in sentences:
            needed = len(sentence) + (1 if current_len > 0 else 0)
            if current_len + needed > self.MAX_CHUNK_SIZE and current:
                chunks.append(" ".join(current))
                current = [sentence]
                current_len = len(sentence)
            else:
                current.append(sentence)
                current_len += needed

        if current:
            chunks.append(" ".join(current))

        return chunks

    def _hard_split(self, text: str) -> list[str]:
        """Hard-split text at MAX_CHUNK_SIZE boundaries."""
        chunks: list[str] = []
        while len(text) > self.MAX_CHUNK_SIZE:
            # Try to break at a word boundary near the limit
            split_at = self.MAX_CHUNK_SIZE
            last_space = text.rfind(" ", 0, split_at)
            if last_space > 0 and last_space > split_at * 0.5:
                split_at = last_space
            chunks.append(text[:split_at].strip())
            text = text[split_at:].strip()

        if text:
            chunks.append(text)

        return chunks
