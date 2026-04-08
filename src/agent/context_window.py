"""Manages the sliding context window for the LLM meditation agent.

Tracks only spoken text (not pre-generated text) in the conversation context,
ensuring clean deviation handling.
"""

from __future__ import annotations

import copy
from typing import Any


class ContextWindow:
    """Manages the LLM conversation context with configurable size limits.

    Only spoken text enters the context window. The system prompt is set once
    and never changes. Messages are pruned automatically when limits are exceeded.
    """

    def __init__(self, max_chunks: int = 8, max_tokens: int = 4000) -> None:
        """Initialize with configurable limits.

        Args:
            max_chunks: Maximum number of non-system messages to retain.
            max_tokens: Maximum estimated tokens before auto-pruning occurs.
        """
        self._max_chunks = max_chunks
        self._max_tokens = max_tokens
        self._system_prompt: str | None = None
        self._messages: list[dict[str, str]] = []

    @property
    def system_prompt(self) -> str | None:
        """The system prompt (set once, never changes)."""
        return self._system_prompt

    @property
    def messages(self) -> list[dict[str, str]]:
        """The conversation messages (system + spoken context)."""
        return self._messages

    def set_system_prompt(self, prompt: str) -> None:
        """Set the system prompt. Should be called once at initialization.

        Args:
            prompt: The system prompt text.
        """
        self._system_prompt = prompt
        self._messages = [{"role": "system", "content": prompt}]

    def add_spoken_chunk(self, text: str) -> None:
        """Add a spoken chunk to the context as an assistant message.

        Auto-prunes if chunk count or estimated tokens exceed limits.

        Args:
            text: The spoken text chunk.
        """
        self._messages.append({"role": "assistant", "content": text})
        if self.estimated_token_count() > self._max_tokens:
            self.prune()
        else:
            self._enforce_chunk_limit()

    def _enforce_chunk_limit(self) -> None:
        """Remove oldest non-system messages when chunk count exceeds max_chunks."""
        non_system = [m for m in self._messages if m["role"] != "system"]
        if len(non_system) <= self._max_chunks:
            return
        to_remove = len(non_system) - self._max_chunks
        # Protect last user message
        if non_system[-1]["role"] == "user":
            non_system = non_system[:-1]
        for _ in range(min(to_remove, len(non_system))):
            removed = non_system.pop(0)
            self._messages.remove(removed)

    def add_user_message(self, text: str) -> None:
        """Add a user message, typically for deviation requests.

        Args:
            text: The user's message text.
        """
        self._messages.append({"role": "user", "content": text})

    def prune(self) -> None:
        """Enforce max_chunks and max_tokens limits.

        Always keeps the system message. Removes oldest assistant messages first.
        Never removes the most recent user message (it might be a deviation request).
        """
        if not self._messages:
            return

        # Separate system message(s) from the rest
        system_msgs: list[dict[str, str]] = []
        rest: list[dict[str, str]] = []

        for msg in self._messages:
            if msg["role"] == "system":
                system_msgs.append(msg)
            else:
                rest.append(msg)

        if not rest:
            return

        # Protect the most recent user message if it exists
        last_msg = rest[-1]
        protected: list[dict[str, str]] = []
        removable: list[dict[str, str]] = rest

        if last_msg["role"] == "user":
            protected = [last_msg]
            removable = rest[:-1]

        # Enforce max_chunks on removable + protected
        while len(removable) + len(protected) > self._max_chunks and removable:
            removable.pop(0)

        # Enforce max_tokens: remove oldest assistant messages first
        combined = removable + protected
        while (
            self._estimated_token_count_from_list(system_msgs + combined)
            > self._max_tokens
        ):
            # Find the oldest assistant message in removable
            removed = False
            for i, msg in enumerate(removable):
                if msg["role"] == "assistant":
                    removable.pop(i)
                    removed = True
                    break
            if not removed:
                # No more assistant messages to remove; try removing from protected
                # (only if it's not a user message, which we must preserve)
                if protected and protected[0]["role"] != "user":
                    protected.pop(0)
                else:
                    break
            combined = removable + protected

        self._messages = system_msgs + combined

    def get_messages(self) -> list[dict[str, str]]:
        """Return the full message list ready for LLM API call.

        Returns:
            List of message dicts in OpenAI chat format.
        """
        return list(self._messages)

    def estimated_token_count(self) -> int:
        """Rough estimate of token count using char_count / 4 heuristic.

        Returns:
            Estimated number of tokens.
        """
        return self._estimated_token_count_from_list(self._messages)

    def clear_context(self) -> None:
        """Remove all non-system messages, keeping only the system prompt."""
        if self._system_prompt is not None:
            self._messages = [{"role": "system", "content": self._system_prompt}]
        else:
            self._messages = []

    def snapshot(self) -> list[dict[str, str]]:
        """Return a deep copy of current messages for deviation state capture.

        Returns:
            Deep copy of the message list.
        """
        return copy.deepcopy(self._messages)

    def restore(self, messages: list[dict[str, str]]) -> None:
        """Replace messages with a snapshot.

        Args:
            messages: A previously captured snapshot to restore.
        """
        self._messages = copy.deepcopy(messages)

    def _estimated_token_count_from_list(self, msgs: list[dict[str, str]]) -> int:
        """Estimate tokens from a list of messages.

        Args:
            msgs: List of message dicts.

        Returns:
            Estimated token count.
        """
        all_text = " ".join(msg.get("content", "") for msg in msgs)
        return len(all_text) // 4
