"""Handles mid-session trajectory changes by capturing spoken context,
clearing the buffer, regenerating with LLM, and replacing the text buffer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from src.types import MeditationReference
    from src.agent.context_window import ContextWindow
    from src.agent.text_buffer import TextBuffer


class LLMClientProtocol(Protocol):
    """Protocol for the LLM client used by the deviation handler."""

    async def generate(self, messages: list[dict]) -> str:
        """Generate text from a list of chat messages.

        Args:
            messages: List of message dicts in OpenAI chat format.

        Returns:
            The generated text response.
        """
        ...


@dataclass
class DeviationResult:
    """Result of a deviation handling operation."""

    success: bool
    old_playhead: int
    new_chunks: int
    deviation_request: str
    error_message: str | None = None


class DeviationHandler:
    """Orchestrates mid-session deviation flow.

    Captures current state, clears the buffer, requests new content from the
    LLM incorporating the user's request, and replaces the text buffer so
    audio continues seamlessly.
    """

    def __init__(
        self,
        reference: MeditationReference,
        context_window: ContextWindow,
        text_buffer: TextBuffer,
        llm_client: LLMClientProtocol,
    ) -> None:
        """Initialize with the core session components.

        Args:
            reference: The meditation reference for style matching.
            context_window: The LLM context window manager.
            text_buffer: The text buffer to replace on deviation.
            llm_client: Async LLM client with a ``generate(messages) -> str`` method.
        """
        self._reference = reference
        self._context_window = context_window
        self._text_buffer = text_buffer
        self._llm_client = llm_client

    async def handle(self, user_request: str) -> DeviationResult:
        """Handle a mid-session deviation request.

        Flow:
        1. Capture current state (context snapshot, playhead position).
        2. Build deviation prompt from reference, spoken context, and user request.
        3. Add the deviation prompt as a user message to the context window.
        4. Generate new script via the LLM client.
        5. Replace the text buffer with the new script.
        6. Reset playhead to 0.
        7. Return the result.

        Args:
            user_request: The user's deviation request (e.g., "do a body scan").

        Returns:
            DeviationResult with success status and metadata.
        """
        old_playhead = self._text_buffer.playhead
        self._context_window.snapshot()
        spoken_context = self._text_buffer.spoken_chunks

        from src.agent.prompt_builder import build_deviation_prompt

        deviation_prompt = build_deviation_prompt(
            self._reference, spoken_context, user_request
        )

        self._context_window.add_user_message(deviation_prompt)

        try:
            new_script = await self._llm_client.generate(
                self._context_window.get_messages()
            )
        except Exception as e:
            return DeviationResult(
                success=False,
                old_playhead=old_playhead,
                new_chunks=0,
                deviation_request=user_request,
                error_message=str(e),
            )

        self._text_buffer.replace(new_script, self._reference.pacing)

        return DeviationResult(
            success=True,
            old_playhead=old_playhead,
            new_chunks=len(self._text_buffer.chunks),
            deviation_request=user_request,
        )
