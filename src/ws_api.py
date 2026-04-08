"""WebSocket API for real-time meditation session control.

Provides a WebSocket server on port 8765 (configurable) that allows clients to
start meditation sessions, request mid-session deviations, and receive real-time
progress updates.

Message types (client -> server):
    - start: Begin a new meditation session
    - deviation: Request a mid-session trajectory change
    - stop: Stop an active session
    - status: Query session status

Message types (server -> client):
    - session_started: Confirmation with session metadata
    - chunk_progress: Real-time TTS chunk processing updates
    - deviation_accepted: Deviation successfully processed
    - deviation_rejected: Deviation failed with error
    - session_complete: Session finished naturally
    - session_stopped: Session stopped by client or disconnect
    - error: Error notification
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import websockets

from src.types import (
    MeditationReference,
    MeditationStyle,
    SessionRequest,
    SessionState,
)
from src.config import EngineConfig
from src.agent import (
    TextBuffer,
    ContextWindow,
    DeviationHandler,
    DeviationResult,
    build_system_prompt,
    build_continuation_prompt,
)
from src.agent.deviation import LLMClientProtocol
from src.audio import (
    FilterGraphBuilder,
    MusicEvent,
    VoiceoverEvent,
    BinauralEvent,
    DuckingEvent,
)
from src.tts import TtsClient, TtsResult, TtsError, VoiceProfileRegistry

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# LLM Client stub for production use
# ---------------------------------------------------------------------------


class OpenAICompatLLMClient:
    """Async LLM client compatible with OpenAI-compatible APIs.

    Implements the LLMClientProtocol required by DeviationHandler.
    """

    def __init__(self, api_key: str, model: str, base_url: str | None = None) -> None:
        import httpx

        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url or "https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(120.0),
        )

    async def generate(self, messages: list[dict]) -> str:
        """Generate text from chat messages."""
        resp = await self._client.post(
            "/chat/completions",
            json={"model": self._model, "messages": messages},
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# MeditationSession — orchestrates one session lifecycle
# ---------------------------------------------------------------------------


@dataclass
class SessionComponents:
    """Core components needed for session execution."""

    config: EngineConfig
    reference: MeditationReference
    text_buffer: TextBuffer
    context_window: ContextWindow
    tts_client: TtsClient
    llm_client: LLMClientProtocol
    deviation_handler: DeviationHandler


class MeditationSession:
    """Manages a single meditation session's lifecycle.

    Handles async generation, TTS, audio mixing, and deviation requests.
    """

    def __init__(
        self,
        session_id: str,
        request: SessionRequest,
        reference: MeditationReference,
        config: EngineConfig | None = None,
    ) -> None:
        self.session_id = session_id
        self.request = request
        self.reference = reference
        self.config = config or EngineConfig.default()

        # State
        self.state = SessionState(
            session_id=session_id,
            request=request,
            reference=reference,
        )

        # Internal
        self._text_buffer: TextBuffer | None = None
        self._context_window: ContextWindow | None = None
        self._tts_client: TtsClient | None = None
        self._deviation_handler: DeviationHandler | None = None
        self._llm_client: LLMClientProtocol | None = None
        self._stop_event = asyncio.Event()
        self._started = False
        self._start_time: float = 0.0

    @property
    def is_playing(self) -> bool:
        return self.state.is_playing

    @property
    def playhead(self) -> int:
        return self.state.playhead

    @property
    def total_chunks(self) -> int:
        return self.state.total_chunks

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    async def _initialize(self) -> SessionComponents:
        """Initialize all session components (lazy, once)."""
        if self._text_buffer is not None:
            return SessionComponents(
                config=self.config,
                reference=self.reference,
                text_buffer=self._text_buffer,
                context_window=self._context_window,  # type: ignore[arg-type]
                tts_client=self._tts_client,  # type: ignore[arg-type]
                llm_client=self._llm_client,  # type: ignore[arg-type]
                deviation_handler=self._deviation_handler,  # type: ignore[arg-type]
            )

        # TTS client
        self._tts_client = TtsClient(
            base_url=self.config.tts_base_url,
            cache_dir=self.config.tts_cache_dir,
        )

        # LLM client (from environment or defaults)
        import os

        api_key = os.environ.get("OPENAI_API_KEY", "")
        model = self.config.llm_model
        self._llm_client = OpenAICompatLLMClient(api_key=api_key, model=model)

        # Context window
        self._context_window = ContextWindow(
            max_chunks=self.config.max_context_chunks,
        )
        self._context_window.set_system_prompt(
            build_system_prompt(self.reference, self.request.duration_minutes)
        )

        # Generate initial script via LLM
        system_prompt = build_system_prompt(
            self.reference, self.request.duration_minutes
        )
        initial_response = await self._llm_client.generate(
            [{"role": "system", "content": system_prompt}]
        )

        # Text buffer
        self._text_buffer = TextBuffer(initial_response, self.reference.pacing)
        self.state.total_chunks = len(self._text_buffer.chunks)

        # Deviation handler
        self._deviation_handler = DeviationHandler(
            reference=self.reference,
            context_window=self._context_window,
            text_buffer=self._text_buffer,
            llm_client=self._llm_client,
        )

        return SessionComponents(
            config=self.config,
            reference=self.reference,
            text_buffer=self._text_buffer,
            context_window=self._context_window,
            tts_client=self._tts_client,
            llm_client=self._llm_client,
            deviation_handler=self._deviation_handler,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def start(
        self,
        progress_callback: Any = None,
    ) -> dict[str, Any]:
        """Start the session asynchronously.

        Returns metadata for the session_started response.
        The actual chunk processing runs in the background via ``run_loop``.
        """
        components = await self._initialize()
        self._started = True
        self._start_time = time.monotonic()
        self.state.is_playing = True

        return {
            "session_id": self.session_id,
            "reference": {
                "id": self.reference.id,
                "name": self.reference.name,
                "collection": self.reference.collection,
                "category": self.reference.category,
            },
            "estimated_chunks": self.state.total_chunks,
        }

    async def run_loop(self, ws_send: Any) -> None:
        """Main processing loop: fetch chunks, TTS, mix, send progress.

        Args:
            ws_send: Async callable to send JSON messages to the client.
        """
        if self._text_buffer is None:
            return

        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        voiceover_events: list[VoiceoverEvent] = []
        chunk_index = 0

        while not self._stop_event.is_set():
            if self._text_buffer.is_complete():
                # Buffer exhausted — try to generate more (continuation)
                if self._context_window and self._llm_client:
                    prompt = build_continuation_prompt(
                        self.reference,
                        self._text_buffer.spoken_chunks[-5:]
                        if self._text_buffer.spoken_chunks
                        else [],
                    )
                    self._context_window.add_user_message(prompt)
                    try:
                        continuation = await self._llm_client.generate(
                            self._context_window.get_messages()
                        )
                        self._text_buffer.replace(continuation, self.reference.pacing)
                        self.state.total_chunks += len(self._text_buffer.chunks)
                    except Exception as e:
                        logger.error("Continuation generation failed: %s", e)
                        break
                else:
                    break

            chunk = self._text_buffer.get_next_chunk()
            if chunk is None:
                break

            # TTS generation
            tts_result: TtsResult | None = None
            try:
                voice_profile_id = self.request.voice_profile_id or "default"
                voice_profile = None
                if self._tts_client:
                    registry_path = self.config.voice_profile_registry
                    if Path(registry_path).exists():
                        registry = VoiceProfileRegistry(registry_path)
                        voice_profile = registry.get(voice_profile_id)

                output_path = str(
                    output_dir / f"{self.session_id}_chunk_{chunk_index}.wav"
                )
                if voice_profile and self._tts_client:
                    tts_result = await self._tts_client.generate(
                        voice_profile_id=voice_profile_id,
                        text=chunk,
                        output_path=output_path,
                        voice_profile=voice_profile,
                    )
                else:
                    # Fallback: use default profile from registry
                    if self._tts_client:
                        registry_path = self.config.voice_profile_registry
                        if Path(registry_path).exists():
                            registry = VoiceProfileRegistry(registry_path)
                            profiles = registry.list()
                            if profiles:
                                vp = profiles[0]
                                tts_result = await self._tts_client.generate(
                                    voice_profile_id=vp.id,
                                    text=chunk,
                                    output_path=output_path,
                                    voice_profile=vp,
                                )
            except TtsError as e:
                logger.error("TTS error on chunk %d: %s", chunk_index, e)
                # Continue to next chunk even if TTS fails
            except Exception as e:
                logger.error("Unexpected error on chunk %d: %s", chunk_index, e)

            if self._text_buffer:
                self._text_buffer.mark_spoken()
                self.state.playhead = self._text_buffer.playhead

            if tts_result:
                voiceover_events.append(
                    VoiceoverEvent(
                        path=tts_result.output_path,
                        start_ms=0,
                    )
                )

            # Send progress update
            text_preview = chunk[:80] + ("..." if len(chunk) > 80 else "")
            await ws_send(
                json.dumps(
                    {
                        "type": "chunk_progress",
                        "session_id": self.session_id,
                        "chunk_index": chunk_index,
                        "total_chunks": self.state.total_chunks,
                        "text_preview": text_preview,
                    }
                )
            )

            chunk_index += 1

        # Session complete — render final audio if any chunks were processed
        if voiceover_events and self._text_buffer:
            try:
                self._render_audio(voiceover_events)
            except Exception as e:
                logger.error("Audio rendering failed: %s", e)

        duration_seconds = time.monotonic() - self._start_time
        self.state.is_playing = False

        await ws_send(
            json.dumps(
                {
                    "type": "session_complete",
                    "session_id": self.session_id,
                    "total_chunks": chunk_index,
                    "duration_seconds": round(duration_seconds, 2),
                }
            )
        )

    async def handle_deviation(self, user_request: str) -> dict[str, Any]:
        """Handle a mid-session deviation request.

        Returns deviation_accepted or deviation_rejected response data.
        """
        if self._deviation_handler is None:
            return {
                "type": "deviation_rejected",
                "session_id": self.session_id,
                "error": "Session not initialized",
            }

        old_playhead = self.state.playhead

        try:
            result: DeviationResult = await self._deviation_handler.handle(user_request)

            if result.success:
                self.state.playhead = 0
                self.state.total_chunks += result.new_chunks
                self.state.is_deviation = True
                return {
                    "type": "deviation_accepted",
                    "session_id": self.session_id,
                    "old_playhead": old_playhead,
                    "new_chunks": result.new_chunks,
                }
            else:
                return {
                    "type": "deviation_rejected",
                    "session_id": self.session_id,
                    "error": result.error_message or "Deviation generation failed",
                }
        except Exception as e:
            logger.error("Deviation handling failed: %s", e)
            return {
                "type": "deviation_rejected",
                "session_id": self.session_id,
                "error": str(e),
            }

    async def stop(self) -> None:
        """Stop the session gracefully."""
        self._stop_event.set()
        self.state.is_playing = False
        logger.info("Session %s stopped", self.session_id)

    def _render_audio(self, voiceover_events: list[VoiceoverEvent]) -> None:
        """Render the final mixed audio file."""
        builder = FilterGraphBuilder(duration_s=self.reference.total_duration_seconds)

        if self.config.loudnorm_enabled:
            builder.with_loudnorm()

        builder.with_voiceover(voiceover_events)
        builder.with_binaural(
            BinauralEvent(
                carrier_freq_hz=self.reference.binaural.carrier_freq_hz,
                beat_freq_hz=self.reference.binaural.beat_freq_hz,
                duration_s=self.reference.total_duration_seconds,
            )
        )

        # Add ambient music if available
        ambient_dir = Path(self.config.ambient_music_dir)
        if ambient_dir.exists():
            music_files = list(ambient_dir.glob("*.mp3")) + list(
                ambient_dir.glob("*.wav")
            )
            if music_files:
                builder.with_music([MusicEvent(path=str(music_files[0]), volume=0.3)])
                builder.with_ducking(
                    DuckingEvent(
                        reduction_db=self.config.ducking_reduction_db,
                        attack_ms=self.config.ducking_attack_ms,
                        release_ms=self.config.ducking_release_ms,
                    )
                )

        builder.with_fade_out(
            start_s=self.reference.total_duration_seconds
            - self.config.fade_out_duration,
            duration_s=self.config.fade_out_duration,
        )

        filter_complex = builder.build()
        output_path = str(Path(self.config.output_dir) / f"{self.session_id}_final.wav")

        # Collect unique input paths
        inputs: list[str] = [ve.path for ve in voiceover_events]
        if ambient_dir.exists():
            music_files = list(ambient_dir.glob("*.mp3")) + list(
                ambient_dir.glob("*.wav")
            )
            if music_files:
                inputs.insert(0, str(music_files[0]))

        if filter_complex and inputs:
            from src.audio import render

            render(
                filter_complex,
                inputs,
                output_path,
                self.reference.total_duration_seconds,
            )


# ---------------------------------------------------------------------------
# WebSocket Server
# ---------------------------------------------------------------------------


class SessionManager:
    """In-memory session registry with cleanup on disconnect."""

    def __init__(self) -> None:
        self._sessions: dict[str, MeditationSession] = {}
        self._tasks: dict[str, asyncio.Task] = {}

    def create_session(
        self,
        request: SessionRequest,
        reference: MeditationReference,
        config: EngineConfig,
    ) -> MeditationSession:
        session_id = str(uuid.uuid4())
        session = MeditationSession(
            session_id=session_id,
            request=request,
            reference=reference,
            config=config,
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> MeditationSession | None:
        return self._sessions.get(session_id)

    def register_task(self, session_id: str, task: asyncio.Task) -> None:
        self._tasks[session_id] = task

    async def cleanup_session(self, session_id: str) -> None:
        """Fade out and stop a session (e.g., on client disconnect)."""
        session = self._sessions.get(session_id)
        if session:
            await session.stop()
        task = self._tasks.pop(session_id, None)
        if task and not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._sessions.pop(session_id, None)

    @property
    def active_sessions(self) -> dict[str, MeditationSession]:
        return dict(self._sessions)


# Global session manager
_manager = SessionManager()


def _load_reference(
    style: MeditationStyle,
    category: str | None = None,
    config: EngineConfig | None = None,
) -> MeditationReference:
    """Load a meditation reference JSON matching the style and optionally category."""
    cfg = config or EngineConfig.default()
    refs_dir = Path(cfg.references_dir)

    if not refs_dir.exists():
        raise FileNotFoundError(f"References directory not found: {refs_dir}")

    candidates: list[Path] = []
    for json_file in refs_dir.rglob("*.json"):
        candidates.append(json_file)

    if not candidates:
        raise ValueError("No meditation references found")

    # Filter by style (collection name)
    style_matches = [
        p
        for p in candidates
        if style.value in str(p.parent.name) or style.value in str(p)
    ]
    if not style_matches:
        # Fall back to first available reference
        style_matches = candidates

    # Filter by category if specified
    if category:
        cat_matches = []
        for p in style_matches:
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if data.get("category", "").lower() == category.lower():
                    cat_matches.append(p)
            except (json.JSONDecodeError, OSError):
                continue
        if cat_matches:
            style_matches = cat_matches

    # Load the first matching reference
    ref_path = style_matches[0]
    data = json.loads(ref_path.read_text(encoding="utf-8"))
    return _parse_reference(data)


def _parse_reference(data: dict[str, Any]) -> MeditationReference:
    """Parse a dict into a MeditationReference."""
    from src.types import (
        ToneConfig,
        PacingConfig,
        LanguageConfig,
        TrajectoryConfig,
        BinauralConfig,
        Brainwave,
    )

    tone_data = data.get("tone", {})
    pacing_data = data.get("pacing", {})
    lang_data = data.get("language", {})
    traj_data = data.get("trajectory", {})
    binaural_data = data.get("binaural", {})

    # Parse brainwave enum
    brainwave_str = binaural_data.get("brainwave", "alpha")
    try:
        brainwave = Brainwave(brainwave_str)
    except ValueError:
        brainwave = Brainwave.ALPHA

    return MeditationReference(
        id=data.get("id", "unknown"),
        name=data.get("name", "Unknown"),
        collection=data.get("collection", "unknown"),
        category=data.get("category", "general"),
        total_duration_seconds=int(data.get("total_duration_seconds", 600)),
        total_phrases=data.get("total_phrases", 100),
        tone=ToneConfig(
            energy=tone_data.get("energy", "moderate"),
            warmth=tone_data.get("warmth", "neutral"),
            formality=tone_data.get("formality", "guided"),
            description=tone_data.get("description", ""),
        ),
        pacing=PacingConfig(
            avg_speaking_rate_wpm=float(pacing_data.get("avg_speaking_rate_wpm", 98)),
            instruction_pause_seconds=float(
                pacing_data.get("instruction_pause_seconds", 4.2)
            ),
            body_scan_pause_seconds=float(
                pacing_data.get("body_scan_pause_seconds", 8.5)
            ),
            countdown_pause_seconds=float(
                pacing_data.get("countdown_pause_seconds", 2.1)
            ),
            act_boundaries=pacing_data.get("act_boundaries", []),
        ),
        language=LanguageConfig(
            sentence_style=lang_data.get("sentence_style", "natural"),
            perspective=lang_data.get("perspective", "second"),
            common_phrases=lang_data.get("common_phrases", []),
            structural_patterns=lang_data.get("structural_patterns", []),
            repetition_rate=float(lang_data.get("repetition_rate", 0.35)),
            avg_sentence_length_words=int(
                lang_data.get("avg_sentence_length_words", 12)
            ),
        ),
        trajectory=TrajectoryConfig(
            opening=traj_data.get("opening", "Gentle induction"),
            deepening=traj_data.get("deepening", "Progressive relaxation"),
            transitions=traj_data.get("transitions", "Smooth transitions"),
            deviation_handling=traj_data.get(
                "deviation_handling",
                "Flow naturally into the new request without acknowledging the change",
            ),
        ),
        binaural=BinauralConfig(
            brainwave=brainwave,
            carrier_freq_hz=float(binaural_data.get("carrier_freq_hz", 120)),
            beat_freq_hz=float(binaural_data.get("beat_freq_hz", 10)),
        ),
    )


async def _handle_ws(websocket: websockets.ServerConnection) -> None:
    """Handle a single WebSocket connection."""
    current_session_id: str | None = None

    async def send_json(msg: str) -> None:
        try:
            await websocket.send(msg)
        except websockets.ConnectionClosed:
            pass

    try:
        async for raw_message in websocket:
            try:
                msg = json.loads(raw_message)
            except json.JSONDecodeError:
                await send_json(
                    json.dumps(
                        {
                            "type": "error",
                            "message": "Invalid JSON",
                            "session_id": current_session_id,
                        }
                    )
                )
                continue

            msg_type = msg.get("type")

            if msg_type == "start":
                current_session_id = await _handle_start(msg, send_json)

            elif msg_type == "deviation":
                await _handle_deviation(msg, send_json)

            elif msg_type == "stop":
                await _handle_stop(msg, send_json)
                current_session_id = None

            elif msg_type == "status":
                await _handle_status(msg, send_json)

            else:
                await send_json(
                    json.dumps(
                        {
                            "type": "error",
                            "message": f"Unknown message type: {msg_type}",
                            "session_id": current_session_id,
                        }
                    )
                )

    except websockets.ConnectionClosed:
        logger.info("Client disconnected")
    finally:
        # Clean up session on disconnect
        if current_session_id:
            session = _manager.get_session(current_session_id)
            if session and session.is_playing:
                logger.info(
                    "Client disconnected during session %s — cleaning up",
                    current_session_id,
                )
                await _manager.cleanup_session(current_session_id)
                try:
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "session_stopped",
                                "session_id": current_session_id,
                            }
                        )
                    )
                except websockets.ConnectionClosed:
                    pass


async def _handle_start(msg: dict, send_json: Any) -> str | None:
    """Handle session start request."""
    try:
        style_str = msg.get("style", "silva-method")
        try:
            style = MeditationStyle(style_str)
        except ValueError:
            style = MeditationStyle.SILVA_METHOD

        duration_minutes = int(msg.get("duration_minutes", 10))
        user_request = msg.get("user_request", "relaxation")
        voice_profile_id = msg.get("voice_profile_id")

        config = EngineConfig.default()

        # Load reference
        reference = _load_reference(style, user_request, config)

        # Create session
        request = SessionRequest(
            style=style,
            duration_minutes=duration_minutes,
            user_request=user_request,
            voice_profile_id=voice_profile_id,
        )

        session = _manager.create_session(request, reference, config)

        # Start session
        start_info = await session.start(progress_callback=send_json)

        await send_json(json.dumps(start_info))

        # Run the session loop in background
        loop_task = asyncio.create_task(session.run_loop(send_json))
        _manager.register_task(session.session_id, loop_task)

        return session.session_id

    except FileNotFoundError as e:
        await send_json(
            json.dumps({"type": "error", "message": str(e), "session_id": None})
        )
        return None
    except Exception as e:
        logger.error("Start session failed: %s", e, exc_info=True)
        await send_json(
            json.dumps({"type": "error", "message": str(e), "session_id": None})
        )
        return None


async def _handle_deviation(msg: dict, send_json: Any) -> None:
    """Handle deviation request."""
    session_id = msg.get("session_id")
    user_request = msg.get("user_request", "")

    if not session_id:
        await send_json(
            json.dumps(
                {"type": "error", "message": "session_id required", "session_id": None}
            )
        )
        return

    session = _manager.get_session(session_id)
    if not session:
        await send_json(
            json.dumps(
                {
                    "type": "error",
                    "message": f"Session {session_id} not found",
                    "session_id": session_id,
                }
            )
        )
        return

    result = await session.handle_deviation(user_request)
    await send_json(json.dumps(result))


async def _handle_stop(msg: dict, send_json: Any) -> None:
    """Handle session stop request."""
    session_id = msg.get("session_id")
    if not session_id:
        await send_json(
            json.dumps(
                {"type": "error", "message": "session_id required", "session_id": None}
            )
        )
        return

    session = _manager.get_session(session_id)
    if not session:
        await send_json(
            json.dumps(
                {
                    "type": "error",
                    "message": f"Session {session_id} not found",
                    "session_id": session_id,
                }
            )
        )
        return

    await session.stop()
    await _manager.cleanup_session(session_id)
    await send_json(json.dumps({"type": "session_stopped", "session_id": session_id}))


async def _handle_status(msg: dict, send_json: Any) -> None:
    """Handle session status request."""
    session_id = msg.get("session_id")
    if not session_id:
        await send_json(
            json.dumps(
                {"type": "error", "message": "session_id required", "session_id": None}
            )
        )
        return

    session = _manager.get_session(session_id)
    if not session:
        await send_json(
            json.dumps(
                {
                    "type": "error",
                    "message": f"Session {session_id} not found",
                    "session_id": session_id,
                }
            )
        )
        return

    await send_json(
        json.dumps(
            {
                "type": "status",
                "session_id": session.session_id,
                "is_playing": session.is_playing,
                "playhead": session.playhead,
                "total_chunks": session.total_chunks,
                "is_deviation": session.state.is_deviation,
            }
        )
    )


def run_server(host: str = "0.0.0.0", port: int = 8765) -> None:
    """Start the WebSocket meditation session server.

    Args:
        host: Bind address (default: 0.0.0.0).
        port: Bind port (default: 8765).

    The server handles graceful shutdown on SIGINT/SIGTERM.
    """
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    stop_event = asyncio.Event()

    def _signal_handler() -> None:
        logger.info("Shutdown signal received")
        stop_event.set()

    # Register signal handlers
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _signal_handler)

    async def main() -> None:
        async with websockets.serve(
            _handle_ws,
            host,
            port,
            ping_interval=20,
            ping_timeout=10,
        ) as server:
            logger.info("Meditation WebSocket server running on ws://%s:%d", host, port)
            await stop_event.wait()
            logger.info("Shutting down server...")
            server.close()
            await server.wait_closed()

            # Clean up all active sessions
            for session_id in list(_manager.active_sessions.keys()):
                await _manager.cleanup_session(session_id)

            logger.info("Server stopped")

    try:
        loop.run_until_complete(main())
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    run_server()
