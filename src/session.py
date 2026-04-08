"""Session manager that orchestrates the meditation engine.

Ties together all components — prompt builder, text buffer, context window,
deviation handler, TTS client, audio mixer, ambient selector, binaural
generator — into a cohesive session flow.
"""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
import subprocess
import uuid
import wave
from pathlib import Path
from typing import Any, Protocol

from src.agent.context_window import ContextWindow
from src.agent.deviation import DeviationHandler, DeviationResult
from src.agent.prompt_builder import build_system_prompt
from src.agent.text_buffer import TextBuffer
from src.audio.ambient import AmbientLibrary, select_for_session
from src.audio.binaural import generate_and_save
from src.audio.mixer import (
    BinauralEvent,
    DuckingEvent,
    FilterGraphBuilder,
    MusicEvent,
    VoiceoverEvent,
    render,
)
from src.config import EngineConfig
from src.tts.client import TtsClient
from src.tts.profiles import VoiceProfileRegistry
from src.types import (
    BinauralConfig,
    Brainwave,
    LanguageConfig,
    MeditationReference,
    MeditationStyle,
    PacingConfig,
    SessionRequest,
    SessionState,
    ToneConfig,
    TrajectoryConfig,
)

logger = logging.getLogger(__name__)

_STYLE_TO_DIR: dict[MeditationStyle, str] = {
    MeditationStyle.SILVA_METHOD: "silva-method-exercises",
    MeditationStyle.SHADOW_REALM: "advancing-witches-craft",
}


def _load_reference_from_file(path: str) -> MeditationReference:
    raw = Path(path).read_text(encoding="utf-8")
    data: dict[str, Any] = json.loads(raw)

    binaural = data["binaural"]
    pacing = data["pacing"]
    tone = data["tone"]
    language = data["language"]
    trajectory = data["trajectory"]

    brainwave = Brainwave(binaural["brainwave"])
    act_boundaries = [int(b) for b in pacing.get("act_boundaries", [])]

    return MeditationReference(
        id=data["id"],
        name=data["name"],
        collection=data["collection"],
        category=data["category"],
        total_duration_seconds=int(data["total_duration_seconds"]),
        total_phrases=int(data["total_phrases"]),
        tone=ToneConfig(
            energy=tone["energy"],
            warmth=tone["warmth"],
            formality=tone["formality"],
            description=tone["description"],
        ),
        pacing=PacingConfig(
            avg_speaking_rate_wpm=float(pacing["avg_speaking_rate_wpm"]),
            instruction_pause_seconds=float(pacing["instruction_pause_seconds"]),
            body_scan_pause_seconds=float(pacing["body_scan_pause_seconds"]),
            countdown_pause_seconds=float(pacing["countdown_pause_seconds"]),
            act_boundaries=act_boundaries,
        ),
        language=LanguageConfig(
            sentence_style=language["sentence_style"],
            perspective=language["perspective"],
            common_phrases=language.get("common_phrases", []),
            structural_patterns=language.get("structural_patterns", []),
            repetition_rate=float(language.get("repetition_rate", 0.35)),
            avg_sentence_length_words=int(
                language.get("avg_sentence_length_words", 12)
            ),
        ),
        trajectory=TrajectoryConfig(
            opening=trajectory["opening"],
            deepening=trajectory["deepening"],
            transitions=trajectory["transitions"],
            deviation_handling=trajectory["deviation_handling"],
        ),
        binaural=BinauralConfig(
            brainwave=brainwave,
            carrier_freq_hz=float(binaural["carrier_freq_hz"]),
            beat_freq_hz=float(binaural.get("beat_freq_hz", 10.0)),
        ),
    )


def _find_first_reference(
    style: MeditationStyle, references_dir: str
) -> MeditationReference | None:
    dir_name = _STYLE_TO_DIR.get(style)
    if dir_name is None:
        return None

    ref_dir = Path(references_dir) / dir_name
    if not ref_dir.is_dir():
        return None

    json_files = sorted(ref_dir.glob("*.json"))
    if not json_files:
        return None

    return _load_reference_from_file(str(json_files[0]))


class LLMGenerateProtocol(Protocol):
    async def generate(self, messages: list[dict]) -> str: ...


def _concat_tts_chunks(chunk_paths: list[str], output_path: str) -> bool:
    if not chunk_paths:
        return False

    concat_file = output_path + ".concat.txt"
    with open(concat_file, "w", encoding="utf-8") as f:
        for cp in chunk_paths:
            escaped = cp.replace("'", r"'\''")
            f.write(f"file '{escaped}'\n")

    try:
        result = _run_ffmpeg(
            [
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                concat_file,
                "-c",
                "copy",
                "-y",
                output_path,
            ]
        )
        return result == 0
    finally:
        Path(concat_file).unlink(missing_ok=True)


def _run_ffmpeg(extra_args: list[str]) -> int:
    cmd = ["ffmpeg", *extra_args]
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        logger.warning("FFmpeg exited %d: %s", result.returncode, result.stderr[:500])
    return result.returncode


class MeditationSession:
    """Orchestrates the full meditation session lifecycle.

    Manages concurrent sessions, each running as an independent asyncio task.
    """

    def __init__(
        self,
        config: EngineConfig,
        llm_client: LLMGenerateProtocol | None = None,
        ambient_lib: AmbientLibrary | None = None,
    ) -> None:
        self._config = config
        self._llm_client = llm_client
        self._sessions: dict[str, SessionState] = {}
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}

        self._tts_client = TtsClient(
            base_url=config.tts_base_url,
            cache_dir=config.tts_cache_dir,
        )
        self._voice_registry = VoiceProfileRegistry(config.voice_profile_registry)
        self._ambient_lib = ambient_lib or AmbientLibrary.from_directory(
            config.ambient_music_dir
        )
        Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    async def start(self, request: SessionRequest) -> str:
        session_id = str(uuid.uuid4())

        reference = _find_first_reference(request.style, self._config.references_dir)
        if reference is None:
            raise FileNotFoundError(
                f"No reference found for style '{request.style.value}' "
                f"in {self._config.references_dir}"
            )

        state = SessionState(
            session_id=session_id,
            request=request,
            reference=reference,
            is_playing=True,
            is_deviation=False,
        )
        self._sessions[session_id] = state

        stop_event = asyncio.Event()
        self._stop_events[session_id] = stop_event

        task = asyncio.create_task(
            self._run_session(session_id, reference, request),
            name=f"session-{session_id[:8]}",
        )
        self._tasks[session_id] = task
        task.add_done_callback(
            lambda t: self._sessions.get(session_id)
            and setattr(self._sessions[session_id], "is_playing", False)
        )

        logger.info(
            "Started session %s (style=%s, duration=%dm)",
            session_id[:8],
            request.style.value,
            request.duration_minutes,
        )
        return session_id

    async def handle_deviation(
        self, session_id: str, user_request: str
    ) -> DeviationResult:
        state = self._sessions.get(session_id)
        if state is None:
            return DeviationResult(
                success=False,
                old_playhead=0,
                new_chunks=0,
                deviation_request=user_request,
                error_message=f"Unknown session: {session_id}",
            )

        if self._llm_client is None:
            return DeviationResult(
                success=False,
                old_playhead=state.playhead,
                new_chunks=0,
                deviation_request=user_request,
                error_message="No LLM client configured",
            )

        state.is_deviation = True

        context_window = ContextWindow(max_chunks=self._config.max_context_chunks)
        context_window.set_system_prompt(
            build_system_prompt(state.reference, state.request.duration_minutes)
        )
        for chunk in state.spoken_chunks:
            context_window.add_spoken_chunk(chunk)

        text_buffer = TextBuffer("", state.reference.pacing)

        handler = DeviationHandler(
            reference=state.reference,
            context_window=context_window,
            text_buffer=text_buffer,
            llm_client=self._llm_client,
        )

        result = await handler.handle(user_request)
        logger.info(
            "Deviation for session %s: success=%s, new_chunks=%d",
            session_id[:8],
            result.success,
            result.new_chunks,
        )
        return result

    async def stop(self, session_id: str) -> None:
        state = self._sessions.get(session_id)
        if state is None:
            logger.warning("Stop called for unknown session: %s", session_id[:8])
            return

        stop_event = self._stop_events.get(session_id)
        if stop_event:
            stop_event.set()

        state.is_playing = False

        task = self._tasks.get(session_id)
        if task and not task.done():
            try:
                await asyncio.wait_for(task, timeout=30.0)
            except asyncio.TimeoutError:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        logger.info("Stopped session %s", session_id[:8])

    async def _run_session(
        self,
        session_id: str,
        reference: MeditationReference,
        request: SessionRequest,
    ) -> None:
        state = self._sessions[session_id]
        output_dir = Path(self._config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        system_prompt = build_system_prompt(reference, request.duration_minutes)
        logger.debug("System prompt built (%d chars)", len(system_prompt))

        context_window = ContextWindow(max_chunks=self._config.max_context_chunks)
        context_window.set_system_prompt(system_prompt)

        if self._llm_client is not None:
            logger.info(
                "Generating meditation script for session %s...", session_id[:8]
            )
            llm_response = await self._llm_client.generate(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": (
                            f"Generate a {request.duration_minutes}-minute "
                            f"{reference.category} meditation"
                        ),
                    },
                ]
            )
            logger.info("LLM script generated (%d chars)", len(llm_response))
        else:
            llm_response = "Close your eyes and take a deep breath. Relax completely."
            logger.info("No LLM client — using placeholder script")

        buffer = TextBuffer(llm_response, reference.pacing)
        state.total_chunks = len(buffer.chunks)
        logger.info("Text buffer ready: %d chunks", state.total_chunks)

        duration_seconds = request.duration_minutes * 60
        binaural_path = str(output_dir / "binaural.wav")

        brainwave_value = reference.binaural.brainwave.value
        generate_and_save(
            brainwave_value,
            reference.binaural.carrier_freq_hz,
            duration_seconds,
            binaural_path,
        )
        logger.info("Binaural audio saved: %s", binaural_path)

        ambient_track = select_for_session(
            request.style,
            reference.category,
            request.duration_minutes,
            self._ambient_lib,
        )
        ambient_path = (
            str(Path(self._config.ambient_music_dir) / ambient_track.path)
            if ambient_track
            else None
        )

        voice_profile_id = request.voice_profile_id or "calm_instructor"
        voice_profile = self._voice_registry.get(voice_profile_id)
        if voice_profile is None:
            voice_profile = self._voice_registry.get("calm_instructor")
        if voice_profile is None:
            profiles = self._voice_registry.list()
            voice_profile = profiles[0] if profiles else None

        tts_chunk_paths: list[str] = []
        chunk_index = 0
        stop_event = self._stop_events.get(session_id, asyncio.Event())

        while not stop_event.is_set():
            chunk = buffer.get_next_chunk()
            if chunk is None:
                break

            chunk_index += 1
            tts_output = str(output_dir / f"chunk_{chunk_index:04d}.wav")

            if voice_profile is not None:
                try:
                    tts_result = await self._tts_client.generate(
                        voice_profile_id=voice_profile.id,
                        text=chunk,
                        output_path=tts_output,
                        voice_profile=voice_profile,
                    )
                    logger.debug(
                        "TTS chunk %d generated: %s (cached=%s)",
                        chunk_index,
                        tts_result.output_path,
                        tts_result.cached,
                    )
                except Exception as e:
                    logger.warning(
                        "TTS failed for chunk %d: %s — skipping", chunk_index, e
                    )
                    buffer.mark_spoken()
                    state.spoken_chunks.append(chunk)
                    context_window.add_spoken_chunk(chunk)
                    state.playhead = buffer.playhead
                    continue
            else:
                logger.warning(
                    "No voice profile — silent placeholder for chunk %d", chunk_index
                )
                _create_silent_wav(tts_output, duration_ms=500)
                tts_result = None

            buffer.mark_spoken()
            context_window.add_spoken_chunk(chunk)
            tts_chunk_paths.append(tts_output)

            state.playhead = buffer.playhead
            state.spoken_chunks = list(buffer.spoken_chunks)

            logger.info("Chunk %d/%d processed", chunk_index, state.total_chunks)

        if stop_event.is_set():
            logger.info(
                "Session %s stopped early at chunk %d", session_id[:8], chunk_index
            )

        if tts_chunk_paths:
            voiceover_path = str(output_dir / "voiceover.wav")
            _concat_tts_chunks(tts_chunk_paths, voiceover_path)

            total_duration = duration_seconds
            fade_start = max(0.0, total_duration - self._config.fade_out_duration)

            graph = (
                FilterGraphBuilder(duration_s=total_duration)
                .with_loudnorm()
                .with_voiceover([VoiceoverEvent(path=voiceover_path, gain_db=0.0)])
                .with_binaural(
                    BinauralEvent(
                        carrier_freq_hz=reference.binaural.carrier_freq_hz,
                        beat_freq_hz=reference.binaural.beat_freq_hz,
                        duration_s=total_duration,
                    )
                )
                .with_fade_out(
                    start_s=fade_start, duration_s=self._config.fade_out_duration
                )
            )

            if ambient_path and Path(ambient_path).exists():
                graph.with_music([MusicEvent(path=ambient_path, volume=0.3)])
                if self._config.loudnorm_enabled:
                    graph.with_ducking(
                        DuckingEvent(
                            reduction_db=self._config.ducking_reduction_db,
                            attack_ms=self._config.ducking_attack_ms,
                            release_ms=self._config.ducking_release_ms,
                        )
                    )

            filter_complex = graph.build()
            final_output = str(output_dir / f"session_{session_id[:8]}.m4a")

            inputs: list[str] = [voiceover_path, binaural_path]
            if ambient_path and Path(ambient_path).exists():
                inputs.append(ambient_path)

            if filter_complex:
                exit_code, elapsed = render(
                    filter_complex=filter_complex,
                    inputs=inputs,
                    output_path=final_output,
                    duration_s=total_duration,
                )
                if exit_code == 0:
                    logger.info(
                        "Final mix saved: %s (rendered in %.1fs)", final_output, elapsed
                    )
                else:
                    logger.error(
                        "FFmpeg render failed with exit code %d for session %s",
                        exit_code,
                        session_id[:8],
                    )
            else:
                shutil.copy2(voiceover_path, final_output)
                logger.info("Voiceover saved as final output: %s", final_output)

            for chunk_path in tts_chunk_paths:
                try:
                    Path(chunk_path).unlink(missing_ok=True)
                except OSError:
                    pass
        else:
            logger.warning(
                "No TTS chunks produced for session %s — skipping final mix",
                session_id[:8],
            )

        state.is_playing = False
        logger.info("Session %s complete", session_id[:8])


def _create_silent_wav(output_path: str, duration_ms: int = 500) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    sample_rate = 22050
    n_frames = int(sample_rate * duration_ms / 1000)
    silence = b"\x00" * (n_frames * 2)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(silence)
