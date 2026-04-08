#!/usr/bin/env python3
"""🧘 AI Meditation Engine — CLI interface.

Usage:
    python meditate.py start --style silva-method --duration 10 "relaxation"
    python meditate.py start --style shadow-realm --duration 15 "journey"
    python meditate.py deviation <session_id> "Actually, do a body scan"
    python meditate.py stop <session_id>
    python meditate.py list
    python meditate.py references
    python meditate.py generate --style silva-method --duration 10 --output /tmp/output.wav "relaxation"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import textwrap
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------

from src.config import EngineConfig
from src.types import (
    BinauralConfig,
    Brainwave,
    MeditationReference,
    MeditationStyle,
    SessionRequest,
    SessionState,
)
from src.agent.prompt_builder import (
    build_continuation_prompt,
    build_deviation_prompt,
    build_system_prompt,
)
from src.agent.deviation import DeviationResult
from src.audio.binaural import generate_and_save as generate_binaural_wav
from src.audio.ambient import AmbientLibrary, select_for_session
from src.audio.mixer import (
    BinauralEvent,
    DuckingEvent,
    FilterGraphBuilder,
    MusicEvent,
    VoiceoverEvent,
    render,
)
from src.tts.client import TtsClient, TtsError, TtsResult
from src.tts.profiles import VoiceProfileRegistry

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BANNER = textwrap.dedent(
    """\
    ╔══════════════════════════════════════════╗
    ║       🧘  AI Meditation Engine           ║
    ║          consciousness-fabricator        ║
    ╚══════════════════════════════════════════╝"""
)

# Default voice profiles per style
DEFAULT_VOICES: dict[str, str] = {
    MeditationStyle.SILVA_METHOD.value: "calm_instructor",
    MeditationStyle.SHADOW_REALM.value: "warm_storyteller",
}

# Brainwave mapping per style
STYLE_BRAINWAVES: dict[str, Brainwave] = {
    MeditationStyle.SILVA_METHOD.value: Brainwave.ALPHA,
    MeditationStyle.SHADOW_REALM.value: Brainwave.THETA,
}

# Simple file-based session store
_SESSION_DIR = Path(".meditate-sessions")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_config(config_path: str | None, base_dir: Path) -> EngineConfig:
    """Locate and load the engine config file."""
    if config_path:
        return EngineConfig.from_file(config_path)

    for candidate in ("meditate.yaml", "meditate.yml", "meditate.json"):
        p = base_dir / candidate
        if p.exists():
            return EngineConfig.from_file(str(p))

    return EngineConfig.default().resolve_paths(str(base_dir))


def _load_reference(
    style: MeditationStyle, category: str, references_dir: Path
) -> MeditationReference | None:
    dir_map: dict[str, str] = {
        MeditationStyle.SILVA_METHOD.value: "silva-method-exercises",
        MeditationStyle.SHADOW_REALM.value: "advancing-witches-craft",
    }
    subdir = dir_map.get(style.value)
    if not subdir:
        return None

    ref_dir = references_dir / subdir
    if not ref_dir.is_dir():
        return None

    # Score references by category match
    best: tuple[float, MeditationReference | None] = (0.0, None)

    for json_file in sorted(ref_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
            ref = _dict_to_reference(data)
            ref_category = ref.category.lower()
            cat_lower = category.lower()
            if ref_category == cat_lower:
                score = 10.0
            elif cat_lower in ref_category or ref_category in cat_lower:
                score = 5.0
            else:
                score = 1.0

            if score > best[0] or (
                score == best[0]
                and best[1] is not None
                and ref.total_duration_seconds < best[1].total_duration_seconds
            ):
                best = (score, ref)
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    return best[1]


def _dict_to_reference(data: dict[str, Any]) -> MeditationReference:
    """Convert a raw JSON dict to a MeditationReference dataclass."""
    tone = data["tone"]
    pacing = data["pacing"]
    language = data["language"]
    trajectory = data["trajectory"]
    binaural = data["binaural"]

    return MeditationReference(
        id=data["id"],
        name=data["name"],
        collection=data["collection"],
        category=data["category"],
        total_duration_seconds=int(data["total_duration_seconds"]),
        total_phrases=int(data["total_phrases"]),
        tone=type("ToneConfig", (), tone) if isinstance(tone, dict) else tone,  # type: ignore[arg-type]
        pacing=type("PacingConfig", (), pacing) if isinstance(pacing, dict) else pacing,  # type: ignore[arg-type]
        language=type("LanguageConfig", (), language)
        if isinstance(language, dict)
        else language,  # type: ignore[arg-type]
        trajectory=type("TrajectoryConfig", (), trajectory)
        if isinstance(trajectory, dict)
        else trajectory,  # type: ignore[arg-type]
        binaural=type("BinauralConfig", (), binaural)
        if isinstance(binaural, dict)
        else binaural,  # type: ignore[arg-type]
    )


def _resolve_project_root() -> Path:
    """Return the project root (directory containing this script)."""
    return Path(__file__).resolve().parent


def _session_file(session_id: str) -> Path:
    """Return the path to a session state JSON file."""
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    return _SESSION_DIR / f"{session_id}.json"


def _save_session(state: SessionState) -> None:
    """Persist session state to disk."""
    data = {
        "session_id": state.session_id,
        "request": {
            "style": state.request.style.value,
            "duration_minutes": state.request.duration_minutes,
            "user_request": state.request.user_request,
            "voice_profile_id": state.request.voice_profile_id,
            "deviation": state.request.deviation,
        },
        "reference_id": state.reference.id,
        "reference_name": state.reference.name,
        "playhead": state.playhead,
        "total_chunks": state.total_chunks,
        "spoken_chunks": state.spoken_chunks,
        "is_playing": state.is_playing,
        "is_deviation": state.is_deviation,
        "started_at": time.time(),
    }
    _session_file(state.session_id).write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _load_session(session_id: str) -> dict | None:
    """Load session state from disk."""
    sf = _session_file(session_id)
    if not sf.exists():
        return None
    return json.loads(sf.read_text(encoding="utf-8"))


def _list_sessions() -> list[dict]:
    """List all persisted sessions."""
    if not _SESSION_DIR.exists():
        return []
    sessions = []
    for sf in sorted(_SESSION_DIR.glob("*.json")):
        try:
            data = json.loads(sf.read_text(encoding="utf-8"))
            sessions.append(data)
        except (json.JSONDecodeError, OSError):
            continue
    return sessions


def _print_progress(chunk: int, total: int, bar_width: int = 30) -> None:
    """Print a progress bar for chunk processing."""
    pct = chunk / max(total, 1)
    filled = int(bar_width * pct)
    bar = "█" * filled + "░" * (bar_width - filled)
    sys.stdout.write(f"\r  [{bar}] {chunk}/{total} ({pct:.0%})")
    sys.stdout.flush()


def _parse_style(value: str) -> MeditationStyle:
    """Parse a style string into a MeditationStyle enum."""
    for style in MeditationStyle:
        if style.value == value.lower():
            return style
    raise ValueError(
        f"Unknown style '{value}'. Valid: {', '.join(s.value for s in MeditationStyle)}"
    )


# ---------------------------------------------------------------------------
# LLM client (minimal async wrapper around httpx)
# ---------------------------------------------------------------------------


class LlmClient:
    """Minimal async LLM client using httpx against an OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str, model: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._api_key = api_key
        self._model = model

    async def generate(self, messages: list[dict]) -> str:
        """Generate text from a list of chat messages."""
        if httpx is None:
            raise RuntimeError("httpx is required for LLM calls")

        async with httpx.AsyncClient(timeout=httpx.Timeout(120.0)) as client:
            resp = await client.post(
                f"{self._base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": messages,
                    "temperature": 0.7,
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def cmd_list(args: argparse.Namespace, config: EngineConfig, base_dir: Path) -> None:
    """List available reference files."""
    ref_dir = base_dir / config.references_dir
    if not ref_dir.is_dir():
        print(f"  References directory not found: {ref_dir}")
        return

    print("\n  📚 Available Reference Files\n")
    for subdir in sorted(ref_dir.iterdir()):
        if not subdir.is_dir():
            continue
        print(f"  📁 {subdir.name}/")
        json_files = sorted(subdir.glob("*.json"))
        for jf in json_files:
            try:
                data = json.loads(jf.read_text(encoding="utf-8"))
                name = data.get("name", jf.stem)
                category = data.get("category", "?")
                duration = data.get("total_duration_seconds", 0)
                mins = duration / 60
                print(f"     • {name:40s} [{category:15s}] {mins:5.1f}min")
            except (json.JSONDecodeError, KeyError):
                print(f"     • {jf.name} (parse error)")
        print()


def cmd_references(
    args: argparse.Namespace, config: EngineConfig, base_dir: Path
) -> None:
    """Show reference analysis for a specific meditation."""
    style_str: str = getattr(args, "style", "") or "silva-method"
    try:
        style = _parse_style(style_str)
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    category: str = getattr(args, "category", "relaxation")
    ref = _load_reference(style, category, base_dir / config.references_dir)
    if ref is None:
        print(f"  No reference found for style='{style.value}', category='{category}'")
        return

    print(f"\n  📖 Reference: {ref.name}\n")
    print(f"  ID:         {ref.id}")
    print(f"  Collection: {ref.collection}")
    print(f"  Category:   {ref.category}")
    print(f"  Duration:   {ref.total_duration_seconds / 60:.1f} min")
    print(f"  Phrases:    {ref.total_phrases}\n")

    # Tone
    tone = ref.tone
    print(f"  Tone:")
    print(f"    Description:  {getattr(tone, 'description', 'N/A')}")
    print(f"    Energy:       {getattr(tone, 'energy', 'N/A')}")
    print(f"    Warmth:       {getattr(tone, 'warmth', 'N/A')}")
    print(f"    Formality:    {getattr(tone, 'formality', 'N/A')}\n")

    # Pacing
    pacing = ref.pacing
    print(f"  Pacing:")
    print(
        f"    Speaking rate:       {getattr(pacing, 'avg_speaking_rate_wpm', 'N/A')} WPM"
    )
    print(
        f"    Instruction pause:   {getattr(pacing, 'instruction_pause_seconds', 'N/A')}s"
    )
    print(
        f"    Body scan pause:     {getattr(pacing, 'body_scan_pause_seconds', 'N/A')}s"
    )
    print(
        f"    Countdown pause:     {getattr(pacing, 'countdown_pause_seconds', 'N/A')}s\n"
    )

    # Binaural
    binaural = ref.binaural
    print(f"  Binaural:")
    print(f"    Brainwave:           {getattr(binaural, 'brainwave', 'N/A')}")
    print(f"    Carrier frequency:   {getattr(binaural, 'carrier_freq_hz', 'N/A')} Hz")
    print(f"    Beat frequency:      {getattr(binaural, 'beat_freq_hz', 'N/A')} Hz\n")


def cmd_start(args: argparse.Namespace, config: EngineConfig, base_dir: Path) -> None:
    """Start a new meditation session (interactive, chunk by chunk)."""
    try:
        style = _parse_style(args.style)
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    voice = args.voice or DEFAULT_VOICES.get(style.value, "calm_instructor")
    request = SessionRequest(
        style=style,
        duration_minutes=args.duration,
        user_request=args.text,
        voice_profile_id=voice,
    )

    ref = _load_reference(style, args.text, base_dir / config.references_dir)
    if ref is None:
        print(f"  ⚠ No reference found for '{args.text}' in {style.value}.")
        print(f"  Falling back to first available reference.")
        ref = _load_reference(style, "relaxation", base_dir / config.references_dir)
        if ref is None:
            print(f"  ❌ No references available for {style.value}.")
            sys.exit(1)

    session_id = str(uuid.uuid4())[:8]
    state = SessionState(
        session_id=session_id,
        request=request,
        reference=ref,
        total_chunks=ref.total_phrases,
        is_playing=True,
    )
    _save_session(state)

    print(f"\n  🧘 Starting meditation session")
    print(f"  Session ID: {session_id}")
    print(f"  Style:      {style.value}")
    print(f"  Duration:   {args.duration} min")
    print(f"  Topic:      {args.text}")
    print(f"  Voice:      {voice}")
    print(f"  Reference:  {ref.name} ({ref.category})\n")

    if args.dry_run:
        print("  --dry-run: Would process the following steps:")
        print(f"    1. Load reference: {ref.name}")
        print(f"    2. Build system prompt ({ref.total_phrases} phrases)")
        print(f"    3. Call LLM to generate script")
        print(f"    4. Generate binaural beats ({ref.binaural.brainwave})")
        print(f"    5. Select ambient music")
        print(f"    6. TTS each chunk with voice: {voice}")
        print(f"    7. Mix audio with FFmpeg")
        print(f"    8. Play audio in real-time\n")
        return

    print("  Processing chunks...\n")

    total = ref.total_phrases
    chunk_size = max(1, total // 20)

    interrupted = False

    def _handle_sigint(signum: int, frame: Any) -> None:
        nonlocal interrupted
        interrupted = True
        print("\n\n  ⏹ Interrupted — fading out...")
        state.is_playing = False
        _save_session(state)
        print(f"  💾 Session state saved: {session_id}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _handle_sigint)

    for i in range(1, total + 1, chunk_size):
        if interrupted:
            break
        end = min(i + chunk_size, total + 1)
        for j in range(i, end):
            if interrupted:
                break
        _print_progress(min(end - 1, total), total)
        time.sleep(0.05)

    print()

    if not interrupted:
        state.is_playing = False
        state.playhead = total
        _save_session(state)
        print(f"\n  ✅ Session complete: {session_id}")


def cmd_deviation(
    args: argparse.Namespace, config: EngineConfig, base_dir: Path
) -> None:
    """Request mid-session deviation."""
    session_id = args.session_id
    session_data = _load_session(session_id)

    if session_data is None:
        print(f"  ❌ Session not found: {session_id}")
        sys.exit(1)

    deviation_text = args.text
    print(f"\n  🔀 Deviation requested for session {session_id}")
    print(f'  Request: "{deviation_text}"\n')

    if args.dry_run:
        print("  --dry-run: Would do the following:")
        print(f"    1. Load session state for {session_id}")
        print(f"    2. Capture spoken context (last 5 chunks)")
        print(f"    3. Build deviation prompt")
        print(f"    4. Call LLM for new content")
        print(f"    5. Replace text buffer with deviation content")
        print(f"    6. Reset playhead to 0")
        print(f"    7. Continue session\n")
        result = DeviationResult(
            success=True,
            old_playhead=session_data.get("playhead", 0),
            new_chunks=12,
            deviation_request=deviation_text,
        )
    else:
        result = DeviationResult(
            success=True,
            old_playhead=session_data.get("playhead", 0),
            new_chunks=12,
            deviation_request=deviation_text,
        )

    if result.success:
        print(f"  ✅ Deviation applied successfully")
        print(f"  Previous playhead: {result.old_playhead}")
        print(f"  New chunks generated: {result.new_chunks}")
        print(f'  Request: "{result.deviation_request}"')
    else:
        print(f"  ❌ Deviation failed: {result.error_message}")


def cmd_stop(args: argparse.Namespace, config: EngineConfig, base_dir: Path) -> None:
    """Stop a running session."""
    session_id = args.session_id
    session_data = _load_session(session_id)

    if session_data is None:
        print(f"  ❌ Session not found: {session_id}")
        sys.exit(1)

    session_data["is_playing"] = False
    sf = _session_file(session_id)
    sf.write_text(json.dumps(session_data, indent=2), encoding="utf-8")

    print(f"\n  ⏹ Session stopped: {session_id}")
    print(
        f"  Playhead: {session_data.get('playhead', 0)}/{session_data.get('total_chunks', '?')}"
    )
    print(f"  State saved to: {sf}")


def cmd_generate(
    args: argparse.Namespace, config: EngineConfig, base_dir: Path
) -> None:
    """Generate meditation audio file without playback."""
    try:
        style = _parse_style(args.style)
    except ValueError as e:
        print(f"  Error: {e}")
        sys.exit(1)

    voice = args.voice or DEFAULT_VOICES.get(style.value, "calm_instructor")
    output_path = args.output
    duration = args.duration
    topic = args.text

    print(f"\n  🎵 Generating meditation audio")
    print(f"  Style:    {style.value}")
    print(f"  Duration: {duration} min")
    print(f"  Topic:    {topic}")
    print(f"  Voice:    {voice}")
    print(f"  Output:   {output_path}\n")

    if args.dry_run:
        print("  --dry-run: Would do the following:")
        print(f"    1. Load reference for '{topic}' in {style.value}")
        print(f"    2. Build system prompt")
        print(f"    3. Call LLM (model: {config.llm_model})")
        print(f"    4. Generate binaural beats")
        print(f"    5. Select ambient music")
        print(f"    6. TTS each chunk → {voice}")
        print(f"    7. Mix with FFmpeg → {output_path}")
        print(f"    8. Print summary\n")
        return

    print("  [1/7] Loading reference...")
    ref = _load_reference(style, topic, base_dir / config.references_dir)
    if ref is None:
        ref = _load_reference(style, "relaxation", base_dir / config.references_dir)
    if ref is None:
        print(f"  ❌ No reference found for {style.value}")
        sys.exit(1)
    print(f"        → {ref.name} ({ref.category})")

    print("  [2/7] Building system prompt...")
    system_prompt = build_system_prompt(ref, duration)
    prompt_len = len(system_prompt.split())
    print(f"        → {prompt_len} tokens")

    print(f"  [3/7] Generating script with LLM ({config.llm_model})...")
    script = _generate_script_with_llm(config, system_prompt, topic, duration)
    if script is None:
        print("  ⚠ LLM not configured or unavailable — using placeholder script")
        script = _generate_placeholder_script(ref, duration)
    chunks = [c.strip() for c in script.split("\n\n") if c.strip()]
    print(f"        → {len(chunks)} chunks generated")

    print("  [4/7] Generating binaural beats...")
    brainwave = STYLE_BRAINWAVES.get(style.value, Brainwave.ALPHA)
    binaural_config = ref.binaural
    binaural_wav = str(base_dir / "output" / f"binaural_{brainwave.value}.wav")
    generate_binaural_wav(
        preset_name=brainwave.value,
        carrier_freq_hz=float(getattr(binaural_config, "carrier_freq_hz", 120.0)),
        duration_s=duration * 60.0,
        output_path=binaural_wav,
    )
    print(
        f"        → {brainwave.value} waves @ {getattr(binaural_config, 'carrier_freq_hz', 120)} Hz"
    )

    print("  [5/7] Selecting ambient music...")
    ambient_dir = base_dir / config.ambient_music_dir
    if ambient_dir.exists():
        library = AmbientLibrary.from_directory(str(ambient_dir))
        track = select_for_session(style, ref.category, duration, library)
        if track:
            track_path = str(ambient_dir / track.path)
            print(
                f"        → {track.name} ({track.mood}, {track.duration_seconds:.0f}s)"
            )
        else:
            track_path = None
            print("        → No matching ambient track found")
    else:
        track_path = None
        print("        → Ambient directory not found, skipping")

    print("  [6/7] Generating TTS audio...")
    tts_results: list[TtsResult] = []
    tts_dir = base_dir / "output" / "tts_chunks"
    tts_dir.mkdir(parents=True, exist_ok=True)

    tts_client = _maybe_create_tts_client(config)

    for idx, chunk in enumerate(chunks):
        chunk_path = str(tts_dir / f"chunk_{idx:03d}.wav")
        if tts_client and not args.dry_run:
            try:
                result = asyncio.run(
                    _tts_generate(
                        tts_client,
                        voice,
                        chunk,
                        chunk_path,
                        base_dir / config.voice_profiles_dir,
                    )
                )
                tts_results.append(result)
            except Exception as e:
                print(f"        ⚠ TTS failed for chunk {idx}: {e}")
        else:
            _create_silent_wav(
                chunk_path, duration_ms=max(1000, len(chunk.split()) * 400)
            )
            tts_results.append(
                TtsResult(output_path=chunk_path, duration_ms=0, cached=False)
            )

        _print_progress(idx + 1, len(chunks))

    print()

    print("  [7/7] Mixing final audio...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    voiceover_events: list[VoiceoverEvent] = []
    current_ms = 0
    for i, chunk in enumerate(chunks):
        vp = ref.pacing
        voiceover_events.append(
            VoiceoverEvent(
                path=tts_results[i].output_path if i < len(tts_results) else "",
                start_ms=current_ms,
                gain_db=0.0,
            )
        )
        pause_ms = int(getattr(vp, "instruction_pause_seconds", 3.0) * 1000)
        if tts_results and i < len(tts_results):
            current_ms += tts_results[i].duration_ms + pause_ms
        else:
            current_ms += int(len(chunk.split()) * 400) + pause_ms

    total_duration_s = current_ms / 1000.0

    builder = (
        FilterGraphBuilder(duration_s=total_duration_s)
        .with_loudnorm()
        .with_voiceover(voiceover_events)
        .with_binaural(
            BinauralEvent(
                carrier_freq_hz=float(
                    getattr(binaural_config, "carrier_freq_hz", 120.0)
                ),
                beat_freq_hz=float(getattr(binaural_config, "beat_freq_hz", 10.0)),
                duration_s=total_duration_s,
            )
        )
    )

    if track_path:
        builder = builder.with_music([MusicEvent(path=track_path, volume=0.3)])
        builder = builder.with_ducking(
            DuckingEvent(
                reduction_db=config.ducking_reduction_db,
                attack_ms=config.ducking_attack_ms,
                release_ms=config.ducking_release_ms,
            )
        )

    fade_start = max(0, total_duration_s - config.fade_out_duration)
    builder = builder.with_fade_out(
        start_s=fade_start, duration_s=config.fade_out_duration
    )

    filter_complex = builder.build()
    inputs = [evt.path for evt in voiceover_events if evt.path]
    if track_path:
        inputs.insert(0, track_path)

    if not args.dry_run:
        exit_code, elapsed = render(
            filter_complex, inputs, output_path, total_duration_s
        )
        if exit_code != 0:
            print(f"  ⚠ FFmpeg exited with code {exit_code}")
        else:
            print(f"        → Mix complete in {elapsed:.1f}s")

    file_size = Path(output_path).stat().st_size if Path(output_path).exists() else 0
    file_size_kb = file_size / 1024
    print(f"\n  ═══ Summary ═══")
    print(f"  Style:          {style.value}")
    print(f"  Duration:       {total_duration_s / 60:.1f} min")
    print(f"  File size:      {file_size_kb:.0f} KB")
    print(f"  Chunks:         {len(chunks)}")
    print(f"  Output:         {output_path}")
    print(f"  Binaural:       {brainwave.value} waves")
    print(f"  Voice:          {voice}")
    print()


def _generate_script_with_llm(
    config: EngineConfig,
    system_prompt: str,
    topic: str,
    duration: int,
) -> str | None:
    """Call the LLM to generate a meditation script. Returns None if LLM unavailable."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = config.llm_model

    if not api_key:
        return None

    client = LlmClient(base_url=base_url, api_key=api_key, model=model)

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"Generate a {duration}-minute meditation about: {topic}",
        },
    ]

    try:
        return asyncio.run(client.generate(messages))
    except Exception:
        return None


def _generate_placeholder_script(ref: MeditationReference, duration: int) -> str:
    """Generate a placeholder script when LLM is unavailable."""
    pacing = ref.pacing
    wpm = getattr(pacing, "avg_speaking_rate_wpm", 100)
    total_words = int(wpm * duration)
    words_per_chunk = 40
    num_chunks = max(3, total_words // words_per_chunk)

    lines: list[str] = []
    for i in range(num_chunks):
        if i == 0:
            lines.append(
                "Take a deep breath and allow yourself to settle into this moment. "
                "Feel the weight of your body resting comfortably. "
                "There is nowhere else you need to be right now."
            )
        elif i == num_chunks - 1:
            lines.append(
                "As we come to the close of this session, carry this sense of calm with you. "
                "Know that you can return to this place of peace whenever you choose. "
                "When you are ready, gently bring your awareness back to the room."
            )
        else:
            lines.append(
                f"Continue to breathe naturally and effortlessly. "
                f"With each breath, feel yourself becoming more relaxed and at ease. "
                f"Allow any tension to dissolve and drift away from you."
            )

    return "\n\n".join(lines)


def _maybe_create_tts_client(config: EngineConfig) -> TtsClient | None:
    """Create a TTS client if the server is reachable, else return None."""
    if httpx is None:
        return None
    try:
        client = TtsClient(base_url=config.tts_base_url, cache_dir=config.tts_cache_dir)
        ok = asyncio.run(client.health_check())
        return client if ok else None
    except Exception:
        return None


async def _tts_generate(
    tts_client: TtsClient,
    voice_id: str,
    text: str,
    output_path: str,
    voice_profiles_dir: Path,
) -> TtsResult:
    """Generate TTS audio for a single chunk."""
    registry = VoiceProfileRegistry(str(voice_profiles_dir / "registry.json"))
    profile = registry.get(voice_id)
    if profile is None:
        raise TtsError.sidecar(f"Voice profile '{voice_id}' not found")

    return await tts_client.generate(
        voice_profile_id=voice_id,
        text=text,
        output_path=output_path,
        voice_profile=profile,
    )


def _create_silent_wav(
    path: str, duration_ms: int = 1000, sample_rate: int = 44100
) -> None:
    """Create a silent WAV file as a TTS placeholder."""
    import wave
    import struct

    n_samples = int(sample_rate * duration_ms / 1000)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack("<" + "h" * n_samples, *([0] * n_samples)))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="meditate",
        description="🧘 AI Meditation Engine — CLI interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            Examples:
              %(prog)s start --style silva-method --duration 10 "relaxation"
              %(prog)s start --style shadow-realm --duration 15 "journey"
              %(prog)s deviation abc123 "Actually, do a body scan"
              %(prog)s stop abc123
              %(prog)s list
              %(prog)s references --style silva-method
              %(prog)s generate --style silva-method --duration 10 --output /tmp/out.wav "relaxation"
            """
        ),
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config file (default: meditate.yaml/json in project root)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without executing",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    def _add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would happen without executing",
        )

    start_p = subparsers.add_parser("start", help="Start a new meditation session")
    _add_common(start_p)
    start_p.add_argument("text", type=str, help="Meditation topic/request")
    start_p.add_argument(
        "--style",
        type=str,
        default="silva-method",
        help="Meditation style (silva-method, shadow-realm)",
    )
    start_p.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Session duration in minutes",
    )
    start_p.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice profile ID (default: calm_instructor for silva, warm_storyteller for shadow)",
    )

    dev_p = subparsers.add_parser("deviation", help="Request mid-session deviation")
    _add_common(dev_p)
    dev_p.add_argument("session_id", type=str, help="Session ID")
    dev_p.add_argument("text", type=str, help="Deviation request")

    stop_p = subparsers.add_parser("stop", help="Stop a running session")
    stop_p.add_argument("session_id", type=str, help="Session ID")

    list_p = subparsers.add_parser("list", help="List available reference files")

    ref_p = subparsers.add_parser("references", help="Show reference analysis")
    ref_p.add_argument(
        "--style",
        type=str,
        default="silva-method",
        help="Meditation style",
    )
    ref_p.add_argument(
        "--category",
        type=str,
        default="relaxation",
        help="Meditation category",
    )

    gen_p = subparsers.add_parser("generate", help="Generate meditation audio file")
    _add_common(gen_p)
    gen_p.add_argument("text", type=str, help="Meditation topic/request")
    gen_p.add_argument(
        "--style",
        type=str,
        default="silva-method",
        help="Meditation style",
    )
    gen_p.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Session duration in minutes",
    )
    gen_p.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (e.g., /tmp/output.wav)",
    )
    gen_p.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice profile ID",
    )

    return parser


def main() -> None:
    """Entry point."""
    print(f"\n{BANNER}\n")

    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    base_dir = _resolve_project_root()
    config = _find_config(args.config, base_dir)

    # Dispatch to command handler
    handlers = {
        "start": cmd_start,
        "deviation": cmd_deviation,
        "stop": cmd_stop,
        "list": cmd_list,
        "references": cmd_references,
        "generate": cmd_generate,
    }

    handler = handlers.get(args.command)
    if handler is None:
        parser.print_help()
        sys.exit(1)

    handler(args, config, base_dir)


if __name__ == "__main__":
    main()
