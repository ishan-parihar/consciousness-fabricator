# Consciousness Fabricator — Architecture

> Research prototype: an AI meditation-audio generator that composes guided
> sessions from a text prompt — voice-cloned narration (Qwen 0.6B via the
> Voicebox TTS container), binaural beats, ambient soundscapes, and real-time
> mid-session deviation — mixed to a final WAV with FFmpeg.
>
> Companion: `README.md` (quick start + CLI reference). This doc covers the
> runtime structure, session lifecycle, and the design decisions behind the
> pipeline.

## 1. Component map

```
meditate.py                    CLI entry point (argparse surface: list/generate/start/deviation/stop/references)
meditate.yaml                  EngineConfig: TTS base URL, cache dir, voice profiles, LLM model, fade-out
src/
├── config.py                  EngineConfig loader (YAML → typed config)
├── types.py                   Shared dataclasses + enums (meditation styles, brainwave targets)
├── session.py                 MeditationSession orchestrator — owns a session's lifecycle
├── ws_api.py                  WebSocket API (port 8765) for interactive/live sessions
├── tts/
│   ├── client.py              Voicebox TTS HTTP client (POST /tts/generate, disk cache)
│   └── profiles.py            Voice profile registry (voices/registry.json + ref audio)
├── agent/
│   ├── prompt_builder.py      LLM system prompts per style
│   ├── text_buffer.py         Script → chunking for TTS-friendly segments
│   ├── context_window.py      Sliding context window for long sessions
│   └── deviation.py           Mid-session redirection logic
└── audio/
    ├── binaural.py            Binaural beat generator (Alpha 10 Hz, Theta 6 Hz, …)
    ├── ambient.py             Mood-matched ambient track selection
    └── mixer.py               FFmpeg multi-track mixing + loudness normalization + fade-out
```

## 2. Session lifecycle

A meditation session is the core runtime object (`MeditationSession`).

1. **Load reference** — resolve the requested style (e.g. `silva-method`) to a
   reference document in `references/`.
2. **Generate script** — the LLM (OpenAI-compatible API; placeholder script if
   unconfigured) turns the reference + user prompt into a guided narration,
   chunked by `text_buffer.py` into TTS-friendly segments.
3. **Narrate** — each chunk is sent to Voicebox (`POST /tts/generate`) in
   x-vector-only mode for clean voice cloning from a short reference clip;
   results are cached to `tts_cache_dir` so re-runs avoid re-synthesis.
4. **Build the sound bed** — `binaural.py` synthesizes the style's carrier
   frequency (Alpha 10 Hz for Silva, Theta 6 Hz for Shadow Realm) and
   `ambient.py` selects a mood-matched track.
5. **Mix** — `mixer.py` composites narration + beats + ambient through FFmpeg,
   applies loudness normalization and the configured `fade_out_duration`,
   and writes the final WAV.

### 2.1 Interactive sessions and deviation

`start` opens a live session (chunks played in real time). While running, a
`deviation <session_id> "<new direction>"` call regenerates the *remaining*
script and audio chunks from the new instruction while preserving the current
emotional flow and pacing — the earlier chunks stay untouched. This is what
makes a session a fabricator rather than a fixed tape.

## 3. TTS backend (Voicebox)

Voicebox runs as a Docker container (`voicebox/docker-compose.gpu.yml`) and
exposes an HTTP health check at `http://localhost:17493/health`.

- **Model**: Qwen 0.6B-Base, GPU bfloat16, ~2 GB VRAM (RTX 2060 8 GB OK).
- **Modes**: `x_vector_only_mode` (voice clone, RTF ~0.14×) is primary; ICL mode
  was rejected because reference audio bleeds into the output; preset voices
  (`Qwen CustomVoice 0.6B`) and a CPU fallback (`Chatterbox-Turbo`) are
  secondary paths.
- **Voices**: `voices/registry.json` entries reference a `ref_audio` clip
  (WAV, 24 kHz, ~5–30 s) and a transcript. See README "Adding Custom Voices".

## 4. Design decisions

- **TTS caching** — synthesized chunks are keyed and cached on disk; the
  dominant cost of iteration (GPU TTS) is paid once per script.
- **Chunked narration** — TTS quality degrades on very long prompts, so the
  script is split into segments sized for stable synthesis; the sliding
  context window keeps LLM prompts coherent across a long session.
- **Deterministic sound bed** — binaural generation is pure signal synthesis;
  style → brainwave mapping lives in one place (`types.py` + `binaural.py`)
  so adding a style is data, not pipeline surgery.
- **Graceful degradation** — no LLM key → placeholder script; no ref audio →
  clear error listing `voices/ref/`; no FFmpeg → explicit install guidance.
  The system always fails with an actionable message, never a silent stub.

## 5. Failure modes & troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| TTS connection refused | Voicebox container down | `docker ps \| grep voicebox`; restart compose |
| No audio in output | Voice profile missing ref audio | check `voices/ref/` + `registry.json` |
| FFmpeg not found | FFmpeg missing | install via system package manager |
| Script looks generic | LLM not configured | set `OPENAI_API_KEY` / `OPENAI_BASE_URL` |
