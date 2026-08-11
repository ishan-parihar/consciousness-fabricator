<!-- T2I HERO SPEC — Subject: a consciousness fabricator — a text prompt on the left weaving through Qwen-0.6B voice cloning and binaural-beat oscillators into a finished guided-meditation session (voice waveform + ambient soundscape layers); neural-thread loom motif. Composition: prompt → loom → meditation sphere. Palette: deep violet #1e1b4b → meditation teal #2dd4bf → binaural gold #f59e0b. Style: dark mystical flat vector, woven neural threads, no text. 16:9. -->

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python)
![LOC](https://img.shields.io/badge/LOC-9.2K-informational?style=flat-square)
[![CI](https://github.com/ishan-parihar/consciousness-fabricator/actions/workflows/ci.yml/badge.svg)](https://github.com/ishan-parihar/consciousness-fabricator/actions/workflows/ci.yml)
![GPU](https://img.shields.io/badge/GPU-Qwen%200.6B-ff6f00)
![FFmpeg](https://img.shields.io/badge/FFmpeg-6+-red?logo=ffmpeg)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-orange)


**AI-powered meditation audio generator. Produces guided meditation sessions with voice-cloned narration, binaural beats, and ambient soundscapes — all from a text prompt.**

> **The Problem**: Generating guided meditation audio usually requires either expensive human voice actors or robotic TTS that lacks the intimate, nuanced delivery needed for deep relaxation. The challenge was to create a system that could not only clone a warm, human-like voice but also synchronize it with psycho-acoustic elements like binaural beats and ambient music to induce specific mental states.

## Engineering Highlights

### Voice-Cloned Narrator with Qwen-TTS
I integrated a GPU-accelerated voice-cloning engine based on Qwen 0.6B, achieving high-fidelity, emotionally resonant narration. By utilizing x-vector-only mode, I enabled clean voice cloning from short reference samples, delivering a "warm storyteller" feel that rivals human narration while maintaining a real-time factor (RTF) of ~0.14x.

### Psycho-Acoustic Audio Pipeline
I built a multi-track audio mixer using FFmpeg that synchronizes voice-cloned narration with binaural beats (e.g., Alpha waves for the Silva Method) and mood-matched ambient soundscapes. This creates a cohesive, immersive audio environment designed to drive the listener toward specific brainwave states.

### Dynamic Session Deviation
I implemented a "mid-session deviation" feature that allows the system to change the direction of a meditation in real-time based on user input. The system dynamically regenerates the remaining script and audio chunks while maintaining the current emotional flow and pacing, allowing for truly interactive guided experiences.

### Installation

```bash
git clone https://github.com/ishan-parihar/consciousness-fabricator.git
cd consciousness-fabricator
pip install -r requirements.txt
```

### Generate Your First Meditation

```bash
python meditate.py generate --style silva-method --duration 10 --output output/relaxation.wav "relaxation"
```

This generates a 10-minute guided meditation with:
- **Voice-cloned narration** (Qwen 0.6B-Base on GPU)
- **Binaural beats** (alpha waves for Silva Method)
- **Ambient music** (mood-matched soundtrack)
- **Mixed final audio** with loudness normalization and fade-out

## Voicebox Setup

Voicebox runs as a Docker container providing the TTS backend. It must be running before generating meditations.

```bash
cd voicebox
docker compose -f docker-compose.gpu.yml up -d
```

Verify it's running:

```bash
curl http://localhost:17493/health
```

### GPU Requirements

- **Qwen 0.6B-Base** — requires ~2GB VRAM (works on RTX 2060 8GB)
- Models are auto-downloaded on first use (~5GB disk)

### Adding Custom Voices

1. Place reference audio in `voices/ref/` (WAV, 24kHz, ~5-30 seconds)
2. Create a profile in `voices/registry.json`:

```json
{
  "my_voice": {
    "id": "my_voice",
    "provider": "qwen3-tts",
    "mode": "voice_clone",
    "ref_audio": "voices/ref/my_voice_ref.wav",
    "ref_text": "Transcript of the reference audio",
    "language": "English",
    "description": "Description of the voice",
    "sample_rate": 24000,
    "created_at": "2026-04-08T00:00:00"
  }
}
```

## Usage

### CLI Commands

```bash
# List available meditation references


python meditate.py list

# Generate meditation audio file
python meditate.py generate --style silva-method --duration 10 \
  --output output/session.wav "relaxation"

# Use a specific voice profile
python meditate.py generate --style shadow-realm --duration 15 \
  --voice warm_storyteller --output output/journey.wav "inner journey"

# Start an interactive session (chunks played in real-time)
python meditate.py start --style silva-method --duration 10 "relaxation"

# Mid-session deviation (change direction)
python meditate.py deviation <session_id> "do a body scan instead"

# Stop a running session
python meditate.py stop <session_id>

# Show reference analysis
python meditate.py references --style silva-method --category relaxation
```

### Voice Profiles

| Profile ID | Style | Description |
|---|---|---|
| `calm_instructor` | Silva Method | Calm, methodical instructor voice |
| `warm_storyteller` | Shadow Realm | Warm, intimate storyteller voice |

### Meditation Styles

| Style | Brainwave | Default Voice |
|---|---|---|
| `silva-method` | Alpha (10Hz) | calm_instructor |
| `shadow-realm` | Theta (6Hz) | warm_storyteller |

## Configuration

Edit `meditate.yaml` to customize:

```yaml
tts_base_url: "http://localhost:17493"   # Voicebox TTS server
tts_cache_dir: ".tts-cache"              # TTS output cache
voice_profiles_dir: "voices"             # Voice profiles directory
output_dir: "output"                     # Generated audio output
ambient_music_dir: "assets/ambient"      # Ambient music tracks
llm_model: "gpt-4o"                      # LLM for script generation
fade_out_duration: 5.0                   # Final fade-out (seconds)
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    meditate.py (CLI)                     │
│  ┌───────────┐  ┌─────────────┐  ┌──────────────────┐   │
│  │ Reference │  │ LLM Client  │  │ Audio Pipeline   │   │
│  │ Loader    │  │ (optional)  │  │ (FFmpeg mixer)   │   │
│  └─────┬─────┘  └──────┬──────┘  └────────┬─────────┘   │
│        │               │                   │             │
│  ┌─────▼───────────────▼───────────────────▼─────────┐   │
│  │              Meditation Pipeline                   │   │
│  │  1. Load reference → 2. Generate script (LLM)     │   │
│  │  3. Split into chunks → 4. TTS per chunk          │   │
│  │  5. Generate binaural beats → 6. Select ambient   │   │
│  │  7. Mix everything with FFmpeg → output WAV       │   │
│  └───────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
                           │
                  POST /tts/generate
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Voicebox (Docker :17493)                    │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Qwen 0.6B-Base (GPU, bfloat16)                  │   │
│  │  x_vector_only_mode → clean voice cloning         │   │
│  │  RTF ~0.14x (7x faster than real-time)            │   │
│  └──────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Project Structure

```
consciousness-fabricator/
├── meditate.py                 # CLI entry point
├── meditate.yaml               # Engine configuration
├── requirements.txt            # Python dependencies
├── src/
│   ├── config.py               # EngineConfig loader
│   ├── types.py                # Shared dataclasses + enums
│   ├── session.py              # MeditationSession orchestrator
│   ├── ws_api.py               # WebSocket API (port 8765)
│   ├── tts/
│   │   ├── client.py           # Voicebox TTS HTTP client
│   │   └── profiles.py         # Voice profile registry
│   ├── agent/
│   │   ├── prompt_builder.py   # LLM system prompts
│   │   ├── text_buffer.py      # Script chunking
│   │   ├── context_window.py   # Sliding context window
│   │   └── deviation.py        # Mid-session changes
│   └── audio/
│       ├── binaural.py         # Binaural beat generator
│       ├── ambient.py          # Ambient track selection
│       └── mixer.py            # FFmpeg audio mixing
├── voices/
│   ├── registry.json           # Voice profile registry
│   └── ref/                    # Reference audio files
├── references/                 # Generated meditation references
│   ├── silva-method-exercises/
│   └── advancing-witches-craft/
├── meditation-repo/            # Source meditation content
└── assets/                     # Ambient music + SFX
```

## TTS Performance

| Model | Mode | Device | RTF | Notes |
|---|---|---|---|---|
| Qwen 0.6B-Base | x_vector (voice clone) | GPU | ~0.14x | Primary — clean output |
| Qwen 0.6B-Base | ICL | GPU | ~0.14x | ❌ Context bleeding from ref audio |
| Qwen CustomVoice 0.6B | Preset voices | GPU | ~2.1x | 9 built-in speakers |
| Chatterbox-Turbo | Voice clone | CPU | ~5.7x | Fallback, English only |

## Troubleshooting

**TTS connection refused** — Make sure voicebox is running:
```bash
docker ps | grep voicebox
curl http://localhost:17493/health
```

**No audio in output** — Check that voice profile ref audio exists:
```bash
ls -la voices/ref/
cat voices/registry.json
```

**FFmpeg not found** — Install FFmpeg:
```bash
# Arch Linux
sudo pacman -S ffmpeg
# Ubuntu/Debian
sudo apt install ffmpeg
```

**LLM not configured** — Meditation generates with placeholder script if no API key:
```bash
# Set your OpenAI-compatible API key (optional — without one a placeholder
# script is generated)
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

---

Developed by [Ishan Parihar](https://github.com/ishanparihar)

---

## ☕ Support & Sponsorship

If you find this project useful, consider supporting ongoing development:

[![Sponsor](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?style=flat-square&logo=github)](https://github.com/sponsors/ishan-parihar)
[![Donate](https://img.shields.io/badge/Donate-Razorpay-3395FF?style=flat-square)](https://rzp.io/rzp/ishan-parihar)

Your support funds new features, releases, and infrastructure for the whole ecosystem.