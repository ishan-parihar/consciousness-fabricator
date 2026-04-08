"""Configuration loading for the meditation engine."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore[import-untyped]

    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False
    yaml = None  # type: ignore[assignment]


@dataclass
class EngineConfig:
    """Top-level engine configuration."""

    # TTS
    tts_base_url: str = "http://localhost:8000"
    tts_cache_dir: str = ".tts-cache"

    # Voices
    voice_profiles_dir: str = "voices"
    voice_profile_registry: str = "voices/registry.json"

    # Audio
    output_dir: str = "output"
    ambient_music_dir: str = "assets/ambient"
    sfx_dir: str = "assets/sfx"

    # References
    references_dir: str = "references"

    # Audio mixing
    loudnorm_enabled: bool = True
    fade_out_duration: float = 5.0
    ducking_reduction_db: float = -20.0
    ducking_attack_ms: int = 50
    ducking_release_ms: int = 200

    # Agent
    max_context_chunks: int = 8
    llm_model: str = "gpt-4o"

    @classmethod
    def from_file(cls, path: str) -> "EngineConfig":
        """Load configuration from a YAML or JSON file."""
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        raw = p.read_text(encoding="utf-8")
        data: dict[str, Any]

        if _HAS_YAML and yaml is not None and (p.suffix in (".yml", ".yaml")):
            data = yaml.safe_load(raw)
        else:
            data = json.loads(raw)

        return cls(**data)

    @classmethod
    def default(cls) -> "EngineConfig":
        """Return a default configuration instance."""
        return cls()

    def resolve_paths(self, base_dir: str) -> "EngineConfig":
        """Convert all relative path fields to absolute paths based on base_dir."""
        base = Path(base_dir).resolve()
        path_fields = [
            "tts_cache_dir",
            "voice_profiles_dir",
            "voice_profile_registry",
            "output_dir",
            "ambient_music_dir",
            "sfx_dir",
            "references_dir",
        ]
        updates: dict[str, str] = {}
        for f in path_fields:
            p = Path(getattr(self, f))
            if not p.is_absolute():
                updates[f] = str((base / p).resolve())
        return EngineConfig(**{**asdict(self), **updates})
