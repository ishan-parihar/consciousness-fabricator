"""Voice profile dataclass and JSON-based registry for TTS voice cloning."""

from __future__ import annotations

import json
import sys
import warnings
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class VoiceProfile:
    """A voice profile for TTS voice cloning."""

    id: str
    provider: str
    mode: str
    model: str
    ref_audio: str
    ref_text: str
    language: str
    description: Optional[str]
    sample_rate: int
    created_at: str


class VoiceProfileRegistry:
    """JSON-based registry for loading, saving, and managing voice profiles."""

    def __init__(self, registry_path: str) -> None:
        path = Path(registry_path)
        if path.exists():
            try:
                content = path.read_text(encoding="utf-8")
                data = json.loads(content)
                self._profiles: dict[str, dict] = data
            except (json.JSONDecodeError, OSError) as e:
                print(
                    f"[WARN] Failed to parse voice profile registry at {path}: {e}",
                    file=sys.stderr,
                )
                self._profiles = {}
        else:
            self._profiles = {}
        self._registry_path = path

    def list(self) -> list[VoiceProfile]:
        """Return all voice profiles as a list."""
        return [self._from_dict(v) for v in self._profiles.values()]

    def get(self, profile_id: str) -> Optional[VoiceProfile]:
        """Return a voice profile by ID, or None if not found."""
        data = self._profiles.get(profile_id)
        if data is None:
            return None
        return self._from_dict(data)

    def add(self, profile: VoiceProfile) -> None:
        """Insert a profile and auto-save."""
        self._profiles[profile.id] = asdict(profile)
        self.save()

    def remove(self, profile_id: str) -> Optional[VoiceProfile]:
        """Remove a profile by ID, auto-save if found. Returns removed profile or None."""
        data = self._profiles.pop(profile_id, None)
        if data is not None:
            self.save()
            return self._from_dict(data)
        return None

    def save(self) -> None:
        """Write the registry as pretty-printed JSON, creating parent dirs if needed."""
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self._profiles, indent=2, ensure_ascii=False)
        self._registry_path.write_text(content, encoding="utf-8")

    @classmethod
    def create(
        cls,
        profile_id: str,
        provider: str,
        mode: str,
        model: str,
        ref_audio: str,
        ref_text: str,
        language: str,
        description: Optional[str] = None,
        sample_rate: int = 22050,
    ) -> VoiceProfile:
        """Factory method to create a new VoiceProfile with auto-generated created_at."""
        return VoiceProfile(
            id=profile_id,
            provider=provider,
            mode=mode,
            model=model,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=language,
            description=description,
            sample_rate=sample_rate,
            created_at=datetime.now(timezone.utc).isoformat(),
        )

    @staticmethod
    def _from_dict(data: dict) -> VoiceProfile:
        """Convert a dict to a VoiceProfile instance."""
        return VoiceProfile(**data)
