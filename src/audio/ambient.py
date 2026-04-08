"""
Ambient music selection and management for meditation sessions.

Loads track metadata from a JSON index or scans a directory for audio files.
Provides filtering and selection based on mood, duration, and style tags.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.types import MeditationStyle

# ---------------------------------------------------------------------------
# Audio file extensions recognized by from_directory
# ---------------------------------------------------------------------------
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg"}

# ---------------------------------------------------------------------------
# Mood keywords inferred from filenames
# ---------------------------------------------------------------------------
_MOOD_KEYWORDS = {
    "calm": ["calm", "calming", "gentle", "peaceful", "soft", "quiet", "still"],
    "meditative": ["meditat", "theta", "deep", "trance", "focus"],
    "mystical": ["mystic", "cosmic", "space", "ether", "celestial", "shadow"],
    "healing": ["heal", "warm", "reiki", "chakra", "light", "renew"],
    "energizing": ["energ", "uplift", "bright", "awake", "morning"],
}


def _infer_mood(filename: str) -> str:
    """Infer mood from filename keywords. Falls back to 'neutral'."""
    lower = filename.lower()
    for mood, keywords in _MOOD_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return mood
    return "neutral"


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AmbientTrack:
    """Single ambient music track."""

    id: str
    name: str
    path: str
    mood: str
    duration_seconds: float
    style_tags: list[str] = field(default_factory=list)


@dataclass
class AmbientLibrary:
    """Collection of ambient tracks with fast lookup and selection."""

    tracks: list[AmbientTrack]
    _index: dict[str, AmbientTrack] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._index = {track.id: track for track in self.tracks}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    @classmethod
    def from_json(cls, path: str | Path) -> "AmbientLibrary":
        """Load tracks from a music index JSON file.

        Expected JSON format:
        {
            "tracks": [
                {
                    "id": "track-id",
                    "name": "Track Name",
                    "path": "relative/path.mp3",
                    "mood": "calm",
                    "duration_seconds": 600.0,
                    "style_tags": ["nature", "rain"]
                }
            ]
        }

        Args:
            path: Path to the JSON index file.

        Returns:
            AmbientLibrary with loaded tracks.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        tracks = [
            AmbientTrack(
                id=t["id"],
                name=t["name"],
                path=t["path"],
                mood=t["mood"],
                duration_seconds=float(t["duration_seconds"]),
                style_tags=list(t.get("style_tags", [])),
            )
            for t in data["tracks"]
        ]
        return cls(tracks=tracks)

    @classmethod
    def from_directory(cls, path: str | Path) -> "AmbientLibrary":
        """Scan a directory for audio files and build a library.

        Track metadata:
        - id: filename without extension
        - name: filename without extension (title-cased)
        - path: relative file path
        - mood: inferred from keywords in the filename
        - duration_seconds: 0.0 (unknown without audio analysis)
        - style_tags: parent directory name as single tag

        Args:
            path: Directory to scan for audio files.

        Returns:
            AmbientLibrary with discovered tracks.
        """
        base = Path(path)
        tracks: list[AmbientTrack] = []

        for root, _dirs, files in os.walk(base):
            for filename in sorted(files):
                ext = Path(filename).suffix.lower()
                if ext not in AUDIO_EXTENSIONS:
                    continue

                relative = Path(root, filename).relative_to(base)
                stem = Path(filename).stem
                parent_dir = relative.parent.name if relative.parent.name else "ambient"

                tracks.append(
                    AmbientTrack(
                        id=stem,
                        name=re.sub(r"[-_]+", " ", stem).title(),
                        path=str(relative),
                        mood=_infer_mood(stem),
                        duration_seconds=0.0,
                        style_tags=[parent_dir] if parent_dir else [],
                    )
                )

        return cls(tracks=tracks)

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def select(
        self,
        mood: str | None = None,
        min_duration: float | None = None,
        style_tags: list[str] | None = None,
    ) -> AmbientTrack | None:
        """Select a track matching the given criteria.

        Selection priority:
        1. Filter out tracks below min_duration (if specified)
        2. Score remaining tracks by mood match (+3) and style tag matches (+1 per tag)
        3. Return the highest-scoring track, or None if no candidates

        Args:
            mood: Preferred mood (e.g., "calm", "mystical").
            min_duration: Minimum duration in seconds.
            style_tags: Preferred style tags (e.g., ["nature", "rain"]).

        Returns:
            Best-matching AmbientTrack or None.
        """
        candidates = self.tracks

        if min_duration is not None:
            candidates = [t for t in candidates if t.duration_seconds >= min_duration]

        if not candidates:
            return None

        def _score(track: AmbientTrack) -> int:
            score = 0
            if mood is not None and track.mood == mood:
                score += 3
            if style_tags:
                track_tags_lower = {tag.lower() for tag in track.style_tags}
                score += sum(1 for tag in style_tags if tag.lower() in track_tags_lower)
            return score

        scored = [(t, _score(t)) for t in candidates]
        # Sort by score descending; ties broken by original order
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[0][0]

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, id: str) -> AmbientTrack | None:
        """Look up a track by its ID.

        Args:
            id: Track identifier.

        Returns:
            AmbientTrack if found, None otherwise.
        """
        return self._index.get(id)

    def list(
        self,
        mood: str | None = None,
        style_tags: list[str] | None = None,
    ) -> list[AmbientTrack]:
        """List tracks with optional filters.

        Args:
            mood: Filter by mood (exact match).
            style_tags: Filter by style tags (track must have ALL specified tags).

        Returns:
            Matching tracks in original order.
        """
        results = self.tracks

        if mood is not None:
            results = [t for t in results if t.mood == mood]

        if style_tags:
            tags_set = {tag.lower() for tag in style_tags}
            results = [
                t for t in results if tags_set <= {tag.lower() for tag in t.style_tags}
            ]

        return results

    def count(self) -> int:
        """Return total number of tracks in the library."""
        return len(self.tracks)


# ---------------------------------------------------------------------------
# Session helper
# ---------------------------------------------------------------------------


def select_for_session(
    style: MeditationStyle,
    category: str,
    duration_minutes: int,
    library: AmbientLibrary,
) -> AmbientTrack | None:
    """Pick ambient music based on meditation style and category.

    Mood mapping:
    - Silva Method + relaxation → "calm"
    - Silva Method + deepening/projection → "meditative"
    - Silva Method + healing → "healing"
    - Shadow Realm (any category) → "mystical"

    Args:
        style: The MeditationStyle enum value.
        category: Session category (e.g., "relaxation", "deepening", "healing").
        duration_minutes: Target session duration in minutes.
        library: AmbientLibrary to select from.

    Returns:
        Best-matching AmbientTrack or None.
    """
    category_lower = category.lower()

    if style == MeditationStyle.SHADOW_REALM:
        preferred_mood = "mystical"
    elif style == MeditationStyle.SILVA_METHOD:
        if category_lower == "relaxation":
            preferred_mood = "calm"
        elif category_lower in ("deepening", "projection"):
            preferred_mood = "meditative"
        elif category_lower == "healing":
            preferred_mood = "healing"
        else:
            preferred_mood = None
    else:
        preferred_mood = None

    min_duration = duration_minutes * 60.0

    return library.select(
        mood=preferred_mood,
        min_duration=min_duration,
    )
