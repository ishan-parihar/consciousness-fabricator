"""Tests for ambient music library: mood inference, loading, selection."""

import json

import pytest

from src.audio.ambient import (
    AmbientTrack,
    AmbientLibrary,
    select_for_session,
    _infer_mood,
)
from src.types import MeditationStyle


# ---------------------------------------------------------------------------
# TestMoodInference
# ---------------------------------------------------------------------------


class TestMoodInference:
    """Mood inference from filename keywords."""

    def test_calm_keywords(self):
        assert _infer_mood("calm_ocean_waves") == "calm"
        assert _infer_mood("gentle_breeze") == "calm"

    def test_meditative_keywords(self):
        assert _infer_mood("theta_deep_focus") == "meditative"

    def test_mystical_keywords(self):
        assert _infer_mood("cosmic_void") == "mystical"
        assert _infer_mood("shadow_realm") == "mystical"

    def test_healing_keywords(self):
        assert _infer_mood("reiki_healing") == "healing"

    def test_unknown_returns_neutral(self):
        assert _infer_mood("random_audio_01") == "neutral"


# ---------------------------------------------------------------------------
# TestAmbientLibraryJSON
# ---------------------------------------------------------------------------


class TestAmbientLibraryJSON:
    """Loading library from JSON index."""

    def test_load_from_json(self, tmp_path):
        index = tmp_path / "music.json"
        index.write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "id": "track-01",
                            "name": "Ocean Waves",
                            "path": "ocean.wav",
                            "mood": "calm",
                            "duration_seconds": 600.0,
                            "style_tags": ["nature"],
                        }
                    ]
                }
            )
        )
        library = AmbientLibrary.from_json(index)
        assert library.count() == 1

    def test_track_accessible(self, tmp_path):
        index = tmp_path / "music.json"
        index.write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "id": "track-01",
                            "name": "Ocean Waves",
                            "path": "ocean.wav",
                            "mood": "calm",
                            "duration_seconds": 600.0,
                            "style_tags": ["nature"],
                        }
                    ]
                }
            )
        )
        library = AmbientLibrary.from_json(index)
        track = library.get("track-01")
        assert track is not None
        assert track.name == "Ocean Waves"

    def test_get_missing_track(self, tmp_path):
        index = tmp_path / "music.json"
        index.write_text(json.dumps({"tracks": []}))
        library = AmbientLibrary.from_json(index)
        assert library.get("nonexistent") is None


# ---------------------------------------------------------------------------
# TestAmbientLibraryDirectory
# ---------------------------------------------------------------------------


class TestAmbientLibraryDirectory:
    """Loading library by scanning a directory."""

    def test_scan_directory(self, tmp_path):
        audio_dir = tmp_path / "ambient"
        audio_dir.mkdir()
        (audio_dir / "track1.mp3").touch()
        (audio_dir / "track2.wav").touch()
        (audio_dir / "notes.txt").touch()

        library = AmbientLibrary.from_directory(audio_dir)
        assert library.count() == 2

    def test_track_id_from_filename(self, tmp_path):
        audio_dir = tmp_path / "ambient"
        audio_dir.mkdir()
        (audio_dir / "calm_ocean.mp3").touch()

        library = AmbientLibrary.from_directory(audio_dir)
        track = library.get("calm_ocean")
        assert track is not None

    def test_mood_inferred_from_filename(self, tmp_path):
        audio_dir = tmp_path / "ambient"
        audio_dir.mkdir()
        (audio_dir / "calm_ocean.mp3").touch()

        library = AmbientLibrary.from_directory(audio_dir)
        track = library.get("calm_ocean")
        assert track is not None
        assert track.mood == _infer_mood("calm_ocean")


# ---------------------------------------------------------------------------
# TestAmbientLibrarySelection
# ---------------------------------------------------------------------------


class TestAmbientLibrarySelection:
    """Track selection and filtering."""

    @staticmethod
    def _make_library():
        return AmbientLibrary(
            tracks=[
                AmbientTrack(
                    id="calm-1",
                    name="Calm One",
                    path="calm1.mp3",
                    mood="calm",
                    duration_seconds=600.0,
                    style_tags=["nature"],
                ),
                AmbientTrack(
                    id="calm-2",
                    name="Calm Two",
                    path="calm2.mp3",
                    mood="calm",
                    duration_seconds=300.0,
                    style_tags=["rain"],
                ),
                AmbientTrack(
                    id="mystical-1",
                    name="Mystical One",
                    path="mystical1.mp3",
                    mood="mystical",
                    duration_seconds=900.0,
                    style_tags=["cosmic"],
                ),
                AmbientTrack(
                    id="short-1",
                    name="Short One",
                    path="short1.mp3",
                    mood="neutral",
                    duration_seconds=60.0,
                    style_tags=[],
                ),
            ]
        )

    def test_select_by_mood(self):
        library = self._make_library()
        track = library.select(mood="calm")
        assert track is not None
        assert track.mood == "calm"

    def test_select_by_min_duration(self):
        library = self._make_library()
        track = library.select(min_duration=500)
        assert track is not None
        assert track.duration_seconds >= 500

    def test_select_filters_by_duration(self):
        library = self._make_library()
        track = library.select(min_duration=500)
        # short-1 (60s) should be excluded
        assert track is None or track.id != "short-1"

    def test_list_by_mood(self):
        library = self._make_library()
        calm_tracks = library.list(mood="calm")
        assert len(calm_tracks) == 2

    def test_list_by_style_tags(self):
        library = self._make_library()
        nature_tracks = library.list(style_tags=["nature"])
        assert len(nature_tracks) == 1
        assert nature_tracks[0].id == "calm-1"


# ---------------------------------------------------------------------------
# TestSelectForSession
# ---------------------------------------------------------------------------


class TestSelectForSession:
    """Session-based track selection."""

    def test_shadow_realm_selects_mystical(self, tmp_path):
        index = tmp_path / "music.json"
        index.write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "id": "mystical-1",
                            "name": "Cosmic Void",
                            "path": "cosmic.mp3",
                            "mood": "mystical",
                            "duration_seconds": 600.0,
                            "style_tags": [],
                        }
                    ]
                }
            )
        )
        library = AmbientLibrary.from_json(index)
        track = select_for_session(
            style=MeditationStyle.SHADOW_REALM,
            category="any",
            duration_minutes=5,
            library=library,
        )
        assert track is not None
        assert track.mood == "mystical"

    def test_silva_relaxation_selects_calm(self, tmp_path):
        index = tmp_path / "music.json"
        index.write_text(
            json.dumps(
                {
                    "tracks": [
                        {
                            "id": "calm-1",
                            "name": "Gentle Breeze",
                            "path": "breeze.mp3",
                            "mood": "calm",
                            "duration_seconds": 600.0,
                            "style_tags": [],
                        }
                    ]
                }
            )
        )
        library = AmbientLibrary.from_json(index)
        track = select_for_session(
            style=MeditationStyle.SILVA_METHOD,
            category="relaxation",
            duration_minutes=5,
            library=library,
        )
        assert track is not None
        assert track.mood == "calm"
