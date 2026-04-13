"""Tests for VoiceProfileRegistry and VoiceProfileFactory."""

import json
import pytest
from pathlib import Path

from src.tts.profiles import VoiceProfile, VoiceProfileRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_profile():
    """A minimal VoiceProfile for testing."""
    return VoiceProfile(
        id="test_voice",
        provider="qwen3-tts",
        mode="voice_clone",
        model="qwen-0.6b-base",
        ref_audio="voices/ref/test.wav",
        ref_text="Hello, this is a test.",
        language="English",
        description="Test voice profile",
        sample_rate=22050,
        created_at="2026-04-08T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# TestVoiceProfileRegistry
# ---------------------------------------------------------------------------


class TestVoiceProfileRegistry:
    def test_empty_registry(self, tmp_path):
        """load empty JSON {}, list() returns empty list."""
        registry_file = tmp_path / "empty_registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        assert registry.list() == []

    def test_add_profile(self, tmp_path, sample_profile):
        """add a profile, list() returns 1 profile with correct ID."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        registry.add(sample_profile)

        profiles = registry.list()
        assert len(profiles) == 1
        assert profiles[0].id == "test_voice"

    def test_get_profile(self, tmp_path, sample_profile):
        """add then get by ID returns matching profile."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        registry.add(sample_profile)

        result = registry.get("test_voice")
        assert result is not None
        assert result.id == "test_voice"
        assert result.provider == "qwen3-tts"

    def test_get_missing_profile(self, tmp_path):
        """get("nonexistent") returns None."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        assert registry.get("nonexistent") is None

    def test_remove_profile(self, tmp_path, sample_profile):
        """remove returns the profile, list() is now empty."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        registry.add(sample_profile)

        removed = registry.remove("test_voice")
        assert removed is not None
        assert removed.id == "test_voice"
        assert registry.list() == []

    def test_remove_missing_profile(self, tmp_path):
        """remove("nonexistent") returns None."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        assert registry.remove("nonexistent") is None

    def test_persistence(self, tmp_path, sample_profile):
        """add profile, create new registry from same file, profile is present."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        registry = VoiceProfileRegistry(str(registry_file))
        registry.add(sample_profile)

        # Load a fresh registry from the same file
        fresh_registry = VoiceProfileRegistry(str(registry_file))
        profiles = fresh_registry.list()
        assert len(profiles) == 1
        assert profiles[0].id == "test_voice"

    def test_invalid_json_warns(self, tmp_path, capsys):
        """load "not json" file, list() returns empty list (no crash)."""
        registry_file = tmp_path / "bad_registry.json"
        registry_file.write_text("not json")

        registry = VoiceProfileRegistry(str(registry_file))
        assert registry.list() == []

        # Verify a warning was printed to stderr
        captured = capsys.readouterr()
        assert "[WARN]" in captured.err

    def test_missing_file_starts_empty(self, tmp_path):
        """load nonexistent file path, list() returns empty list."""
        registry_file = tmp_path / "nonexistent_registry.json"

        registry = VoiceProfileRegistry(str(registry_file))
        assert registry.list() == []


# ---------------------------------------------------------------------------
# TestVoiceProfileFactory
# ---------------------------------------------------------------------------


class TestVoiceProfileFactory:
    def test_create_sets_created_at(self, tmp_path):
        """Registry.create() returns profile with created_at in ISO format (contains 'T')."""
        registry_file = tmp_path / "registry.json"
        registry_file.write_text("{}")

        profile = VoiceProfileRegistry.create(
            profile_id="factory_voice",
            provider="qwen3-tts",
            mode="voice_clone",
            model="qwen-0.6b-base",
            ref_audio="voices/ref/factory.wav",
            ref_text="Factory created voice.",
            language="English",
        )

        assert "T" in profile.created_at
