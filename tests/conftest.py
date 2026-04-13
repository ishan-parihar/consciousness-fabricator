"""Shared pytest fixtures for consciousness-fabricator tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from src.types import (
    MeditationStyle,
    Brainwave,
    BinauralConfig,
    PacingConfig,
    ToneConfig,
    LanguageConfig,
    TrajectoryConfig,
    MeditationReference,
    SessionRequest,
    SessionState,
)
from src.config import EngineConfig


# ---------------------------------------------------------------------------
# Config fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_tone():
    """Minimal ToneConfig fixture."""
    return ToneConfig(
        energy="moderate",
        warmth="neutral",
        formality="guided",
        description="Balanced meditation guidance",
    )


@pytest.fixture
def sample_pacing():
    """PacingConfig matching typical Silva Method meditation."""
    return PacingConfig(
        avg_speaking_rate_wpm=98.0,
        instruction_pause_seconds=4.2,
        body_scan_pause_seconds=8.5,
        countdown_pause_seconds=2.1,
        act_boundaries=[],
    )


@pytest.fixture
def sample_language():
    """LanguageConfig for descriptive meditation style."""
    return LanguageConfig(
        sentence_style="descriptive-narrative",
        perspective="second-person-guided",
        common_phrases=["take a deep breath", "relax your body", "go deeper"],
        structural_patterns=["deepening progression", "relaxation cycles"],
        repetition_rate=0.35,
        avg_sentence_length_words=12,
    )


@pytest.fixture
def sample_trajectory():
    """TrajectoryConfig for standard meditation flow."""
    return TrajectoryConfig(
        opening="Gentle induction with body relaxation",
        deepening="Progressive relaxation through numbered phases",
        transitions="Smooth transitions between meditation states",
        deviation_handling="Flow naturally without acknowledging the change",
    )


@pytest.fixture
def sample_binaural():
    """BinauralConfig for alpha brainwave state."""
    return BinauralConfig(
        brainwave=Brainwave.ALPHA,
        carrier_freq_hz=120.0,
        beat_freq_hz=10.0,
    )


# ---------------------------------------------------------------------------
# Composite fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_reference(
    sample_tone, sample_pacing, sample_language, sample_trajectory, sample_binaural
):
    """Complete MeditationReference for testing."""
    return MeditationReference(
        id="test-meditation",
        name="Test Meditation",
        collection="test-collection",
        category="relaxation",
        total_duration_seconds=600,
        total_phrases=100,
        tone=sample_tone,
        pacing=sample_pacing,
        language=sample_language,
        trajectory=sample_trajectory,
        binaural=sample_binaural,
    )


@pytest.fixture
def sample_session_request():
    """SessionRequest for starting a meditation."""
    return SessionRequest(
        style=MeditationStyle.SILVA_METHOD,
        duration_minutes=10,
        user_request="relaxation",
        voice_profile_id="calm_instructor",
    )


@pytest.fixture
def sample_session_state(sample_reference, sample_session_request):
    """SessionState for an active session."""
    return SessionState(
        session_id="test-session-id",
        request=sample_session_request,
        reference=sample_reference,
    )


@pytest.fixture
def default_config():
    """Default EngineConfig."""
    return EngineConfig.default()


# ---------------------------------------------------------------------------
# Mock fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_llm_client():
    """AsyncMock LLM client implementing LLMClientProtocol."""
    client = AsyncMock()
    client.generate.return_value = "This is a test meditation script. Relax your body and mind. Take a deep breath and let go of tension. Feel yourself going deeper into a state of calm awareness."
    return client


@pytest.fixture
def mock_tts_client():
    """Mock TTS client with realistic responses."""
    from src.tts import TtsResult

    client = AsyncMock()
    client.generate.return_value = TtsResult(
        output_path="/tmp/test_chunk.wav",
        duration_ms=5000,
        cached=False,
    )
    return client


# ---------------------------------------------------------------------------
# Filesystem fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def references_dir(tmp_path):
    """Create a temporary references directory with a sample JSON file."""
    ref_dir = tmp_path / "references" / "test-collection"
    ref_dir.mkdir(parents=True)

    ref_data = {
        "id": "test-meditation",
        "name": "Test Meditation",
        "collection": "test-collection",
        "category": "relaxation",
        "total_duration_seconds": 600,
        "total_phrases": 100,
        "tone": {
            "energy": "moderate",
            "warmth": "neutral",
            "formality": "guided",
            "description": "Balanced meditation guidance",
        },
        "pacing": {
            "avg_speaking_rate_wpm": 98.0,
            "instruction_pause_seconds": 4.2,
            "body_scan_pause_seconds": 8.5,
            "countdown_pause_seconds": 2.1,
            "act_boundaries": [],
            "act_structure": {},
            "gap_distribution": {},
            "avg_gap_seconds": 3.5,
            "median_gap_seconds": 3.0,
            "max_gap_seconds": 12.0,
            "min_gap_seconds": 0.5,
        },
        "language": {
            "sentence_style": "descriptive-narrative",
            "perspective": "second-person-guided",
            "common_phrases": ["take a deep breath", "relax your body"],
            "structural_patterns": ["deepening progression"],
            "repetition_rate": 0.35,
            "avg_sentence_length_words": 12,
            "imperative_ratio": 0.3,
            "total_sentences": 50,
        },
        "trajectory": {
            "opening": "Gentle induction",
            "deepening": "Progressive relaxation",
            "transitions": "Smooth transitions",
            "deviation_handling": "Flow naturally",
        },
        "binaural": {
            "brainwave": "alpha",
            "carrier_freq_hz": 120.0,
            "beat_freq_hz": 10.0,
            "description": "Alpha range for relaxed awareness",
        },
    }

    ref_path = ref_dir / "test-meditation.json"
    ref_path.write_text(json.dumps(ref_data, indent=2))

    return tmp_path / "references"
