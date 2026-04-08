"""Shared dataclasses and enums for the meditation engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class MeditationStyle(str, Enum):
    """Meditation style archetypes."""

    SILVA_METHOD = "silva-method"
    SHADOW_REALM = "shadow-realm"


class Brainwave(str, Enum):
    """Brainwave types with frequency ranges."""

    DELTA = "delta"  # 0.5-4 Hz
    THETA = "theta"  # 4-8 Hz
    ALPHA = "alpha"  # 8-12 Hz
    BETA = "beta"  # 12-30 Hz
    GAMMA = "gamma"  # 30-100 Hz


@dataclass
class BinauralConfig:
    """Binaural beat configuration."""

    brainwave: Brainwave
    carrier_freq_hz: float = 120.0
    beat_freq_hz: float = 10.0

    @property
    def description(self) -> str:
        """Human-readable description per brainwave type."""
        descriptions: dict[Brainwave, str] = {
            Brainwave.DELTA: "Deep sleep and healing (0.5-4 Hz)",
            Brainwave.THETA: "Deep meditation and creativity (4-8 Hz)",
            Brainwave.ALPHA: "Relaxed awareness and calm (8-12 Hz)",
            Brainwave.BETA: "Active thinking and focus (12-30 Hz)",
            Brainwave.GAMMA: "Heightened perception (30-100 Hz)",
        }
        return descriptions.get(self.brainwave, f"{self.brainwave.value} waves")


@dataclass
class PacingConfig:
    """Pacing configuration for meditation delivery."""

    avg_speaking_rate_wpm: float = 98.0
    instruction_pause_seconds: float = 4.2
    body_scan_pause_seconds: float = 8.5
    countdown_pause_seconds: float = 2.1
    act_boundaries: list[int] = field(default_factory=list)


@dataclass
class ToneConfig:
    """Tone configuration for meditation voice."""

    energy: str  # "low", "very_low", "medium"
    warmth: str  # "warm", "neutral", "cool"
    formality: str  # "instructor", "intimate", "conversational"
    description: str


@dataclass
class LanguageConfig:
    """Language and style configuration."""

    sentence_style: str
    perspective: str
    common_phrases: list[str] = field(default_factory=list)
    structural_patterns: list[str] = field(default_factory=list)
    repetition_rate: float = 0.35
    avg_sentence_length_words: int = 12


@dataclass
class TrajectoryConfig:
    """Session arc configuration."""

    opening: str
    deepening: str
    transitions: str
    deviation_handling: str


@dataclass
class MeditationReference:
    """Reference instruction, loaded from JSON."""

    id: str
    name: str
    collection: str
    category: str
    total_duration_seconds: int
    total_phrases: int
    tone: ToneConfig
    pacing: PacingConfig
    language: LanguageConfig
    trajectory: TrajectoryConfig
    binaural: BinauralConfig


@dataclass
class SessionRequest:
    """Session request provided by the user."""

    style: MeditationStyle
    duration_minutes: int
    user_request: str  # e.g., "relaxation", "problem solving", "do a body scan"
    voice_profile_id: str | None = None
    deviation: bool = False


@dataclass
class SessionState:
    """Runtime session state."""

    session_id: str  # UUID
    request: SessionRequest
    reference: MeditationReference
    playhead: int = 0  # Index in text buffer
    total_chunks: int = 0
    spoken_chunks: list[str] = field(default_factory=list)
    is_playing: bool = False
    is_deviation: bool = False
