"""Qwen3-TTS HTTP client with voice cloning support and SHA256-based caching.

Uses ``httpx`` for async HTTP communication with the TTS sidecar server.
"""

from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .profiles import VoiceProfile


@dataclass
class TtsResult:
    """Result of a successful TTS generation call."""

    output_path: str
    duration_ms: int
    cached: bool


class TtsError(Exception):
    """Exception raised by the TTS client.

    Subtypes:
        Http -- HTTP request failure (carries status code and body when available).
        Io   -- Filesystem / I/O failure.
        Sidecar -- Logical error from the TTS sidecar (bad response, missing ref audio, etc.).
    """

    @classmethod
    def http(cls, status_code: int, body: str) -> "TtsError":
        return cls(f"HTTP {status_code}: {body}")

    @classmethod
    def io(cls, message: str) -> "TtsError":
        return cls(f"IO error: {message}")

    @classmethod
    def sidecar(cls, message: str) -> "TtsError":
        return cls(f"TTS sidecar error: {message}")


class TtsClient:
    """Async client for Qwen3-TTS with voice cloning and disk caching.

    Parameters:
        base_url: Base URL of the TTS sidecar (e.g. ``"http://localhost:8080"``).
        cache_dir: Directory path for SHA256-based audio cache.
    """

    def __init__(self, base_url: str, cache_dir: str) -> None:
        self._base_url = base_url.rstrip("/")
        self._cache_dir = Path(cache_dir)
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def health_check(self) -> bool:
        """Check if the TTS sidecar server is reachable.

        Tries ``/health`` first, falls back to the root endpoint.
        """
        try:
            resp = await self._client.get(f"{self._base_url}/health")
            if resp.is_success:
                return True
        except httpx.RequestError:
            pass

        try:
            resp = await self._client.get(self._base_url)
            return resp.is_success
        except httpx.RequestError:
            return False

    async def generate(
        self,
        voice_profile_id: str,
        text: str,
        output_path: str,
        speed: float = 1.0,
        pitch: float = 0.0,
        volume: float = 1.0,
        format: str = "wav",
        voice_profile: VoiceProfile | None = None,
    ) -> TtsResult:
        """Generate speech from text using voice cloning.

        Checks the disk cache first; on a miss, POSTs to ``/generate`` with
        multipart form data, decodes the base64 audio response, and writes
        the WAV to both *output_path* and the cache directory.
        """
        key = self._cache_key(voice_profile_id, text, speed, pitch, volume)

        cached = self._get_cached_path(key, format)
        if cached is not None:
            output = Path(output_path)
            output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cached, output)
            duration_ms = self.extract_audio_duration(cached) or 0
            return TtsResult(
                output_path=output_path,
                duration_ms=duration_ms,
                cached=True,
            )

        if voice_profile is None:
            raise TtsError.sidecar("voice_profile is required for cache miss")

        ref_path = Path(voice_profile.ref_audio)
        if not ref_path.exists():
            raise TtsError.sidecar(
                f"Cannot read ref audio {voice_profile.ref_audio}: file not found"
            )
        ref_bytes = ref_path.read_bytes()

        multipart_data: dict[str, Any] = {
            "text": text,
            "ref_text": voice_profile.ref_text,
            "language": voice_profile.language.lower(),
            "mode": "voice_clone",
            "xvec_only": "true",
            "non_streaming_mode": "true",
        }

        files = {
            "ref_audio": ("ref.wav", ref_bytes, "audio/wav"),
        }

        resp = await self._client.post(
            f"{self._base_url}/generate",
            data=multipart_data,
            files=files,
        )

        if not resp.is_success:
            raise TtsError.http(resp.status_code, resp.text)

        json_resp: dict[str, Any] = resp.json()

        error = json_resp.get("error")
        if error is not None:
            raise TtsError.sidecar(str(error))

        audio_b64 = json_resp.get("audio_b64")
        if audio_b64 is None:
            raise TtsError.sidecar("TTS response missing audio_b64 field")

        audio_bytes = base64.b64decode(audio_b64)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(audio_bytes)

        cache_path = self._cache_dir / f"{key}.wav"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(audio_bytes)

        duration_ms_raw = json_resp.get("duration_ms")
        if duration_ms_raw is not None:
            duration_ms = int(duration_ms_raw)
        else:
            duration_ms = self.extract_audio_duration(output) or 0

        return TtsResult(
            output_path=output_path,
            duration_ms=duration_ms,
            cached=False,
        )

    @staticmethod
    def extract_audio_duration(path: Path | str) -> int | None:
        """Extract audio duration in **milliseconds** from a file via ffprobe.

        Returns ``None`` when ffprobe fails or the path is invalid.
        """
        try:
            result = subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    str(path),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                return None
            dur_str = result.stdout.strip()
            if not dur_str:
                return None
            return int(float(dur_str) * 1000)
        except (subprocess.TimeoutExpired, FileNotFoundError, ValueError, OSError):
            return None

    @staticmethod
    def estimate_duration(text: str, speed: float) -> int:
        """Estimate speech duration in milliseconds based on word count and speed.

        Uses a base rate of 2.5 words/second, adjusted by *speed*.
        """
        words = len(text.split())
        base_rate = 2.5
        adjusted = base_rate * speed
        if adjusted <= 0.0:
            return 0
        return int((words / adjusted) * 1000)

    @staticmethod
    def _cache_key(
        voice_id: str, text: str, speed: float, pitch: float, volume: float
    ) -> str:
        """SHA256-based cache key: first 16 hex characters of ``voice_id|text|speed|pitch|volume``."""
        raw = f"{voice_id}|{text}|{speed}|{pitch}|{volume}"
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return digest[:16]

    def _get_cached_path(self, key: str, format: str) -> Path | None:
        """Return the cache file path if it exists, else ``None``."""
        path = self._cache_dir / f"{key}.{format}"
        if path.exists():
            return path
        return None
