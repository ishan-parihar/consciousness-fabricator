"""Tests for TTS HTTP client (src/tts/client.py)."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from src.tts.client import TtsClient, TtsResult, TtsError


# ---------------------------------------------------------------------------
# TestTtsClientCacheKey
# ---------------------------------------------------------------------------


class TestTtsClientCacheKey:
    """Tests for TtsClient._cache_key static method."""

    def test_same_inputs_same_key(self):
        key1 = TtsClient._cache_key("voice_1", "Hello world", 1.0, 0.0, 1.0)
        key2 = TtsClient._cache_key("voice_1", "Hello world", 1.0, 0.0, 1.0)
        assert key1 == key2

    def test_different_text_different_key(self):
        key1 = TtsClient._cache_key("voice_1", "Hello world", 1.0, 0.0, 1.0)
        key2 = TtsClient._cache_key("voice_1", "Goodbye world", 1.0, 0.0, 1.0)
        assert key1 != key2

    def test_different_speed_different_key(self):
        key1 = TtsClient._cache_key("voice_1", "Hello world", 1.0, 0.0, 1.0)
        key2 = TtsClient._cache_key("voice_1", "Hello world", 2.0, 0.0, 1.0)
        assert key1 != key2

    def test_key_is_16_chars(self):
        key = TtsClient._cache_key("voice_1", "Hello world", 1.0, 0.0, 1.0)
        assert len(key) == 16


# ---------------------------------------------------------------------------
# TestTtsClientEstimateDuration
# ---------------------------------------------------------------------------


class TestTtsClientEstimateDuration:
    """Tests for TtsClient.estimate_duration static method."""

    def test_basic_estimation(self):
        # 10 words at speed 1.0 (base rate 2.5 w/s) = 4000ms
        result = TtsClient.estimate_duration(
            "one two three four five six seven eight nine ten", 1.0
        )
        assert result == 4000

    def test_speed_adjustment(self):
        # 10 words at speed 2.0 (5.0 w/s) = 2000ms
        result = TtsClient.estimate_duration(
            "one two three four five six seven eight nine ten", 2.0
        )
        assert result == 2000

    def test_empty_text(self):
        result = TtsClient.estimate_duration("", 1.0)
        assert result == 0


# ---------------------------------------------------------------------------
# TestTtsClientHealthCheck
# ---------------------------------------------------------------------------


class TestTtsClientHealthCheck:
    """Tests for TtsClient.health_check async method."""

    @pytest.mark.asyncio
    async def test_health_check_success(self):
        mock_response = MagicMock()
        mock_response.is_success = True

        mock_client = AsyncMock()
        mock_client.get.return_value = mock_response

        with patch("src.tts.client.httpx.AsyncClient", return_value=mock_client):
            client = TtsClient(base_url="http://localhost:8080", cache_dir="/tmp/cache")
            result = await client.health_check()
            assert result is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self):
        import httpx

        mock_client = AsyncMock()
        mock_client.get.side_effect = httpx.RequestError("Connection refused")

        with patch("src.tts.client.httpx.AsyncClient", return_value=mock_client):
            client = TtsClient(base_url="http://localhost:8080", cache_dir="/tmp/cache")
            result = await client.health_check()
            assert result is False


# ---------------------------------------------------------------------------
# TestTtsErrors
# ---------------------------------------------------------------------------


class TestTtsErrors:
    """Tests for TtsError factory methods."""

    def test_http_error(self):
        err = TtsError.http(500, "Internal Server Error")
        assert "HTTP 500" in str(err)
        assert "Internal Server Error" in str(err)

    def test_io_error(self):
        err = TtsError.io("File not found")
        assert "IO error" in str(err)
        assert "File not found" in str(err)

    def test_sidecar_error(self):
        err = TtsError.sidecar("Missing audio_b64")
        assert "TTS sidecar error" in str(err)
        assert "Missing audio_b64" in str(err)
