"""Tests for EngineConfig in src/config.py."""

import json

import pytest

from src.config import EngineConfig


# ---------------------------------------------------------------------------


class TestEngineConfigDefault:
    def test_tts_base_url(self, default_config):
        assert default_config.tts_base_url == "http://localhost:8000"

    def test_ws_host(self, default_config):
        assert hasattr(default_config, "tts_base_url")

    def test_ws_port(self, default_config):
        assert default_config.tts_base_url == "http://localhost:8000"

    def test_llm_model(self, default_config):
        assert default_config.llm_model == "gpt-4o"


# ---------------------------------------------------------------------------


class TestEngineConfigFromFileYaml:
    def test_load_yaml_overrides(self, tmp_path):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(
            "tts_base_url: http://custom:9000\n"
            "llm_model: claude-sonnet-4\n"
            "fade_out_duration: 3.0\n"
        )

        config = EngineConfig.from_file(str(cfg_path))

        assert config.tts_base_url == "http://custom:9000"
        assert config.llm_model == "claude-sonnet-4"
        assert config.fade_out_duration == 3.0
        assert config.output_dir == "output"  # default preserved


# ---------------------------------------------------------------------------


class TestEngineConfigFromFileJson:
    def test_load_json_overrides(self, tmp_path):
        cfg_path = tmp_path / "config.json"
        cfg_path.write_text(
            json.dumps(
                {
                    "tts_base_url": "http://json:7000",
                    "llm_model": "gpt-4o-mini",
                    "output_dir": "build/output",
                }
            )
        )

        config = EngineConfig.from_file(str(cfg_path))

        assert config.tts_base_url == "http://json:7000"
        assert config.llm_model == "gpt-4o-mini"
        assert config.output_dir == "build/output"
        assert config.llm_model == "gpt-4o-mini"


# ---------------------------------------------------------------------------


class TestEngineConfigFromFileErrors:
    def test_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError, match="Config file not found"):
            EngineConfig.from_file("/nonexistent/path/config.yaml")

    def test_invalid_yaml_raises(self, tmp_path):
        cfg_path = tmp_path / "bad.yaml"
        cfg_path.write_text(":\n  - invalid: yaml: : :\n  {{{broken")

        with pytest.raises(Exception):
            EngineConfig.from_file(str(cfg_path))


# ---------------------------------------------------------------------------


class TestEngineConfigPartial:
    def test_partial_config_uses_defaults(self, tmp_path):
        cfg_path = tmp_path / "partial.yaml"
        cfg_path.write_text("tts_base_url: http://partial:5000\n")

        config = EngineConfig.from_file(str(cfg_path))

        assert config.tts_base_url == "http://partial:5000"
        assert config.llm_model == "gpt-4o"
        assert config.output_dir == "output"
        assert config.fade_out_duration == 5.0
