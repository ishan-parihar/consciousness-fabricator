"""Tests for src/audio/binaural.py — binaural beat generator."""

import numpy as np
import pytest

from src.audio.binaural import (
    BRAINWAVE_PRESETS,
    generate_binaural,
    generate_binaural_with_preset,
    generate_and_save,
    save_binaural,
)


# ---------------------------------------------------------------------------
# TestBrainwavePresets
# ---------------------------------------------------------------------------


class TestBrainwavePresets:
    def test_all_presets_exist(self):
        expected = {"delta", "theta", "alpha", "beta", "gamma"}
        assert expected.issubset(set(BRAINWAVE_PRESETS.keys()))

    def test_preset_has_required_fields(self):
        for name, preset in BRAINWAVE_PRESETS.items():
            assert isinstance(preset["beat_freq"], float), (
                f"{name}: beat_freq must be float"
            )
            assert isinstance(preset["description"], str), (
                f"{name}: description must be str"
            )


# ---------------------------------------------------------------------------
# TestGenerateBinaural
# ---------------------------------------------------------------------------


class TestGenerateBinaural:
    def test_output_shape_is_stereo(self):
        audio = generate_binaural(120.0, 10.0, 1.0)
        assert audio.shape[1] == 2

    def test_output_duration(self):
        audio = generate_binaural(120.0, 10.0, 1.0, sample_rate=44100)
        assert audio.shape[0] == 44100

    def test_output_dtype(self):
        audio = generate_binaural(120.0, 10.0, 1.0)
        assert audio.dtype == np.float32

    def test_amplitude_range(self):
        audio = generate_binaural(120.0, 10.0, 1.0)
        assert audio.max() <= 0.5
        assert audio.min() >= -0.5

    def test_different_frequencies(self):
        audio1 = generate_binaural(120.0, 10.0, 1.0)
        audio2 = generate_binaural(120.0, 20.0, 1.0)
        assert audio1.shape == audio2.shape

    def test_zero_duration(self):
        audio = generate_binaural(120.0, 10.0, 0.0)
        assert audio.shape == (0, 2)


# ---------------------------------------------------------------------------
# TestGenerateWithPreset
# ---------------------------------------------------------------------------


class TestGenerateWithPreset:
    def test_alpha_preset(self):
        audio = generate_binaural_with_preset("alpha", 120.0, 1.0)
        assert audio.shape[1] == 2

    def test_theta_preset(self):
        audio = generate_binaural_with_preset("theta", 120.0, 1.0)
        assert audio.shape[1] == 2

    def test_invalid_preset_raises(self):
        with pytest.raises(KeyError):
            generate_binaural_with_preset("invalid", 120.0, 1.0)

    def test_all_presets_generate(self):
        for name in BRAINWAVE_PRESETS:
            audio = generate_binaural_with_preset(name, 120.0, 0.5, sample_rate=44100)
            assert audio.shape == (22050, 2), f"{name}: expected (22050, 2)"


# ---------------------------------------------------------------------------
# TestSaveBinaural
# ---------------------------------------------------------------------------


class TestSaveBinaural:
    def test_save_as_wav(self, tmp_path):
        audio = generate_binaural(120.0, 10.0, 0.5)
        output = tmp_path / "test.wav"
        save_binaural(audio, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_save_creates_parent_dirs(self, tmp_path):
        audio = generate_binaural(120.0, 10.0, 0.5)
        output = tmp_path / "sub" / "dir" / "test.wav"
        save_binaural(audio, output)
        assert output.exists()
        assert output.stat().st_size > 0


# ---------------------------------------------------------------------------
# TestGenerateAndSave
# ---------------------------------------------------------------------------


class TestGenerateAndSave:
    def test_full_pipeline(self, tmp_path):
        output = tmp_path / "output.wav"
        generate_and_save("alpha", 120.0, 0.5, output)
        assert output.exists()
        assert output.stat().st_size > 0
