"""Tests for src/agent/prompt_builder.py."""

import pytest
from src.agent.prompt_builder import (
    build_system_prompt,
    build_deviation_prompt,
    build_continuation_prompt,
)


# ---------------------------------------------------------------------------
# TestBuildSystemPrompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    def test_includes_tone_info(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "moderate" in prompt
        assert "neutral" in prompt
        assert "guided" in prompt

    def test_includes_pacing_info(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "98" in prompt
        assert "4.2" in prompt

    def test_includes_language_info(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "descriptive-narrative" in prompt
        assert "second-person-guided" in prompt

    def test_includes_binaural_info(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "120" in prompt
        assert "10" in prompt

    def test_includes_trajectory_info(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "Gentle induction" in prompt
        assert "Progressive relaxation" in prompt

    def test_duration_based_word_estimate(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=15)
        assert "1470" in prompt

    def test_no_markdown_formatting(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "Do NOT use markdown" in prompt

    def test_deviation_handling_included(self, sample_reference):
        prompt = build_system_prompt(sample_reference, duration_minutes=10)
        assert "Flow naturally" in prompt


# ---------------------------------------------------------------------------
# TestBuildDeviationPrompt
# ---------------------------------------------------------------------------


class TestBuildDeviationPrompt:
    def test_includes_spoken_context(self, sample_reference):
        context = ["chunk one", "chunk two"]
        prompt = build_deviation_prompt(
            sample_reference,
            spoken_context=context,
            user_request="do a body scan",
        )
        assert "chunk one" in prompt
        assert "chunk two" in prompt

    def test_includes_user_request(self, sample_reference):
        prompt = build_deviation_prompt(
            sample_reference,
            spoken_context=["some context"],
            user_request="do a body scan instead",
        )
        assert "do a body scan instead" in prompt

    def test_empty_spoken_context(self, sample_reference):
        prompt = build_deviation_prompt(
            sample_reference,
            spoken_context=[],
            user_request="change topic",
        )
        assert "(no prior context)" in prompt

    def test_limits_context_to_recent(self, sample_reference):
        context = [
            "first",
            "second",
            "third",
            "fourth",
            "fifth",
            "sixth",
            "seventh",
            "eighth",
            "ninth",
            "tenth",
        ]
        prompt = build_deviation_prompt(
            sample_reference,
            spoken_context=context,
            user_request="continue",
        )
        for chunk in ["sixth", "seventh", "eighth", "ninth", "tenth"]:
            assert chunk in prompt
        for chunk in ["first", "second", "third", "fourth", "fifth"]:
            assert chunk not in prompt


# ---------------------------------------------------------------------------
# TestBuildContinuationPrompt
# ---------------------------------------------------------------------------


class TestBuildContinuationPrompt:
    def test_includes_spoken_context(self, sample_reference):
        context = ["previous chunk", "last chunk"]
        prompt = build_continuation_prompt(sample_reference, spoken_context=context)
        assert "previous chunk" in prompt
        assert "last chunk" in prompt

    def test_references_trajectory(self, sample_reference):
        prompt = build_continuation_prompt(
            sample_reference,
            spoken_context=["some context"],
        )
        assert "Gentle induction with body relaxation" in prompt
        assert "Progressive relaxation through numbered phases" in prompt
