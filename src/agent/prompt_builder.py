"""Constructs system prompts for the LLM meditation agent from reference JSON files."""

from __future__ import annotations

from src.types import MeditationReference


def build_system_prompt(reference: MeditationReference, duration_minutes: int) -> str:
    """Generate a system prompt that instructs the LLM to produce meditation scripts
    matching the reference style, pacing, tone, and trajectory.

    Args:
        reference: The MeditationReference dataclass with style configuration.
        duration_minutes: Target duration for the generated meditation in minutes.

    Returns:
        A ~500-800 token system prompt string.
    """
    wpm = reference.pacing.avg_speaking_rate_wpm
    tone = reference.tone
    pacing = reference.pacing
    lang = reference.language
    traj = reference.trajectory
    binaural = reference.binaural

    phrases = reference.language.common_phrases[:6]
    phrases_str = ", ".join(f'"{p}"' for p in phrases) if phrases else "None defined"

    patterns = lang.structural_patterns[:5]
    patterns_str = (
        "\n".join(f"  - {p}" for p in patterns)
        if patterns
        else "  Follow natural meditation flow"
    )

    act_str = ""
    if pacing.act_boundaries:
        boundaries = ", ".join(str(b) for b in pacing.act_boundaries)
        act_str = f"\n- Act boundaries at phrase indices: {boundaries}"

    return f"""You are a meditation guide specializing in {reference.name} exercises from the {reference.collection} collection.

## YOUR VOICE
{tone.description}
Energy: {tone.energy} | Warmth: {tone.warmth} | Formality: {tone.formality}

## PACING
- Speak at approximately {wpm:.0f} WPM
- After instructions: pause {pacing.instruction_pause_seconds:.1f} seconds
- After body scan segments: pause {pacing.body_scan_pause_seconds:.1f} seconds
- During countdowns: pause {pacing.countdown_pause_seconds:.1f} seconds between numbers{act_str}

## LANGUAGE STYLE
{lang.sentence_style}
Use phrases like: {phrases_str}
Perspective: {lang.perspective}
Average sentence length: {lang.avg_sentence_length_words} words
Repetition rate: {lang.repetition_rate:.0%} — repeat key phrases for deepening effect

Structural patterns:
{patterns_str}

## SESSION STRUCTURE
Opening: {traj.opening}
Deepening: {traj.deepening}
Transitions: {traj.transitions}

## BINAURAL BEATS
{binaural.description}
Carrier frequency: {binaural.carrier_freq_hz:.0f} Hz | Beat frequency: {binaural.beat_freq_hz:.0f} Hz
Write the script with awareness that these frequencies are playing underneath — do not reference them aloud.

## OUTPUT FORMAT
Write the meditation script as plain text. Use natural paragraph breaks to indicate where pauses should occur. Each paragraph represents one TTS chunk. Do NOT use markdown, headers, bullet points, or special formatting. Do NOT include timing markers like [pause 3s] or (breathe). Do NOT number paragraphs.

## DEVIATION HANDLING
{traj.deviation_handling}

## CONSTRAINTS
- Never mention you are an AI, assistant, or language model
- Never use meta-language like "now we will move to" or "next we'll explore"
- Stay in the meditation frame at all times
- Never break the instructor persona
- Do not acknowledge the user by name unless the context demands it
- Do not explain what you are doing — just guide
- Generate approximately {duration_minutes} minutes of content at {wpm:.0f} WPM (roughly {int(wpm * duration_minutes)} words total)"""


def build_deviation_prompt(
    reference: MeditationReference,
    spoken_context: list[str],
    user_request: str,
) -> str:
    """Build the user-facing prompt when the user requests a deviation mid-session.

    Includes recent spoken context so the LLM can smoothly transition from where
    it left off to the new request while maintaining the reference's style.

    Args:
        reference: The MeditationReference for style matching.
        spoken_context: The last several chunks already spoken (typically 3-5).
        user_request: What the user is asking for now.

    Returns:
        A prompt string for the LLM to generate a continuation that incorporates
        the deviation.
    """
    context_str = (
        "\n".join(spoken_context[-5:]) if spoken_context else "(no prior context)"
    )

    return f"""The user has requested a change mid-session.

## RECENT SPOKEN CONTEXT
{context_str}

## USER REQUEST
{user_request}

## YOUR TASK
Generate the next portion of the meditation that smoothly transitions from the recent context above into fulfilling the user's request. Maintain the {reference.name} style — {reference.tone.description}. Do not acknowledge the deviation explicitly; simply flow into it naturally as a skilled meditation guide would.

Write plain text with natural paragraph breaks. No markdown, no timing markers."""


def build_continuation_prompt(
    reference: MeditationReference,
    spoken_context: list[str],
) -> str:
    """Build the prompt for generating the next chunk when the buffer runs low.

    This is NOT a deviation — the user just needs more content in the same
    trajectory. The LLM should continue in the established style.

    Args:
        reference: The MeditationReference for style matching.
        spoken_context: The last several chunks already spoken (typically 3-5).

    Returns:
        A prompt string for the LLM to generate the next portion of meditation.
    """
    context_str = (
        "\n".join(spoken_context[-5:]) if spoken_context else "(no prior context)"
    )

    return f"""Continue the meditation in the same style and trajectory.

## RECENT SPOKEN CONTEXT
{context_str}

## YOUR TASK
Generate the next portion of the {reference.name} meditation, flowing naturally from the recent context above. Maintain the established pace, tone ({reference.tone.description}), and structure. If approaching a natural section boundary (deepening, transition, closing), follow the reference trajectory:
- Opening: {reference.trajectory.opening}
- Deepening: {reference.trajectory.deepening}
- Transitions: {reference.trajectory.transitions}

Write plain text with natural paragraph breaks. No markdown, no timing markers."""
