#!/usr/bin/env python3
"""
Extract optimal voice cloning reference clips from meditation audio.

Uses SRT phrase timing data to identify 30-60 second segments with:
- High speech density (minimal gaps between phrases)
- Continuous speech flow (no long pauses > 3s)
- Clean voice quality (prefers Silva Method for calm instructor)

Outputs:
- WAV files in voices/ref/ (24kHz, 16-bit, mono)
- Matching reference text for each clip
- Quality report with gap analysis
"""

import re
import sys
import json
import os
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple, Optional

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
SRT_DIR = PROJECT_ROOT / "meditation-repo" / "srt-phrase"
AUDIO_DIRS = [
    PROJECT_ROOT
    / "meditation-repo"
    / "collections"
    / "silva-method-exercises"
    / "audio",
    PROJECT_ROOT
    / "meditation-repo"
    / "collections"
    / "advancing-witches-craft"
    / "audio",
]
OUTPUT_DIR = PROJECT_ROOT / "voices" / "ref"

# Extraction parameters
MIN_CLIP_DURATION = 30.0  # seconds
MAX_CLIP_DURATION = 60.0  # seconds
MAX_GAP_THRESHOLD = 3.0  # seconds - gaps longer than this break continuity
TARGET_GAP_DENSITY = 0.3  # ratio of gap time to total clip time (lower = denser speech)
MIN_PHRASES_PER_CLIP = 5  # minimum phrases in a clip for voice cloning quality


@dataclass
class SRTEntry:
    """Single phrase from SRT file."""

    index: int
    start_sec: float
    end_sec: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def gap_after(self) -> float:
        """Gap between this phrase end and next phrase start (set later)."""
        return getattr(self, "_gap_after", 0.0)


@dataclass
class ClipCandidate:
    """A potential voice cloning reference clip."""

    start_sec: float
    end_sec: float
    entries: List[SRTEntry]
    total_speech_sec: float
    total_gap_sec: float
    gap_density: float
    phrase_count: int
    avg_words_per_sec: float
    text: str

    @property
    def duration(self) -> float:
        return self.end_sec - self.start_sec

    @property
    def quality_score(self) -> float:
        """Higher is better. Combines speech density, phrase count, and duration fit."""
        # Prefer clips with speech density > 70%
        density_score = max(0, 1.0 - self.gap_density) * 40
        # Prefer 5-15 phrases (enough variety, not too long)
        phrase_score = min(30, self.phrase_count * 3)
        # Prefer clips in 35-50 second range (sweet spot)
        duration_fit = 1.0 - abs(self.duration - 42.5) / 30.0
        duration_score = max(0, duration_fit) * 30
        return density_score + phrase_score + duration_score


def parse_timestamp(ts: str) -> float:
    """Convert SRT timestamp to seconds."""
    match = re.match(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})", ts.strip())
    if not match:
        return 0.0
    h, m, s, ms = match.groups()
    return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0


def parse_srt(srt_path: Path) -> List[SRTEntry]:
    """Parse SRT phrase file into entries."""
    content = srt_path.read_text(encoding="utf-8")
    entries = []

    # Split by double newlines (SRT block separator)
    blocks = re.split(r"\n\s*\n", content.strip())

    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 3:
            continue

        # First line: index
        try:
            index = int(lines[0].strip())
        except ValueError:
            continue

        # Second line: timestamps
        ts_match = re.match(
            r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})",
            lines[1].strip(),
        )
        if not ts_match:
            continue

        start_sec = parse_timestamp(ts_match.group(1))
        end_sec = parse_timestamp(ts_match.group(2))

        # Remaining lines: text
        text = " ".join(line.strip() for line in lines[2:] if line.strip())

        entries.append(
            SRTEntry(index=index, start_sec=start_sec, end_sec=end_sec, text=text)
        )

    # Calculate gaps between consecutive entries
    for i in range(len(entries) - 1):
        entries[i]._gap_after = entries[i + 1].start_sec - entries[i].end_sec

    return entries


def find_best_clips(
    entries: List[SRTEntry], target_duration: float = 45.0
) -> List[ClipCandidate]:
    """
    Find the best 30-60 second clips from SRT entries.

    Strategy: Use a sliding window that grows until it hits MAX_CLIP_DURATION
    or the gap density becomes too high. Scores each candidate and returns top ones.
    """
    if not entries:
        return []

    candidates = []

    for start_idx in range(len(entries)):
        current_entries = []
        current_speech = 0.0
        current_gap = 0.0
        current_end = entries[start_idx].start_sec

        for end_idx in range(start_idx, len(entries)):
            entry = entries[end_idx]
            current_entries.append(entry)
            current_speech += entry.duration
            current_end = entry.end_sec

            # Add gap after this entry (if not last)
            if end_idx < len(entries) - 1:
                gap = entry.gap_after
                current_gap += gap

            total_duration = current_end - entries[start_idx].start_sec

            # Skip if too short
            if total_duration < MIN_CLIP_DURATION:
                continue

            # Check if we've exceeded max duration
            if total_duration > MAX_CLIP_DURATION:
                # Remove last entry and finalize this candidate
                if len(current_entries) > 1:
                    last = current_entries.pop()
                    current_speech -= last.duration
                    if current_entries:
                        current_gap -= last.gap_after
                        current_end = current_entries[-1].end_sec
                    total_duration = current_end - entries[start_idx].start_sec
                break

            # Check gap density (skip if too gappy)
            if total_duration > 0:
                gap_density = current_gap / total_duration
            else:
                gap_density = 1.0

            if gap_density > TARGET_GAP_DENSITY * 2:
                continue

            # Check minimum phrases
            if len(current_entries) < MIN_PHRASES_PER_CLIP:
                continue

            # Check no single gap is too long
            max_single_gap = max((e.gap_after for e in current_entries[:-1]), default=0)
            if max_single_gap > MAX_GAP_THRESHOLD * 2:
                continue

            # Calculate text stats
            text = " ".join(e.text for e in current_entries)
            word_count = len(text.split())
            avg_wps = word_count / total_duration if total_duration > 0 else 0

            candidate = ClipCandidate(
                start_sec=entries[start_idx].start_sec,
                end_sec=current_end,
                entries=list(current_entries),
                total_speech_sec=current_speech,
                total_gap_sec=current_gap,
                gap_density=gap_density,
                phrase_count=len(current_entries),
                avg_words_per_sec=avg_wps,
                text=text,
            )
            candidates.append(candidate)

    # Sort by quality score and return top candidates
    candidates.sort(key=lambda c: c.quality_score, reverse=True)
    return candidates


def format_time(sec: float) -> str:
    """Format seconds as HH:MM:SS.mmm for ffmpeg."""
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def extract_audio_clip(
    audio_path: Path, start_sec: float, end_sec: float, output_path: Path
) -> bool:
    """Extract audio clip using ffmpeg."""
    duration = end_sec - start_sec

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-ss",
        format_time(start_sec),
        "-t",
        str(duration),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-sample_fmt",
        "s16",
        "-f",
        "wav",
        str(output_path),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ✗ FFmpeg error: {result.stderr[:200]}")
        return False

    return True


def get_audio_duration(audio_path: Path) -> float:
    """Get audio duration using ffprobe."""
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        str(audio_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return 0.0

    try:
        data = json.loads(result.stdout)
        return float(data["format"]["duration"])
    except (json.JSONDecodeError, KeyError):
        return 0.0


def map_srt_to_audio(srt_name: str) -> Optional[Path]:
    """Map SRT filename to audio file path."""
    # Extract base name without extension
    base = Path(srt_name).stem

    # Try each audio directory
    for audio_dir in AUDIO_DIRS:
        if not audio_dir.exists():
            continue

        for audio_file in audio_dir.iterdir():
            if audio_file.suffix.lower() not in (".mp3", ".flac", ".wav", ".m4a"):
                continue

            # Normalize names for comparison
            audio_base = audio_file.stem.lower().replace(" ", "-").replace("_", "-")

            # Direct match
            if base == audio_base:
                return audio_file

            # Partial match (SRT name is substring of audio name)
            if base.replace("-", "") in audio_base.replace("-", ""):
                return audio_file

            # Audio name is substring of SRT name
            if audio_base.replace("-", "") in base.replace("-", ""):
                return audio_file

    return None


def analyze_all_srt_files() -> dict:
    """Analyze all SRT files and return results by archetype."""
    results = {"silva": [], "shadow": [], "all_candidates": []}

    srt_files = sorted(SRT_DIR.glob("*.srt"))

    for srt_file in srt_files:
        entries = parse_srt(srt_file)
        if not entries:
            continue

        # Determine archetype
        is_silva = "silva" in srt_file.name.lower() or any(
            k in srt_file.name.lower()
            for k in [
                "long-relax",
                "illuminate",
                "source-connection",
                "visioning",
                "problem-solving",
                "projection",
                "intuition",
                "plants",
                "healing",
                "fantastic",
                "scanning",
                "laboratory",
                "hollow",
                "nonphysical",
                "exploring-tunnel",
            ]
        )
        archetype = "silva" if is_silva else "shadow"

        # Map to audio file
        audio_path = map_srt_to_audio(srt_file.name)
        if not audio_path:
            print(f"  ⚠ No audio match for {srt_file.name}")
            continue

        audio_duration = get_audio_duration(audio_path)

        # Find best clips
        candidates = find_best_clips(entries)

        # Get top 3 candidates
        top_candidates = candidates[:3]

        file_result = {
            "srt_file": srt_file.name,
            "audio_file": audio_path.name,
            "audio_path": audio_path,
            "audio_duration": audio_duration,
            "archetype": archetype,
            "total_phrases": len(entries),
            "total_duration": entries[-1].end_sec - entries[0].start_sec
            if entries
            else 0,
            "avg_wpm": sum(
                len(e.text.split()) / max(e.duration, 0.1) * 60 for e in entries
            )
            / max(len(entries), 1),
            "top_candidates": top_candidates,
        }

        results[archetype].append(file_result)
        results["all_candidates"].extend([(file_result, c) for c in top_candidates])

    return results


def print_analysis_report(results: dict):
    """Print a summary report of all analyzed files."""
    print("\n" + "=" * 80)
    print("VOICE CLIP EXTRACTION ANALYSIS REPORT")
    print("=" * 80)

    for archetype in ["silva", "shadow"]:
        files = results[archetype]
        print(f"\n{'─' * 60}")
        print(f"  {archetype.upper()} METHOD ({len(files)} files)")
        print(f"{'─' * 60}")

        for f in files:
            print(f"\n  📁 {f['audio_file']}")
            print(
                f"     Duration: {f['audio_duration'] / 60:.1f}min | "
                f"Phrases: {f['total_phrases']} | "
                f"Speaking Rate: {f['avg_wpm']:.0f} WPM"
            )

            if f["top_candidates"]:
                print(f"     Top clip candidates:")
                for i, c in enumerate(f["top_candidates"]):
                    print(
                        f"       {i + 1}. [{format_time(c.start_sec)} → {format_time(c.end_sec)}] "
                        f"Duration: {c.duration:.1f}s | "
                        f"Phrases: {c.phrase_count} | "
                        f"Gap Density: {c.gap_density:.1%} | "
                        f"Quality: {c.quality_score:.0f}/100"
                    )
                    print(f"          Text preview: {c.text[:80]}...")


def select_best_clips_for_profiles(
    results: dict,
) -> List[Tuple[str, ClipCandidate, Path, str]]:
    """
    Select the best clips for each voice profile.

    Returns list of (profile_id, clip, audio_path, srt_file_name)
    """
    selections = []

    # For Silva Method (calm_instructor): pick highest quality Silva clip
    silva_files = [f for f in results["silva"] if f["top_candidates"]]
    if silva_files:
        # Sort all Silva candidates by quality
        all_silva = [(f, c) for f in silva_files for c in f["top_candidates"]]
        all_silva.sort(key=lambda x: x[1].quality_score, reverse=True)

        best_file, best_clip = all_silva[0]
        selections.append(
            (
                "calm_instructor",
                best_clip,
                best_file["audio_path"],
                best_file["srt_file"],
            )
        )

    # For Shadow Realm (warm_storyteller): pick highest quality Shadow clip
    shadow_files = [f for f in results["shadow"] if f["top_candidates"]]
    if shadow_files:
        all_shadow = [(f, c) for f in shadow_files for c in f["top_candidates"]]
        all_shadow.sort(key=lambda x: x[1].quality_score, reverse=True)

        best_file, best_clip = all_shadow[0]
        selections.append(
            (
                "warm_storyteller",
                best_clip,
                best_file["audio_path"],
                best_file["srt_file"],
            )
        )

    return selections


def extract_selected_clips(
    selections: List[Tuple[str, ClipCandidate, Path, str]],
) -> List[dict]:
    """Extract the selected clips and save to voices/ref/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    results = []

    for profile_id, clip, audio_path, srt_file in selections:
        output_wav = OUTPUT_DIR / f"{profile_id}_ref.wav"
        text_file = OUTPUT_DIR / f"{profile_id}_ref.txt"

        print(f"\n🎙️  Extracting {profile_id}:")
        print(f"   Source: {audio_path.name}")
        print(
            f"   Segment: {format_time(clip.start_sec)} → {format_time(clip.end_sec)} "
            f"({clip.duration:.1f}s)"
        )
        print(f"   Phrases: {clip.phrase_count}")
        print(f"   Gap density: {clip.gap_density:.1%}")
        print(f"   Output: {output_wav}")

        # Extract audio
        success = extract_audio_clip(
            audio_path, clip.start_sec, clip.end_sec, output_wav
        )
        if not success:
            print(f"   ✗ Failed to extract audio")
            continue

        # Verify output
        if output_wav.exists():
            output_duration = get_audio_duration(output_wav)
            file_size = output_wav.stat().st_size / (1024 * 1024)
            print(f"   ✓ Extracted: {output_duration:.1f}s, {file_size:.1f}MB")

            # Save reference text
            text_file.write_text(clip.text, encoding="utf-8")
            print(f"   ✓ Reference text: {text_file} ({len(clip.text.split())} words)")

            results.append(
                {
                    "profile_id": profile_id,
                    "wav_path": str(output_wav),
                    "text_path": str(text_file),
                    "text": clip.text,
                    "duration": output_duration,
                    "source_audio": audio_path.name,
                    "start_sec": clip.start_sec,
                    "end_sec": clip.end_sec,
                }
            )
        else:
            print(f"   ✗ Output file not created")

    return results


def generate_voicebox_setup_script(results: List[dict]) -> str:
    """Generate a Python script to set up voice-box profiles."""
    script = '''#!/usr/bin/env python3
"""
Auto-generated script to create voice-box profiles from extracted reference clips.

Run this after the voice-box server is running:
    python scripts/setup_voicebox_profiles.py
"""

import httpx
import sys
from pathlib import Path

VOICEBOX_URL = "http://127.0.0.1:17493"

PROFILES = [
'''

    for r in results:
        script += f'''    {{
        "profile_id": "{r["profile_id"]}",
        "name": "{r["profile_id"].replace("_", " ").title()}",
        "description": "Extracted from {r["source_audio"]} ({r["start_sec"]:.0f}s-{r["end_sec"]:.0f}s)",
        "wav_path": "{r["wav_path"]}",
        "text_path": "{r["text_path"]}",
    }},
'''

    script += ''']


async def setup_profiles():
    """Create voice-box profiles and add reference samples."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        for profile in PROFILES:
            print(f"\\n🎙️  Setting up profile: {profile['name']}")

            # Check if profile already exists
            try:
                resp = await client.get(f"{VOICEBOX_URL}/profiles")
                existing = resp.json()
                existing_ids = [p['id'] for p in existing]

                if profile['profile_id'] in existing_ids:
                    print(f"   Profile already exists, skipping creation")
                    profile_id = profile['profile_id']
                else:
                    # Create profile
                    resp = await client.post(f"{VOICEBOX_URL}/profiles", json={{
                        "name": profile['name'],
                        "description": profile['description'],
                        "language": "en"
                    }})
                    resp.raise_for_status()
                    profile_id = resp.json()['id']
                    print(f"   ✓ Created profile: {profile_id}")

            except Exception as e:
                print(f"   ✗ Error creating profile: {{e}}")
                continue

            # Add reference sample
            wav_path = Path(profile['wav_path'])
            text_path = Path(profile['text_path'])

            if not wav_path.exists():
                print(f"   ✗ WAV file not found: {{wav_path}}")
                continue

            if not text_path.exists():
                print(f"   ✗ Text file not found: {{text_path}}")
                continue

            reference_text = text_path.read_text(encoding='utf-8')

            try:
                with open(wav_path, 'rb') as f:
                    files = {{
                        'file': (wav_path.name, f, 'audio/wav')
                    }}
                    data = {{
                        'reference_text': reference_text
                    }}
                    resp = await client.post(
                        f"{VOICEBOX_URL}/profiles/{{profile_id}}/samples",
                        files=files,
                        data=data
                    )
                    resp.raise_for_status()
                    print(f"   ✓ Added reference sample: {{len(reference_text.split())}} words")

            except Exception as e:
                print(f"   ✗ Error adding sample: {{e}}")
                continue

            # Test generation
            try:
                test_text = reference_text[:100] + "..."
                resp = await client.post(f"{VOICEBOX_URL}/generate", json={{
                    "profile_id": profile_id,
                    "text": test_text,
                    "engine": "qwen",
                    "model_size": "1.7B"
                }})
                resp.raise_for_status()
                gen_id = resp.json()['id']
                print(f"   ✓ Test generation started: {{gen_id}}")

            except Exception as e:
                print(f"   ⚠ Test generation failed: {{e}}")

    print("\\n✅ Profile setup complete!")


if __name__ == "__main__":
    import asyncio
    asyncio.run(setup_profiles())
'''
    return script


def main():
    """Main entry point."""
    print("🎙️  Voice Clip Extraction Tool")
    print("=" * 40)

    # Step 1: Analyze all SRT files
    print("\n📊 Analyzing SRT timing data...")
    results = analyze_all_srt_files()

    # Step 2: Print analysis report
    print_analysis_report(results)

    # Step 3: Select best clips for each profile
    print("\n" + "=" * 80)
    print("\n🎯 SELECTING BEST CLIPS FOR VOICE PROFILES")
    print("=" * 80)

    selections = select_best_clips_for_profiles(results)

    if not selections:
        print("\n❌ No suitable clips found. Check SRT files and audio mapping.")
        sys.exit(1)

    for profile_id, clip, audio_path, srt_file in selections:
        print(f"\n  {profile_id}:")
        print(f"    Source: {audio_path.name}")
        print(
            f"    Segment: {format_time(clip.start_sec)} → {format_time(clip.end_sec)}"
        )
        print(f"    Duration: {clip.duration:.1f}s")
        print(f"    Phrases: {clip.phrase_count}")
        print(f"    Text: {clip.text[:100]}...")

    # Step 4: Extract clips
    print("\n" + "=" * 80)
    print("\n🔪 EXTRACTING CLIPS")
    print("=" * 80)

    extraction_results = extract_selected_clips(selections)

    if not extraction_results:
        print("\n❌ No clips extracted successfully.")
        sys.exit(1)

    # Step 5: Generate voice-box setup script
    setup_script = generate_voicebox_setup_script(extraction_results)
    setup_script_path = PROJECT_ROOT / "scripts" / "setup_voicebox_profiles.py"
    setup_script_path.parent.mkdir(parents=True, exist_ok=True)
    setup_script_path.write_text(setup_script, encoding="utf-8")

    print(f"\n✅ Generated voice-box setup script: {setup_script_path}")
    print(f"\n📋 Next steps:")
    print(f"   1. Review extracted clips in {OUTPUT_DIR}/")
    print(f"   2. Run: python {setup_script_path}")
    print(f"   3. Test voice cloning quality")

    # Summary
    print(f"\n{'=' * 80}")
    print("EXTRACTION SUMMARY")
    print(f"{'=' * 80}")
    for r in extraction_results:
        print(f"\n  Profile: {r['profile_id']}")
        print(f"  WAV: {r['wav_path']}")
        print(f"  Text: {r['text_path']}")
        print(f"  Duration: {r['duration']:.1f}s")
        print(f"  Source: {r['source_audio']}")
        print(f"  Segment: {r['start_sec']:.1f}s - {r['end_sec']:.1f}s")


if __name__ == "__main__":
    main()
