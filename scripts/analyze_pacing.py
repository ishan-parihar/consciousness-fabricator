#!/usr/bin/env python3
"""Parse SRT phrase files and TXT transcripts, extract pacing data, generate reference JSON.

Usage:
  python scripts/analyze_pacing.py \
    --input meditation-repo/srt-phrase/02-13-long-relax.srt \
    --txt meditation-repo/txt/02-13-long-relax.txt \
    --output references/

  python scripts/analyze_pacing.py --batch meditation-repo/ --output references/
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median


# ---------------------------------------------------------------------------
# SRT parsing
# ---------------------------------------------------------------------------

_SRT_BLOCK_RE = re.compile(
    r"(\d+)\s*\n"
    r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n"
    r"(.*?)(?=\n\s*\n|\n\s*$|\Z)",
    re.DOTALL,
)

_TS_RE = re.compile(r"(\d{2}):(\d{2}):(\d{2}),(\d{3})")


def _ts_to_ms(ts: str) -> float:
    m = _TS_RE.match(ts)
    if not m:
        return 0.0
    h, mi, s, ms = (int(x) for x in m.groups())
    return h * 3600_000 + mi * 60_000 + s * 1000 + ms


def parse_srt(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        content = f.read()

    phrases = []
    for m in _SRT_BLOCK_RE.finditer(content):
        idx = int(m.group(1))
        start_ms = _ts_to_ms(m.group(2))
        end_ms = _ts_to_ms(m.group(3))
        raw_text = m.group(4).strip()
        text = re.sub(r"\s+", " ", raw_text)
        phrases.append(
            {
                "index": idx,
                "start_ms": start_ms,
                "end_ms": end_ms,
                "start_s": start_ms / 1000,
                "end_s": end_ms / 1000,
                "text": text,
                "word_count": len(text.split()) if text.strip() else 0,
            }
        )
    return phrases


# ---------------------------------------------------------------------------
# TXT reading
# ---------------------------------------------------------------------------


def read_txt(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as f:
        return f.read().strip()


# ---------------------------------------------------------------------------
# Pacing analysis
# ---------------------------------------------------------------------------


def _classify_gap(seconds: float) -> str:
    if seconds <= 2:
        return "short"
    elif seconds <= 5:
        return "medium"
    elif seconds <= 10:
        return "long"
    else:
        return "act"


def compute_pacing(phrases: list[dict]) -> dict:
    if not phrases:
        return {}

    total_duration_s = phrases[-1]["end_s"] - phrases[0]["start_s"]

    speaking_rates = []
    for p in phrases:
        dur_min = (p["end_s"] - p["start_s"]) / 60.0
        if dur_min > 0 and p["word_count"] > 0:
            speaking_rates.append(p["word_count"] / dur_min)

    avg_wpm = round(mean(speaking_rates)) if speaking_rates else 0

    gaps = []
    gap_classifications = {"short": 0, "medium": 0, "long": 0, "act": 0}
    act_boundaries = []

    for i in range(1, len(phrases)):
        gap = phrases[i]["start_s"] - phrases[i - 1]["end_s"]
        gaps.append(gap)
        cat = _classify_gap(gap)
        gap_classifications[cat] += 1
        if gap > 8:
            act_boundaries.append(round(phrases[i - 1]["end_s"], 1))

    avg_pause = round(mean(gaps), 1) if gaps else 0.0

    n = len(phrases)
    opening_end = int(n * 0.2)
    body_scan_end = int(n * 0.7)
    return_end = n

    opening_gaps = [g for i, g in enumerate(gaps) if i < opening_end]
    body_gaps = [g for i, g in enumerate(gaps) if opening_end <= i < body_scan_end]
    return_gaps = [g for i, g in enumerate(gaps) if i >= body_scan_end]

    def _avg(ls):
        return round(mean(ls), 1) if ls else 0.0

    acts = []
    if act_boundaries:
        boundaries = [phrases[0]["start_s"]] + act_boundaries + [phrases[-1]["end_s"]]
        for i in range(len(boundaries) - 1):
            acts.append(
                {
                    "start": round(boundaries[i], 1),
                    "end": round(boundaries[i + 1], 1),
                    "duration": round(boundaries[i + 1] - boundaries[i], 1),
                }
            )

    return {
        "avg_speaking_rate_wpm": avg_wpm,
        "instruction_pause_seconds": _avg(opening_gaps),
        "body_scan_pause_seconds": _avg(body_gaps),
        "countdown_pause_seconds": _avg(return_gaps),
        "act_boundaries": act_boundaries,
        "act_structure": _build_act_structure(phrases, act_boundaries),
        "gap_distribution": {
            "short": {"range": "0-2s", "count": gap_classifications["short"]},
            "medium": {"range": "2-5s", "count": gap_classifications["medium"]},
            "long": {"range": "5-10s", "count": gap_classifications["long"]},
            "act": {"range": "10s+", "count": gap_classifications["act"]},
        },
        "avg_gap_seconds": round(mean(gaps), 2) if gaps else 0.0,
        "median_gap_seconds": round(median(gaps), 2) if gaps else 0.0,
        "max_gap_seconds": round(max(gaps), 2) if gaps else 0.0,
        "min_gap_seconds": round(min(gaps), 2) if gaps else 0.0,
    }


def _build_act_structure(phrases: list[dict], act_boundaries: list[float]) -> dict:
    if not act_boundaries:
        total = phrases[-1]["end_s"] - phrases[0]["start_s"]
        return {
            "single_act": {
                "phrase_range": [0, len(phrases)],
                "duration_s": round(total, 1),
                "percentage": 1.0,
            }
        }

    total = phrases[-1]["end_s"] - phrases[0]["start_s"]
    boundaries = [phrases[0]["start_s"]] + act_boundaries + [phrases[-1]["end_s"]]
    labels = ["opening", "deepening", "body", "return", "closing", "extra"]

    result = {}
    for i in range(len(boundaries) - 1):
        start_s = boundaries[i]
        end_s = boundaries[i + 1]
        start_idx = 0
        end_idx = len(phrases)
        for j, p in enumerate(phrases):
            if p["start_s"] >= start_s:
                start_idx = j
                break
        for j in range(len(phrases) - 1, -1, -1):
            if phrases[j]["end_s"] <= end_s:
                end_idx = j + 1
                break

        dur = end_s - start_s
        pct = round(dur / total, 2) if total > 0 else 0
        label = labels[i] if i < len(labels) else f"act_{i}"
        result[label] = {
            "phrase_range": [start_idx, end_idx],
            "duration_s": round(dur, 1),
            "percentage": pct,
        }

    return result


# ---------------------------------------------------------------------------
# Language analysis
# ---------------------------------------------------------------------------


def compute_language(phrases: list[dict], txt: str) -> dict:
    all_text = " ".join(p["text"] for p in phrases).strip()
    if not all_text:
        return {}

    words = re.findall(r"[a-zA-Z']+", all_text.lower())
    ngrams_2 = [" ".join(words[i : i + 2]) for i in range(len(words) - 1)]
    ngrams_3 = [" ".join(words[i : i + 3]) for i in range(len(words) - 2)]

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "of",
        "for",
        "is",
        "it",
        "you",
        "your",
        "that",
        "this",
        "as",
        "will",
        "are",
        "be",
        "with",
        "all",
        "from",
        "not",
        "do",
        "no",
    }

    counter_2 = Counter(ngrams_2)
    counter_3 = Counter(ngrams_3)

    meaningful_2 = [
        (ng, c)
        for ng, c in counter_2.most_common(30)
        if not all(w in stop for w in ng.split())
    ]
    top_bigrams = [ng for ng, _ in meaningful_2[:10]]

    meaningful_3 = [
        (ng, c)
        for ng, c in counter_3.most_common(20)
        if not all(w in stop for w in ng.split())
    ]
    top_trigrams = [ng for ng, _ in meaningful_3[:5]]

    common_phrases = top_bigrams[:5] + top_trigrams[:5]

    full_text = txt if txt else all_text
    sentences = re.split(r"[.!?]+", full_text)
    sentences = [s.strip() for s in sentences if s.strip()]

    sentence_lengths = [len(re.findall(r"[a-zA-Z']+", s)) for s in sentences]
    avg_sentence_len = round(mean(sentence_lengths), 1) if sentence_lengths else 0

    imperative_starts = {
        "take",
        "find",
        "close",
        "relax",
        "feel",
        "now",
        "allow",
        "begin",
        "concentrate",
        "visualize",
        "notice",
        "imagine",
        "bring",
        "move",
        "place",
        "let",
        "just",
        "assume",
        "drift",
        "sense",
        "remain",
        "enter",
        "use",
        "count",
        "repeat",
    }

    imperative_count = sum(
        1
        for s in sentences
        if s and s.split()[0].lower().rstrip(".") in imperative_starts
    )
    total_sentences = max(len(sentences), 1)
    imperative_ratio = round(imperative_count / total_sentences, 2)

    if imperative_ratio > 0.4:
        sentence_style = "imperative-directive"
    elif imperative_ratio > 0.25:
        sentence_style = "mixed-imperative"
    else:
        sentence_style = "descriptive-narrative"

    # Perspective
    you_count = len(re.findall(r"\byou\b", full_text, re.IGNORECASE))
    i_count = len(re.findall(r"\bI\b", full_text)) + len(
        re.findall(r"\bmy\b", full_text, re.IGNORECASE)
    )
    we_count = len(re.findall(r"\bwe\b", full_text, re.IGNORECASE))

    if you_count > i_count * 2:
        perspective = "second-person-guided"
    elif i_count > you_count:
        perspective = "first-person-narrative"
    else:
        perspective = "mixed"

    phrase_texts = [p["text"].lower().strip().rstrip(".") for p in phrases]
    phrase_counter = Counter(phrase_texts)
    repeated = sum(1 for c in phrase_counter.values() if c > 1)
    repetition_rate = round(repeated / max(len(phrase_counter), 1), 2)

    patterns = []
    if imperative_ratio > 0.3:
        patterns.append("command-based guidance")
    if any("deeper" in ft for ft in phrase_texts):
        patterns.append("deepening progression")
    if any("relax" in ft for ft in phrase_texts):
        patterns.append("relaxation cycles")
    if any(re.search(r"\b\d+\b", ft) for ft in phrase_texts[:5]):
        patterns.append("numbered sequence")
    if any("count" in ft for ft in phrase_texts):
        patterns.append("countdown element")
    if we_count > 3:
        patterns.append("inclusive framing")
    if not patterns:
        patterns.append("free-form narrative")

    return {
        "sentence_style": sentence_style,
        "perspective": perspective,
        "common_phrases": common_phrases[:10],
        "structural_patterns": patterns,
        "repetition_rate": repetition_rate,
        "avg_sentence_length_words": avg_sentence_len,
        "imperative_ratio": imperative_ratio,
        "total_sentences": len(sentences),
    }


# ---------------------------------------------------------------------------
# Tone inference
# ---------------------------------------------------------------------------


def infer_tone(phrases: list[dict], pacing: dict, language: dict) -> dict:
    wpm = pacing.get("avg_speaking_rate_wpm", 0)

    if wpm < 90:
        energy = "very-low"
    elif wpm < 110:
        energy = "low"
    elif wpm < 140:
        energy = "moderate"
    else:
        energy = "high"

    txt_words = " ".join(p["text"] for p in phrases).lower()
    warmth_words = sum(
        1
        for w in [
            "comfortable",
            "gentle",
            "soft",
            "peaceful",
            "warm",
            "safe",
            "loved",
            "beautiful",
            "wonderful",
        ]
        if w in txt_words
    )
    if warmth_words > 5:
        warmth = "warm"
    elif warmth_words > 2:
        warmth = "neutral-warm"
    else:
        warmth = "neutral"

    if language.get("sentence_style") == "imperative-directive":
        formality = "instructor"
    elif language.get("perspective") == "first-person-narrative":
        formality = "narrative"
    else:
        formality = "guided"

    descriptions = {
        "instructor": "Direct instructional tone with clear commands and structured guidance",
        "narrative": "Story-driven narrative tone with descriptive imagery",
        "guided": "Balanced guidance mixing instruction with descriptive elements",
    }

    return {
        "energy": energy,
        "warmth": warmth,
        "formality": formality,
        "description": descriptions.get(formality, "Meditation guidance tone"),
    }


# ---------------------------------------------------------------------------
# Binaural inference
# ---------------------------------------------------------------------------


def infer_binaural(collection: str, txt: str) -> dict:
    if collection == "silva-method-exercises":
        return {
            "brainwave": "alpha",
            "carrier_freq_hz": 120,
            "beat_freq_hz": 10.0,
            "description": "Alpha range (8-12 Hz) for relaxed awareness and learning, typical of Silva Method exercises",
        }
    else:
        txt_lower = txt.lower()
        if any(w in txt_lower for w in ["deep", "void", "shadow", "magic", "myst"]):
            brainwave = "theta"
            freq = 6.0
            desc = "Theta range (4-8 Hz) for deep meditation and visualization"
        else:
            brainwave = "alpha-theta"
            freq = 8.0
            desc = "Alpha-theta border (7-9 Hz) for relaxed meditation"

        return {
            "brainwave": brainwave,
            "carrier_freq_hz": 120,
            "beat_freq_hz": freq,
            "description": desc,
        }


# ---------------------------------------------------------------------------
# Trajectory inference
# ---------------------------------------------------------------------------


def infer_trajectory(phrases: list[dict], act_structure: dict, language: dict) -> dict:
    opening_desc = "Guided entry with relaxation induction"
    deepening_desc = "Progressive deepening through structured phases"
    transition_desc = "Smooth transitions between meditation states"
    deviation_desc = "Built-in handling for mind wandering with gentle redirection"

    if act_structure:
        n_acts = len(act_structure)
        if n_acts > 2:
            deepening_desc = f"Multi-phase deepening across {n_acts} distinct sections"
            transition_desc = "Structured transitions between well-defined phases"
        elif n_acts > 1:
            deepening_desc = "Binary structure with clear opening and return phases"
        else:
            deepening_desc = "Single continuous meditation flow"

    all_text = " ".join(p["text"] for p in phrases).lower()
    if "count" in all_text and ("1" in all_text or "one" in all_text):
        transition_desc = "Countdown-based transitions for deepening and return"
    if any(w in all_text for w in ["wandering", "distract", "attention", "focus"]):
        deviation_desc = "Explicit attention redirection for wandering mind"

    return {
        "opening": opening_desc,
        "deepening": deepening_desc,
        "transitions": transition_desc,
        "deviation_handling": deviation_desc,
    }


# ---------------------------------------------------------------------------
# Collection and category detection
# ---------------------------------------------------------------------------

# Known mappings
_KNOWN_COLLECTIONS = {
    "02-13-long-relax": "silva-method-exercises",
    "05-problem-solving-hollow-viewing": "silva-method-exercises",
    "10-04-mental-laboratory": "silva-method-exercises",
    "12-hollow-viewing-daytime": "silva-method-exercises",
    "05-the-avatar": "silva-method-exercises",
    "06-projection-exercise": "silva-method-exercises",
    "07-intuition-inanimate-objects": "silva-method-exercises",
    "08-projection-plants-animals": "silva-method-exercises",
    "09-scanning-human-body": "silva-method-exercises",
    "01-flow-twilight-pool": "advancing-witches-craft",
    "01-exploring-the-tunnel": "advancing-witches-craft",
    "02-illuminate-your-path": "advancing-witches-craft",
    "02-balance-circle-possibilities": "advancing-witches-craft",
    "03-source-connection": "advancing-witches-craft",
    "03-strength-looking-glass": "advancing-witches-craft",
    "04-visioning-exercise": "advancing-witches-craft",
    "04a-reflection-right-hand-path": "advancing-witches-craft",
    "04b-reflection-left-hand-path": "advancing-witches-craft",
    "08-09-healing-the-past": "advancing-witches-craft",
    "09-fantastic-voyage": "advancing-witches-craft",
    "cd2-04-nonphysical-friends": "advancing-witches-craft",
}

_KNOWN_CATEGORIES = {
    "02-13-long-relax": "relaxation",
    "05-problem-solving-hollow-viewing": "problem-solving",
    "10-04-mental-laboratory": "mental-laboratory",
    "12-hollow-viewing-daytime": "hollow-viewing",
    "05-the-avatar": "projection",
    "06-projection-exercise": "projection",
    "07-intuition-inanimate-objects": "intuition",
    "08-projection-plants-animals": "projection",
    "09-scanning-human-body": "body-scan",
    "01-flow-twilight-pool": "shadow-realm",
    "01-exploring-the-tunnel": "tunnel-exploration",
    "02-illuminate-your-path": "illumination",
    "02-balance-circle-possibilities": "balance",
    "03-source-connection": "source-connection",
    "03-strength-looking-glass": "strength",
    "04-visioning-exercise": "visioning",
    "04a-reflection-right-hand-path": "reflection",
    "04b-reflection-left-hand-path": "reflection",
    "08-09-healing-the-past": "healing",
    "09-fantastic-voyage": "voyage",
    "cd2-04-nonphysical-friends": "guide-connection",
}


def detect_collection(filename: str, phrases: list[dict]) -> str:
    base = Path(filename).stem
    if base in _KNOWN_COLLECTIONS:
        return _KNOWN_COLLECTIONS[base]

    all_text = " ".join(p["text"] for p in phrases).lower()
    silva_markers = [
        "level three",
        "level two",
        "level 3",
        "level 2",
        "three-to-one",
        "3-2-1",
        "countdown deepening",
    ]
    if any(m in all_text for m in silva_markers):
        return "silva-method-exercises"

    return "advancing-witches-craft"


def detect_category(filename: str) -> str:
    base = Path(filename).stem
    if base in _KNOWN_CATEGORIES:
        return _KNOWN_CATEGORIES[base]
    parts = base.replace("-", " ").split()
    return parts[-1] if parts else "meditation"


def human_readable_name(filename: str) -> str:
    base = Path(filename).stem
    name = re.sub(r"^(?:cd\d+-)?\d+(?:-\d+)?-", "", base)
    return name.replace("-", " ").title()


# ---------------------------------------------------------------------------
# Single-file processing
# ---------------------------------------------------------------------------


def process_file(srt_path: str, txt_path: str, output_dir: str) -> str:
    phrases = parse_srt(srt_path)
    txt = read_txt(txt_path)

    if not phrases:
        print(f"  WARNING: No phrases parsed from {srt_path}", file=sys.stderr)
        return ""

    collection = detect_collection(srt_path, phrases)
    category = detect_category(srt_path)
    filename_base = Path(srt_path).stem
    total_duration = round(phrases[-1]["end_s"] - phrases[0]["start_s"], 1)

    pacing = compute_pacing(phrases)
    language = compute_language(phrases, txt)
    tone = infer_tone(phrases, pacing, language)
    binaural = infer_binaural(collection, txt)
    trajectory = infer_trajectory(phrases, pacing.get("act_structure", {}), language)

    result = {
        "id": filename_base,
        "name": human_readable_name(srt_path),
        "collection": collection,
        "category": category,
        "total_duration_seconds": round(total_duration, 1),
        "total_phrases": len(phrases),
        "tone": tone,
        "pacing": pacing,
        "language": language,
        "trajectory": trajectory,
        "binaural": binaural,
    }

    out_path = Path(output_dir) / collection / f"{filename_base}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return str(out_path)


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def batch_process(repo_dir: str, output_dir: str) -> list[str]:
    srt_dir = Path(repo_dir) / "srt-phrase"
    txt_dir = Path(repo_dir) / "txt"

    if not srt_dir.exists():
        print(f"ERROR: SRT directory not found: {srt_dir}", file=sys.stderr)
        return []

    srt_files = sorted(srt_dir.glob("*.srt"))
    if not srt_files:
        print(f"ERROR: No .srt files found in {srt_dir}", file=sys.stderr)
        return []

    print(f"Batch processing {len(srt_files)} files from {srt_dir}")
    results = []

    for srt_file in srt_files:
        txt_file = txt_dir / f"{srt_file.stem}.txt"
        print(f"  Processing: {srt_file.name}", end=" ")
        try:
            out = process_file(str(srt_file), str(txt_file), output_dir)
            if out:
                print(f"-> {out}")
                results.append(out)
            else:
                print("SKIPPED (no phrases)")
        except Exception as e:
            print(f"ERROR: {e}", file=sys.stderr)

    print(f"\nDone: {len(results)}/{len(srt_files)} files processed")
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Parse SRT phrase files and TXT transcripts, extract pacing data, generate reference JSON."
    )
    parser.add_argument(
        "--input",
        "-i",
        help="Path to a single SRT file",
    )
    parser.add_argument(
        "--txt",
        "-t",
        help="Path to corresponding TXT transcript (used with --input)",
    )
    parser.add_argument(
        "--batch",
        "-b",
        help="Path to repository root (processes all SRT files in srt-phrase/)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="references",
        help="Output directory for reference JSON files (default: references/)",
    )

    args = parser.parse_args()

    if not args.input and not args.batch:
        parser.error("Either --input or --batch is required")

    if args.batch:
        batch_process(args.batch, args.output)
    elif args.input:
        if not os.path.exists(args.input):
            print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
            sys.exit(1)

        txt_path = args.txt or ""
        if not txt_path:
            base = Path(args.input).stem
            candidates = [
                Path(args.input).parent.parent / "txt" / f"{base}.txt",
                Path(args.input).parent / f"{base}.txt",
            ]
            for c in candidates:
                if c.exists():
                    txt_path = str(c)
                    break

        if not txt_path or not os.path.exists(txt_path):
            print(
                f"WARNING: No TXT file found for {args.input}, proceeding with SRT only",
                file=sys.stderr,
            )
            txt_path = ""

        out = process_file(args.input, txt_path, args.output)
        if out:
            print(f"Output: {out}")
        else:
            print("ERROR: No output generated", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()
