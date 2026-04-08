#!/usr/bin/env python3
"""Transcribe audio files using faster-whisper on GPU."""

import os
import sys
import time
import torch
from pathlib import Path

MEDIA_DIR = (
    sys.argv[1]
    if len(sys.argv) > 1
    else str(
        Path(__file__).parent.parent
        / "meditation-repo"
        / "Advancing The Witches Craft Meditations"
    )
)

AUDIO_EXTS = {".mp3", ".flac", ".wav", ".m4a", ".ogg", ".aac"}


def fmt_ts(s):
    ms = round((s % 1) * 1000) % 1000
    s_int = int(s)
    return "%02d:%02d:%02d,%03d" % (s_int // 3600, (s_int % 3600) // 60, s_int % 60, ms)


def transcribe_file(audio_path, out_dir):
    from faster_whisper import WhisperModel

    stem = Path(audio_path).stem
    text_out = out_dir / f"{stem}.txt"
    word_srt = out_dir / f"{stem}.word.srt"
    phrase_srt = out_dir / f"{stem}.phrase.srt"

    log = lambda m: print(f"[transcribe] {m}", file=sys.stderr, flush=True)
    log(f"Loading model (int8_float16, GPU)...")
    model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")

    log(f"Transcribing: {audio_path}")
    start = time.time()
    segments_iter, info = model.transcribe(
        str(audio_path),
        language=None,
        beam_size=5,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    segments = list(segments_iter)
    elapsed = time.time() - start
    log(
        f"Done in {elapsed:.0f}s ({elapsed / 60:.1f}min). Detected: {info.language} ({info.language_probability:.0%})"
    )

    all_words = []
    all_text_parts = []
    for seg in segments:
        all_text_parts.append(seg.text.strip())
        if seg.words:
            all_words.extend(seg.words)

    full_text = " ".join(all_text_parts)
    text_out.write_text(full_text + "\n", encoding="utf-8")

    with open(word_srt, "w", encoding="utf-8") as f:
        for i, w in enumerate(all_words, 1):
            f.write(f"{i}\n{fmt_ts(w.start)} --> {fmt_ts(w.end)}\n{w.word.strip()}\n\n")

    def group_words(words, max_words=12, max_chars=64, max_gap=0.6):
        groups = []
        cur_words = []
        cur_start = cur_end = None
        for w in words:
            t = w.word.strip()
            if not t:
                continue
            if cur_start is None:
                cur_start = w.start
                cur_end = w.end
                cur_words = [t]
                continue
            gap = w.start - (cur_end or w.start)
            combined = " ".join(cur_words)
            next_len = len(combined) + 1 + len(t)
            if gap > max_gap or len(cur_words) >= max_words or next_len > max_chars:
                groups.append((" ".join(cur_words), cur_start, cur_end))
                cur_start = w.start
                cur_end = w.end
                cur_words = [t]
            else:
                cur_words.append(t)
                cur_end = w.end
        if cur_words:
            groups.append((" ".join(cur_words), cur_start, cur_end))
        return groups

    groups = group_words(all_words)
    with open(phrase_srt, "w", encoding="utf-8") as f:
        for i, (text, s, e) in enumerate(groups, 1):
            f.write(f"{i}\n{fmt_ts(s)} --> {fmt_ts(e)}\n{text}\n\n")

    log(f"Word SRT: {len(all_words)} words, Phrase SRT: {len(groups)} phrases")
    del model
    import gc

    gc.collect()
    torch.cuda.empty_cache()
    return str(phrase_srt)


def main():
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_file():
        audio_path = Path(sys.argv[1])
        out_dir = audio_path.parent
        transcribe_file(audio_path, out_dir)
        return

    media_dir = (
        Path(MEDIA_DIR).resolve()
        if len(sys.argv) > 1
        else (Path(__file__).parent.parent / "meditation-repo").resolve()
    )
    if not media_dir.exists():
        print(f"Error: {media_dir} not found", file=sys.stderr)
        sys.exit(1)

    files = []
    for f in media_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in AUDIO_EXTS:
            if not f.parent.joinpath(f.stem + ".phrase.srt").exists():
                files.append(f)
    files.sort()
    if not files:
        print(f"No untranscribed audio files in {media_dir}", file=sys.stderr)
        sys.exit(0)

    print(
        f"Found {len(files)} untranscribed audio files in {media_dir}", file=sys.stderr
    )
    results = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {f.name}", file=sys.stderr)
        srt = transcribe_file(f, f.parent)
        results.append((f.name, srt))

    print(f"\n=== All done ===", file=sys.stderr)
    for name, srt in results:
        print(f"  {name} -> {srt}", file=sys.stderr)


if __name__ == "__main__":
    main()
