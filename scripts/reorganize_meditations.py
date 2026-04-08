#!/usr/bin/env python3
"""Reorganize meditation collections into artifact-type directories."""

import shutil
from pathlib import Path

REPO = Path(__file__).parent.parent / "meditation-repo"
COLLECTIONS_DIR = REPO / "collections"

COLLECTIONS = {
    "silva-method-exercises": {
        "files": {
            "01-exploring-the-tunnel": "01 - Exercise 1 - Exploring the Tunnel",
            "02-illuminate-your-path": "02_illuminate_your_path_with_significant_points_of_reference_05",
            "02-13-long-relax": "02-13 Long Relax Exercise",
            "03-source-connection": "03_source_connection_excercise_05",
            "04-visioning-exercise": "04_visioning_exercise_find_your_purpose_07",
            "05-problem-solving-hollow-viewing": "05_problem_solving_hollow_viewing_night_08",
            "06-projection-exercise": "06_projection_exercise_03",
            "07-intuition-inanimate-objects": "07_intuition_and_inanimate_objects_05",
            "08-projection-plants-animals": "08_projection_into_plants_and_animals_03",
            "08-09-healing-the-past": "08-09 Healing the Past Exercise",
            "09-fantastic-voyage": "09_09_Fantastic_Voyage_&_Learning_Points_of_Reference_Exercise",
            "09-scanning-human-body": "09_scanning_the_human_body_06",
            "10-04-mental-laboratory": "10-04 Creation of Mental Laboratory Exercise",
            "12-hollow-viewing-daytime": "12_hollow_viewing_daytime_02",
            "cd2-04-nonphysical-friends": "CD2 - 4 - Nonphysical friends",
        },
    },
    "advancing-witches-craft": {
        "files": {
            "01-flow-twilight-pool": "01-Flow_ The Twilight Pool",
            "02-balance-circle-possibilities": "02-Balance_ The Circle of Possibilities",
            "03-strength-looking-glass": "03-Strength_ Through the Looking Glass",
            "04a-reflection-right-hand-path": "04A-Reflection_ The Hall of Mirrors Right Hand Path",
            "04b-reflection-left-hand-path": "04B-Reflection_ The Hall of Mirrors Left Hand Path",
            "05-the-avatar": "05-The Avatar_ And Then There Were Two",
        },
    },
}

AUDIO_EXTS = [".mp3", ".MP3", ".flac", ".wav", ".m4a", ".ogg", ".aac"]


def reorganize():
    for collection_slug, collection in COLLECTIONS.items():
        collection_dir = COLLECTIONS_DIR / collection_slug

        for subdir in ("audio", "srt-phrase", "srt-word", "txt"):
            (collection_dir / subdir).mkdir(parents=True, exist_ok=True)

        for track_slug, audio_stem in collection["files"].items():
            track_dir = collection_dir / track_slug
            if not track_dir.exists():
                continue

            for f in track_dir.iterdir():
                if not f.is_file():
                    continue

                name = f.name
                suffixes = "".join(f.suffixes)
                if f.suffix in AUDIO_EXTS:
                    dst = collection_dir / "audio" / name
                elif suffixes.endswith(".phrase.srt"):
                    dst = collection_dir / "srt-phrase" / f"{track_slug}.srt"
                elif suffixes.endswith(".word.srt"):
                    dst = collection_dir / "srt-word" / f"{track_slug}.srt"
                elif f.suffix == ".txt":
                    dst = collection_dir / "txt" / f"{track_slug}.txt"
                else:
                    continue

                shutil.move(str(f), str(dst))

            if not any(track_dir.iterdir()):
                track_dir.rmdir()

    print("Reorganization complete.")
    for p in sorted((COLLECTIONS_DIR).rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(COLLECTIONS_DIR)}")


if __name__ == "__main__":
    reorganize()
