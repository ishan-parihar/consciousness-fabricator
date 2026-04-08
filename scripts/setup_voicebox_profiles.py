#!/usr/bin/env python3
"""Create voice-box profiles from extracted reference clips."""

import asyncio
import httpx
from pathlib import Path

VOICEBOX_URL = "http://127.0.0.1:17493"

PROFILES = [
    {
        "profile_id": "calm_instructor",
        "name": "Calm Instructor",
        "description": "Silva Method meditation voice - extracted from Fantastic Voyage exercise",
        "wav_path": "voices/ref/calm_instructor_ref.wav",
        "text_path": "voices/ref/calm_instructor_ref.txt",
    },
    {
        "profile_id": "warm_storyteller",
        "name": "Warm Storyteller",
        "description": "Shadow Realm journey voice - extracted from The Avatar exercise",
        "wav_path": "voices/ref/warm_storyteller_ref.wav",
        "text_path": "voices/ref/warm_storyteller_ref.txt",
    },
]

PROJECT_ROOT = Path(__file__).parent.parent


async def setup_profiles():
    async with httpx.AsyncClient(timeout=60.0) as client:
        for profile in PROFILES:
            print(f"\nSetting up profile: {profile['name']}")

            try:
                resp = await client.get(f"{VOICEBOX_URL}/profiles")
                resp.raise_for_status()
                existing = resp.json()
                existing_ids = [p["id"] for p in existing]

                if profile["profile_id"] in existing_ids:
                    print(f"   Profile already exists, using existing")
                    profile_id = profile["profile_id"]
                else:
                    resp = await client.post(
                        f"{VOICEBOX_URL}/profiles",
                        json={
                            "name": profile["name"],
                            "description": profile["description"],
                            "language": "en",
                        },
                    )
                    resp.raise_for_status()
                    profile_id = resp.json()["id"]
                    print(f"   Created profile: {profile_id}")

            except Exception as e:
                print(f"   Error creating profile: {e}")
                continue

            wav_path = PROJECT_ROOT / profile["wav_path"]
            text_path = PROJECT_ROOT / profile["text_path"]

            if not wav_path.exists():
                print(f"   WAV file not found: {wav_path}")
                continue

            if not text_path.exists():
                print(f"   Text file not found: {text_path}")
                continue

            reference_text = text_path.read_text(encoding="utf-8")

            try:
                with open(wav_path, "rb") as f:
                    files = {"file": (wav_path.name, f, "audio/wav")}
                    data = {"reference_text": reference_text}
                    resp = await client.post(
                        f"{VOICEBOX_URL}/profiles/{profile_id}/samples",
                        files=files,
                        data=data,
                    )
                    resp.raise_for_status()
                    print(
                        f"   Added reference sample: {len(reference_text.split())} words"
                    )

            except Exception as e:
                print(f"   Error adding sample: {e}")
                continue

            try:
                test_text = "Welcome to this meditation. Take a deep breath and relax."
                resp = await client.post(
                    f"{VOICEBOX_URL}/generate",
                    json={
                        "profile_id": profile_id,
                        "text": test_text,
                        "engine": "qwen",
                        "model_size": "1.7B",
                    },
                )
                resp.raise_for_status()
                gen_id = resp.json()["id"]
                print(f"   Test generation queued: {gen_id}")

                await wait_for_generation(client, gen_id)

            except Exception as e:
                print(f"   Test generation failed: {e}")

    print("\nProfile setup complete!")


async def wait_for_generation(client, gen_id, poll_interval=2.0, max_wait=120.0):
    import time

    start = time.time()
    while time.time() - start < max_wait:
        resp = await client.get(f"{VOICEBOX_URL}/generate/{gen_id}/status")
        resp.raise_for_status()
        status = resp.json()

        if status.get("status") == "completed":
            duration = status.get("duration", 0)
            print(f"   Generation complete: {duration:.1f}s audio")
            return True
        elif status.get("status") == "failed":
            print(f"   Generation failed: {status.get('error', 'unknown')}")
            return False

        await asyncio.sleep(poll_interval)

    print(f"   Generation timed out after {max_wait}s")
    return False


if __name__ == "__main__":
    asyncio.run(setup_profiles())
