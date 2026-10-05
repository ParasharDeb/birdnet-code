"""End-to-end smoke test: start the app in-process and push audio through it.

Uses a synthetic noise clip, so it asserts the plumbing works (upload, decode,
inference, filtering, serialisation) — not that BirdNET recognises anything.

    python -m scripts.smoke_test [path/to/real_recording.wav]
"""

from __future__ import annotations

import sys
import wave
from pathlib import Path
from tempfile import mkdtemp


def make_noise_wav(path: Path, seconds: int = 10, rate: int = 48_000) -> Path:
    """A band-limited noise clip — valid audio that shouldn't match any species."""
    import numpy as np

    rng = np.random.default_rng(0)
    samples = rng.normal(0, 0.05, seconds * rate)
    pcm = np.clip(samples, -1, 1)
    pcm = (pcm * 32767).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return path


def main() -> int:
    from fastapi.testclient import TestClient

    from app.main import app

    audio = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else make_noise_wav(Path(mkdtemp(prefix="birdnet-smoke-")) / "noise.wav")
    )
    print(f"Audio: {audio}\n")

    with TestClient(app) as client:  # lifespan loads the model here
        health = client.get("/health").json()
        print(f"/health  -> {health}\n")
        if not health["model_loaded"]:
            print("Model did not load; aborting.")
            return 2

        species = client.get("/species").json()
        print(f"/species -> {species['model_labels_total']} labels in model")
        for item in species["targets"]:
            mark = "OK     " if item["available_in_model"] else "MISSING"
            print(f"  {mark} {item['emoji']} {item['common_name']:<22} {item['model_label']}")
        if species["unavailable"]:
            print(f"\n  Not in this model: {', '.join(species['unavailable'])}")
        print()

        with audio.open("rb") as fh:
            resp = client.post(
                "/analyze",
                files={"file": (audio.name, fh, "audio/wav")},
                data={"min_confidence": "0.1", "include_non_target": "true"},
            )
        if resp.status_code != 200:
            print(f"/analyze -> {resp.status_code} {resp.text}")
            return 1

        body = resp.json()
        print(
            f"/analyze -> {body['duration_seconds']}s audio analysed in "
            f"{body['analysis_seconds']}s, {body['total_detections']} raw detections"
        )
        for row in body["species_summary"]:
            state = (
                f"{row['detection_count']}x, max {row['max_confidence']}"
                if row["detected"]
                else ("-" if row["available_in_model"] else "not in model")
            )
            print(f"  {row['emoji']} {row['common_name']:<22} {state}")

        others = body.get("other_detections") or []
        if others:
            print("\n  Other species heard (top 5):")
            for det in others[:5]:
                print(f"    {det['common_name']:<32} {det['confidence']}")

        # Error paths should be rejected cleanly rather than 500.
        bad = client.post("/analyze", files={"file": ("x.txt", b"nope", "text/plain")})
        assert bad.status_code == 415, bad.status_code
        empty = client.post("/analyze", files={"file": ("x.wav", b"", "audio/wav")})
        assert empty.status_code == 400, empty.status_code
        print("\nRejects bad extension (415) and empty upload (400).")

    print("\nSmoke test passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
