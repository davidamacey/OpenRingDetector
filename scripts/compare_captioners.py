"""Compare Gemma 3 4B vs Moondream 1.8B for CCTV captioning on CPU."""

from __future__ import annotations

import base64
import time
from pathlib import Path

import requests

OLLAMA_URL = "http://localhost:9551"
PROMPT = (
    "Describe this security camera image in 1-2 short sentences. "
    "List people (count, actions), vehicles (type, color), and activity. "
    "Do not start with 'Here is' or 'This image shows'. Be direct and factual."
)

TEST_IMAGES = [
    ("Driveway (cleaner car)", "cleaner_video-20230918/frame_0151.jpg"),
    ("Side view 1", "images/Side_-_2023-09-30_09-42-25.jpg"),
    ("Side view 2", "images/Side_-_2023-10-01_09-53-36.jpg"),
    ("Side night", "images/Side_-_2023-09-20_02-06-44.jpg"),
]

MODELS = ["gemma3:4b", "moondream:1.8b"]


def caption(model: str, image_path: Path) -> tuple[str, float]:
    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    start = time.perf_counter()
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 150},
        },
        timeout=300,
    )
    elapsed = time.perf_counter() - start
    resp.raise_for_status()
    text = resp.json().get("message", {}).get("content", "").strip()
    return text, elapsed


def main():
    base = Path("/mnt/nvm/repos/ring_detector")

    # Warm up each model once
    print("Warming up models...")
    warmup_img = base / TEST_IMAGES[0][1]
    for model in MODELS:
        print(f"  Loading {model}...")
        caption(model, warmup_img)
    print()

    results: dict[str, list] = {m: [] for m in MODELS}

    for label, rel_path in TEST_IMAGES:
        img = base / rel_path
        if not img.exists():
            print(f"SKIP {label}: {img} not found")
            continue

        print(f"{'=' * 70}")
        print(f"IMAGE: {label} ({rel_path})")
        print(f"{'=' * 70}")

        for model in MODELS:
            text, elapsed = caption(model, img)
            results[model].append(elapsed)
            print(f"\n  [{model}] ({elapsed:.1f}s)")
            print(f"  {text}")

        print()

    # Summary
    print(f"\n{'=' * 70}")
    print("SPEED SUMMARY (CPU-only inference)")
    print(f"{'=' * 70}")
    for model in MODELS:
        times = results[model]
        if times:
            avg = sum(times) / len(times)
            print(f"  {model:20s}  avg={avg:.1f}s  min={min(times):.1f}s  max={max(times):.1f}s")


if __name__ == "__main__":
    main()
