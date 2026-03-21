"""Scene captioning via a local vision LLM (Ollama).

Generates natural-language descriptions of camera snapshots, e.g.:
  "A white van is parked in the driveway. A person in a blue jacket
   is walking toward the front door carrying a bucket."

Supported models (via Ollama):
  - gemma3:4b       — best quality/VRAM ratio (~2.6 GB INT4), recommended
  - qwen2.5vl:3b    — excellent quality (~4-5 GB INT4)
  - moondream:1.8b   — ultra-fast, minimal VRAM (~2.5 GB INT4)

Set CAPTIONER_MODEL and OLLAMA_URL in .env to configure.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

import requests

from ring_detector.config import settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Describe this security camera image in 1-2 short sentences. "
    "List people (count, actions), vehicles (type, color), and activity. "
    "Do not start with 'Here is' or 'This image shows'. Be direct and factual."
)


def caption_image(image_path: str | Path) -> str | None:
    """Generate a scene description from a snapshot using a local vision LLM.

    Returns the caption string, or None if captioning is disabled/fails.
    """
    if not settings.captioner.enabled:
        return None

    image_path = Path(image_path)
    if not image_path.is_file():
        log.warning("Caption requested but image not found: %s", image_path)
        return None

    try:
        image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")

        response = requests.post(
            f"{settings.captioner.ollama_url}/api/chat",
            json={
                "model": settings.captioner.model,
                "messages": [
                    {
                        "role": "user",
                        "content": SYSTEM_PROMPT,
                        "images": [image_b64],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 150,
                },
            },
            timeout=settings.captioner.timeout,
        )
        response.raise_for_status()
        data = response.json()
        caption = data.get("message", {}).get("content", "").strip()

        # Strip common LLM preamble
        for prefix in (
            "Here's a description",
            "Here is a description",
            "This image shows",
            "The image shows",
        ):
            if caption.lower().startswith(prefix.lower()):
                # Find the first colon or period after the prefix and skip it
                idx = caption.find(":", len(prefix))
                if idx == -1:
                    idx = caption.find(".", len(prefix))
                if idx != -1:
                    caption = caption[idx + 1 :].strip()

        if caption:
            log.info("Caption: %s", caption[:100])
        return caption or None

    except requests.ConnectionError:
        log.warning(
            "Ollama not reachable at %s — captioning disabled",
            settings.captioner.ollama_url,
        )
        return None
    except requests.Timeout:
        log.warning("Caption timed out after %ds", settings.captioner.timeout)
        return None
    except Exception:
        log.exception("Caption generation failed")
        return None


MULTI_FRAME_PROMPT = (
    "These are frames from a security camera video showing a motion event. "
    "Describe what is happening in 2-3 short sentences. "
    "List people (count, actions, appearance), vehicles (type, color, movement), and activity. "
    "Do not start with 'Here is' or 'This image shows'. Be direct and factual."
)


def caption_frames(frames: list[bytes | str | Path], max_frames: int = 5) -> str | None:
    """Generate a scene description from multiple video frames.

    Sends up to `max_frames` images in a single Ollama request for a comprehensive
    caption that captures the full event (arrival, movement, departure).

    Args:
        frames: List of image file paths or raw bytes.
        max_frames: Maximum number of frames to send (limits VRAM/latency).

    Returns the caption string, or None if captioning is disabled/fails.
    """
    if not settings.captioner.enabled:
        return None

    if not frames:
        return None

    # Sample evenly if we have more frames than max
    if len(frames) > max_frames:
        step = len(frames) / max_frames
        indices = [int(i * step) for i in range(max_frames)]
        frames = [frames[i] for i in indices]

    images_b64 = []
    for frame in frames:
        try:
            data = Path(frame).read_bytes() if isinstance(frame, str | Path) else frame
            images_b64.append(base64.b64encode(data).encode("utf-8").decode("utf-8"))
        except Exception:
            log.warning("Failed to encode frame for captioning", exc_info=True)

    if not images_b64:
        return None

    try:
        response = requests.post(
            f"{settings.captioner.ollama_url}/api/chat",
            json={
                "model": settings.captioner.model,
                "messages": [
                    {
                        "role": "user",
                        "content": MULTI_FRAME_PROMPT,
                        "images": images_b64,
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 200,
                },
            },
            timeout=settings.captioner.timeout,
        )
        response.raise_for_status()
        data = response.json()
        caption = data.get("message", {}).get("content", "").strip()

        # Strip common LLM preamble
        for prefix in (
            "Here's a description",
            "Here is a description",
            "This image shows",
            "The image shows",
        ):
            if caption.lower().startswith(prefix.lower()):
                idx = caption.find(":", len(prefix))
                if idx == -1:
                    idx = caption.find(".", len(prefix))
                if idx != -1:
                    caption = caption[idx + 1 :].strip()

        if caption:
            log.info("Multi-frame caption: %s", caption[:120])
        return caption or None

    except requests.ConnectionError:
        log.warning("Ollama not reachable — multi-frame captioning skipped")
        return None
    except requests.Timeout:
        log.warning("Multi-frame caption timed out after %ds", settings.captioner.timeout)
        return None
    except Exception:
        log.exception("Multi-frame caption generation failed")
        return None


def is_available() -> bool:
    """Check if the captioning service is reachable and the model is loaded."""
    if not settings.captioner.enabled:
        return False
    try:
        resp = requests.get(
            f"{settings.captioner.ollama_url}/api/tags",
            timeout=5,
        )
        resp.raise_for_status()
        models = [m["name"] for m in resp.json().get("models", [])]
        available = any(settings.captioner.model in m for m in models)
        if not available:
            log.warning(
                "Model '%s' not found in Ollama. Available: %s",
                settings.captioner.model,
                ", ".join(models) or "none",
            )
        return available
    except Exception:
        return False
