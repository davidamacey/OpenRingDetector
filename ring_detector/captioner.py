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
    "You are a security camera assistant. Describe what you see in this camera snapshot "
    "in 1-2 concise sentences. Focus on: people (count, appearance, actions), vehicles "
    "(type, color, location), and any notable activity. Be factual, not speculative."
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
