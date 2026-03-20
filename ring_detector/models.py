"""ML model loading and management."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import open_clip
import torch
from ultralytics import YOLO

from ring_detector.config import settings

log = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    detect_model: YOLO
    clip_model: torch.nn.Module       # CLIP ViT-B/32 visual encoder
    clip_preprocess: object           # torchvision transform for CLIP input
    face_app: object | None           # insightface.app.FaceAnalysis or None
    device: torch.device


def load_models(device: str | None = None) -> LoadedModels:
    """Load all detection and embedding models onto the specified device."""
    device = device or settings.model.device
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Loading models on %s", dev)

    # Object detection — YOLO11
    detect_model = YOLO(settings.model.yolo_model_path, task="detect")
    detect_model.to(dev)
    log.info("YOLO11 loaded: %s", settings.model.yolo_model_path)

    # Vehicle embedding — CLIP ViT-B/32 (OpenAI weights, 512-dim output)
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai"
    )
    clip_model = clip_model.eval().to(dev)
    log.info("CLIP ViT-B/32 loaded for vehicle embeddings (512-dim)")

    # Face detection + recognition — InsightFace (SCRFD + ArcFace r100, 512-dim)
    face_app = None
    try:
        from insightface.app import FaceAnalysis

        ctx_id = int(str(dev).split(":")[-1]) if "cuda" in str(dev) else -1
        face_app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=ctx_id, det_size=(640, 640))
        log.info("InsightFace loaded (SCRFD + ArcFace r100, 512-dim)")
    except Exception:
        log.warning("InsightFace failed to load — face detection disabled", exc_info=True)

    log.info("All models loaded")
    return LoadedModels(
        detect_model=detect_model,
        clip_model=clip_model,
        clip_preprocess=clip_preprocess,
        face_app=face_app,
        device=dev,
    )
