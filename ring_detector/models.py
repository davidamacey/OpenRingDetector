"""ML model loading and management."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from ultralytics import YOLO

from ring_detector.config import settings

log = logging.getLogger(__name__)


@dataclass
class LoadedModels:
    detect_model: YOLO
    embed_model: YOLO
    face_mtcnn: MTCNN
    face_resnet: InceptionResnetV1
    device: torch.device


def load_models(device: str | None = None) -> LoadedModels:
    """Load all detection and embedding models onto the specified device."""
    device = device or settings.model.device
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    log.info("Loading models on %s", dev)

    model_path = settings.model.yolo_model_path

    # Object detection model
    detect_model = YOLO(model_path, task="detect")
    detect_model.to(dev)

    # Embedding model (YOLO backbone without detection head)
    embed_model = YOLO(model_path, task="detect")
    embed_model.to(dev)
    embed_model.model.model = embed_model.model.model[:-1]

    # Face detection (MTCNN)
    face_mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=dev,
        keep_all=True,
    )

    # Face embedding (InceptionResnetV1)
    face_resnet = InceptionResnetV1(pretrained="vggface2").eval().to(dev)

    log.info("All models loaded")
    return LoadedModels(
        detect_model=detect_model,
        embed_model=embed_model,
        face_mtcnn=face_mtcnn,
        face_resnet=face_resnet,
        device=dev,
    )
