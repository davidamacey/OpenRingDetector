"""Abstract face detector interface with local ONNX and Triton HTTP backends.

Usage:
    from ring_detector.face_detector import create_face_detector
    detector = create_face_detector(settings)
    if detector:
        results = detector.detect_and_embed(image_bgr, min_face_size=50)
        for r in results:
            print(r.score, r.embedding[:5])

Backends:
    local  (default) — SCRFD-10G + ArcFace w600k_r50 via ONNX Runtime
    triton           — Sends inference to triton-api HTTP endpoint
                       Falls back to local if Triton is unreachable.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Single detected face with box, landmarks, aligned crop, and 512-dim embedding."""

    box: np.ndarray           # [4] xyxy in pixel coords
    score: float              # Detection confidence [0, 1]
    landmarks: np.ndarray     # [5, 2] five-point landmarks in pixel coords
    aligned_crop: np.ndarray  # [112, 112, 3] BGR uint8, Umeyama-aligned
    embedding: list[float]    # 512-dim ArcFace L2-normalized embedding


class FaceDetector(ABC):
    """Abstract interface for face detection and recognition."""

    @abstractmethod
    def detect_and_embed(
        self,
        image_bgr: np.ndarray,
        min_face_size: int = 50,
        det_thresh: float = 0.5,
    ) -> list[FaceResult]:
        """Detect faces in image_bgr and return embeddings.

        Args:
            image_bgr: BGR uint8 image [H, W, 3]
            min_face_size: Minimum face bounding-box side length (pixels)
            det_thresh: Detection confidence threshold [0, 1]

        Returns:
            List of FaceResult, one per detected face passing thresholds.
            Returns empty list on any error (failures logged as warnings, not raised).
        """


class LocalFaceDetector(FaceDetector):
    """SCRFD-10G + ArcFace w600k_r50 running locally via ONNX Runtime."""

    def __init__(self, detector_session, recognizer_session) -> None:
        self._detector = detector_session    # ort.InferenceSession for SCRFD
        self._recognizer = recognizer_session  # ort.InferenceSession for ArcFace

    def detect_and_embed(
        self,
        image_bgr: np.ndarray,
        min_face_size: int = 50,
        det_thresh: float = 0.5,
    ) -> list[FaceResult]:
        try:
            from ring_detector.face_utils import (
                INPUT_SIZE,
                SCRFD_INPUT_NAME,
                _STRIDE_MAP,
                align_faces_batch,
                decode_scrfd_outputs,
                preprocess_scrfd,
                run_arcface,
            )

            blob, det_scale = preprocess_scrfd(image_bgr, INPUT_SIZE)
            raw_outs = self._detector.run(None, {SCRFD_INPUT_NAME: blob})
            output_names = [name for stride in [8, 16, 32] for name in _STRIDE_MAP[stride]]
            net_outs = dict(zip(output_names, raw_outs))

            boxes, scores, landmarks = decode_scrfd_outputs(
                net_outs, det_scale, det_thresh=det_thresh
            )

            if len(boxes) == 0:
                return []

            if min_face_size > 0:
                face_sizes = np.minimum(boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
                keep = face_sizes >= min_face_size
                boxes, scores, landmarks = boxes[keep], scores[keep], landmarks[keep]

            if len(boxes) == 0:
                log.debug("All faces filtered by min_face_size=%d", min_face_size)
                return []

            aligned = align_faces_batch(image_bgr, landmarks)
            embeddings_np = run_arcface(self._recognizer, aligned)

            results = []
            for box, score, lmk, crop, emb in zip(
                boxes, scores, landmarks, aligned, embeddings_np, strict=True
            ):
                results.append(
                    FaceResult(
                        box=box,
                        score=float(score),
                        landmarks=lmk,
                        aligned_crop=crop,
                        embedding=emb.tolist(),
                    )
                )

            log.debug("LocalFaceDetector: %d face(s) detected", len(results))
            return results

        except Exception:
            log.warning("LocalFaceDetector: detection failed (non-fatal)", exc_info=True)
            return []


class TritonFaceDetector(FaceDetector):
    """Face detector/recognizer that calls the triton-api HTTP endpoint.

    Expects triton-api serving POST /v1/faces/recognize with:
      - multipart file field 'image' (JPEG/PNG)
      - query param 'confidence' (float)

    Response: {"num_faces": N, "faces": [...], "embeddings": [[512 floats], ...],
               "image": {"width": W, "height": H}}
    Box coordinates and landmarks are normalized [0, 1] and are de-normalized here.
    """

    def __init__(self, http_url: str) -> None:
        self._url = http_url.rstrip("/")

    def detect_and_embed(
        self,
        image_bgr: np.ndarray,
        min_face_size: int = 50,
        det_thresh: float = 0.5,
    ) -> list[FaceResult]:
        try:
            import cv2
            import requests

            _, buf = cv2.imencode(".jpg", image_bgr)
            jpeg_bytes = buf.tobytes()

            resp = requests.post(
                f"{self._url}/v1/faces/recognize",
                params={"confidence": det_thresh},
                files={"image": ("image.jpg", jpeg_bytes, "image/jpeg")},
                timeout=15,
            )
            resp.raise_for_status()
            data = resp.json()

            faces = data.get("faces", [])
            embeddings = data.get("embeddings", [])
            img_info = data.get("image", {})
            w = img_info.get("width", image_bgr.shape[1])
            h = img_info.get("height", image_bgr.shape[0])

            results = []
            for face, emb in zip(faces, embeddings, strict=False):
                box_norm = face.get("box", [0.0, 0.0, 0.0, 0.0])
                box = np.array(
                    [box_norm[0] * w, box_norm[1] * h, box_norm[2] * w, box_norm[3] * h],
                    dtype=np.float32,
                )
                fw = max(0.0, box[2] - box[0])
                fh = max(0.0, box[3] - box[1])

                if min_face_size > 0 and min(fw, fh) < min_face_size:
                    continue

                lmk_flat = face.get("landmarks", [])
                if len(lmk_flat) >= 10:
                    lmk = np.array(lmk_flat[:10], dtype=np.float32).reshape(5, 2)
                    lmk[:, 0] *= w
                    lmk[:, 1] *= h
                else:
                    lmk = np.zeros((5, 2), dtype=np.float32)

                # Align face locally for a consistent aligned_crop
                try:
                    from ring_detector.face_utils import align_face

                    aligned_crop = align_face(image_bgr, lmk)
                except Exception:
                    aligned_crop = np.zeros((112, 112, 3), dtype=np.uint8)

                results.append(
                    FaceResult(
                        box=box,
                        score=float(face.get("score", 0.0)),
                        landmarks=lmk,
                        aligned_crop=aligned_crop,
                        embedding=emb,
                    )
                )

            log.debug("TritonFaceDetector: %d face(s) detected", len(results))
            return results

        except Exception:
            log.warning("TritonFaceDetector: detection failed (non-fatal)", exc_info=True)
            return []


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_face_detector(cfg) -> FaceDetector | None:
    """Create a FaceDetector from settings.

    Returns None if face detection is disabled.
    FACE_BACKEND=triton falls back to local if Triton is unreachable.
    FACE_BACKEND=local (default) loads SCRFD + ArcFace ONNX sessions.
    """
    if not cfg.face.enabled:
        log.info("Face detection disabled (ENABLE_FACE_DETECTION=false)")
        return None

    if cfg.face.backend == "triton":
        return _create_triton_detector(cfg)

    return _create_local_detector(cfg)


def _create_local_detector(cfg) -> FaceDetector | None:
    """Load SCRFD + ArcFace ONNX sessions and return a LocalFaceDetector."""
    from pathlib import Path

    import torch

    scrfd_path = Path(cfg.model.scrfd_model_path)
    arcface_path = Path(cfg.model.arcface_model_path)

    if not scrfd_path.is_file():
        log.warning(
            "SCRFD model not found at %s — face detection disabled. "
            "Run: bash scripts/download_face_models.sh",
            scrfd_path,
        )
        return None

    if not arcface_path.is_file():
        log.warning(
            "ArcFace model not found at %s — face detection disabled. "
            "Run: bash scripts/download_face_models.sh",
            arcface_path,
        )
        return None

    try:
        import onnxruntime as ort

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if torch.cuda.is_available()
            else ["CPUExecutionProvider"]
        )
        detector_session = ort.InferenceSession(str(scrfd_path), providers=providers)
        recognizer_session = ort.InferenceSession(str(arcface_path), providers=providers)

        active_provider = detector_session.get_providers()[0]
        log.info("Face models loaded: SCRFD-10G + ArcFace w600k_r50 (%s)", active_provider)
        return LocalFaceDetector(detector_session, recognizer_session)

    except Exception:
        log.warning(
            "Face models failed to load — face detection disabled for this session",
            exc_info=True,
        )
        return None


def _create_triton_detector(cfg) -> FaceDetector | None:
    """Try to connect to triton-api; fall back to local on failure."""
    import requests

    http_url = cfg.face.triton_http_url.rstrip("/")
    log.info("FACE_BACKEND=triton — connecting to %s", http_url)

    try:
        resp = requests.get(f"{http_url}/health", timeout=5)
        resp.raise_for_status()
        log.info("Triton face API reachable at %s", http_url)
        return TritonFaceDetector(http_url)
    except Exception as exc:
        log.warning(
            "Triton face API unreachable (%s) — falling back to local mode", exc
        )
        return _create_local_detector(cfg)
