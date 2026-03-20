"""Detection, embedding, and face recognition pipeline."""

from __future__ import annotations

import logging
from datetime import datetime
from gc import collect
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from ring_detector.image_utils import pad_to_square
from ring_detector.models import LoadedModels

log = logging.getLogger(__name__)


def clear_gpu_memory() -> None:
    collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def chunk(lst: list, n: int):
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


def get_chunks(*lists, chunk_size: int):
    return zip(*(chunk(lst, chunk_size) for lst in lists), strict=False)


# --- Object Detection ---


def run_detection(
    models: LoadedModels,
    resized_images: list[np.ndarray],
    image_paths: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, list[np.ndarray]]:
    """Run YOLO detection and return (metadata_df, detections_df, cropped_images)."""
    results = models.detect_model.predict(
        resized_images,
        imgsz=640,
        device=models.device,
        stream=False,
        verbose=False,
    )

    df_metadata_list = []
    df_detections_list = []
    img_crops = []
    det_idx = 0

    for m_index, result in enumerate(results):
        file_uuid = uuid4().hex
        meta = pd.DataFrame(
            {
                "file_uuid": file_uuid,
                "date": datetime.now(),
                "height": result.orig_shape[0],
                "width": result.orig_shape[1],
            },
            index=[m_index],
        )
        df_metadata_list.append(meta)

        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = models.detect_model.names[cls]
            conf = int(box.conf[0] * 100)
            bx = box.xywhn.tolist()[0]
            det_uuid = uuid4().hex

            det = pd.DataFrame(
                {
                    "uuid": det_uuid,
                    "file_uuid": file_uuid,
                    "class_name": class_name,
                    "class_id": cls,
                    "confidence": conf,
                    "xcenter": bx[0],
                    "ycenter": bx[1],
                    "width": bx[2],
                    "height": bx[3],
                },
                index=[det_idx],
            )
            df_detections_list.append(det)

            # Crop detected objects
            x1, y1, x2, y2 = (int(c) for c in box.xyxy.tolist()[0])
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(result.orig_img.shape[1], x2)
            y2 = min(result.orig_img.shape[0], y2)
            img_crops.append(result.orig_img[y1:y2, x1:x2])
            det_idx += 1

    df_meta = pd.concat(df_metadata_list) if df_metadata_list else pd.DataFrame()
    df_meta["path"] = image_paths[: len(df_meta)]
    df_dets = pd.concat(df_detections_list) if df_detections_list else pd.DataFrame()

    del results
    clear_gpu_memory()
    return df_meta, df_dets, img_crops


# --- CLIP Embeddings (vehicle matching) ---


def compute_clip_embeddings(
    models: LoadedModels,
    images: list[np.ndarray],
) -> list[list[float]]:
    """Compute L2-normalized 512-dim CLIP ViT-B/32 embeddings from BGR images.

    Replaces the old YOLO-backbone approach with a proper image-text encoder.
    Suitable for vehicle similarity matching via pgvector cosine distance.
    """
    if not images:
        return []

    tensors = []
    for img_bgr in images:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        tensors.append(models.clip_preprocess(pil_img))

    batch = torch.stack(tensors).to(models.device)
    with torch.no_grad():
        features = models.clip_model.encode_image(batch)

    normed = F.normalize(features.float(), p=2, dim=1)
    return normed.cpu().tolist()


# --- Face Detection (SCRFD + ArcFace) ---


def detect_faces_simple(
    models: LoadedModels,
    image_bgr: np.ndarray,
    min_face_size: int = 50,
) -> tuple[list[np.ndarray], list[list[float]]]:
    """Detect faces in a single BGR image.

    Delegates to models.face_detector (LocalFaceDetector or TritonFaceDetector).

    Returns (aligned_face_crops, face_embeddings) as parallel lists.
    Both lists are empty when face_detector is None or no faces are found.
    All failures are logged as warnings, not raised.
    """
    if models.face_detector is None:
        return [], []

    try:
        results = models.face_detector.detect_and_embed(image_bgr, min_face_size=min_face_size)
        if not results:
            return [], []
        crops = [r.aligned_crop for r in results]
        embeddings = [r.embedding for r in results]
        log.debug("detect_faces_simple: %d face(s) found", len(results))
        return crops, embeddings
    except Exception:
        log.warning("Face detection failed (non-fatal)", exc_info=True)
        return [], []


def detect_faces(
    models: LoadedModels,
    images_bgr: list[np.ndarray],
    image_paths: list[str],
    resized_images: list[np.ndarray],  # kept for API compatibility (unused)
    uuid_list: list[str],
    batch_size: int = 25,              # kept for API compatibility (unused)
) -> tuple[list[np.ndarray], list[list[float]], pd.DataFrame]:
    """Detect faces in a batch of images.

    Delegates to models.face_detector per image.

    Returns (aligned_face_crops_bgr, face_embeddings, face_detections_df).
    Aligned crops are 112×112 BGR uint8. Embeddings are 512-dim ArcFace vectors.
    Returns ([], [], empty DataFrame) if face_detector is None.
    """
    if models.face_detector is None:
        return [], [], pd.DataFrame()

    all_crops: list[np.ndarray] = []
    all_embeddings: list[list[float]] = []
    records: list[dict] = []

    for img, path, file_uuid in zip(images_bgr, image_paths, uuid_list, strict=False):
        try:
            results = models.face_detector.detect_and_embed(img)
            if not results:
                continue

            h, w = img.shape[:2]
            for r in results:
                all_crops.append(r.aligned_crop)
                all_embeddings.append(r.embedding)
                x1, y1, x2, y2 = r.box
                fw = max(0.0, x2 - x1)
                fh = max(0.0, y2 - y1)
                records.append(
                    {
                        "uuid": uuid4().hex,
                        "path": path,
                        "file_uuid": file_uuid,
                        "class_name": "face",
                        "class_id": None,
                        "confidence": int(r.score * 100),
                        "xcenter": max(0.0, (x1 + fw / 2) / w),
                        "ycenter": max(0.0, (y1 + fh / 2) / h),
                        "width": max(0.0, fw / w),
                        "height": max(0.0, fh / h),
                    }
                )
        except Exception:
            log.warning("Face detection failed for %s (non-fatal)", path, exc_info=True)

    df_faces = pd.DataFrame(records) if records else pd.DataFrame()
    return all_crops, all_embeddings, df_faces


def compute_face_embeddings(
    models: LoadedModels,
    face_crops: list[np.ndarray],
) -> list[list[float]]:
    """Compute ArcFace 512-dim embeddings from aligned face crops.

    Deprecated: detect_faces() now returns embeddings directly. This function
    is kept for backward compatibility; prefer detect_faces_simple() instead.
    """
    if not face_crops or models.face_detector is None:
        return []

    from ring_detector.face_detector import LocalFaceDetector
    from ring_detector.face_utils import run_arcface

    if not isinstance(models.face_detector, LocalFaceDetector):
        log.warning("compute_face_embeddings: not supported for non-local backends")
        return []

    batch = np.stack(face_crops)
    return run_arcface(models.face_detector._recognizer, batch).tolist()


# --- Similarity Matching ---


def find_vectors_above_threshold(
    vectors: list[list[float]],
    reference_vector: list[float],
    threshold: float = 0.85,
) -> list[int]:
    """Return indices of vectors whose dot product with reference exceeds threshold."""
    vecs = np.array(vectors)
    ref = np.array(reference_vector)
    scores = np.dot(vecs, ref)
    return list(np.where(scores > threshold)[0])


# --- Full Pipeline ---


def process_batch(
    models: LoadedModels,
    image_paths: list[str],
    resized_images: list[np.ndarray],
    padded_images: list[np.ndarray],
    session=None,
    batch_size: int = 50,
) -> dict:
    """Run full detection + embedding + face pipeline on a batch.

    All data (metadata, detections, embeddings) stored in PostgreSQL via pgvector.
    Returns summary dict with counts.
    """
    from ring_detector import database

    if not image_paths:
        return {"images": 0, "detections": 0, "faces": 0}

    # 1. Object detection
    log.info("Running object detection on %d images", len(image_paths))
    df_meta, df_dets, img_crops = run_detection(models, resized_images, image_paths)

    # 2. Crop embeddings (CLIP)
    crop_embeddings = []
    if len(df_dets) > 0 and img_crops:
        log.info("Computing CLIP embeddings for %d detections", len(img_crops))
        padded_crops = [pad_to_square(c) for c in img_crops]
        crop_embeddings = compute_clip_embeddings(models, padded_crops)
        del padded_crops
        clear_gpu_memory()

    # 3. Full image embeddings (CLIP)
    log.info("Computing full image CLIP embeddings")
    full_embeddings = compute_clip_embeddings(models, padded_images)

    # 4. Store everything in PostgreSQL
    if session is not None:
        log.info("Saving to PostgreSQL")
        database.insert_metadata_bulk(session, df_meta.to_dict("records"))
        if len(df_dets) > 0:
            database.insert_detections_bulk(session, df_dets.to_dict("records"))

        database.insert_embeddings_bulk(
            session,
            df_meta["file_uuid"].tolist(),
            full_embeddings,
            embed_type="full_image",
        )

        if crop_embeddings:
            database.insert_embeddings_bulk(
                session,
                df_dets["file_uuid"].tolist(),
                crop_embeddings,
                embed_type="detection",
                labels=df_dets["class_name"].tolist(),
            )

    del full_embeddings, crop_embeddings
    clear_gpu_memory()

    # 5. Face detection + embedding (SCRFD + ArcFace or Triton)
    if models.face_detector is None:
        return {"images": len(image_paths), "detections": len(df_dets), "faces": 0}

    log.info("Running face detection")
    uuid_list = df_meta["file_uuid"].tolist()
    face_crops, face_vectors, df_faces = detect_faces(
        models, padded_images, image_paths, resized_images, uuid_list
    )

    num_faces = 0
    if face_crops:
        log.info("Storing %d face embedding(s)", len(face_crops))
        if session is not None and len(df_faces) > 0:
            db_faces = df_faces.drop(columns=["path"]).to_dict("records")
            database.insert_detections_bulk(session, db_faces)

            database.insert_face_embeddings_bulk(
                session,
                df_faces["file_uuid"].tolist(),
                df_faces["path"].tolist(),
                face_vectors,
            )

        num_faces = len(face_crops)
        del face_crops, face_vectors
        clear_gpu_memory()

    return {
        "images": len(image_paths),
        "detections": len(df_dets),
        "faces": num_faces,
    }
