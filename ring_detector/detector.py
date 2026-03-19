"""Detection, embedding, and face recognition pipeline."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from gc import collect
from math import ceil
from uuid import uuid4

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ring_detector.image_utils import pad_to_square
from ring_detector.models import LoadedModels

log = logging.getLogger(__name__)


@dataclass
class ImageData:
    img_file_path: str
    img_face_box: np.ndarray
    img_uuid_org: str
    img_face_prob: float
    img_org_shape: tuple
    img_resized_shape: tuple


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


# --- Embeddings ---


def compute_yolo_embeddings(
    models: LoadedModels,
    images: list[np.ndarray],
) -> list[list[float]]:
    """Compute L2-normalized embeddings from YOLO backbone."""
    if not images:
        return []

    # Normalize and stack
    tensors = []
    for img in images:
        if img.max() > 1:
            t = torch.tensor(img / 255.0, dtype=torch.float32)
        else:
            t = torch.tensor(img, dtype=torch.float32)
        tensors.append(t)

    batch = torch.stack(tensors).to(models.device)
    batch = batch.permute(0, 3, 1, 2)  # BHWC → BCHW

    with torch.no_grad():
        out = models.embed_model.model(batch)

    batch_size, num_ch, _, _ = out.shape
    reshaped = out.view(batch_size, num_ch, -1)
    mean_vals = reshaped.mean(dim=2)
    normed = F.normalize(mean_vals, p=2, dim=1)
    return normed.tolist()


def compute_face_embeddings(
    models: LoadedModels,
    face_crops: list[torch.Tensor],
) -> list[list[float]]:
    """Compute L2-normalized face embeddings from InceptionResnetV1."""
    if not face_crops:
        return []
    batch = torch.stack(face_crops).to(models.device)
    with torch.no_grad():
        embeddings = models.face_resnet(batch)
    normed = F.normalize(embeddings, p=2, dim=1)
    return normed.tolist()


# --- Face Detection ---


def detect_faces(
    models: LoadedModels,
    padded_images_bgr: list[np.ndarray],
    image_paths: list[str],
    resized_images: list[np.ndarray],
    uuid_list: list[str],
    batch_size: int = 25,
) -> tuple[list[torch.Tensor], pd.DataFrame]:
    """Detect faces in padded images. Returns (face_crops, face_detections_df).

    Uses adaptive batch sizing to handle CUDA OOM errors.
    """
    # Convert BGR → RGB for MTCNN
    rgb_images = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in padded_images_bgr]

    face_crops: list[torch.Tensor] = []
    image_data_list: list[ImageData] = []
    current_batch_size = batch_size

    # Adaptive batch loop — reduces batch size on OOM
    while current_batch_size > 0:
        try:
            num_batches = ceil(len(rgb_images) / current_batch_size)
            for face_imgs, uuids, files, org_imgs in tqdm(
                zip(
                    chunk(rgb_images, current_batch_size),
                    chunk(uuid_list, current_batch_size),
                    chunk(image_paths, current_batch_size),
                    chunk(resized_images, current_batch_size),
                    strict=False,
                ),
                desc="Face detection",
                total=num_batches,
            ):
                try:
                    crops_batch, boxes_batch, probs_batch = models.face_mtcnn(
                        face_imgs, return_prob=True
                    )
                    for idx, (faces, boxes, probs) in enumerate(
                        zip(crops_batch, boxes_batch, probs_batch, strict=True)
                    ):
                        if isinstance(faces, torch.Tensor) and isinstance(boxes, np.ndarray):
                            for crop, box, prob in zip(faces, boxes, probs, strict=True):
                                face_crops.append(crop)
                                image_data_list.append(
                                    ImageData(
                                        img_file_path=files[idx],
                                        img_face_box=box,
                                        img_uuid_org=uuids[idx],
                                        img_face_prob=prob,
                                        img_org_shape=org_imgs[idx].shape[:2],
                                        img_resized_shape=face_imgs[idx].shape[:2],
                                    )
                                )
                except (torch.cuda.OutOfMemoryError, RuntimeError):
                    torch.cuda.empty_cache()
                    collect()
                    current_batch_size //= 2
                    log.warning("CUDA OOM, reducing face batch size to %d", current_batch_size)
                    break
            else:
                break  # Completed without OOM
        except Exception:
            log.exception("Face detection failed")
            break

    # Convert face coordinates to detection records
    if not image_data_list:
        return [], pd.DataFrame()

    records = []
    for data in image_data_list:
        orig_h, orig_w = data.img_org_shape
        x1, y1, x2, y2 = data.img_face_box
        eff_w = orig_w * (data.img_resized_shape[1] / max(data.img_org_shape))
        eff_h = orig_h * (data.img_resized_shape[0] / max(data.img_org_shape))

        x1_r = x1 / eff_w * orig_w
        y1_r = y1 / eff_h * orig_h
        x2_r = x2 / eff_w * orig_w
        y2_r = y2 / eff_h * orig_h

        w = x2_r - x1_r
        h = y2_r - y1_r
        xc = x1_r + w / 2
        yc = y1_r + h / 2

        records.append(
            {
                "uuid": uuid4().hex,
                "path": data.img_file_path,
                "file_uuid": data.img_uuid_org,
                "class_name": "face",
                "class_id": None,
                "confidence": int(data.img_face_prob * 100),
                "xcenter": max(0, xc / orig_w),
                "ycenter": max(0, yc / orig_h),
                "width": max(0, w / orig_w),
                "height": max(0, h / orig_h),
            }
        )

    df_faces = pd.DataFrame(records)
    return face_crops, df_faces


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

    # 2. Crop embeddings
    crop_embeddings = []
    if len(df_dets) > 0 and img_crops:
        log.info("Computing crop embeddings for %d detections", len(img_crops))
        padded_crops = [pad_to_square(c) for c in img_crops]
        crop_embeddings = compute_yolo_embeddings(models, padded_crops)
        del padded_crops
        clear_gpu_memory()

    # 3. Full image embeddings
    log.info("Computing full image embeddings")
    full_embeddings = compute_yolo_embeddings(models, padded_images)

    # 4. Store everything in PostgreSQL
    if session is not None:
        log.info("Saving to PostgreSQL")
        database.insert_metadata_bulk(session, df_meta.to_dict("records"))
        if len(df_dets) > 0:
            database.insert_detections_bulk(session, df_dets.to_dict("records"))

        # Full image embeddings
        database.insert_embeddings_bulk(
            session,
            df_meta["file_uuid"].tolist(),
            full_embeddings,
            embed_type="full_image",
        )

        # Crop embeddings
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

    # 5. Face detection + embedding
    log.info("Running face detection")
    uuid_list = df_meta["file_uuid"].tolist()
    face_crops, df_faces = detect_faces(
        models, padded_images, image_paths, resized_images, uuid_list, batch_size
    )

    num_faces = 0
    if face_crops:
        log.info("Computing face embeddings for %d faces", len(face_crops))
        face_vectors = compute_face_embeddings(models, face_crops)

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
