"""SCRFD face detection and ArcFace recognition utilities.

CPU post-processing for SCRFD-10G face detection with 5-point landmarks,
Umeyama affine face alignment, and ArcFace 512-dim embedding inference.

Pipeline:
    image_bgr
    → preprocess_scrfd()            # letterbox + normalize
    → SCRFD ONNX session.run()      # 9 raw output tensors
    → decode_scrfd_outputs()        # anchor decode + NMS → boxes, scores, landmarks
    → align_faces_batch()           # Umeyama affine → 112×112 aligned crops
    → preprocess_for_arcface()      # BGR→RGB, CHW, (x-127.5)/128
    → ArcFace ONNX session.run()    # 512-dim L2-normalized embeddings

Adapted from triton-api/src/utils/scrfd_decode.py and face_align.py.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# SCRFD model constants
# ---------------------------------------------------------------------------

FPN_STRIDES = [8, 16, 32]
NUM_ANCHORS = 2
INPUT_SIZE = 640

# Raw ONNX output tensor names (from scrfd_10g_bnkps.onnx)
# Order: score_{8,16,32}, bbox_{8,16,32}, kps_{8,16,32}
SCRFD_INPUT_NAME = "input.1"
ARCFACE_INPUT_NAME = "input"

# Maps stride → (score_name, bbox_name, kps_name) in the ONNX model
_STRIDE_MAP: dict[int, tuple[str, str, str]] = {
    8: ("448", "451", "454"),
    16: ("471", "474", "477"),
    32: ("494", "497", "500"),
}

# ---------------------------------------------------------------------------
# ArcFace face alignment constants
# ---------------------------------------------------------------------------

ARCFACE_SIZE = 112

# Reference 5-point landmarks on a 112×112 canvas.
# Landmark order: left_eye, right_eye, nose, left_mouth, right_mouth.
ARCFACE_REF = np.array(
    [
        [38.2946, 51.6963],  # left eye
        [73.5318, 51.5014],  # right eye
        [56.0252, 71.7366],  # nose tip
        [41.5493, 92.3655],  # left mouth corner
        [70.7299, 92.2041],  # right mouth corner
    ],
    dtype=np.float64,
)


# ---------------------------------------------------------------------------
# SCRFD preprocessing
# ---------------------------------------------------------------------------


def preprocess_scrfd(img_bgr: np.ndarray, input_size: int = INPUT_SIZE) -> tuple[np.ndarray, float]:
    """Letterbox-resize image and normalize for SCRFD inference.

    Returns:
        blob: [1, 3, input_size, input_size] FP32 NCHW, RGB, (x-127.5)/128
        det_scale: scale factor used to map output coords back to original image
    """
    import cv2

    h, w = img_bgr.shape[:2]
    if h >= w:
        new_h = input_size
        new_w = int(w * input_size / h)
    else:
        new_w = input_size
        new_h = int(h * input_size / w)

    det_scale = float(new_h) / h

    resized = cv2.resize(img_bgr, (new_w, new_h))

    canvas = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    canvas[:new_h, :new_w] = resized

    # BGR→RGB, normalize, NHWC→NCHW
    blob = cv2.dnn.blobFromImage(
        canvas, 1.0 / 128.0, (input_size, input_size), (127.5, 127.5, 127.5), swapRB=True
    )
    return blob, det_scale


# ---------------------------------------------------------------------------
# SCRFD anchor generation + bbox/kps decoding + NMS
# ---------------------------------------------------------------------------


def _generate_anchors(h: int, w: int, stride: int) -> np.ndarray:
    """Generate anchor centers for one FPN level. Returns [N*NUM_ANCHORS, 2]."""
    centers = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
    centers = (centers * stride).reshape(-1, 2)
    if NUM_ANCHORS > 1:
        centers = np.stack([centers] * NUM_ANCHORS, axis=1).reshape(-1, 2)
    return centers


def _distance2bbox(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode (left, top, right, bottom) distances from anchor centers to xyxy boxes."""
    return np.stack(
        [
            points[:, 0] - distance[:, 0],
            points[:, 1] - distance[:, 1],
            points[:, 0] + distance[:, 2],
            points[:, 1] + distance[:, 3],
        ],
        axis=-1,
    )


def _distance2kps(points: np.ndarray, distance: np.ndarray) -> np.ndarray:
    """Decode (dx, dy) keypoint offsets from anchor centers. Returns [N, 5, 2]."""
    kps = np.zeros((len(points), 5, 2), dtype=np.float32)
    for i in range(5):
        kps[:, i, 0] = points[:, 0] + distance[:, i * 2]
        kps[:, i, 1] = points[:, 1] + distance[:, i * 2 + 1]
    return kps


def _nms(dets: np.ndarray, threshold: float = 0.4) -> list[int]:
    """Greedy NMS. dets: [N, 5] (x1, y1, x2, y2, score)."""
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        ovr = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(ovr <= threshold)[0] + 1]

    return keep


def decode_scrfd_outputs(
    net_outs: dict[str, np.ndarray],
    det_scale: float,
    input_size: int = INPUT_SIZE,
    det_thresh: float = 0.5,
    nms_thresh: float = 0.4,
    max_faces: int = 128,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode raw SCRFD ONNX output tensors into boxes, scores, and 5-point landmarks.

    Args:
        net_outs: {onnx_output_name: array} from session.run()
        det_scale: scale returned by preprocess_scrfd()
        det_thresh: minimum detection confidence
        nms_thresh: NMS IoU threshold
        max_faces: cap on returned faces

    Returns:
        boxes:     [M, 4] xyxy in original image coordinates
        scores:    [M]    confidence scores
        landmarks: [M, 5, 2] in original image coordinates
    """
    all_scores, all_bboxes, all_kpss = [], [], []

    for stride in FPN_STRIDES:
        score_name, bbox_name, kps_name = _STRIDE_MAP[stride]
        scores_raw = net_outs[score_name]  # [N, 1] or [B, N, 1]
        bbox_preds = net_outs[bbox_name]  # [N, 4]
        kps_preds = net_outs[kps_name]  # [N, 10]

        # Strip batch dim if present (e.g. [1, N, 1] → [N, 1])
        if scores_raw.ndim == 3:
            scores_raw = scores_raw[0]
            bbox_preds = bbox_preds[0]
            kps_preds = kps_preds[0]

        h = w = input_size // stride
        scores = scores_raw.flatten()

        anchor_centers = _generate_anchors(h, w, stride)

        # Scale predictions by stride
        bbox_preds = bbox_preds * stride
        kps_preds = kps_preds * stride

        pos = np.where(scores >= det_thresh)[0]
        if len(pos) == 0:
            continue

        bboxes = _distance2bbox(anchor_centers[pos], bbox_preds[pos])
        kpss = _distance2kps(anchor_centers[pos], kps_preds[pos])

        all_scores.append(scores[pos])
        all_bboxes.append(bboxes)
        all_kpss.append(kpss)

    if not all_scores:
        return np.array([]), np.array([]), np.array([])

    scores = np.concatenate(all_scores)
    bboxes = np.concatenate(all_bboxes)
    kpss = np.concatenate(all_kpss)

    # Scale back to original image space
    bboxes /= det_scale
    kpss /= det_scale

    # Sort by confidence, apply NMS
    order = scores.argsort()[::-1]
    scores, bboxes, kpss = scores[order], bboxes[order], kpss[order]

    pre_det = np.hstack((bboxes, scores[:, None])).astype(np.float32)
    keep = _nms(pre_det, nms_thresh)[:max_faces]

    return bboxes[keep], scores[keep], kpss[keep]


# ---------------------------------------------------------------------------
# Umeyama face alignment
# ---------------------------------------------------------------------------


def _umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Estimate similarity transform (rotation + scale + translation) mapping src → dst.

    Returns [2, 3] affine matrix for cv2.warpAffine.
    """
    n, d = src.shape
    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    src_demean = src - src_mean
    dst_demean = dst - dst_mean

    A = dst_demean.T @ src_demean / n
    U, S, Vt = np.linalg.svd(A)

    diag = np.ones(d, dtype=np.float64)
    if np.linalg.det(A) < 0:
        diag[d - 1] = -1

    T = np.eye(d + 1, dtype=np.float64)
    src_var = src_demean.var(axis=0).sum()
    scale = (S * diag).sum() / src_var

    T[:d, :d] = U @ np.diag(diag) @ Vt
    T[:d, :d] *= scale
    T[:d, d] = dst_mean - T[:d, :d] @ src_mean

    return T[:2, :]


def align_face(img_bgr: np.ndarray, landmarks: np.ndarray, size: int = ARCFACE_SIZE) -> np.ndarray:
    """Align a single face using 5-point Umeyama similarity transform.

    Args:
        img_bgr: BGR image [H, W, 3]
        landmarks: [5, 2] detected landmark coordinates (pixel space)
        size: output crop size (default 112 for ArcFace)

    Returns:
        Aligned BGR crop [size, size, 3]
    """
    import cv2

    dst = ARCFACE_REF if size == ARCFACE_SIZE else ARCFACE_REF * (size / ARCFACE_SIZE)
    M = _umeyama(landmarks.astype(np.float64), dst)
    return cv2.warpAffine(img_bgr, M.astype(np.float32), (size, size), borderValue=0.0)


def align_faces_batch(
    img_bgr: np.ndarray,
    landmarks_batch: np.ndarray,
    size: int = ARCFACE_SIZE,
) -> np.ndarray:
    """Align multiple faces from one image.

    Args:
        img_bgr: BGR image [H, W, 3]
        landmarks_batch: [N, 5, 2] landmark coordinates
        size: output size (default 112 for ArcFace)

    Returns:
        [N, size, size, 3] aligned BGR crops
    """
    if len(landmarks_batch) == 0:
        return np.zeros((0, size, size, 3), dtype=np.uint8)
    out = np.zeros((len(landmarks_batch), size, size, 3), dtype=np.uint8)
    for i, lmk in enumerate(landmarks_batch):
        out[i] = align_face(img_bgr, lmk, size)
    return out


# ---------------------------------------------------------------------------
# ArcFace preprocessing + inference
# ---------------------------------------------------------------------------


def preprocess_for_arcface(aligned_bgr: np.ndarray) -> np.ndarray:
    """Convert aligned BGR crops to ArcFace input format.

    Args:
        aligned_bgr: [N, 112, 112, 3] uint8 BGR aligned crops

    Returns:
        [N, 3, 112, 112] FP32, RGB, (x - 127.5) / 128
    """
    if len(aligned_bgr) == 0:
        return np.zeros((0, 3, ARCFACE_SIZE, ARCFACE_SIZE), dtype=np.float32)
    rgb = aligned_bgr[:, :, :, ::-1].copy()  # BGR → RGB
    chw = rgb.transpose(0, 3, 1, 2).astype(np.float32)  # NHWC → NCHW
    return (chw - 127.5) / 128.0


def run_arcface(session, aligned_bgr: np.ndarray) -> np.ndarray:
    """Run ArcFace ONNX inference on aligned face crops.

    Args:
        session: onnxruntime.InferenceSession for ArcFace
        aligned_bgr: [N, 112, 112, 3] uint8 BGR aligned crops

    Returns:
        [N, 512] L2-normalized FP32 embeddings
    """
    face_batch = preprocess_for_arcface(aligned_bgr)
    raw = session.run(None, {ARCFACE_INPUT_NAME: face_batch})[0]
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    return raw / np.maximum(norms, 1e-10)
