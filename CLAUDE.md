# CLAUDE.md — OpenRingDetector

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**OpenRingDetector** — self-hosted AI replacement for Ring's paid detection features. Listens for Ring motion events via Firebase push, runs YOLO11 detection + optional Gemma 3 captioning on GPU, matches vehicles against saved references, detects and identifies known persons via face recognition, tracks arrivals/departures, and sends push notifications (with snapshot images) via ntfy.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
bash scripts/download_model.sh          # YOLO11m weights
bash scripts/download_face_models.sh    # SCRFD-10G + ArcFace w600k_r50 ONNX models
docker compose up -d
```

## Commands

```bash
ring-watch       # Main event loop (push → detect → match → track → notify)
ring-embed DIR   # Batch process images
ring-ref DIR --name cleaners_car --display-name "Cleaner"
ring-visits      # View visit history (--active, --name, --limit)
ring-refs        # List configured vehicle references
ring-status      # Health check: DB, GPU, model, Ring token, Ollama, archive

ring-face add david photo.jpg --display-name "David"   # Add face profile
ring-face list                                          # List face profiles
ring-face delete david                                  # Remove face profile

ruff check --fix . && ruff format .
pytest -v
```

## Architecture

1. `RingEventListener` (Firebase push) → event queue
2. Dedup + cooldown (30s) → download snapshot → archive video to NAS
3. YOLO11 detects objects → crop vehicles (COCO class 2=car, 7=truck)
4. Optional: Gemma 3 4B caption via Ollama
5. Vehicle crops → 512-dim CLIP embeddings → `match_against_references()` (pgvector cosine)
6. SCRFD-10G detects faces → Umeyama align → ArcFace w600k_r50 → 512-dim embeddings
7. Face embeddings → `match_against_face_profiles()` (pgvector cosine, threshold 0.6)
8. Match → `record_arrival()` + `notify_arrival()` / `notify_known_person()` with snapshot
9. Subsequent motion → `extend_visit()` (resets departure timer)
10. No motion for `DEPARTURE_TIMEOUT` → `record_departure()` + "Time to pay!"

### Package (`ring_detector/`)

| Module | Role |
|--------|------|
| `config.py` | All settings from `.env` |
| `ring_api.py` | Ring auth + Firebase push + async download |
| `models.py` | YOLO11, CLIP ViT-B/32, SCRFD + ArcFace on GPU |
| `face_detector.py` | FaceDetector ABC (local ONNX + Triton backends) |
| `face_utils.py` | SCRFD decode, Umeyama alignment, ArcFace preprocessing |
| `detector.py` | Detection + embedding + face pipeline |
| `captioner.py` | Gemma 3 4B via Ollama (optional) |
| `image_utils.py` | I/O, resize, pad, batch prep |
| `database.py` | SQLAlchemy 2.0 + pgvector — all tables |
| `notifications.py` | ntfy with snapshot attachments |
| `watcher.py` | Main async event loop |
| `cli.py` | All CLI commands |

### Database (PostgreSQL 17 + pgvector)

`metadata`, `detections`, `embeddings` (512-dim CLIP), `face_embeddings` (512-dim ArcFace), `references`, `face_profiles`, `visit_events`

### Model Stack

| Component | Model | Dim |
|-----------|-------|-----|
| Object detection | YOLO11m | — |
| Vehicle embedding | CLIP ViT-B/32 (OpenAI) | 512 |
| Face detection | SCRFD-10G (`scrfd_10g_bnkps.onnx`) | — |
| Face embedding | ArcFace w600k_r50 (`arcface_w600k_r50.onnx`) | 512 |

### Face Recognition Stack (self-contained, no external service)

| Component | Model | Details |
|-----------|-------|---------|
| Detection | SCRFD-10G (`scrfd_10g_bnkps.onnx`) | 5-point landmarks, 95.2%/93.9%/83.1% WiderFace |
| Alignment | Umeyama similarity transform | Maps landmarks to 112×112 ArcFace reference |
| Embedding | ArcFace w600k_r50 (`arcface_w600k_r50.onnx`) | 512-dim L2-normalized, 99.8% LFW |

All inference runs locally via `onnxruntime-gpu` (GPU) or `onnxruntime` (CPU fallback).
Download models once: `bash scripts/download_face_models.sh`

### Key Constraints

- `torch>=2.5` — no upper pin (facenet-pytorch removed)
- `onnxruntime-gpu>=1.19` — SCRFD + ArcFace inference
- Docker: `pgvector/pgvector:pg17` — single container
- Captioner: disabled by default (`CAPTIONER_ENABLED=false`)
- Firebase push — not polling
- Face detection: enabled by default (`ENABLE_FACE_DETECTION=true`)
- Face match threshold: 0.6 cosine similarity (`FACE_MATCH_THRESHOLD`)
- Face min size: 50px (`FACE_MIN_SIZE`) — filters tiny/blurry faces
- DB migration required from v2.0 (576-dim → 512-dim): `docker compose down -v && docker compose up -d`

### Notification Types

| Trigger | Function | Message |
|---------|----------|---------|
| Known vehicle | `notify_arrival` | "Cleaner's car arrived at Front Door (car, David)" |
| Known person (no vehicle) | `notify_known_person` | "David arrived at Front Door" |
| Unknown visitor | `notify_unknown_visitor` | "Unknown visitor at Front Door (person)" |
| Motion only | `notify_motion` | "Motion at Front Door" |
| Doorbell | `notify_ding` | "Someone rang the doorbell at Front Door" |
| Departure | `notify_departure` | "Cleaner left Front Door after ~5 min. Time to pay!" |
