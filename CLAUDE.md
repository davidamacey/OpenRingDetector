# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ring Detector is a self-hosted AI replacement for Ring's paid detection features. It polls Ring doorbell cameras for motion events, downloads snapshots/videos to NAS, runs YOLOv8 object detection and optional face recognition on GPU, matches detections against saved reference vectors (cleaner's car, yard guy, etc.), and sends push notifications via ntfy.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # edit with your credentials
docker compose up -d   # PostgreSQL with pgvector
```

## Commands

```bash
# Watch for motion events and notify (main loop)
ring-watch

# Batch process a directory of images
ring-embed /path/to/images --batch-size 50

# Create a reference (e.g., for a known vehicle)
ring-ref /path/to/photos --name cleaners_car --display-name "Cleaner's Car"

# Lint and format
ruff check --fix . && ruff format .

# Run tests
pytest -v
```

## Architecture

**Single database**: PostgreSQL with pgvector handles all structured data AND vector similarity search. No Milvus/etcd/MinIO needed.

### Pipeline: Motion Event → Notification

1. `watcher.py` polls Ring API for motion events
2. Downloads snapshot + video, archives to `/mnt/nas/ring_archive/` by date
3. YOLOv8 detects objects (cars, trucks, people) in snapshot
4. Detected crops → 576-dim embeddings via YOLOv8 backbone
5. Embeddings compared against references in pgvector
6. Match found → ntfy push notification (e.g., "Cleaner's Car Arrived")

### Package Layout (`ring_detector/`)

| Module | Purpose |
|--------|---------|
| `config.py` | All settings from `.env` — no hardcoded values |
| `ring_api.py` | Async Ring API (auth, video download, motion events) |
| `models.py` | Loads YOLOv8, MTCNN, InceptionResnetV1 onto GPU |
| `detector.py` | Detection + embedding pipeline, adaptive face batch sizing |
| `image_utils.py` | Image I/O, resize, pad to square, batch preparation |
| `database.py` | SQLAlchemy 2.0 + pgvector (metadata, detections, embeddings, references) |
| `notifications.py` | ntfy push notifications |
| `watcher.py` | Main async polling loop |
| `cli.py` | CLI entry points (`ring-embed`, `ring-ref`) |

### Database Tables (pgvector)

- **metadata** — image file info (path, dimensions, date)
- **detections** — YOLO bounding boxes (class, confidence, normalized coords)
- **embeddings** — 576-dim YOLO backbone vectors (full_image, detection, reference)
- **face_embeddings** — 512-dim InceptionResnetV1 face vectors
- **references** — named reference vectors for matching (cleaners_car, yard_guy, etc.)

### Key Conventions

- `ring-doorbell` >= 0.9 is fully async — all Ring API calls use `await`
- torch pinned to `>=2.2,<2.3` due to facenet-pytorch compatibility
- Config exclusively via environment variables (see `.env.example`)
- Docker image is `pgvector/pgvector:pg16` (PostgreSQL 16 + pgvector extension)
- Face detection uses adaptive batch sizing (halves on CUDA OOM) — critical section in `detector.py`
