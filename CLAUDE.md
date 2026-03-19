# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ring Detector is a self-hosted AI replacement for Ring's paid detection features. It listens for motion events via Firebase push (near-instant, no polling), downloads snapshots/videos to NAS, runs YOLOv8 object detection on GPU, matches vehicles against saved reference vectors (cleaner's car, yard guy's truck), tracks arrivals/departures, and sends push notifications via ntfy — including "Time to pay!" when the cleaner leaves.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # edit with your credentials
docker compose up -d   # PostgreSQL with pgvector
```

## Commands

```bash
ring-watch                          # Main loop: push events → detect → track → notify
ring-embed /path/to/images          # Batch process images
ring-ref ./photos --name cleaners_car --display-name "Cleaner"  # Create reference

ruff check --fix . && ruff format . # Lint
pytest -v                           # Test
```

## Architecture

### Event Flow

1. `RingEventListener` (Firebase push) delivers motion events in seconds
2. Downloads snapshot + archives video to `/mnt/nas/ring_archive/` by date
3. YOLOv8 detects objects → crops vehicles (car=2, truck=7 in COCO)
4. Crops → 576-dim embeddings via YOLOv8 backbone (head removed)
5. `match_against_references()` compares against pgvector cosine similarity
6. **Match** → `record_arrival()` + `notify_arrival("Cleaner arrived!")`
7. Subsequent motion extends the visit timer
8. **No motion for `DEPARTURE_TIMEOUT` seconds** → `record_departure()` + `notify_departure("Time to pay!")`

### Watcher Concurrent Tasks (`watcher.py`)

- `_process_events()` — handles push events from the Firebase queue
- `_check_departures()` — every 60s, checks if active visits have timed out
- `_refresh_ring_session()` — hourly Ring token refresh

### Package Layout

| Module | Purpose |
|--------|---------|
| `config.py` | All settings from `.env` — no hardcoded values |
| `ring_api.py` | Ring auth + `RingEventListener` (Firebase push) + async download |
| `models.py` | Loads YOLOv8, MTCNN, InceptionResnetV1 onto GPU |
| `detector.py` | Detection + embedding pipeline, adaptive face batch sizing |
| `image_utils.py` | Image I/O, resize, pad to square, batch preparation |
| `database.py` | SQLAlchemy 2.0 + pgvector — all data + vectors + visit tracking |
| `notifications.py` | ntfy: `notify_arrival()`, `notify_departure()`, `notify_motion()` |
| `watcher.py` | Main async event loop with arrival/departure tracking |
| `cli.py` | CLI entry points (`ring-embed`, `ring-ref`) |

### Database Tables

- **metadata** — image file info (path, dimensions, date)
- **detections** — YOLO bounding boxes (class, confidence, normalized coords)
- **embeddings** — 576-dim vectors (full_image, detection, reference types)
- **face_embeddings** — 512-dim InceptionResnetV1 face vectors
- **references** — named reference vectors for matching (cleaners_car, yard_guy)
- **visit_events** — arrival/departure tracking (arrived_at, departed_at, duration)

### Key Conventions

- Push-based events via `RingEventListener` — NOT polling
- `ring-doorbell` >= 0.9 is fully async — all Ring API calls use `await`
- torch pinned to `>=2.2,<2.3` due to facenet-pytorch compatibility
- Config exclusively via environment variables (see `.env.example`)
- Docker: `pgvector/pgvector:pg16` — single container for all data + vectors
- Face detection uses adaptive batch sizing (halves on CUDA OOM)
