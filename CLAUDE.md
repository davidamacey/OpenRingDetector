# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Ring Detector is a self-hosted AI replacement for Ring's paid detection features. It listens for motion events via Firebase push, downloads snapshots/videos to NAS, runs YOLOv8 detection + optional Gemma 3 scene captioning, matches vehicles against saved references, tracks arrivals/departures, and sends rich push notifications (with snapshot images) via ntfy.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env  # edit with your credentials
docker compose up -d   # PostgreSQL with pgvector
```

## Commands

```bash
ring-watch                          # Main event loop
ring-embed /path/to/images          # Batch process images
ring-ref ./photos --name cleaners_car --display-name "Cleaner"
ring-visits --active                # View visit history
ring-refs                           # List references

ruff check --fix . && ruff format . # Lint
pytest -v                           # Test (8 tests)
```

## Architecture

### Event Flow

1. `RingEventListener` (Firebase push) delivers motion events in seconds
2. Cooldown check (30s default per camera) + event ID deduplication
3. Downloads snapshot + archives video to NAS by date
4. YOLOv8 detects objects → crops vehicles (car=2, truck=7)
5. (Optional) Gemma 3 4B via Ollama generates scene caption
6. Crops → 576-dim embeddings → `match_against_references()` via pgvector
7. **Match** → `record_arrival()` + notify with snapshot attachment
8. Subsequent motion extends visit timer
9. **Departure** (no motion for `DEPARTURE_TIMEOUT`) → notify "Time to pay!"

### Watcher Tasks (`watcher.py`)

- `_process_events()` — handles push events from Firebase queue
- `_check_departures()` — every 60s, checks visit timeouts
- `_refresh_ring_session()` — hourly token refresh

### Package Layout

| Module | Purpose |
|--------|---------|
| `config.py` | All settings from `.env` |
| `ring_api.py` | Ring auth + Firebase push listener + async download |
| `models.py` | Loads YOLOv8, MTCNN, InceptionResnetV1 onto GPU |
| `detector.py` | Detection + embedding pipeline |
| `captioner.py` | Scene captioning via Ollama (Gemma 3 4B) |
| `image_utils.py` | Image I/O, resize, pad, batch preparation |
| `database.py` | SQLAlchemy 2.0 + pgvector — all data + vectors + visits |
| `notifications.py` | ntfy with snapshot attachments |
| `watcher.py` | Async event loop: detect → match → track → notify |
| `cli.py` | CLI: ring-embed, ring-ref, ring-visits, ring-refs |

### Database Tables

- **metadata** — image file info
- **detections** — YOLO bounding boxes
- **embeddings** — 576-dim YOLO backbone vectors (pgvector)
- **face_embeddings** — 512-dim face vectors
- **references** — named reference vectors (cleaners_car, yard_guy)
- **visit_events** — arrival/departure tracking with timestamps

### Key Conventions

- Firebase push events — NOT polling
- `ring-doorbell` >= 0.9 fully async
- torch `>=2.2,<2.3` (facenet-pytorch compat)
- Config via env vars only (`.env.example`)
- Docker: `pgvector/pgvector:pg16` for all data + vectors
- Captioner: Gemma 3 4B via Ollama (optional, ~2.6 GB VRAM)
- Notifications include snapshot images via ntfy PUT
- Motion cooldown per camera (MOTION_COOLDOWN, default 30s)
- Event ID deduplication prevents double-processing
