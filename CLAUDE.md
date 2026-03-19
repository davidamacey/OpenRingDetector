# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Self-hosted AI replacement for Ring's paid detection features. Listens for Ring motion events via Firebase push, runs YOLOv8 detection + optional Gemma 3 captioning on GPU, matches vehicles against saved references, tracks arrivals/departures, and sends push notifications (with snapshot images) via ntfy.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
docker compose up -d
```

## Commands

```bash
ring-watch       # Main event loop (push → detect → match → track → notify)
ring-embed DIR   # Batch process images
ring-ref DIR --name cleaners_car --display-name "Cleaner"
ring-visits      # View visit history (--active, --name, --limit)
ring-refs        # List configured references
ring-status      # Health check: DB, GPU, model, Ring token, Ollama, archive

ruff check --fix . && ruff format .
pytest -v
```

## Architecture

1. `RingEventListener` (Firebase push) → event queue
2. Dedup + cooldown (30s) → download snapshot → archive video to NAS
3. YOLOv8 detects objects → crop vehicles (COCO class 2=car, 7=truck)
4. Optional: Gemma 3 4B caption via Ollama
5. Crops → 576-dim embeddings → `match_against_references()` (pgvector cosine)
6. Match → `record_arrival()` + `notify_arrival()` with snapshot
7. Subsequent motion → `extend_visit()` (resets departure timer)
8. No motion for `DEPARTURE_TIMEOUT` → `record_departure()` + "Time to pay!"

### Package (`ring_detector/`)

| Module | Role |
|--------|------|
| `config.py` | All settings from `.env` |
| `ring_api.py` | Ring auth + Firebase push + async download |
| `models.py` | YOLOv8, MTCNN, InceptionResnetV1 on GPU |
| `detector.py` | Detection + embedding pipeline |
| `captioner.py` | Gemma 3 4B via Ollama (optional) |
| `image_utils.py` | I/O, resize, pad, batch prep |
| `database.py` | SQLAlchemy 2.0 + pgvector — all tables |
| `notifications.py` | ntfy with snapshot attachments |
| `watcher.py` | Main async event loop |
| `cli.py` | All CLI commands |

### Database (PostgreSQL + pgvector)

`metadata`, `detections`, `embeddings` (576-dim), `face_embeddings` (512-dim), `references`, `visit_events`

### Key Constraints

- `torch>=2.2,<2.3` — facenet-pytorch compat
- Docker: `pgvector/pgvector:pg16` — single container
- Captioner: disabled by default (`CAPTIONER_ENABLED=false`)
- Firebase push — not polling
