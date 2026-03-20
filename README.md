# OpenRingDetector

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Open-source, self-hosted AI replacement for Ring's paid detection features. Runs entirely on your own hardware — no Ring subscription required for AI features.

OpenRingDetector listens for Ring camera motion events via Firebase push, downloads snapshots, runs local YOLO11 object detection on GPU, matches vehicles against saved references, recognises known faces, tracks arrivals and departures, and sends push notifications via [ntfy](https://ntfy.sh).

## Features

- **No Ring subscription** — AI detection runs on your own GPU
- **Near-instant alerts** — Firebase push events, not polling (sub-second delivery)
- **Vehicle recognition** — match specific vehicles (cleaner's car, yard guy's truck) against saved photo references
- **Face detection & recognition** — identify known people via SCRFD + ArcFace (ONNX, GPU-accelerated)
- **Arrival + departure tracking** — "Cleaner arrived!" then "Cleaner left after ~45 min. Time to pay!"
- **Smart notifications** — snapshot images attached, detection summary included
- **Visit logging** — all visits stored in PostgreSQL with timestamps
- **Scene captioning** — optional Gemma 3 4B via Ollama for natural-language descriptions
- **Automatic archiving** — videos and snapshots saved to NAS by date
- **Vector similarity search** — pgvector in PostgreSQL 17 (no separate vector DB)
- **GPU accelerated** — YOLO11 detection + CLIP embeddings on NVIDIA GPU via CUDA

## Architecture

```
Ring Camera
    |  Firebase push (motion event)
    v
Download snapshot + archive video to NAS
    ↓
YOLO11 Detection → Identify objects (cars, trucks, people)
    ↓
Crop vehicles → Compute embeddings (512-dim, CLIP ViT-B/32)
    ↓
Compare against references → pgvector cosine similarity
    ↓
Match found?
  YES → Record arrival, notify "Cleaner arrived!" + snapshot
        Track visit, on departure → "Time to pay!"
  NO  → "Unknown visitor" or "Motion detected: car, person"

Face crops → ArcFace 512-dim embeddings → match face_profiles
  Known  → "David arrived at Front Door"
  Unknown → snapshot notification
```

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support
- Docker (for PostgreSQL 17 with pgvector)
- A Ring doorbell or floodlight camera

### Install

```bash
git clone https://github.com/davidamacey/OpenRingDetector.git
cd OpenRingDetector

python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env: set DB_PASSWORD, RING_CAMERA_NAME, NTFY_URL
```

### Download Model Weights

```bash
# YOLO11m detection weights
bash scripts/download_model.sh

# Face detection + recognition models (SCRFD + ArcFace ONNX)
bash scripts/download_face_models.sh
```

### Start Infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL 17 (with pgvector).

### Ring Authentication

```bash
# First-time auth (requires 2FA code from Ring app)
python -m ring_doorbell.cli --auth
# Token saved to ./tokens/token.cache
```

### Create Vehicle References

Take 10–20 photos of each vehicle you want to recognise:

```bash
ring-ref ./photos/cleaners_car --name cleaners_car --display-name "Cleaner"
ring-ref ./photos/yard_guy     --name yard_guy     --display-name "Yard Guy"
```

### Add Face Profiles (optional)

```bash
ring-face add david  photo.jpg --display-name "David"
ring-face add sarah  photo.jpg --display-name "Sarah"
```

### Watch

```bash
ring-watch
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `ring-watch` | Main event loop: push → detect → match → track → notify |
| `ring-embed DIR` | Batch-process images into the embedding store |
| `ring-ref DIR --name key --display-name "Label"` | Create/update a vehicle reference |
| `ring-refs` | List all vehicle references |
| `ring-face add NAME FILE` | Add a face profile |
| `ring-face list` | List all face profiles |
| `ring-visits` | View visit history (`--active`, `--name NAME`, `--limit N`) |
| `ring-status` | Health check: DB, GPU, models, Ring token, Ollama, archive |

## How It Works

1. **Push Event**: Ring camera detects motion → Firebase delivers event in seconds
2. **Snapshot**: Downloads camera snapshot + archives video to NAS
3. **Detection**: YOLO11 identifies objects (cars, trucks, people, etc.)
4. **Embedding**: Vehicle crops → 512-dim feature vectors via CLIP ViT-B/32
5. **Matching**: Vectors compared against references via pgvector cosine similarity
6. **Arrival**: Match found → record visit, send "Cleaner arrived!" notification
7. **Tracking**: Subsequent motion extends the visit timer
8. **Departure**: No motion for 5 min (configurable) → "Cleaner left after ~45 min. Time to pay!"

## Model Stack

| Component | Model | Notes |
|-----------|-------|-------|
| Object detection | YOLO11m | Cars, trucks, people — COCO 80 classes |
| Vehicle embedding | CLIP ViT-B/32 (OpenAI) | 512-dim, cosine similarity |
| Face detection | SCRFD-10GF (InsightFace) | 5-point landmarks |
| Face embedding | ArcFace r100 (InsightFace) | 512-dim, 99.6% LFW accuracy |

## Configuration

All settings are read from `.env` (copy from `.env.example`):

| Variable | Default | Description |
|----------|---------|-------------|
| `RING_CAMERA_NAME` | | Name of your Ring camera |
| `RING_TOKEN_PATH` | `./tokens/token.cache` | Ring auth token path |
| `RING_FCM_CREDENTIALS_PATH` | `./tokens/fcm_credentials.json` | FCM push credentials |
| `DEPARTURE_TIMEOUT` | `300` | Seconds without motion before departure |
| `MOTION_COOLDOWN` | `30` | Seconds to suppress duplicate events |
| `ARCHIVE_DIR` | `/mnt/nas/ring_archive` | Video/snapshot archive path |
| `NTFY_URL` | | ntfy topic URL |
| `TORCH_DEVICE` | `cuda:0` | GPU device for inference |
| `YOLO_MODEL_PATH` | `./models/yolo11m.pt` | Path to YOLO weights |
| `ENABLE_FACE_DETECTION` | `true` | Enable SCRFD + ArcFace pipeline |
| `FACE_MATCH_THRESHOLD` | `0.6` | Cosine similarity threshold for face match |
| `CAPTIONER_ENABLED` | `false` | Enable Gemma 3 4B scene captioning |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5433` | PostgreSQL port |

## Project Structure

```
ring_detector/          # Python package (internal name kept for compatibility)
├── __init__.py         # Package version
├── config.py           # All settings from .env
├── ring_api.py         # Ring auth + Firebase push event listener
├── models.py           # ML model loading (YOLO11, CLIP, InsightFace)
├── detector.py         # Detection + embedding pipeline
├── face_detector.py    # FaceDetector ABC (local ONNX + Triton backends)
├── face_utils.py       # SCRFD decoder, Umeyama alignment, ArcFace preprocessing
├── captioner.py        # Gemma 3 4B via Ollama (optional)
├── image_utils.py      # Image I/O, resize, pad
├── database.py         # PostgreSQL 17 + pgvector (all tables)
├── notifications.py    # ntfy push notifications
├── watcher.py          # Main async event loop
└── cli.py              # CLI entry points

scripts/
├── download_model.sh         # YOLO11m weights
└── download_face_models.sh   # SCRFD + ArcFace ONNX models

docker-compose.yml      # PostgreSQL 17 + pgvector container
```

## Database

PostgreSQL 17 + pgvector. Single container (`pgvector/pgvector:pg17`), port 5433.

| Table | Contents |
|-------|----------|
| `metadata` | Raw event metadata |
| `detections` | Per-frame detection results |
| `embeddings` | 512-dim CLIP vehicle embeddings |
| `references` | Saved vehicle reference embeddings |
| `face_embeddings` | 512-dim ArcFace embeddings per detection |
| `face_profiles` | Saved face references per known person |
| `visit_events` | Arrival/departure log with timestamps |

### Database Migration Note

If upgrading from `v2.0.x`, the vehicle embedding dimension changed from 576-dim (YOLO backbone) to 512-dim (CLIP ViT-B/32). You must recreate the database:

```bash
docker compose down -v   # ⚠ drops all data
docker compose up -d
ring-ref ./photos/... --name ... --display-name "..."  # rebuild references
```

## Development

```bash
ruff check --fix . && ruff format .
pytest -v
```

## License

OpenRingDetector is released under the [GNU Affero General Public License v3.0](LICENSE).

This project uses [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics) which is licensed under AGPL-3.0. Use of this project in a commercial product requires a separate Ultralytics commercial license.
