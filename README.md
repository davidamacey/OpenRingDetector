# OpenRingDetector

[![License: AGPL-3.0](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

Open-source, self-hosted AI replacement for Ring's paid detection features. Runs entirely on your own hardware — no Ring subscription required for AI features.

OpenRingDetector listens for Ring camera motion events via Firebase push, downloads snapshots, runs local YOLOv8 object detection on GPU, matches vehicles against saved references, recognises known faces, tracks arrivals and departures, and sends push notifications via [ntfy](https://ntfy.sh).

## Features

- **No Ring subscription** — AI detection runs on your own GPU
- **Near-instant alerts** — Firebase push events, not polling (sub-second delivery)
- **Vehicle recognition** — match specific vehicles (cleaner's car, yard guy's truck) against saved photo references
- **Face detection & recognition** — identify known people via SCRFD + ArcFace (ONNX, GPU-accelerated)
- **Arrival + departure tracking** — "Cleaner arrived!" then "Cleaner left after ~45 min. Time to pay!"
- **Smart notifications** — snapshot images attached, detection summary included
- **Visit logging** — all visits stored in PostgreSQL with timestamps
- **Scene captioning** — optional Gemma 3 4B via Ollama for natural-language descriptions
- **Vector similarity search** — pgvector in PostgreSQL (no separate vector DB)
- **GPU accelerated** — YOLOv8 + ArcFace on NVIDIA GPU via CUDA

## Architecture

```
Ring Camera
    |  Firebase push (motion event)
    v
Download snapshot + archive video to NAS
    |
    v
YOLOv8 Detection  -->  Identify objects (cars, trucks, people)
    |
    v
Crop vehicles  -->  576-dim embeddings (YOLOv8 backbone)
    |
    v
pgvector cosine similarity against saved references
    |
  Match?
  YES --> record_arrival() + notify "Cleaner arrived!" + snapshot
          extend_visit() on subsequent motion
          After DEPARTURE_TIMEOUT --> "Cleaner left. Time to pay!"
  NO  --> "Unknown visitor" or generic motion alert

Face crops  -->  ArcFace 512-dim embeddings  -->  match face_profiles
  Known  --> "David arrived at Front Door"
  Unknown --> snapshot notification
```

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA
- Docker (for PostgreSQL + pgvector)
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

### Start Infrastructure

```bash
docker compose up -d
```

### Download Model Weights

```bash
bash scripts/download_model.sh          # YOLOv8m weights
bash scripts/download_face_models.sh    # SCRFD + ArcFace ONNX models
```

### Authenticate with Ring

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
| `YOLO_MODEL_PATH` | `./models/yolov8m.pt` | YOLOv8 weights |
| `ENABLE_FACE_DETECTION` | `true` | Enable SCRFD + ArcFace pipeline |
| `FACE_MATCH_THRESHOLD` | `0.6` | Cosine similarity threshold for face match |
| `CAPTIONER_ENABLED` | `false` | Enable Gemma 3 4B scene captioning |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama endpoint |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5433` | PostgreSQL port |

## Project Structure

```
ring_detector/          # Python package (internal name kept for compatibility)
├── config.py           # All settings from .env
├── ring_api.py         # Ring auth + Firebase push event listener
├── models.py           # ML model loading (YOLO, SCRFD, ArcFace)
├── detector.py         # Detection + embedding pipeline
├── face_utils.py       # SCRFD decoder, Umeyama alignment, ArcFace preprocessing
├── captioner.py        # Gemma 3 4B via Ollama (optional)
├── image_utils.py      # Image I/O, resize, pad
├── database.py         # PostgreSQL + pgvector (all tables)
├── notifications.py    # ntfy push notifications
├── watcher.py          # Main async event loop
└── cli.py              # CLI entry points

scripts/
├── download_model.sh         # YOLOv8m weights
└── download_face_models.sh   # SCRFD + ArcFace ONNX models

db-init-scripts/
└── initdb.sql          # pgvector extension + schema

docker-compose.yml      # PostgreSQL + pgvector container
```

## Database

PostgreSQL 16 + pgvector. Single container (`pgvector/pgvector:pg16`), port 5433.

| Table | Contents |
|-------|----------|
| `metadata` | Raw event metadata |
| `detections` | Per-frame detection results |
| `embeddings` | 576-dim YOLOv8 vehicle embeddings |
| `references` | Saved vehicle reference embeddings |
| `face_embeddings` | 512-dim ArcFace embeddings per detection |
| `face_profiles` | Saved face references per known person |
| `visit_events` | Arrival/departure log with timestamps |

## Development

```bash
ruff check --fix . && ruff format .
pytest -v
```

## License

OpenRingDetector is released under the [GNU Affero General Public License v3.0](LICENSE).

This project uses [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) which is licensed under AGPL-3.0. Use of this project in a commercial product requires a separate Ultralytics commercial license.
