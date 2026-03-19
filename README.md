# Ring Detector

Custom AI-powered motion detection and notifications for Ring doorbell cameras — without paying for Ring Protect Plus AI features.

Ring Detector polls your Ring cameras for motion events, downloads snapshots/videos, runs local object detection (YOLOv8) and optional face recognition, matches against your saved reference images (e.g., your cleaner's car, yard guy's truck), and sends push notifications to your phone via [ntfy](https://ntfy.sh).

## Features

- **No Ring subscription needed** for AI features — runs entirely on your own hardware
- **Custom vehicle recognition** — train on photos of specific vehicles (cleaner, yard guy, etc.)
- **Face recognition** (optional) — identify known people via MTCNN + InceptionResnetV1
- **Push notifications** via ntfy — works on any phone with the ntfy app
- **Automatic archiving** — videos and snapshots saved to NAS with date-organized folders
- **Vector similarity search** — powered by Milvus for fast embedding lookups
- **GPU accelerated** — YOLOv8 detection runs on NVIDIA GPU

## Architecture

```
Ring Camera → [poll for motion events]
    ↓
Download snapshot/video → Archive to /mnt/nas/ring_archive/
    ↓
YOLOv8 Detection → Identify objects (cars, trucks, people, etc.)
    ↓
Crop detections → Compute embeddings (YOLOv8 backbone, 576-dim)
    ↓
Compare against saved references → Milvus similarity search
    ↓
Match found? → ntfy push notification ("Cleaner arrived!")
No match?   → ntfy notification ("Unknown visitor at Front Door")
```

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support
- Docker (for PostgreSQL + Milvus)
- A Ring doorbell camera with a cached auth token

### Installation

```bash
git clone https://github.com/davidamacey/ring-detector.git
cd ring-detector

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -e ".[dev]"

# Copy and edit config
cp .env.example .env
# Edit .env with your database passwords, camera name, ntfy URL, etc.
```

### Start Infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL, Milvus (with etcd + MinIO), and pgAdmin.

### Ring Authentication

```bash
# First-time auth (requires 2FA)
python -m ring_doorbell.cli --auth

# Token is saved to ./tokens/token.cache
```

### Create a Reference (e.g., Cleaner's Car)

Take 10-20 photos of the vehicle you want to recognize and put them in a directory:

```bash
ring-ref ./photos/cleaners_car --name "Cleaner's Car"
```

### Start Watching

```bash
ring-watch
```

This polls your Ring camera every 60 seconds (configurable) for motion events, runs detection, and sends notifications.

### Batch Process Existing Images

```bash
ring-embed /path/to/images --batch-size 50
```

## Configuration

All config is via environment variables (or `.env` file):

| Variable | Default | Description |
|----------|---------|-------------|
| `RING_CAMERA_NAME` | | Name of your Ring camera |
| `RING_POLL_INTERVAL` | `60` | Seconds between motion checks |
| `RING_TOKEN_PATH` | `./tokens/token.cache` | Path to Ring auth token |
| `ARCHIVE_DIR` | `/mnt/nas/ring_archive` | Where to save videos/snapshots |
| `NTFY_URL` | | Your ntfy topic URL |
| `TORCH_DEVICE` | `cuda:0` | GPU device for inference |
| `YOLO_MODEL_PATH` | `./models/yolov8m.pt` | Path to YOLO weights |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5433` | PostgreSQL port |
| `MILVUS_HOST` | `localhost` | Milvus host |
| `MILVUS_PORT` | `19530` | Milvus port |

## Project Structure

```
ring_detector/
├── __init__.py          # Package version
├── config.py            # Settings from environment
├── ring_api.py          # Async Ring API (auth, download, events)
├── models.py            # ML model loading (YOLO, MTCNN, ResNet)
├── detector.py          # Detection + embedding pipeline
├── image_utils.py       # Image I/O, resize, pad
├── database.py          # PostgreSQL (SQLAlchemy 2.0)
├── vector_db.py         # Milvus vector operations
├── notifications.py     # ntfy push notifications
├── watcher.py           # Main event loop (poll → detect → notify)
└── cli.py               # CLI entry points
```

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5433 | Metadata and detection storage |
| Milvus | 19530 | Vector similarity search |
| MinIO | 9054/9055 | Milvus object storage backend |
| pgAdmin | 6060 | Database admin UI |

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Lint and format
ruff check --fix .
ruff format .

# Run tests
pytest -v
```

## How It Works

1. **Polling**: `ring-watch` checks your Ring camera for new motion events
2. **Download**: When motion is detected, downloads a snapshot (and optionally video)
3. **Detection**: YOLOv8 identifies objects in the image (cars, trucks, people, etc.)
4. **Embedding**: Detected objects are cropped and run through the YOLOv8 backbone to produce 576-dimensional feature vectors
5. **Matching**: Vectors are compared against your saved references using cosine similarity
6. **Notification**: If a match is found (e.g., cleaner's car), sends a targeted push notification

## License

MIT
