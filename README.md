# Ring Detector

Custom AI-powered motion detection and notifications for Ring doorbell cameras — without paying for Ring Protect Plus AI features.

Ring Detector listens for motion events from your Ring cameras via Firebase push notifications, downloads snapshots, runs local YOLOv8 object detection on GPU, matches vehicles against your saved references (cleaner's car, yard guy's truck, etc.), tracks arrivals and departures, and sends push notifications to your phone via [ntfy](https://ntfy.sh).

## Features

- **No Ring subscription needed** for AI features — runs entirely on your own hardware
- **Near-instant alerts** — Firebase push events, not polling (sub-second delivery)
- **Custom vehicle recognition** — train on photos of specific vehicles
- **Arrival + departure tracking** — "Cleaner arrived!" then "Cleaner left after ~45 min. Time to pay!"
- **Visit logging** — all visits stored in PostgreSQL with timestamps
- **Face recognition** (optional) — identify known people via MTCNN + InceptionResnetV1
- **Push notifications** via ntfy — works on any phone
- **Automatic archiving** — videos and snapshots saved to NAS by date
- **Vector similarity search** — pgvector in PostgreSQL (no separate vector DB)
- **GPU accelerated** — YOLOv8 detection on NVIDIA GPU

## Architecture

```
Ring Camera
    ↓ Firebase push (motion event)
Download snapshot + archive video to NAS
    ↓
YOLOv8 Detection → Identify objects (cars, trucks, people)
    ↓
Crop vehicles → Compute embeddings (576-dim, YOLOv8 backbone)
    ↓
Compare against references → pgvector cosine similarity
    ↓
Match found?
  YES → Record arrival, notify "Cleaner arrived!"
        Track visit, on departure → "Time to pay!"
  NO  → "Unknown visitor" or "Motion detected: car, person"
```

## Quick Start

### Prerequisites

- Python 3.12+
- NVIDIA GPU with CUDA support
- Docker (for PostgreSQL with pgvector)
- A Ring doorbell camera

### Installation

```bash
git clone https://github.com/davidamacey/ring-detector.git
cd ring-detector

python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Edit .env with your database passwords, camera name, ntfy URL, etc.
```

### Start Infrastructure

```bash
docker compose up -d
```

This starts PostgreSQL (with pgvector) and pgAdmin.

### Ring Authentication

```bash
# First-time auth (requires 2FA)
python -m ring_doorbell.cli --auth

# Token is saved to ./tokens/token.cache
# FCM credentials auto-generated on first run of ring-watch
```

### Create References

Take 10-20 photos of each vehicle you want to recognize:

```bash
# Cleaner's vehicle
ring-ref ./photos/cleaners_car --name cleaners_car --display-name "Cleaner"

# Yard guy's truck
ring-ref ./photos/yard_guy --name yard_guy --display-name "Yard Guy"
```

### Start Watching

```bash
ring-watch
```

This connects to Ring via Firebase push events (near-instant), runs detection on every motion event, and tracks arrivals/departures.

### Batch Process Existing Images

```bash
ring-embed /path/to/images --batch-size 50
```

## How It Works

1. **Push Event**: Ring camera detects motion → Firebase delivers event in seconds
2. **Snapshot**: Downloads camera snapshot + archives video to NAS
3. **Detection**: YOLOv8 identifies objects (cars, trucks, people, etc.)
4. **Embedding**: Vehicle crops → 576-dim feature vectors via YOLOv8 backbone
5. **Matching**: Vectors compared against references via pgvector cosine similarity
6. **Arrival**: Match found → record visit, send "Cleaner arrived!" notification
7. **Tracking**: Subsequent motion extends the visit timer
8. **Departure**: No motion for 5 min (configurable) → "Cleaner left after ~45 min. Time to pay!"

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `RING_CAMERA_NAME` | | Name of your Ring camera |
| `RING_TOKEN_PATH` | `./tokens/token.cache` | Path to Ring auth token |
| `RING_FCM_CREDENTIALS_PATH` | `./tokens/fcm_credentials.json` | FCM push credentials |
| `DEPARTURE_TIMEOUT` | `300` | Seconds without motion before departure |
| `ARCHIVE_DIR` | `/mnt/nas/ring_archive` | Where to save videos/snapshots |
| `NTFY_URL` | | Your ntfy topic URL |
| `TORCH_DEVICE` | `cuda:0` | GPU device for inference |
| `YOLO_MODEL_PATH` | `./models/yolov8m.pt` | Path to YOLO weights |
| `DB_HOST` | `localhost` | PostgreSQL host |
| `DB_PORT` | `5433` | PostgreSQL port |

## Project Structure

```
ring_detector/
├── __init__.py          # Package version
├── config.py            # Settings from environment
├── ring_api.py          # Ring API + Firebase push event listener
├── models.py            # ML model loading (YOLO, MTCNN, ResNet)
├── detector.py          # Detection + embedding pipeline
├── image_utils.py       # Image I/O, resize, pad
├── database.py          # PostgreSQL + pgvector (all data + vectors)
├── notifications.py     # ntfy push notifications (arrival, departure, etc.)
├── watcher.py           # Main event loop (push events → detect → track → notify)
└── cli.py               # CLI entry points (ring-embed, ring-ref)
```

## Docker Services

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL + pgvector | 5433 | All data: metadata, detections, embeddings, visits |
| pgAdmin | 6060 | Database admin UI |

## Development

```bash
pip install -e ".[dev]"

ruff check --fix .
ruff format .

pytest -v
```

## License

MIT
