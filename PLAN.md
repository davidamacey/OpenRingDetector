# Ring Detector — Implementation Plan

## Current Status

**v2.0 is feature-complete and deployable.** All modules are implemented with real
working code. This document tracks what's done, known limitations, and the backlog.

---

## What's Done

### Core Pipeline
- [x] Firebase push-based event listener (`ring_doorbell.RingEventListener`) — no polling
- [x] Ring OAuth token caching and auto-refresh
- [x] FCM credentials persist across restarts
- [x] Per-camera motion cooldown (default 30 s)
- [x] Ring event ID deduplication (rolling window of 500)
- [x] Snapshot download on every motion event
- [x] Video archiving to NAS in background (non-blocking)
- [x] YOLOv8 object detection (car, truck, person, etc.)
- [x] Vehicle crop extraction from YOLO bounding boxes
- [x] 576-dim YOLOv8 backbone embeddings for crops
- [x] pgvector cosine similarity matching against named references
- [x] Arrival detection + visit recording (`visit_events` table)
- [x] Departure detection via inactivity timeout (default 5 min)
- [x] "Time to pay!" departure notification
- [x] ntfy push notifications with snapshot image attachments
- [x] VLM scene captioning via Ollama (Gemma 3 4B, optional)
- [x] Face detection (MTCNN) + face embedding (InceptionResnetV1 / VGGFace2)
- [x] Graceful SIGTERM/SIGINT shutdown (tasks cancel within seconds)
- [x] Automatic Firebase listener reconnection monitoring (every 5 min)

### CLI Commands
- [x] `ring-watch` — main event loop
- [x] `ring-embed DIR` — batch embed a directory of images
- [x] `ring-ref DIR --name X --display-name Y` — create a vehicle reference
- [x] `ring-visits` — view visit history (`--active`, `--name`, `--limit`)
- [x] `ring-refs` — list all configured references
- [x] `ring-status` — health check (DB, GPU, YOLO model, Ring token, Ollama)

### Infrastructure
- [x] PostgreSQL 16 + pgvector Docker container (`pgvector/pgvector:pg16`)
- [x] `docker-compose.yml` — DB service with healthcheck
- [x] `docker-compose.override.yml` — optional Ollama service (commented out)
- [x] `Dockerfile` — containerized ring-detector app (with CUDA 12.1)
- [x] `tokens/.gitkeep` — tokens directory tracked in git (contents gitignored)
- [x] `models/.gitkeep` — models directory (YOLO weights, gitignored)
- [x] `scripts/download_model.sh` — YOLOv8m weights download helper
- [x] `.pre-commit-config.yaml` — ruff check + format hooks

### Tests (22 total, all passing)
- [x] `test_config.py` — settings defaults and DB URL format
- [x] `test_captioner.py` — disabled-by-default, availability check
- [x] `test_image_utils.py` — pad_to_square (landscape, portrait, square)
- [x] `test_watcher.py` — dedup, cooldown, detection summary, shutdown helper
- [x] `test_notifications.py` — message content, priority, snapshot attachment

---

## Deployment Checklist

Before `ring-watch` can run, complete these one-time steps:

### 1. Install dependencies
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — at minimum set:
#   DB_PASSWORD, RING_CAMERA_NAME, NTFY_URL
```

### 3. Download YOLO model
```bash
bash scripts/download_model.sh
```

### 4. Start PostgreSQL
```bash
docker compose up -d
```

### 5. Authenticate with Ring (one-time, requires 2FA)
```bash
source .venv/bin/activate
python -m ring_doorbell.cli --auth
# Follow prompts — enters email, password, then 2FA code.
# Token saved to ./tokens/token.cache
```

### 6. (Optional) Create vehicle references
```bash
ring-ref ./photos/cleaners_car --name cleaners_car --display-name "Cleaner"
ring-ref ./photos/yard_guy    --name yard_guy      --display-name "Yard Guy"
```

### 7. Start the watcher
```bash
ring-watch
# Or as a background service: see systemd section below
```

### 8. Verify health
```bash
ring-status
```

---

## Systemd Service (Optional)

```ini
# /etc/systemd/system/ring-detector.service
[Unit]
Description=Ring Detector
After=network.target docker.service
Requires=docker.service

[Service]
Type=simple
User=superdave
WorkingDirectory=/mnt/nvm/repos/ring_detector
EnvironmentFile=/mnt/nvm/repos/ring_detector/.env
ExecStart=/mnt/nvm/repos/ring_detector/.venv/bin/ring-watch
Restart=on-failure
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now ring-detector
journalctl -fu ring-detector
```

---

## Docker Compose (Full Stack)

To run the detector itself in Docker alongside PostgreSQL:

```bash
# Add to docker-compose.yml under services:
#
#   detector:
#     build: .
#     restart: unless-stopped
#     env_file: .env
#     volumes:
#       - ./tokens:/app/tokens
#       - ./models:/app/models
#       - /mnt/nas/ring_archive:/mnt/nas/ring_archive
#     depends_on:
#       db:
#         condition: service_healthy
#     deploy:
#       resources:
#         reservations:
#           devices:
#             - driver: nvidia
#               count: 1
#               capabilities: [gpu]
```

Note: GPU passthrough requires `nvidia-container-toolkit` installed on the host.

---

## Backlog / Future Work

### Near-term
- [ ] **`ring-auth` CLI command** — wrap `python -m ring_doorbell.cli --auth` into a
      first-class `ring-auth` entry point with better UX
- [ ] **Person reference matching** — the face recognition pipeline is implemented
      (MTCNN + InceptionResnetV1) but not yet wired into the arrival/departure flow
      for people; currently only vehicles are matched against references
- [ ] **Multi-camera departure tracking** — departures are tracked per reference name,
      not per camera; if someone is seen by two cameras the visit logic may double-count
- [ ] **Configurable match threshold** — `MATCH_THRESHOLD` env var to tune cosine
      similarity cutoff (currently hardcoded at 0.15 cosine distance = 0.85 similarity)

### Medium-term
- [ ] **Web dashboard** — SvelteKit UI to view live events, visit history, manage
      references, and preview camera snapshots
- [ ] **Reference management via CLI** — `ring-ref delete NAME` and `ring-ref list`
      with embedding counts
- [ ] **Batch re-embed** — re-process archived images after adding new references
      (useful for bootstrapping after initial deployment)
- [ ] **Multi-GPU inference** — distribute detection across A6000 ×2 + 3080 Ti for
      higher throughput when processing multiple cameras simultaneously
- [ ] **LPR (license plate reading)** — integrate a lightweight LPR model as an
      additional matching signal alongside visual embeddings

### Long-term
- [ ] **Home Assistant integration** — publish events to MQTT for HA automations
- [ ] **Webhook output** — configurable POST to any URL on arrival/departure
- [ ] **Ring Alarm integration** — arm/disarm alarm based on visitor identity

---

## Known Limitations

1. **Active visit + new visitor** — when `get_active_visits()` returns results,
   `_handle_motion` extends those visits and returns early without re-detecting.
   A new, unknown visitor arriving while the cleaner is there will be missed until
   the cleaner's visit expires. Fix: run detection when motion comes from a camera
   that has no active visit on that specific camera.

2. **Ring subscription required for video** — `async_recording_download` requires
   an active Ring Protect subscription. Snapshot-only mode works without one.

3. **Torch version pin** — `torch>=2.2,<2.3` is required for `facenet-pytorch` compat.
   Cannot upgrade to torch 2.3+ without first updating facenet-pytorch.

4. **YOLO embedding dimension** — the 576-dim embedding is backbone-specific to
   YOLOv8m. Changing model size (yolov8s, yolov8l) changes the embedding dimension,
   requiring a database migration and re-embedding all references.

5. **No HEIC support in Docker** — `pyheif` requires libheif which is not in the
   slim Docker image; HEIC images only work in the native venv setup.
