# Ring Detector Web Dashboard — Implementation Prompt

Use this prompt to build the full web dashboard in a fresh Claude Code session.

---

## Prompt

Build a web dashboard for the OpenRingDetector project. Single user, no auth. The dashboard provides real-time monitoring, event history, vehicle/face profile management, and analytics for a Ring doorbell AI detection system.

### Existing Codebase

The backend data layer already exists in `ring_detector/database.py`:
- **Tables:** `metadata`, `detections`, `embeddings`, `face_embeddings`, `references`, `face_profiles`, `visit_events`
- **Functions:** `get_all_references()`, `get_all_face_profiles()`, `upsert_reference()`, `upsert_face_profile()`, `delete_face_profile()`, `get_active_visits()`, `match_against_references()`, `match_against_face_profiles()`, `record_event()`, `record_arrival()`, `record_departure()`, `run_migrations()`
- **DB:** PostgreSQL 17 + pgvector, connection via `settings.db.url`
- **Config:** `ring_detector/config.py` — all settings from `.env`

The detection pipeline exists in:
- `ring_detector/detector.py` — YOLO detection, CLIP embeddings, face detection
- `ring_detector/watcher.py` — main event loop, `RingWatcher` class
- `ring_detector/notifications.py` — ntfy push notifications
- `ring_detector/captioner.py` — Gemma 3 4B scene captioning via Ollama
- `ring_detector/image_utils.py` — frame extraction, image loading
- `ring_detector/ring_api.py` — Ring auth, snapshots, video download

### Tech Stack

- **Backend API:** FastAPI in `backend/` directory
  - Python 3.12, same venv as ring_detector
  - Import and reuse `ring_detector.database`, `ring_detector.config`, etc. directly
  - WebSocket endpoint for real-time event streaming
  - Use `pg_notify` to bridge database events to WebSocket clients
- **Frontend:** SvelteKit in `frontend/` directory
  - TypeScript, Tailwind CSS
  - 2-space indentation, `$lib` path aliases
  - `npm` (not yarn/pnpm)
- **Docker:** Add `ring-api` and `ring-frontend` services to `docker-compose.yml`
  - ring-api: port 8000
  - ring-frontend: port 3000

### API Endpoints (FastAPI)

```
GET    /api/health                    — status of DB, GPU, models, Ring token, Ollama
GET    /api/events                    — paginated event history (motion, ding, arrival, departure)
GET    /api/events/{id}               — single event with detections + snapshot
GET    /api/events/live               — WebSocket: real-time event stream via pg_notify
GET    /api/visits                    — visit history (active, completed, filterable by name)
GET    /api/visits/active             — currently active visits
GET    /api/references                — list all vehicle references
POST   /api/references                — create/update reference (upload images, compute CLIP embeddings)
DELETE /api/references/{name}         — delete a vehicle reference
GET    /api/faces                     — list all face profiles
POST   /api/faces                     — create/update face profile (upload image, compute ArcFace embedding)
DELETE /api/faces/{name}              — delete a face profile
GET    /api/snapshots/{path}          — serve snapshot images from archive dir
GET    /api/analytics/summary         — detection counts, visit stats, camera activity (last 24h/7d/30d)
GET    /api/analytics/timeline        — hourly detection counts for charts
GET    /api/config                    — current settings (non-sensitive)
POST   /api/config                    — update settings (writes to .env, reloads config)
POST   /api/test/notify               — send a test notification
```

### Frontend Pages

1. **Live** (`/`) — real-time event feed via WebSocket
   - Latest snapshot with detection overlay (bounding boxes)
   - Active visits panel showing who's currently here
   - Event stream: motion, ding, arrival, departure with timestamps
   - Status indicators: DB, GPU, Ring connection, Ollama

2. **History** (`/history`) — searchable event log
   - Filterable by: event type, camera, date range, reference name
   - Each event shows: timestamp, camera, type, detection summary, caption, snapshot thumbnail
   - Click to expand: full snapshot, detection details, face/vehicle matches
   - Pagination with infinite scroll

3. **Vehicles** (`/vehicles`) — vehicle reference management
   - Grid of current references with representative image
   - Click to view: reference images, match history, similarity stats
   - Add new: drag-and-drop images, set name/display name/category
   - Delete with confirmation

4. **Faces** (`/faces`) — face profile management
   - Grid of face profiles with profile photo
   - Click to view: enrollment photo, match history
   - Add new: upload photo, auto-detect face, set name/display name
   - Delete with confirmation

5. **Analytics** (`/analytics`) — detection statistics
   - Detection count chart (hourly, daily) — bar/line chart
   - Most frequent visitors (vehicle + face)
   - Average visit duration
   - Camera activity heatmap (hour of day vs day of week)
   - Detection class breakdown (pie chart: car, person, truck, etc.)

6. **Settings** (`/settings`) — configuration
   - Ring connection status + re-auth button
   - Detection settings: confidence threshold, cooldown, departure timeout
   - Video analysis: enabled, frame interval, max frames, wait timeout
   - Face detection: enabled, match threshold, min face size
   - Captioner: enabled, model, Ollama URL
   - Notification settings: ntfy URL, attach snapshots
   - Test notification button

### WebSocket Event Bridge

The watcher already calls `record_event()` for every motion/ding/arrival/departure. Add a `pg_notify` trigger on the events table so the API can bridge DB events to WebSocket clients in real-time:

```sql
CREATE OR REPLACE FUNCTION notify_event() RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('ring_events', row_to_json(NEW)::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER event_notify AFTER INSERT ON events
FOR EACH ROW EXECUTE FUNCTION notify_event();
```

The FastAPI WebSocket endpoint listens on `pg_notify('ring_events')` via asyncpg and forwards to connected clients.

### Key Constraints

- No authentication — single user, LAN only
- Reuse existing `ring_detector` package — don't duplicate database models or detection logic
- FastAPI app imports from `ring_detector.database`, `ring_detector.config`, etc.
- Snapshots are served from `settings.storage.archive_dir` — don't copy them
- The watcher (`ring-watch`) runs as a separate process — the API just reads the database
- Use ruff for Python linting (line-length 100), prettier for frontend
- Suppress ruff B008 for FastAPI `Depends()`/`Query()` patterns
- Add `ring-api` and `ring-frontend` entry points to `pyproject.toml`

### File Structure

```
backend/
  __init__.py
  app.py              — FastAPI app factory, CORS, lifespan
  routers/
    events.py          — /api/events, /api/events/live (WebSocket)
    visits.py          — /api/visits
    references.py      — /api/references
    faces.py           — /api/faces
    analytics.py       — /api/analytics
    config.py          — /api/config
    health.py          — /api/health
    snapshots.py       — /api/snapshots (static file serving)
    test.py            — /api/test/notify
frontend/
  src/
    routes/
      +page.svelte          — Live view
      history/+page.svelte  — History
      vehicles/+page.svelte — Vehicle management
      faces/+page.svelte    — Face management
      analytics/+page.svelte — Analytics
      settings/+page.svelte  — Settings
    lib/
      api.ts              — API client (fetch wrapper)
      websocket.ts        — WebSocket connection manager
      stores.ts           — Svelte stores (events, visits, config)
      components/
        EventCard.svelte
        DetectionOverlay.svelte
        VisitPanel.svelte
        StatusBar.svelte
        ImageUploader.svelte
        Chart.svelte
  package.json
  svelte.config.js
  tailwind.config.js
  tsconfig.json
```

### Docker Services (add to docker-compose.yml)

```yaml
ring-api:
  container_name: ring-api
  build:
    context: .
    dockerfile: backend/Dockerfile
  restart: always
  env_file: .env
  ports:
    - "8000:8000"
  depends_on:
    db:
      condition: service_healthy
  command: uvicorn backend.app:app --host 0.0.0.0 --port 8000

ring-frontend:
  container_name: ring-frontend
  build:
    context: ./frontend
  restart: always
  ports:
    - "3000:3000"
  depends_on:
    - ring-api
  environment:
    PUBLIC_API_URL: http://ring-api:8000
```
