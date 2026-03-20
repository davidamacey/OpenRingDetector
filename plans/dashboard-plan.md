# OpenRingDetector Dashboard — Comprehensive Implementation Plan

**Target:** Professional SvelteKit + FastAPI web dashboard
**Audience:** Single user (David), local network, no auth required
**Reference UX:** Frigate NVR, OpenTranscribe
**Created:** 2026-03-20

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Database Schema Additions](#2-database-schema-additions)
3. [Design System](#3-design-system)
4. [Pages — Wireframes & Behavior](#4-pages--wireframes--behavior)
5. [FastAPI Backend Specification](#5-fastapi-backend-specification)
6. [WebSocket Protocol](#6-websocket-protocol)
7. [Frontend Component Hierarchy](#7-frontend-component-hierarchy)
8. [SvelteKit Store Architecture](#8-sveltekit-store-architecture)
9. [Docker & Infrastructure](#9-docker--infrastructure)
10. [Implementation Phases](#10-implementation-phases)
11. [File Structure](#11-file-structure)
12. [Edge Cases & Empty States](#12-edge-cases--empty-states)
13. [Color Tokens Reference](#13-color-tokens-reference)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Browser                                │
│                                                                     │
│   SvelteKit SPA (http://localhost:5173 dev / :80 prod)             │
│   - Tailwind CSS dark theme                                         │
│   - WebSocket client for real-time events                          │
│   - Axios for REST API calls                                        │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ HTTP + WebSocket
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FastAPI Backend                                   │
│                    (http://localhost:8000)                           │
│                                                                     │
│   /api/events       - motion/ding event log (new table)            │
│   /api/visits       - arrival/departure history                    │
│   /api/references   - vehicle profile CRUD                         │
│   /api/faces        - face profile CRUD                            │
│   /api/unmatched    - unmatched detection gallery                  │
│   /api/status       - system health                                │
│   /api/settings     - configuration read/write                     │
│   /api/analytics    - aggregated stats                             │
│   /api/images/{path} - static snapshot serving                     │
│   WS /api/ws        - real-time event broadcast                    │
│                                                                     │
│   Shares same PostgreSQL DB as ring-watch process                  │
│   Also reads .env for settings, can write config overrides         │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ SQLAlchemy
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│             PostgreSQL 16 + pgvector (ring-db container)            │
│                                                                     │
│   metadata        - snapshot image records                         │
│   detections      - YOLO bounding boxes + classes                  │
│   embeddings      - 576-dim YOLO vehicle vectors                   │
│   face_embeddings - 512-dim ArcFace vectors                        │
│   references      - known vehicle profiles                         │
│   visit_events    - arrival/departure records                      │
│   events          - NEW: raw Ring event log (motion/ding/etc)      │
└─────────────────────────────────────────────────────────────────────┘
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
┌─────────────────────┐   ┌─────────────────────────────────────────┐
│   ring-watch        │   │   NAS / ARCHIVE_DIR                     │
│   (Python process)  │   │                                         │
│   - Firebase push   │   │   snapshots/YYYY-MM-DD/*.jpg            │
│   - YOLOv8 detect   │   │   videos/YYYY-MM-DD/*.mp4               │
│   - Face detection  │   │                                         │
│   - Visit tracking  │   └─────────────────────────────────────────┘
│   - ntfy notify     │
└─────────────────────┘
```

**Key Architectural Decision:** The FastAPI backend is a **read-mostly** companion to the existing `ring-watch` process. It does not replace or duplicate the detection pipeline — it only reads from the same PostgreSQL database and serves data to the dashboard. The watcher process continues to own detection and notification logic.

The FastAPI server also listens for internal events via an asyncio queue bridge (populated by a shared in-process event bus, or via PostgreSQL LISTEN/NOTIFY if running as separate containers).

---

## 2. Database Schema Additions

### New Table: `events`

The current schema has no explicit event log — snapshots (`metadata`) exist but have no event type, camera name, or link back to a visit. This new table is the primary feed for the dashboard's live view.

```sql
CREATE TABLE events (
    id          SERIAL PRIMARY KEY,
    event_type  VARCHAR(20) NOT NULL,   -- 'motion' | 'ding' | 'arrival' | 'departure'
    camera_name VARCHAR(100) NOT NULL,
    occurred_at TIMESTAMP NOT NULL DEFAULT NOW(),
    file_uuid   VARCHAR(32) REFERENCES metadata(file_uuid) ON DELETE SET NULL,
    snapshot_path VARCHAR(500),
    detection_summary TEXT,             -- e.g. "car, person x2"
    reference_name VARCHAR(100),        -- populated for arrival/departure
    display_name   VARCHAR(100),        -- populated for arrival/departure
    visit_event_id INTEGER REFERENCES visit_events(id) ON DELETE SET NULL,
    caption     TEXT                    -- Gemma VLM caption when available
);

CREATE INDEX idx_events_occurred_at ON events(occurred_at DESC);
CREATE INDEX idx_events_camera ON events(camera_name);
CREATE INDEX idx_events_type ON events(event_type);
```

**Populated by:** `ring_detector/watcher.py` — call `record_event()` at each stage of `_handle_motion` and on ding.

### New Table: `face_profiles` (already in MEMORY.md, add if not present)

```sql
CREATE TABLE face_profiles (
    uuid         VARCHAR(32) PRIMARY KEY,
    name         VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(100) NOT NULL,
    created_at   TIMESTAMP DEFAULT NOW(),
    sample_image_path VARCHAR(500),      -- best sample face crop for display
    vector       VECTOR(512) NOT NULL    -- mean ArcFace embedding
);
```

### SQLAlchemy ORM Addition (`database.py`)

```python
class Event(Base):
    __tablename__ = "events"

    id             = Column(Integer, primary_key=True, autoincrement=True)
    event_type     = Column(String, nullable=False)
    camera_name    = Column(String, nullable=False)
    occurred_at    = Column(DateTime, default=datetime.now, index=True)
    file_uuid      = Column(String, ForeignKey("metadata.file_uuid", ondelete="SET NULL"), nullable=True)
    snapshot_path  = Column(String, nullable=True)
    detection_summary = Column(String, nullable=True)
    reference_name = Column(String, nullable=True)
    display_name   = Column(String, nullable=True)
    visit_event_id = Column(Integer, ForeignKey("visit_events.id", ondelete="SET NULL"), nullable=True)
    caption        = Column(String, nullable=True)
```

### No other new tables required.

The dashboard can reconstruct unmatched detections by querying `detections` records whose `file_uuid` has no associated `reference_name` in the `events` table.

---

## 3. Design System

### Philosophy
Dark, dense, professional. Like Frigate NVR meets Apple's dark mode. Snapshot images are the primary content — the UI frames them without competing.

### Color Tokens

```css
/* Base palette — add to tailwind.config.ts */
--color-bg:           #0d0f14;   /* page background */
--color-surface:      #151820;   /* sidebar, panels */
--color-card:         #1c1f2e;   /* card/item background */
--color-card-hover:   #232638;   /* card hover */
--color-border:       #2a2d3e;   /* all borders */
--color-border-light: #353952;   /* subtle dividers */

--color-accent:       #3b82f6;   /* primary action, links (blue-500) */
--color-accent-dim:   #1d4ed8;   /* darker accent */
--color-violet:       #8b5cf6;   /* secondary accent (faces) */
--color-green:        #22c55e;   /* arrival, online, success */
--color-amber:        #f59e0b;   /* warning, unknown */
--color-red:          #ef4444;   /* danger, departure, error */
--color-teal:         #14b8a6;   /* vehicle detections */

--color-text-primary:   #f1f5f9;
--color-text-secondary: #94a3b8;
--color-text-muted:     #4b5563;

/* Status badge colors */
--badge-motion:    bg:#1e3a5f text:#60a5fa;     /* blue */
--badge-arrival:   bg:#14532d text:#4ade80;     /* green */
--badge-departure: bg:#450a0a text:#fca5a5;     /* red */
--badge-ding:      bg:#4a1d96 text:#c4b5fd;     /* violet */
--badge-unknown:   bg:#451a03 text:#fcd34d;     /* amber */
--badge-vehicle:   bg:#134e4a text:#2dd4bf;     /* teal */
--badge-person:    bg:#312e81 text:#a5b4fc;     /* indigo */
```

### Typography
- Font: `Inter` (via CDN or self-hosted)
- Monospace: `JetBrains Mono` (for timestamps, IDs)
- Scale: Tailwind defaults (sm=12px, base=14px, lg=16px, xl=18px)

### Spacing & Layout
- Sidebar width: 220px (collapsed: 56px)
- Content max-width: 1440px
- Card border-radius: 8px
- Consistent 16px gutters

### Iconography
- Use `lucide-svelte` for all icons (consistent, tree-shakeable)
- Key icons: `Camera`, `Car`, `User`, `Bell`, `Clock`, `Activity`, `Settings`, `Shield`, `Zap`

---

## 4. Pages — Wireframes & Behavior

---

### 4A. Live Dashboard — `/`

**Purpose:** Primary view — shows real-time event feed + active visit status.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR │              HEADER: "Live Feed"        [Status: ● Online]│
│         │──────────────────────────────────────────────────────────│
│ ● Live  │  ┌──────────────────────────────────┐  ┌───────────────┐ │
│   Feed  │  │  ACTIVE VISITS (if any)           │  │  SYSTEM       │ │
│         │  │  ┌──────────────┐  ┌───────────┐ │  │  STATUS       │ │
│ ⊙ Hist. │  │  │ David ● 12m  │  │ Cleaners  │ │  │               │ │
│         │  │  │ Front Door   │  │ ● 3m      │ │  │ Cameras: 1    │ │
│ ♦ Prof. │  │  └──────────────┘  └───────────┘ │  │ Models: OK    │ │
│         │  └──────────────────────────────────┘  │ DB: OK        │ │
│ ≋ Unmtc │                                         │ VRAM: 8.2GB   │ │
│         │  FILTER BAR: [All Cameras ▾] [All Types ▾] [↺ Live]     │ │
│ ⌗ Analy │                                                          │ │
│         │  ┌──────────────────────────────────────────────────────┐│ │
│ ⚙ Sett. │  │  EVENT FEED (newest at top, infinite scroll)         ││ │
│         │  │                                                       ││ │
│         │  │  ┌──────────────────────────────────────────────────┐││ │
│         │  │  │ ● ARRIVAL  Front Door  14:32:01                  │││ │
│         │  │  │ [snapshot 160×90]  David arrived (car, person)   │││ │
│         │  │  │ [car] [person] [David]                           │││ │
│         │  │  └──────────────────────────────────────────────────┘││ │
│         │  │                                                       ││ │
│         │  │  ┌──────────────────────────────────────────────────┐││ │
│         │  │  │ ◌ MOTION   Front Door  14:31:55                  │││ │
│         │  │  │ [snapshot]  car, person x2                       │││ │
│         │  │  │ [car] [person]                 [Create Profile?] │││ │
│         │  │  └──────────────────────────────────────────────────┘││ │
│         │  │                                                       ││ │
│         │  │  ┌──────────────────────────────────────────────────┐││ │
│         │  │  │ ◎ DING     Back Camera  14:28:11                  │││ │
│         │  │  │ [snapshot]  Doorbell ring                        │││ │
│         │  │  └──────────────────────────────────────────────────┘││ │
│         │  └──────────────────────────────────────────────────────┘│ │
└─────────────────────────────────────────────────────────────────────┘
```

**Event Card Anatomy:**
- Left edge: colored bar (green=arrival, red=departure, blue=motion, violet=ding, amber=unknown)
- Badge: event type
- Camera name + timestamp (relative: "2 min ago", absolute on hover)
- Thumbnail: 160×90 crop of snapshot. Click → full-size overlay with detection boxes drawn as SVG overlays
- Detection tags: badge chips for each unique class (car, person, David, Unknown Visitor)
- Detection summary text (or VLM caption when available)
- For unmatched persons: subtle "Add to profiles →" link (takes to Unmatched page with this crop pre-selected)

**Active Visits Banner:**
- Only shown when `GET /api/visits/active` returns entries
- Each active visit: name + camera + elapsed time (ticking live via JS)
- Click: opens Visit History filtered to that name

**Filters:**
- Camera dropdown: populated from `GET /api/cameras`
- Type: All / Motion / Arrivals / Departures / Doorbell
- Live toggle: when ON, WebSocket drives the feed. When OFF, static snapshot of last N events.

**Empty State:** No events yet → centered illustration with "Waiting for Ring events..." and animated pulsing ring icon.

**Loading State:** Skeleton cards (shimmer effect) while initial data loads.

---

### 4B. Visit History — `/history`

**Purpose:** Full timeline of all visits, filterable and searchable.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR │   "Visit History"           Today: 4 visits │ 3 departures│
│─────────│────────────────────────────────────────────────────────── │
│         │  FILTER BAR                                               │
│         │  [Search: name...] [Date: Today ▾] [Type: All ▾]         │
│         │  [Known / Unknown] [Vehicle / Person / All]               │
│         │                                                           │
│         │  ── TODAY, March 20 ──────────────────────────────────── │
│         │                                                           │
│         │  ┌────────────────────────────────────────────────────── │
│         │  │ [snapshot]  David (My Car)          ● ACTIVE          │
│         │  │             Front Door                                 │
│         │  │  Arrived: 14:32  Duration: ~4 min (ongoing)           │
│         │  │  [View event →]                                        │
│         │  └────────────────────────────────────────────────────── │
│         │                                                           │
│         │  ┌────────────────────────────────────────────────────── │
│         │  │ [snapshot]  Cleaners               ✓ DEPARTED         │
│         │  │             Front Door                                 │
│         │  │  Arrived: 10:15  Departed: 11:22  Duration: 67 min    │
│         │  │  [View event →]                                        │
│         │  └────────────────────────────────────────────────────── │
│         │                                                           │
│         │  ── MARCH 19 ─────────────────────────────────────────── │
│         │  ...                                                      │
│         │                                                           │
│         │  [Load more]                                              │
└─────────────────────────────────────────────────────────────────────┘
```

**Visit Card Anatomy:**
- Snapshot thumbnail (from `visit_events.snapshot_path`) — click to view full size
- Name (display_name), category badge (vehicle/person)
- Camera name
- Arrived / Departed times + computed duration
- Status badge: ACTIVE (green pulsing dot) or DEPARTED (checkmark)
- "View event →" link: opens the corresponding event in the live feed

**Stats Header:**
- Today's summary: N visits, N departures, N unknown visitors
- Longest visit of day

**Filters:**
- Date picker: Today / Yesterday / Last 7 days / Last 30 days / Custom range
- Name search (debounced 300ms)
- Type: Vehicles / Faces / All
- Status: Active / Departed / All

**Empty State:** "No visits in this date range." with suggestion to check if `ring-watch` is running.

---

### 4C. Profiles — `/profiles`

**Purpose:** Manage known vehicle and person profiles.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR │   "Profiles"      [Vehicles (3)]  [People (2)]  [+ Add]  │
│─────────│────────────────────────────────────────────────────────── │
│         │                                                           │
│         │  ── VEHICLES ─────────────────────────────────────────── │
│         │                                                           │
│         │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│         │  │[car img] │  │[car img] │  │[car img] │  │    +     │ │
│         │  │          │  │          │  │          │  │  Add New │ │
│         │  │ My Car   │  │ Cleaners │  │ Mechanic │  │          │ │
│         │  │ 23 visits│  │ 8 visits │  │ 1 visit  │  │          │ │
│         │  │ [Edit]   │  │ [Edit]   │  │ [Edit]   │  │          │ │
│         │  └──────────┘  └──────────┘  └──────────┘  └──────────┘ │
│         │                                                           │
│         │  ── PEOPLE ───────────────────────────────────────────── │
│         │                                                           │
│         │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│         │  │[face img]│  │[face img]│  │    +     │               │
│         │  │          │  │          │  │  Add New │               │
│         │  │ David    │  │ Jane     │  │          │               │
│         │  │ 12 visits│  │ 3 visits │  │          │               │
│         │  │ [Edit]   │  │ [Edit]   │  │          │               │
│         │  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

**Profile Card:**
- Thumbnail: best sample image (vehicle crop or face crop stored in profile)
- Display name (large)
- Category badge (vehicle/person)
- Visit count + last seen timestamp
- [Edit] button: opens ProfileDetailModal
- Long-press / right-click: [Delete profile]

**ProfileDetailModal (slide-over panel):**
```
┌─────────────────────────────────────┐
│ ← Back    David (My Car)       [×] │
│─────────────────────────────────────│
│  [reference photo / face crop]      │
│                                     │
│  Display Name: [____________]       │
│  Category:     [Vehicle ▾]          │
│  Reference Key: my_car              │
│                                     │
│  Match Threshold: [0.85 ──●── 1.0]  │
│  (similarity required for a match)  │
│                                     │
│  ── RECENT DETECTIONS ───────────── │
│  [snap] 14:32 Front Door            │
│  [snap] 10:15 Front Door            │
│  [snap] Yesterday 09:00             │
│                                     │
│  [Save Changes]  [Delete Profile]   │
└─────────────────────────────────────┘
```

**Add Profile Flow (Vehicles):**
1. Click "+" card → "Add Vehicle Profile" modal
2. Upload reference photos (drag-and-drop, multi-file)
3. Preview extracted detection crops
4. Enter display name, category
5. Submit → calls `POST /api/references` which runs YOLO+embedding pipeline
6. Success toast "Profile created: Cleaners"

**Add Profile Flow (Faces):**
1. Click "+" card → "Add Person Profile" modal
2. Upload photo(s) containing the person's face
3. Preview detected face crops
4. Enter display name
5. Submit → calls `POST /api/faces`
6. Success toast "Face profile created: David"

**Empty State:**
- "No profiles yet" with description of what profiles do, with "Add your first profile" CTA

---

### 4D. Unmatched Detections — `/unmatched`

**Purpose:** Gallery of detections that didn't match any known profile. One-click to create profile from detection.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR │   "Unmatched Detections"    [Vehicles (12)]  [Faces (5)]  │
│─────────│────────────────────────────────────────────────────────── │
│         │  Vehicles seen 2+ times in the last 30 days              │
│         │                                                           │
│         │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│         │  │ [car crop]   │  │ [car crop]   │  │ [car crop]   │   │
│         │  │ 8 sightings  │  │ 3 sightings  │  │ 2 sightings  │   │
│         │  │ Last: 2h ago │  │ Last: 1d ago │  │ Last: 3d ago │   │
│         │  │ Front Door   │  │ Front Door   │  │ Back Camera  │   │
│         │  │ [+ Add Prof] │  │ [+ Add Prof] │  │ [+ Add Prof] │   │
│         │  │ [  Dismiss ] │  │ [  Dismiss ] │  │ [  Dismiss ] │   │
│         │  └──────────────┘  └──────────────┘  └──────────────┘   │
│         │                                                           │
│         │  ── FACES (Unmatched) ──────────────────────────────────  │
│         │                                                           │
│         │  ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│         │  │[face crp]│  │[face crp]│  │[face crp]│               │
│         │  │ 3 sights │  │ 1 sights │  │ 1 sights │               │
│         │  │[+Profile]│  │[Dismiss] │  │[Dismiss] │               │
│         │  └──────────┘  └──────────┘  └──────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

**How "Unmatched Vehicles" is derived:**
Query `embeddings` records of type `detection` (crop embeddings) where the associated `event` has no `reference_name`. Group by embedding similarity (cluster visually similar crops). The backend clusters these using pgvector cosine similarity — crops within 0.15 distance of each other are the same vehicle.

**"+ Add Profile" Action:**
1. Opens AddProfileModal pre-filled with that crop
2. User enters display name
3. Submit → creates reference using that crop's embedding as the starting vector
4. Card disappears from Unmatched gallery

**"Dismiss" Action:**
- Marks a hidden_detections record so it doesn't re-appear (need a `dismissed_detections` simple table, or a JSON list in settings)
- Dismissed items shown in a collapsed "Dismissed" section at bottom

**Empty State:** "No unmatched detections" — all vehicles and people are identified.

---

### 4E. Analytics — `/analytics`

**Purpose:** Visual summary of activity patterns.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR │   "Analytics"        [Last 7 days ▾]                      │
│─────────│────────────────────────────────────────────────────────── │
│         │  ┌────────────────────────────────┐  ┌──────────────────┐│
│         │  │ EVENTS PER DAY                 │  │ TOP VISITORS     ││
│         │  │ ███▉▉█████▉ (bar chart)        │  │ David     23 vis ││
│         │  │ Mon Tue Wed Thu Fri Sat Sun     │  │ Cleaners  8 vis  ││
│         │  └────────────────────────────────┘  │ Mechanic  1 vis  ││
│         │                                       └──────────────────┘│
│         │  ┌────────────────────────────────┐  ┌──────────────────┐│
│         │  │ ACTIVITY BY HOUR OF DAY        │  │ DETECTION TYPES  ││
│         │  │ (heatmap: darker = more events)│  │ car     64%      ││
│         │  │ 00 01 02 03 ... 23             │  │ person  30%      ││
│         │  └────────────────────────────────┘  │ truck   6%       ││
│         │                                       └──────────────────┘│
│         │  ┌─────────────────────────────────────────────────────┐  │
│         │  │ VISIT DURATION DISTRIBUTION                         │  │
│         │  │ (histogram: 0-5min, 5-15min, 15-60min, 60+ min)    │  │
│         │  └─────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

**Charts library:** `Chart.js` via `svelte-chartjs` wrapper.

**Endpoints needed:**
- `GET /api/analytics/events-per-day?days=7` → `{date, count}[]`
- `GET /api/analytics/activity-heatmap?days=7` → `{hour, count}[]`
- `GET /api/analytics/top-visitors?days=30&limit=10` → `{display_name, visit_count, last_seen}[]`
- `GET /api/analytics/detection-types?days=7` → `{class_name, count}[]`
- `GET /api/analytics/visit-durations?days=30` → histogram buckets

**Empty State:** "Not enough data yet — check back after a week of monitoring."

---

### 4F. Settings — `/settings`

**Purpose:** View and configure the system. Organized into tabs.

```
┌─────────────────────────────────────────────────────────────────────┐
│ SIDEBAR │   "Settings"   [Detection] [Notifications] [Storage] [System]│
│─────────│────────────────────────────────────────────────────────── │
│         │                                                           │
│         │  ── DETECTION TAB ──────────────────────────────────────  │
│         │                                                           │
│         │  Vehicle Match Threshold  [0.85 ───●──── 1.0]            │
│         │  (higher = stricter matching)                             │
│         │                                                           │
│         │  Motion Cooldown          [____30____] seconds            │
│         │  (ignore duplicate motion events within this window)      │
│         │                                                           │
│         │  Departure Timeout        [____300___] seconds (5 min)   │
│         │                                                           │
│         │  YOLO Model               [yolov8m.pt] ● Loaded          │
│         │  Device                   [cuda:0    ] ● Available        │
│         │                                                           │
│         │  Face Detection           [● Enabled] [SCRFD-10G]        │
│         │  Face Match Threshold     [0.6 ──●─── 1.0]               │
│         │  Min Face Size            [____50____] pixels             │
│         │                                                           │
│         │  Captioner (Gemma 3)      [○ Disabled]                   │
│         │    Ollama URL             [http://localhost:11434]        │
│         │                                                           │
│         │  [Save Detection Settings]                                │
│         │                                                           │
│         │  ── SYSTEM TAB (read-only health) ─────────────────────── │
│         │  [same output as ring-status CLI, refreshed every 30s]   │
└─────────────────────────────────────────────────────────────────────┘
```

**Settings Tabs:**

1. **Detection** — thresholds, cooldowns, model paths, face detection toggle
2. **Notifications** — ntfy URL, attach snapshots toggle, priority levels per event type
3. **Storage** — archive dir path, show disk usage, cleanup old snapshots
4. **System** — live health check (DB connectivity, GPU VRAM, Ring token status, Ollama)

**Implementation note:** Settings reads from the running config (already loaded from `.env`). Writes update a `config_overrides` JSON file or directly patches `.env`. On save, display a warning: "Ring Watch must be restarted to apply changes."

---

## 5. FastAPI Backend Specification

### Project Structure

```
backend/
├── app/
│   ├── main.py              # FastAPI app + startup
│   ├── database.py          # Import shared DB session (from ring_detector.database)
│   ├── websocket.py         # WebSocket hub
│   ├── routers/
│   │   ├── events.py        # GET /api/events, GET /api/events/{id}
│   │   ├── visits.py        # GET /api/visits, /api/visits/active
│   │   ├── references.py    # CRUD /api/references
│   │   ├── faces.py         # CRUD /api/faces
│   │   ├── unmatched.py     # GET /api/unmatched/vehicles, /api/unmatched/faces
│   │   ├── analytics.py     # GET /api/analytics/*
│   │   ├── status.py        # GET /api/status, /api/cameras
│   │   ├── settings.py      # GET/PATCH /api/settings
│   │   └── images.py        # GET /api/images/{path:path}
│   └── schemas.py           # Pydantic response models
├── Dockerfile
└── requirements.txt
```

### Core Dependencies

```
fastapi>=0.111
uvicorn[standard]>=0.29
python-multipart      # file uploads
aiofiles              # async file serving
sqlalchemy>=2.0
pgvector
psycopg2-binary
python-dotenv
```

### Endpoint Specifications

---

#### Events

```
GET /api/events
  Query: camera=str, type=str (motion|ding|arrival|departure), limit=int (default 50),
         offset=int, from=datetime, to=datetime
  Response: { total: int, items: EventResponse[] }

EventResponse {
  id: int
  event_type: "motion" | "ding" | "arrival" | "departure"
  camera_name: str
  occurred_at: datetime (ISO 8601)
  snapshot_url: str | null        # /api/images/{encoded_path}
  detection_summary: str | null
  reference_name: str | null
  display_name: str | null
  caption: str | null
  detections: DetectionResponse[]  # joined from detections table via file_uuid
  visit_id: int | null
}

DetectionResponse {
  class_name: str
  class_id: int
  confidence: float  # 0–100
  xcenter: float     # normalized 0–1
  ycenter: float
  width: float
  height: float
}

GET /api/events/{event_id}
  Response: EventResponse (same as above, with full detections list)
```

---

#### Visits

```
GET /api/visits
  Query: limit=50, offset=0, from=datetime, to=datetime, name=str,
         active_only=bool, type=vehicle|face|all
  Response: { total: int, items: VisitResponse[] }

VisitResponse {
  id: int
  reference_name: str
  display_name: str
  camera_name: str
  arrived_at: datetime
  last_motion_at: datetime
  departed_at: datetime | null
  duration_minutes: int | null     # null if still active
  snapshot_url: str | null
  is_active: bool
}

GET /api/visits/active
  Response: VisitResponse[]

GET /api/visits/{visit_id}
  Response: VisitResponse
```

---

#### References (Vehicle Profiles)

```
GET /api/references
  Response: ReferenceResponse[]

ReferenceResponse {
  uuid: str
  name: str
  display_name: str
  category: str
  visit_count: int         # count from visit_events
  last_seen: datetime | null
  sample_image_url: str | null
}

POST /api/references
  Body: multipart/form-data { name: str, display_name: str, category: str,
                               images: File[] }
  Action: runs YOLO detection + embedding pipeline on uploaded images
  Response: ReferenceResponse

PUT /api/references/{name}
  Body: { display_name?: str, category?: str }
  Response: ReferenceResponse

DELETE /api/references/{name}
  Response: { success: bool }
```

---

#### Face Profiles

```
GET /api/faces
  Response: FaceProfileResponse[]

FaceProfileResponse {
  uuid: str
  name: str
  display_name: str
  created_at: datetime
  visit_count: int
  last_seen: datetime | null
  sample_image_url: str | null
}

POST /api/faces
  Body: multipart/form-data { name: str, display_name: str, image: File }
  Action: runs SCRFD face detection + ArcFace embedding
  Response: FaceProfileResponse

PUT /api/faces/{name}
  Body: { display_name?: str }
  Response: FaceProfileResponse

DELETE /api/faces/{name}
  Response: { success: bool }
```

---

#### Unmatched Detections

```
GET /api/unmatched/vehicles
  Query: min_sightings=2, days=30, limit=50
  Response: UnmatchedVehicle[]

UnmatchedVehicle {
  cluster_id: str       # hash of representative embedding
  sighting_count: int
  first_seen: datetime
  last_seen: datetime
  camera_name: str
  representative_crop_url: str    # best crop image
  all_crop_urls: str[]            # all crops in this cluster
}

GET /api/unmatched/faces
  Query: days=30, limit=50
  Response: UnmatchedFace[]

UnmatchedFace {
  face_embedding_uuid: str
  sighting_count: int
  last_seen: datetime
  camera_name: str
  crop_url: str
}

POST /api/unmatched/dismiss
  Body: { type: "vehicle" | "face", id: str }
  Response: { success: bool }
```

---

#### Analytics

```
GET /api/analytics/events-per-day?days=7
  Response: { date: str, count: int }[]

GET /api/analytics/activity-heatmap?days=7
  Response: { hour: int, count: int }[]

GET /api/analytics/top-visitors?days=30&limit=10
  Response: { display_name: str, visit_count: int, last_seen: datetime }[]

GET /api/analytics/detection-types?days=7
  Response: { class_name: str, count: int, percentage: float }[]

GET /api/analytics/visit-durations?days=30
  Response: { bucket: str, count: int }[]
  # buckets: "0-5min", "5-15min", "15-60min", "1-4hr", "4hr+"
```

---

#### Status

```
GET /api/status
  Response: SystemStatus {
    database: ComponentStatus
    gpu: ComponentStatus
    yolo_model: ComponentStatus
    ring_token: ComponentStatus
    face_models: ComponentStatus
    ollama: ComponentStatus
    archive_dir: ComponentStatus
    watcher_running: bool       # check if ring-watch process is alive via PID file
    uptime_seconds: int         # FastAPI server uptime
  }

ComponentStatus {
  status: "ok" | "warn" | "fail" | "missing" | "off"
  detail: str
}

GET /api/cameras
  Response: { name: str, type: str }[]
  # Reads from Ring API or returns cached camera list from DB
```

---

#### Settings

```
GET /api/settings
  Response: SettingsResponse {
    ring: { camera_name, departure_timeout, cooldown_seconds }
    model: { yolo_model_path, device, batch_size }
    captioner: { enabled, ollama_url, model }
    notify: { ntfy_url, attach_snapshot }
    storage: { archive_dir }
    face: { enabled, match_threshold, min_face_size, backend }
  }

PATCH /api/settings
  Body: partial SettingsResponse (same structure, all fields optional)
  Action: writes to config_overrides.json (not .env directly)
  Response: { success: bool, restart_required: bool }
```

---

#### Image Serving

```
GET /api/images/{path:path}
  Action: serves image from filesystem with path validation
  Security: constrain to ARCHIVE_DIR + models dir only (prevent path traversal)
  Headers: Cache-Control: max-age=86400 (immutable snapshots)
  Response: image/jpeg or image/png
```

---

### WebSocket

```
WS /api/ws
  Protocol: JSON messages
  On connect: send current system status snapshot
  On event: broadcast to all connected clients
```

---

## 6. WebSocket Protocol

### Client → Server Messages

```json
{ "type": "subscribe", "cameras": ["Front Door", "Back Camera"] }
{ "type": "unsubscribe", "cameras": ["Back Camera"] }
{ "type": "ping" }
```

### Server → Client Messages

```json
// New Ring event received
{
  "type": "event",
  "data": {
    "id": 1234,
    "event_type": "arrival",
    "camera_name": "Front Door",
    "occurred_at": "2026-03-20T14:32:01",
    "snapshot_url": "/api/images/snapshots/2026-03-20/Front_Door_14-32-01.jpg",
    "detection_summary": "car, person",
    "display_name": "David",
    "detections": [
      { "class_name": "car", "confidence": 92, "xcenter": 0.45, "ycenter": 0.6, "width": 0.4, "height": 0.3 },
      { "class_name": "person", "confidence": 88, "xcenter": 0.7, "ycenter": 0.5, "width": 0.12, "height": 0.35 }
    ]
  }
}

// Visit started
{
  "type": "visit_started",
  "data": {
    "visit_id": 42,
    "display_name": "David",
    "camera_name": "Front Door",
    "arrived_at": "2026-03-20T14:32:01"
  }
}

// Visit ended
{
  "type": "visit_ended",
  "data": {
    "visit_id": 42,
    "display_name": "David",
    "duration_minutes": 12
  }
}

// System status changed
{
  "type": "status",
  "data": {
    "database": { "status": "ok", "detail": "3 references" },
    "gpu": { "status": "ok", "detail": "RTX A6000 (48 GB)" }
  }
}

// Pong
{ "type": "pong" }
```

### WebSocket Bridge Architecture

The FastAPI server needs to receive events from the `ring-watch` process. Two options:

**Option A (same process — recommended for simplicity):** Run FastAPI inside the same Python process as `ring-watch`. The watcher publishes to an `asyncio.Queue` that the WebSocket hub subscribes to. This requires the watcher to be refactored to expose a shared event bus.

**Option B (separate container — recommended for production):** Use PostgreSQL `LISTEN/NOTIFY`. When `ring-watch` inserts a new `events` row, it runs `pg_notify('ring_events', payload_json)`. The FastAPI backend subscribes with `asyncpg.connect()` and broadcasts to WebSocket clients.

**Recommended: Option B** — cleaner separation, no code changes to ring-watch needed, works across Docker containers.

Implementation sketch:
```python
# In FastAPI startup
async def listen_for_pg_events():
    conn = await asyncpg.connect(settings.db.url)
    await conn.add_listener("ring_events", ws_hub.broadcast_from_pg)

# In ring-watch database.py record_event():
    conn.execute(text(
        "SELECT pg_notify('ring_events', :payload)"
    ), {"payload": json.dumps(event_dict)})
```

---

## 7. Frontend Component Hierarchy

```
src/
├── routes/
│   ├── +layout.svelte          # App shell: sidebar + header
│   ├── +page.svelte            # Live Dashboard
│   ├── history/
│   │   └── +page.svelte        # Visit History
│   ├── profiles/
│   │   └── +page.svelte        # Known Profiles
│   ├── unmatched/
│   │   └── +page.svelte        # Unmatched Detections
│   ├── analytics/
│   │   └── +page.svelte        # Analytics
│   └── settings/
│       └── +page.svelte        # Settings
│
└── lib/
    ├── components/
    │   ├── layout/
    │   │   ├── Sidebar.svelte          # Nav links + collapse toggle
    │   │   ├── SidebarLink.svelte      # Single nav item with icon + label
    │   │   └── Header.svelte           # Page title + status indicator
    │   │
    │   ├── ui/
    │   │   ├── Badge.svelte            # Colored chip: [motion] [arrival] [car]
    │   │   ├── Button.svelte           # Primary/secondary/danger variants
    │   │   ├── Card.svelte             # Surface card with optional hover state
    │   │   ├── EmptyState.svelte       # Centered illustration + message + CTA
    │   │   ├── Modal.svelte            # Backdrop + centered dialog
    │   │   ├── SlideOver.svelte        # Right-panel slide-in (profile detail)
    │   │   ├── Spinner.svelte          # Loading spinner
    │   │   ├── SkeletonCard.svelte     # Shimmer loading placeholder
    │   │   ├── StatusDot.svelte        # ● colored dot for live status
    │   │   ├── Toast.svelte            # Success/error notification
    │   │   └── Tooltip.svelte          # Hover tooltip
    │   │
    │   ├── events/
    │   │   ├── EventFeed.svelte        # Scrollable list of EventCard
    │   │   ├── EventCard.svelte        # Single event: thumbnail + badges + summary
    │   │   ├── EventFilter.svelte      # Camera + type filter bar
    │   │   ├── DetectionOverlay.svelte # SVG bboxes drawn over snapshot image
    │   │   └── SnapshotModal.svelte    # Full-size image viewer with overlays
    │   │
    │   ├── visits/
    │   │   ├── VisitCard.svelte        # Single visit: name + times + duration
    │   │   ├── VisitFeed.svelte        # List of VisitCard grouped by date
    │   │   ├── VisitFilter.svelte      # Date range + name search + type filter
    │   │   └── ActiveVisitBanner.svelte # Top-of-dashboard current active visits
    │   │
    │   ├── profiles/
    │   │   ├── ProfileGrid.svelte      # Responsive grid of ProfileCard
    │   │   ├── ProfileCard.svelte      # Thumbnail + name + visit count
    │   │   ├── ProfileDetailPanel.svelte # SlideOver with edit form + recent detections
    │   │   ├── AddVehicleModal.svelte  # Upload photos + name form
    │   │   └── AddFaceModal.svelte     # Upload photo + name form
    │   │
    │   ├── unmatched/
    │   │   ├── UnmatchedGrid.svelte    # Grid of unmatched crops
    │   │   ├── UnmatchedCard.svelte    # Crop image + sighting count + actions
    │   │   └── CreateFromUnmatched.svelte # Modal: name input → create profile
    │   │
    │   ├── analytics/
    │   │   ├── EventsPerDayChart.svelte  # Bar chart
    │   │   ├── ActivityHeatmap.svelte   # Hour-of-day heatmap
    │   │   ├── TopVisitorsList.svelte   # Ranked list
    │   │   ├── DetectionTypesChart.svelte # Donut chart
    │   │   └── VisitDurationsChart.svelte # Histogram
    │   │
    │   └── status/
    │       ├── SystemStatusPanel.svelte  # Grid of ComponentStatus cards
    │       └── ComponentStatusCard.svelte # Individual component health row
    │
    ├── stores/
    │   ├── websocket.ts        # WebSocket connection + reconnect logic
    │   ├── events.ts           # Event list + real-time updates
    │   ├── visits.ts           # Active + historical visits
    │   ├── status.ts           # System health polling
    │   └── toast.ts            # Toast notification queue
    │
    ├── api/
    │   ├── client.ts           # Axios instance with base URL
    │   ├── events.ts           # API functions for /api/events
    │   ├── visits.ts           # API functions for /api/visits
    │   ├── references.ts       # API functions for /api/references
    │   ├── faces.ts            # API functions for /api/faces
    │   ├── analytics.ts        # API functions for /api/analytics
    │   └── status.ts           # API functions for /api/status
    │
    └── types/
        ├── event.ts            # EventResponse, DetectionResponse
        ├── visit.ts            # VisitResponse
        ├── profile.ts          # ReferenceResponse, FaceProfileResponse
        └── status.ts           # SystemStatus, ComponentStatus
```

---

## 8. SvelteKit Store Architecture

### WebSocket Store (`stores/websocket.ts`)

```typescript
// Manages single WebSocket connection with auto-reconnect
// Dispatches typed messages to subscriber callbacks
export const wsStore = createWebSocketStore({
  url: () => `ws://${window.location.hostname}:8000/api/ws`,
  reconnect: { initialDelay: 1000, maxDelay: 30000, backoff: 2 }
});

// Usage in components:
wsStore.on('event', (data) => eventsStore.prepend(data));
wsStore.on('visit_started', (data) => visitsStore.addActive(data));
wsStore.on('visit_ended', (data) => visitsStore.markDeparted(data));
```

### Events Store (`stores/events.ts`)

```typescript
// Writable list of EventResponse with real-time prepend
// Backed by initial REST fetch + WebSocket updates
export const eventsStore = {
  subscribe,           // Svelte readable
  load(filters),       // GET /api/events → replaces list
  prepend(event),      // WebSocket new event → add to top
  loadMore(),          // Infinite scroll
};
```

### Status Store (`stores/status.ts`)

```typescript
// Polls GET /api/status every 30s + WebSocket status updates
// Provides derived values: isOnline, cameraCount, etc.
export const statusStore = createPollingStore({
  url: '/api/status',
  interval: 30_000
});
```

---

## 9. Docker & Infrastructure

### New Docker Services

Add to `docker-compose.yml`:

```yaml
services:
  # existing:
  db:
    ...

  # NEW: FastAPI dashboard backend
  api:
    container_name: ring-api
    build:
      context: ./backend
      dockerfile: Dockerfile
    restart: always
    env_file: .env
    ports:
      - "${API_PORT:-8000}:8000"
    volumes:
      - ${ARCHIVE_DIR:-/mnt/nas/ring_archive}:/archive:ro  # read-only snapshot serving
      - ./tokens:/tokens:ro
      - ./models:/models:ro
    environment:
      - DB_HOST=db
      - DB_PORT=5432
      - ARCHIVE_DIR=/archive
    depends_on:
      db:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/status"]
      interval: 10s
      timeout: 5s
      retries: 5

  # NEW: SvelteKit frontend (nginx in production)
  frontend:
    container_name: ring-frontend
    build:
      context: ./frontend
      dockerfile: Dockerfile
    restart: always
    ports:
      - "${FRONTEND_PORT:-3000}:80"
    environment:
      - PUBLIC_API_URL=http://api:8000
    depends_on:
      api:
        condition: service_healthy

volumes:
  ring_data:
```

### Backend Dockerfile (`backend/Dockerfile`)

```dockerfile
FROM python:3.12-slim

WORKDIR /app
RUN adduser --disabled-password --uid 1000 appuser

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install -e ..  # installs ring_detector package from repo root

USER appuser
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Frontend Dockerfile (`frontend/Dockerfile`)

```dockerfile
# Build stage
FROM node:22-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Serve stage
FROM nginx:alpine
COPY --from=build /app/build /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
EXPOSE 80
```

### nginx.conf for SPA routing

```nginx
server {
    listen 80;
    root /usr/share/nginx/html;
    index index.html;

    # API proxy
    location /api/ {
        proxy_pass http://api:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }

    # SPA fallback
    location / {
        try_files $uri $uri/ /index.html;
    }
}
```

---

## 10. Implementation Phases

### Phase 1 — Foundation (API + Scaffold + Live Events)

**Goal:** Working live event feed with real-time updates.

**Backend tasks:**
1. Add `events` table to `database.py` with `record_event()` helper
2. Update `watcher.py` to call `record_event()` at each stage
3. Create FastAPI app (`backend/app/main.py`) with:
   - `GET /api/events` endpoint (from `events` + `detections` join)
   - `GET /api/status` endpoint (mirrors `ring-status` CLI)
   - `GET /api/images/{path:path}` endpoint
   - `WS /api/ws` with PostgreSQL LISTEN/NOTIFY bridge
4. Docker setup: `ring-api` service

**Frontend tasks:**
1. SvelteKit project: `npm create svelte@latest frontend` (SPA mode)
2. Tailwind CSS v3 + dark theme config + design tokens
3. `+layout.svelte`: sidebar + header shell
4. `+page.svelte`: Live Dashboard
   - WebSocket store connecting to `/api/ws`
   - Initial load from `GET /api/events`
   - EventCard component with snapshot thumbnail
   - Detection bounding box overlay (SVG)
   - Status bar (cameras online, model loaded)
5. ActiveVisitBanner component

**Deliverable:** `http://localhost:3000` shows live event feed. New Ring events appear instantly.

---

### Phase 2 — Visit History + Profile Management

**Goal:** Full historical context + manage who's who.

**Backend tasks:**
1. `GET /api/visits` + `GET /api/visits/active` endpoints
2. `GET /api/references` + `POST` + `PUT` + `DELETE`
3. `GET /api/faces` + `POST` + `PUT` + `DELETE`
   - POST endpoints run YOLO/ArcFace pipeline on uploaded images
4. `GET /api/cameras` from Ring API (cached for 5 min)

**Frontend tasks:**
1. `/history` page: VisitFeed + VisitFilter + date grouping
2. `/profiles` page: ProfileGrid with tabs (Vehicles / People)
3. ProfileCard + ProfileDetailPanel (SlideOver)
4. AddVehicleModal: file upload + YOLO preview + name form
5. AddFaceModal: face crop preview + name form

**Deliverable:** User can view visit history, add/edit/delete vehicle and face profiles.

---

### Phase 3 — Unmatched Detections Gallery

**Goal:** One-click profile creation from unknown detections.

**Backend tasks:**
1. `GET /api/unmatched/vehicles` — clusters similar embeddings
2. `GET /api/unmatched/faces` — lists face embeddings without profile match
3. `POST /api/unmatched/dismiss`

**Frontend tasks:**
1. `/unmatched` page: Vehicles tab + Faces tab
2. UnmatchedCard: crop image, sighting count, camera, [+ Add Profile] + [Dismiss]
3. CreateFromUnmatched modal: name input → calls POST /api/references or /api/faces
4. Dismissed items section (collapsed by default)

**Deliverable:** Unknown vehicles that appear regularly get a prominent "Add Profile" CTA.

---

### Phase 4 — Settings + Analytics

**Goal:** Full visibility into system health + activity patterns.

**Backend tasks:**
1. All `GET /api/analytics/*` endpoints
2. `GET /api/settings` + `PATCH /api/settings`
3. Settings writes to `config_overrides.json` in project root

**Frontend tasks:**
1. `/analytics` page with all 5 Chart.js charts
2. `/settings` page with 4 tabs
3. ComponentStatusCard with color-coded health indicators
4. Settings save + "restart required" warning toast

**Deliverable:** User can see activity patterns and configure thresholds without touching `.env`.

---

### Phase 5 — Polish

**Goal:** Production quality, feels like a real app.

- Svelte `fly` / `fade` transitions on EventCards appearing in feed
- Skeleton loading states (shimmer) for all data-fetching components
- Responsive layout (collapses sidebar on mobile, stacks panels)
- PWA: `manifest.webmanifest` + service worker for offline fallback page
- Keyboard shortcuts: `?` help overlay, `L` toggle live/paused, `F` focus filter
- Snapshot zoom: pinch-to-zoom in SnapshotModal
- Export: CSV download from `/history` page
- Dark/light mode toggle (dark is default)
- Page titles + favicon (ring/camera icon)
- Print stylesheet for reports

---

## 11. File Structure

**Estimated counts after Phase 5:**

```
backend/                        (~15 files)
  app/
    main.py
    database.py
    websocket.py
    schemas.py
    routers/ (8 files)
  Dockerfile
  requirements.txt

frontend/                       (~70 files)
  src/
    routes/ (7 files)
    lib/
      components/ (~35 .svelte files)
      stores/ (5 .ts files)
      api/ (7 .ts files)
      types/ (4 .ts files)
  package.json
  tailwind.config.ts
  svelte.config.js
  vite.config.ts
  Dockerfile
  nginx.conf
```

**Total new files:** ~85
**Modified existing files:** `database.py` (add Event model), `watcher.py` (add record_event calls), `docker-compose.yml` (add api + frontend services)

---

## 12. Edge Cases & Empty States

| Scenario | Where | Behavior |
|---|---|---|
| Database empty | Live feed | "Waiting for Ring events..." + animated ring pulse |
| No cameras connected | Status bar | Orange warning badge "No cameras" + link to settings |
| Models not loaded | Status bar | Red badge "Models offline" — events still display, detection indicators hidden |
| ring-watch not running | Status bar | Yellow badge "Watcher stopped" — can still browse history |
| Snapshot file deleted | EventCard, VisitCard | Gray "Image unavailable" placeholder with camera icon |
| WebSocket disconnected | Header | Amber "Reconnecting..." badge — reverts to polling every 10s |
| WebSocket reconnects | Header | Badge disappears, feed resumes live mode |
| NAS unreachable | Images | 404 → placeholder shown, no crash |
| No visits in date range | Visit History | "No visits in this period" with different CTA based on filter |
| No profiles configured | Profiles page | "No profiles yet" card explaining what they are and how to add |
| Face detection disabled | Profiles → People tab | Shows "Face detection is disabled" with link to Settings |
| Captioner disabled | Event cards | Show raw detection summary instead of VLM caption |
| CUDA unavailable | Settings | "Using CPU" warning, inference will be slow |
| Ollama unreachable | Settings | "Ollama unreachable" warning with URL shown |
| Upload fails (too large) | Add profile modal | "Image too large, max 10MB" inline error |
| Upload: no vehicle detected | Add vehicle modal | "No vehicle found in these images" with suggestion |
| Upload: no face detected | Add face modal | "No face detected — try a clearer photo" |
| Visit deduplication | Visit History | Same vehicle re-detected within cooldown window: extends visit, no duplicate entry |
| Analytics: <2 days data | Analytics charts | Charts show available data + "More data needed for full charts" notice |

---

## 13. Color Tokens Reference

```typescript
// tailwind.config.ts — extend colors
export default {
  darkMode: 'class',
  content: ['./src/**/*.{svelte,ts}'],
  theme: {
    extend: {
      colors: {
        bg:        '#0d0f14',
        surface:   '#151820',
        card:      '#1c1f2e',
        'card-hover': '#232638',
        border:    '#2a2d3e',
        'border-light': '#353952',
        accent:    '#3b82f6',
        'accent-dim': '#1d4ed8',
        violet:    '#8b5cf6',
        ring: {
          50:  '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        }
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'slide-in':   'slideIn 0.2s ease-out',
        'fade-in':    'fadeIn 0.15s ease-out',
      }
    }
  }
}
```

### Badge Variant Map

```typescript
// components/ui/Badge.svelte
const BADGE_VARIANTS = {
  motion:    'bg-blue-950  text-blue-400  border-blue-800',
  arrival:   'bg-green-950 text-green-400 border-green-800',
  departure: 'bg-red-950   text-red-400   border-red-800',
  ding:      'bg-violet-950 text-violet-400 border-violet-800',
  unknown:   'bg-amber-950 text-amber-400 border-amber-800',
  vehicle:   'bg-teal-950  text-teal-400  border-teal-800',
  person:    'bg-indigo-950 text-indigo-400 border-indigo-800',
  face:      'bg-purple-950 text-purple-400 border-purple-800',
} as const;
```

---

## Summary

| Aspect | Decision |
|---|---|
| Frontend | SvelteKit SPA (adapter-static), served by nginx in prod |
| Styling | Tailwind CSS v3, dark theme only (toggle Phase 5), Inter font |
| Real-time | WebSocket + PostgreSQL LISTEN/NOTIFY bridge |
| Image serving | FastAPI static file endpoint, constrained to ARCHIVE_DIR |
| Profile creation | Upload → in-memory YOLO/ArcFace pipeline → store in DB |
| Settings persistence | `config_overrides.json` (ring-watch reads on startup) |
| Unmatched detection discovery | pgvector cosine clustering, no new ML needed |
| Container count | +2 new containers (ring-api, ring-frontend) |
| Auth | None — single user, local network |
| Phase order | Live feed first, profiles second, gallery third, analytics fourth |
| Charts | Chart.js via svelte-chartjs |
| Icons | lucide-svelte |
| HTTP client | Axios |
