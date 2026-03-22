"""Context builder for chat: tiered summarization, token estimation, prompt assembly.

Queries existing DB tables and assembles compact text context for Gemma 3 4B.
Token budget: ~1800-2000 tokens for DB context within Gemma's 8K window.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from urllib.parse import quote

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.chat_entities import ChatIntent
from ring_detector.database import Event, FaceProfile, Reference, VisitEvent

log = logging.getLogger(__name__)

TOKEN_BUDGET = 2000
SYSTEM_PROMPT_TOKENS = 300
MAX_EVENT_DETAIL = 20
MAX_EVENT_REDUCED = 10

SYSTEM_PROMPT_TEMPLATE = """\
You are the OpenRingDetector assistant. You answer questions about security \
camera activity ONLY using the data provided below. If the data doesn't \
contain enough information, say so. Never invent events, times, or counts.

Current time: {now}

RULES:
- When referencing events, include timestamps
- Durations: "X minutes" for <60min, "X hours Y minutes" for longer
- Times: 12-hour format with am/pm
- Dates: "Today", "Yesterday", or "Weekday, Month Day"
- Use exact numbers, not approximations
- Reference event IDs when mentioning specific events (e.g., "[#42]")

{system_summary}
{context}"""


def _estimate_tokens(content: str) -> int:
    """Rough token estimate: ~3.5 chars per token for English."""
    return int(len(content) / 3.5)


def _format_time(dt: datetime) -> str:
    """Format datetime for display: '3:15 PM' or 'Yesterday 3:15 PM'."""
    now = datetime.now()
    if dt.date() == now.date():
        return dt.strftime("%-I:%M %p")
    if dt.date() == (now - timedelta(days=1)).date():
        return "Yesterday " + dt.strftime("%-I:%M %p")
    return dt.strftime("%a %b %-d %-I:%M %p")


def _snapshot_url(path: str | None) -> str | None:
    if not path:
        return None
    clean = path.lstrip("/")
    return f"/api/images/{quote(clean, safe='/')}"


def _format_duration(minutes: float) -> str:
    if minutes < 60:
        return f"{int(minutes)} minutes"
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    if mins == 0:
        return f"{hours} hours"
    return f"{hours} hours {mins} minutes"


# --- System Summary (always included, ~80 tokens) ---


def _build_system_summary(session: Session) -> str:
    """Quick system overview: camera count, reference count, face count, active visits."""
    cameras = session.execute(
        text("SELECT DISTINCT camera_name FROM events ORDER BY camera_name")
    ).fetchall()
    ref_count = session.query(Reference).count()
    face_count = session.query(FaceProfile).count()
    active = session.query(VisitEvent).filter(VisitEvent.departed_at.is_(None)).count()

    camera_names = ", ".join(r[0] for r in cameras) if cameras else "none"
    lines = [
        f"SYSTEM: Cameras: {camera_names}.",
        f"Known vehicles: {ref_count}. Known faces: {face_count}. Active visits: {active}.",
    ]
    return "\n".join(lines)


# --- Event Context (tiered by time distance) ---


def _get_recent_events_context(
    session: Session,
    from_dt: datetime | None,
    to_dt: datetime | None,
    entity_name: str | None = None,
    camera: str | None = None,
    limit: int = MAX_EVENT_DETAIL,
) -> str:
    """Detailed event rows for recent time range."""
    q = session.query(Event)
    if from_dt:
        q = q.filter(Event.occurred_at >= from_dt)
    if to_dt:
        q = q.filter(Event.occurred_at <= to_dt)
    if entity_name:
        q = q.filter(Event.reference_name == entity_name)
    if camera:
        q = q.filter(Event.camera_name == camera)

    total = q.count()
    events = q.order_by(Event.occurred_at.desc()).limit(limit).all()

    if not events:
        return "EVENTS: No events found for this time range."

    lines = [f"EVENTS ({total} total, showing {len(events)} most recent):"]
    for e in events:
        parts = [
            f"[#{e.id}]",
            _format_time(e.occurred_at),
            e.event_type.upper(),
            f"at {e.camera_name}",
        ]
        if e.display_name:
            parts.append(f"- {e.display_name}")
        if e.detection_summary:
            parts.append(f"({e.detection_summary[:50]})")
        if e.caption:
            parts.append(f'"{e.caption[:60]}"')
        lines.append(" | ".join(parts))

    return "\n".join(lines)


def _get_aggregated_events_context(
    session: Session,
    from_dt: datetime,
    to_dt: datetime,
    entity_name: str | None = None,
) -> str:
    """Daily aggregates for medium-range queries (2-7 days)."""
    params: dict = {"from_dt": from_dt, "to_dt": to_dt}
    entity_clause = ""
    if entity_name:
        entity_clause = "AND reference_name = :entity"
        params["entity"] = entity_name

    rows = session.execute(
        text(f"""
            SELECT DATE(occurred_at) AS d, event_type, COUNT(*) AS cnt
            FROM events
            WHERE occurred_at >= :from_dt AND occurred_at <= :to_dt {entity_clause}
            GROUP BY DATE(occurred_at), event_type
            ORDER BY d DESC, cnt DESC
        """),
        params,
    ).fetchall()

    if not rows:
        return ""

    # Group by date
    by_date: dict[str, list[str]] = {}
    for date_val, etype, cnt in rows:
        d = str(date_val)
        if d not in by_date:
            by_date[d] = []
        by_date[d].append(f"{cnt} {etype}")

    lines = ["DAILY SUMMARY:"]
    for d, parts in by_date.items():
        lines.append(f"  {d}: {', '.join(parts)}")
    return "\n".join(lines)


def _get_period_counts_context(
    session: Session,
    from_dt: datetime,
    to_dt: datetime,
) -> str:
    """Period-level counts for historical queries (8-30 days)."""
    total = (
        session.execute(
            text("SELECT COUNT(*) FROM events WHERE occurred_at >= :f AND occurred_at <= :t"),
            {"f": from_dt, "t": to_dt},
        ).scalar()
        or 0
    )

    arrivals = (
        session.execute(
            text(
                "SELECT COUNT(*) FROM events WHERE occurred_at >= :f AND occurred_at <= :t "
                "AND event_type = 'arrival'"
            ),
            {"f": from_dt, "t": to_dt},
        ).scalar()
        or 0
    )

    label = f"{from_dt.strftime('%b %d')}-{to_dt.strftime('%b %d')}"
    return f"PERIOD ({label}): {total} events, {arrivals} arrivals."


# --- Visit Context ---


def _get_visit_history_context(
    session: Session,
    from_dt: datetime | None,
    to_dt: datetime | None,
    entity_name: str | None = None,
    limit: int = 10,
) -> str:
    """Visit history with durations."""
    q = session.query(VisitEvent)
    if from_dt:
        q = q.filter(VisitEvent.arrived_at >= from_dt)
    if to_dt:
        q = q.filter(VisitEvent.arrived_at <= to_dt)
    if entity_name:
        q = q.filter(VisitEvent.reference_name == entity_name)

    visits = q.order_by(VisitEvent.arrived_at.desc()).limit(limit).all()
    if not visits:
        return "VISITS: No visits found for this time range."

    lines = ["VISITS:"]
    for v in visits:
        dur = ""
        if v.departed_at:
            minutes = (v.departed_at - v.arrived_at).total_seconds() / 60
            dur = f", stayed {_format_duration(minutes)}"
        status = "ACTIVE" if not v.departed_at else f"departed {_format_time(v.departed_at)}"
        lines.append(
            f"  {v.display_name} at {v.camera_name}: "
            f"arrived {_format_time(v.arrived_at)}, {status}{dur}"
        )

    return "\n".join(lines)


def _get_active_visits_context(session: Session) -> str:
    """Currently active visits (departed_at IS NULL)."""
    visits = session.query(VisitEvent).filter(VisitEvent.departed_at.is_(None)).all()
    if not visits:
        return "ACTIVE VISITS: No one is currently here."

    lines = ["ACTIVE VISITS:"]
    now = datetime.now()
    for v in visits:
        dur = (now - v.arrived_at).total_seconds() / 60
        lines.append(
            f"  {v.display_name} at {v.camera_name}: "
            f"arrived {_format_time(v.arrived_at)} ({_format_duration(dur)} ago)"
        )
    return "\n".join(lines)


# --- Statistics Context ---


def _get_statistics_context(
    session: Session,
    from_dt: datetime | None,
    to_dt: datetime | None,
) -> str:
    """Aggregated counts for statistics intents."""
    params: dict = {}
    where = "1=1"
    if from_dt:
        where += " AND occurred_at >= :from_dt"
        params["from_dt"] = from_dt
    if to_dt:
        where += " AND occurred_at <= :to_dt"
        params["to_dt"] = to_dt

    total = (
        session.execute(text(f"SELECT COUNT(*) FROM events WHERE {where}"), params).scalar() or 0
    )

    by_type = session.execute(
        text(
            f"SELECT event_type, COUNT(*) FROM events"
            f" WHERE {where} GROUP BY event_type ORDER BY COUNT(*) DESC"
        ),
        params,
    ).fetchall()

    by_camera = session.execute(
        text(
            f"SELECT camera_name, COUNT(*) FROM events"
            f" WHERE {where} GROUP BY camera_name ORDER BY COUNT(*) DESC"
        ),
        params,
    ).fetchall()

    lines = [f"STATISTICS ({total} total events):"]
    if by_type:
        lines.append("  By type: " + ", ".join(f"{t}: {c}" for t, c in by_type))
    if by_camera:
        lines.append("  By camera: " + ", ".join(f"{cam}: {c}" for cam, c in by_camera))

    # Visit stats
    v_where = "1=1"
    v_params: dict = {}
    if from_dt:
        v_where += " AND arrived_at >= :from_dt"
        v_params["from_dt"] = from_dt
    if to_dt:
        v_where += " AND arrived_at <= :to_dt"
        v_params["to_dt"] = to_dt

    visit_count = (
        session.execute(
            text(f"SELECT COUNT(*) FROM visit_events WHERE {v_where}"), v_params
        ).scalar()
        or 0
    )
    lines.append(f"  Visits: {visit_count}")

    return "\n".join(lines)


# --- Entity Info Context ---


def _get_entity_info_context(session: Session) -> str:
    """Configured references + face profiles."""
    refs = session.query(Reference).all()
    faces = session.query(FaceProfile).all()

    lines = ["CONFIGURED ENTITIES:"]
    if refs:
        lines.append(
            "  Vehicle references: "
            + ", ".join(f"{r.display_name} ({r.name}, {r.category})" for r in refs)
        )
    else:
        lines.append("  Vehicle references: none configured")
    if faces:
        lines.append("  Face profiles: " + ", ".join(f"{f.display_name} ({f.name})" for f in faces))
    else:
        lines.append("  Face profiles: none configured")
    return "\n".join(lines)


# --- Snapshot Lookup ---


def lookup_event_snapshots(session: Session, intent: ChatIntent, limit: int = 4) -> list[dict]:
    """Find snapshot URLs for image intents. Returns list of {url, alt, event_id}."""
    q = session.query(Event).filter(Event.snapshot_path.isnot(None))
    if intent.time_from:
        q = q.filter(Event.occurred_at >= intent.time_from)
    if intent.time_to:
        q = q.filter(Event.occurred_at <= intent.time_to)
    if intent.entity_name:
        q = q.filter(Event.reference_name == intent.entity_name)
    if intent.camera_filter:
        q = q.filter(Event.camera_name == intent.camera_filter)

    events = q.order_by(Event.occurred_at.desc()).limit(limit).all()
    results = []
    for e in events:
        url = _snapshot_url(e.snapshot_path)
        if url:
            results.append(
                {
                    "url": url,
                    "alt": f"{e.camera_name} - {_format_time(e.occurred_at)}",
                    "event_id": e.id,
                }
            )
    return results


def lookup_reference_info(session: Session, name: str) -> dict | None:
    """Look up a reference with visit count and last seen."""
    ref = session.query(Reference).filter_by(name=name).first()
    if not ref:
        return None

    visit_count = session.query(VisitEvent).filter_by(reference_name=name).count()
    last_visit = (
        session.query(VisitEvent)
        .filter_by(reference_name=name)
        .order_by(VisitEvent.arrived_at.desc())
        .first()
    )
    # Find thumbnail
    sample_event = (
        session.query(Event)
        .filter(Event.reference_name == name, Event.snapshot_path.isnot(None))
        .order_by(Event.occurred_at.desc())
        .first()
    )

    return {
        "name": ref.name,
        "display_name": ref.display_name,
        "category": ref.category,
        "image_url": _snapshot_url(sample_event.snapshot_path) if sample_event else None,
        "visit_count": visit_count,
        "last_seen": _format_time(last_visit.arrived_at) if last_visit else None,
    }


def lookup_face_info(session: Session, name: str) -> dict | None:
    """Look up a face profile with visit count and last seen."""
    fp = session.query(FaceProfile).filter_by(name=name).first()
    if not fp:
        return None

    # Face visits are tracked in events with reference_name (from face matches)
    event_count = session.query(Event).filter(Event.reference_name == name).count()
    last_event = (
        session.query(Event)
        .filter(Event.reference_name == name, Event.snapshot_path.isnot(None))
        .order_by(Event.occurred_at.desc())
        .first()
    )

    return {
        "name": fp.name,
        "display_name": fp.display_name,
        "image_url": _snapshot_url(last_event.snapshot_path) if last_event else None,
        "visit_count": event_count,
        "last_seen": _format_time(last_event.occurred_at) if last_event else None,
    }


# --- Main Context Builder ---


def build_chat_context(session: Session, intent: ChatIntent) -> str:
    """Build the full context string for a chat request, respecting token budget."""
    now = datetime.now()
    category = intent.category

    # Determine time span for tiered summarization
    from_dt = intent.time_from or (now - timedelta(hours=24))
    to_dt = intent.time_to or now
    span_days = (to_dt - from_dt).total_seconds() / 86400

    context_parts: list[str] = []

    # Category-specific context
    if category == "active_visits":
        context_parts.append(_get_active_visits_context(session))
        # Also show recent arrivals for context
        context_parts.append(
            _get_recent_events_context(session, now - timedelta(hours=4), now, limit=5)
        )

    elif category in ("visit_history", "vehicle_query", "person_query"):
        context_parts.append(
            _get_visit_history_context(session, from_dt, to_dt, intent.entity_name)
        )
        context_parts.append(
            _get_recent_events_context(
                session, from_dt, to_dt, intent.entity_name, intent.camera_filter
            )
        )

    elif category == "statistics":
        context_parts.append(_get_statistics_context(session, from_dt, to_dt))

    elif category in ("system_info",):
        context_parts.append(_get_entity_info_context(session))
        cameras = session.execute(
            text("SELECT DISTINCT camera_name FROM events ORDER BY camera_name")
        ).fetchall()
        if cameras:
            context_parts.append("CAMERAS: " + ", ".join(r[0] for r in cameras))

    elif category == "doorbell":
        # Filter to ding events only
        q = session.query(Event).filter(Event.event_type == "ding")
        if from_dt:
            q = q.filter(Event.occurred_at >= from_dt)
        if to_dt:
            q = q.filter(Event.occurred_at <= to_dt)
        events = q.order_by(Event.occurred_at.desc()).limit(10).all()
        if events:
            lines = [f"DOORBELL EVENTS ({len(events)}):"]
            for e in events:
                lines.append(f"  [#{e.id}] {_format_time(e.occurred_at)} at {e.camera_name}")
            context_parts.append("\n".join(lines))
        else:
            context_parts.append("DOORBELL: No doorbell events found for this time range.")

    else:
        # recent_events, show_event, describe_event, show_visitors, compare_events, etc.
        # Use tiered summarization based on time span
        if span_days <= 1:
            context_parts.append(
                _get_recent_events_context(
                    session,
                    from_dt,
                    to_dt,
                    intent.entity_name,
                    intent.camera_filter,
                    limit=MAX_EVENT_DETAIL,
                )
            )
        elif span_days <= 7:
            context_parts.append(
                _get_recent_events_context(
                    session,
                    from_dt,
                    to_dt,
                    intent.entity_name,
                    intent.camera_filter,
                    limit=MAX_EVENT_REDUCED,
                )
            )
            context_parts.append(
                _get_aggregated_events_context(session, from_dt, to_dt, intent.entity_name)
            )
        else:
            context_parts.append(
                _get_recent_events_context(
                    session,
                    from_dt,
                    to_dt,
                    intent.entity_name,
                    intent.camera_filter,
                    limit=MAX_EVENT_REDUCED,
                )
            )
            context_parts.append(_get_period_counts_context(session, from_dt, to_dt))

    # Build and check token budget
    context = "\n\n".join(p for p in context_parts if p)
    tokens = _estimate_tokens(context)

    # Overflow handling
    if tokens > TOKEN_BUDGET:
        log.debug("Context overflow (%d tokens), reducing", tokens)
        context_parts_reduced: list[str] = []
        for part in context_parts:
            # Truncate long event listings
            lines = part.split("\n")
            if len(lines) > MAX_EVENT_REDUCED + 1:
                lines = lines[: MAX_EVENT_REDUCED + 1]
                lines.append("  ... (truncated)")
            context_parts_reduced.append("\n".join(lines))
        context = "\n\n".join(p for p in context_parts_reduced if p)

    return context


def build_system_prompt(session: Session, context: str) -> str:
    """Assemble the full system prompt with context."""
    now = datetime.now()
    summary = _build_system_summary(session)
    return SYSTEM_PROMPT_TEMPLATE.format(
        now=now.strftime("%A, %B %-d, %Y %-I:%M %p"),
        system_summary=summary,
        context=context,
    )
