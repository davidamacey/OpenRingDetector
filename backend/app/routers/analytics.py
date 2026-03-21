"""Analytics router — aggregated statistics."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.schemas import (
    AnalyticsDetectionType,
    AnalyticsEventPerDay,
    AnalyticsHeatmap,
    AnalyticsSummary,
    AnalyticsTimeline,
    AnalyticsTopVisitor,
    AnalyticsVisitDuration,
)
from ring_detector.database import get_session

router = APIRouter(tags=["analytics"])


def _get_db():
    session = get_session()
    try:
        yield session
    finally:
        session.close()


@router.get("/analytics/events-per-day", response_model=list[AnalyticsEventPerDay])
def events_per_day(days: int = Query(7, ge=1, le=365), session: Session = Depends(_get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    rows = session.execute(
        text(
            """
            SELECT DATE(occurred_at) AS d, COUNT(*) AS cnt
            FROM events
            WHERE occurred_at >= :cutoff
            GROUP BY DATE(occurred_at)
            ORDER BY d
            """
        ),
        {"cutoff": cutoff},
    ).fetchall()
    return [AnalyticsEventPerDay(date=str(r[0]), count=r[1]) for r in rows]


@router.get("/analytics/activity-heatmap", response_model=list[AnalyticsHeatmap])
def activity_heatmap(days: int = Query(7, ge=1, le=365), session: Session = Depends(_get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    rows = session.execute(
        text(
            """
            SELECT EXTRACT(HOUR FROM occurred_at)::int AS hr, COUNT(*) AS cnt
            FROM events
            WHERE occurred_at >= :cutoff
            GROUP BY hr
            ORDER BY hr
            """
        ),
        {"cutoff": cutoff},
    ).fetchall()
    return [AnalyticsHeatmap(hour=r[0], count=r[1]) for r in rows]


@router.get("/analytics/top-visitors", response_model=list[AnalyticsTopVisitor])
def top_visitors(
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(10, ge=1, le=50),
    session: Session = Depends(_get_db),
):
    cutoff = datetime.now() - timedelta(days=days)
    rows = session.execute(
        text(
            """
            SELECT display_name, COUNT(*) AS cnt, MAX(arrived_at) AS last_seen
            FROM visit_events
            WHERE arrived_at >= :cutoff AND display_name IS NOT NULL
            GROUP BY display_name
            ORDER BY cnt DESC
            LIMIT :limit
            """
        ),
        {"cutoff": cutoff, "limit": limit},
    ).fetchall()
    return [AnalyticsTopVisitor(display_name=r[0], visit_count=r[1], last_seen=r[2]) for r in rows]


@router.get("/analytics/detection-types", response_model=list[AnalyticsDetectionType])
def detection_types(days: int = Query(7, ge=1, le=365), session: Session = Depends(_get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    rows = session.execute(
        text(
            """
            SELECT d.class_name, COUNT(*) AS cnt
            FROM detections d
            JOIN metadata m ON d.file_uuid = m.file_uuid
            WHERE m.created_at >= :cutoff
            GROUP BY d.class_name
            ORDER BY cnt DESC
            """
        ),
        {"cutoff": cutoff},
    ).fetchall()
    total = sum(r[1] for r in rows) or 1
    return [
        AnalyticsDetectionType(class_name=r[0], count=r[1], percentage=round(r[1] / total * 100, 1))
        for r in rows
    ]


@router.get("/analytics/visit-durations", response_model=list[AnalyticsVisitDuration])
def visit_durations(days: int = Query(30, ge=1, le=365), session: Session = Depends(_get_db)):
    cutoff = datetime.now() - timedelta(days=days)
    rows = session.execute(
        text(
            """
            SELECT EXTRACT(EPOCH FROM (departed_at - arrived_at)) / 60 AS dur_min
            FROM visit_events
            WHERE arrived_at >= :cutoff AND departed_at IS NOT NULL
            """
        ),
        {"cutoff": cutoff},
    ).fetchall()

    buckets: dict[str, int] = {
        "0-5min": 0,
        "5-15min": 0,
        "15-60min": 0,
        "1-4hr": 0,
        "4hr+": 0,
    }
    for (dur,) in rows:
        if dur is None:
            continue
        if dur < 5:
            buckets["0-5min"] += 1
        elif dur < 15:
            buckets["5-15min"] += 1
        elif dur < 60:
            buckets["15-60min"] += 1
        elif dur < 240:
            buckets["1-4hr"] += 1
        else:
            buckets["4hr+"] += 1

    return [AnalyticsVisitDuration(bucket=k, count=v) for k, v in buckets.items()]


@router.get("/analytics/summary", response_model=AnalyticsSummary)
def analytics_summary(days: int = Query(7, ge=1, le=365), session: Session = Depends(_get_db)):
    """High-level summary: counts, active visits, avg duration."""
    cutoff = datetime.now() - timedelta(days=days)

    total_events = (
        session.execute(
            text("SELECT COUNT(*) FROM events WHERE occurred_at >= :cutoff"),
            {"cutoff": cutoff},
        ).scalar()
        or 0
    )

    total_visits = (
        session.execute(
            text("SELECT COUNT(*) FROM visit_events WHERE arrived_at >= :cutoff"),
            {"cutoff": cutoff},
        ).scalar()
        or 0
    )

    active_visits = (
        session.execute(
            text("SELECT COUNT(*) FROM visit_events WHERE departed_at IS NULL"),
        ).scalar()
        or 0
    )

    total_detections = (
        session.execute(
            text(
                """
            SELECT COUNT(*) FROM detections d
            JOIN metadata m ON d.file_uuid = m.file_uuid
            WHERE m.created_at >= :cutoff
            """
            ),
            {"cutoff": cutoff},
        ).scalar()
        or 0
    )

    cameras = [
        r[0]
        for r in session.execute(
            text("SELECT DISTINCT camera_name FROM events ORDER BY camera_name")
        ).fetchall()
    ]

    avg_dur = session.execute(
        text(
            """
            SELECT AVG(EXTRACT(EPOCH FROM (departed_at - arrived_at)) / 60)
            FROM visit_events
            WHERE arrived_at >= :cutoff AND departed_at IS NOT NULL
            """
        ),
        {"cutoff": cutoff},
    ).scalar()

    return AnalyticsSummary(
        total_events=total_events,
        total_visits=total_visits,
        active_visits=active_visits,
        total_detections=total_detections,
        cameras=cameras,
        avg_visit_duration_minutes=round(avg_dur, 1) if avg_dur else None,
    )


@router.get("/analytics/timeline", response_model=list[AnalyticsTimeline])
def analytics_timeline(
    days: int = Query(7, ge=1, le=365),
    interval: str = Query("hour", pattern="^(hour|day)$"),
    session: Session = Depends(_get_db),
):
    """Hourly or daily event counts for chart rendering."""
    cutoff = datetime.now() - timedelta(days=days)
    if interval == "hour":
        rows = session.execute(
            text(
                """
                SELECT date_trunc('hour', occurred_at) AS ts, COUNT(*) AS cnt
                FROM events
                WHERE occurred_at >= :cutoff
                GROUP BY ts ORDER BY ts
                """
            ),
            {"cutoff": cutoff},
        ).fetchall()
    else:
        rows = session.execute(
            text(
                """
                SELECT DATE(occurred_at) AS ts, COUNT(*) AS cnt
                FROM events
                WHERE occurred_at >= :cutoff
                GROUP BY DATE(occurred_at) ORDER BY ts
                """
            ),
            {"cutoff": cutoff},
        ).fetchall()

    return [AnalyticsTimeline(timestamp=str(r[0]), count=r[1]) for r in rows]
