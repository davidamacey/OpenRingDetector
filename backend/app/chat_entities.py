"""Intent classification, time parsing, entity resolution, and scope checking.

All keyword-based (no LLM). Deterministic and zero-latency.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from difflib import get_close_matches

from sqlalchemy.orm import Session

from ring_detector.database import FaceProfile, Reference

# --- Intent Categories ---

INTENT_CATEGORIES = [
    "recent_events",
    "vehicle_query",
    "person_query",
    "visit_history",
    "active_visits",
    "statistics",
    "doorbell",
    "system_info",
    "describe_event",
    "show_event",
    "show_reference",
    "show_visitors",
    "compare_events",
    "visual_search",
]

# Image intents (checked first — require snapshot lookup / Gemma multi-modal)
IMAGE_INTENTS = {"describe_event", "show_event", "show_visitors", "compare_events", "visual_search"}

# Domain keywords for scope checking
DOMAIN_KEYWORDS = {
    "motion",
    "arrival",
    "arrive",
    "arrived",
    "departure",
    "depart",
    "departed",
    "left",
    "vehicle",
    "car",
    "truck",
    "van",
    "visitor",
    "visit",
    "person",
    "people",
    "someone",
    "camera",
    "doorbell",
    "ring",
    "door",
    "front",
    "driveway",
    "event",
    "events",
    "detection",
    "detected",
    "seen",
    "saw",
    "came",
    "come",
    "here",
    "home",
    "activity",
    "movement",
    "alert",
    "notification",
    "snapshot",
    "photo",
    "image",
    "video",
    "face",
    "who",
    "cleaner",
    "delivery",
    "package",
    "watcher",
    "watch",
    "today",
    "yesterday",
    "week",
    "morning",
    "evening",
    "night",
    "afternoon",
    "last",
    "recent",
    "latest",
    "how many",
    "count",
    "total",
    "average",
    "active",
    "current",
    "right now",
    "status",
    "setup",
    "configured",
    "show",
    "describe",
    "compare",
    "look",
    "check",
}

# Intent keyword patterns — order matters (image intents first)
_INTENT_PATTERNS: list[tuple[str, list[str]]] = [
    # Image intents (checked first)
    ("describe_event", ["describe", "what's in the photo", "what is in the photo", "analyze"]),
    ("compare_events", ["compare", "side by side", "difference between"]),
    ("visual_search", ["red car", "blue car", "white van", "black truck", "color of"]),
    ("show_reference", ["show me the", "show the", "picture of"]),
    ("show_visitors", ["show.*visitor", "show.*who"]),
    ("show_event", ["show me", "show the last", "show latest"]),
    # Text intents
    ("doorbell", ["doorbell", "rang", "ring the bell", "ding"]),
    (
        "active_visits",
        [
            "anyone here",
            "is anyone",
            "who is here",
            "who's here",
            "anyone home",
            "right now",
            "currently",
            "active visit",
            "still here",
        ],
    ),
    (
        "visit_history",
        [
            "how long",
            "duration",
            "stay",
            "stayed",
            "visit history",
            "when did.*leave",
            "when did.*arrive",
            "when did.*come",
            "when did.*left",
        ],
    ),
    (
        "vehicle_query",
        [
            "vehicle",
            "car",
            "truck",
            "van",
            "bus",
            "what vehicle",
            "which car",
            "parking",
            "parked",
        ],
    ),
    (
        "person_query",
        [
            "who visited",
            "who came",
            "who was",
            "person",
            "people",
            "face",
            "recognize",
            "identified",
            "known",
        ],
    ),
    (
        "statistics",
        [
            "how many",
            "count",
            "total",
            "average",
            "statistics",
            "stats",
            "busiest",
            "most",
            "least",
            "frequency",
        ],
    ),
    (
        "system_info",
        [
            "camera",
            "cameras",
            "set up",
            "setup",
            "configured",
            "references",
            "profiles",
            "system",
            "what do you",
        ],
    ),
    (
        "recent_events",
        [
            "what happened",
            "any activity",
            "anything",
            "events",
            "recent",
            "latest",
            "last few",
            "today",
            "yesterday",
            "this week",
            "this morning",
        ],
    ),
]


@dataclass
class ChatIntent:
    category: str = "recent_events"
    time_from: datetime | None = None
    time_to: datetime | None = None
    entity_name: str | None = None
    entity_display_name: str | None = None
    entity_type: str | None = None  # "reference" | "face_profile"
    camera_filter: str | None = None
    is_image_intent: bool = False
    event_limit: int = 20
    raw_question: str = ""


# --- Time Parsing ---


def parse_time_range(question: str) -> tuple[datetime | None, datetime | None]:
    """Parse natural language time references into a (from, to) datetime range."""
    q = question.lower().strip()
    now = datetime.now()
    today_start = datetime.combine(now.date(), time.min)

    # "last N hours/minutes"
    m = re.search(r"last\s+(\d+)\s+(hour|hr|minute|min)s?", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit in ("hour", "hr"):
            return (now - timedelta(hours=n), now)
        return (now - timedelta(minutes=n), now)

    # "last N days"
    m = re.search(r"last\s+(\d+)\s+days?", q)
    if m:
        return (now - timedelta(days=int(m.group(1))), now)

    # "N days/hours ago"
    m = re.search(r"(\d+)\s+(day|hour|hr)s?\s+ago", q)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        if unit in ("hour", "hr"):
            return (now - timedelta(hours=n + 1), now - timedelta(hours=max(n - 1, 0)))
        dt = now - timedelta(days=n)
        return (datetime.combine(dt.date(), time.min), datetime.combine(dt.date(), time.max))

    # "a few days ago"
    if re.search(r"few days ago", q):
        return (now - timedelta(days=5), now - timedelta(days=2))

    # Named days of week
    day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    for i, name in enumerate(day_names):
        if name in q and "last" not in q:
            # Most recent occurrence of this day
            current_weekday = now.weekday()
            days_back = (current_weekday - i) % 7
            if days_back == 0 and "today" not in q:
                days_back = 7
            target = now - timedelta(days=days_back)
            return (
                datetime.combine(target.date(), time.min),
                datetime.combine(target.date(), time.max),
            )

    # "today" / "this morning" / "tonight" / "this afternoon" / "this evening"
    today_kws = ("today", "this morning", "tonight", "this afternoon", "this evening")
    if any(kw in q for kw in today_kws):
        if "this morning" in q:
            return (today_start, today_start.replace(hour=12))
        if "this afternoon" in q:
            return (today_start.replace(hour=12), today_start.replace(hour=18))
        if "this evening" in q or "tonight" in q:
            return (today_start.replace(hour=18), now)
        return (today_start, now)

    # "yesterday" with optional time-of-day
    if "yesterday" in q:
        yday = today_start - timedelta(days=1)
        if "morning" in q:
            return (yday, yday.replace(hour=12))
        if "afternoon" in q:
            return (yday.replace(hour=12), yday.replace(hour=18))
        if "evening" in q or "night" in q:
            return (yday.replace(hour=18), yday.replace(hour=23, minute=59, second=59))
        return (yday, yday + timedelta(days=1))

    # "this week"
    if "this week" in q:
        monday = today_start - timedelta(days=now.weekday())
        return (monday, now)

    # "last week"
    if "last week" in q:
        this_monday = today_start - timedelta(days=now.weekday())
        last_monday = this_monday - timedelta(days=7)
        return (last_monday, this_monday)

    # "this month"
    if "this month" in q:
        month_start = today_start.replace(day=1)
        return (month_start, now)

    # Specific times: "after 10pm", "before 8am", "at 3pm"
    m = re.search(r"(?:after|since|from)\s+(\d{1,2})\s*(am|pm)", q)
    if m:
        hour = int(m.group(1))
        if m.group(2) == "pm" and hour != 12:
            hour += 12
        if m.group(2) == "am" and hour == 12:
            hour = 0
        return (today_start.replace(hour=hour), now)

    m = re.search(r"(?:before|until|by)\s+(\d{1,2})\s*(am|pm)", q)
    if m:
        hour = int(m.group(1))
        if m.group(2) == "pm" and hour != 12:
            hour += 12
        if m.group(2) == "am" and hour == 12:
            hour = 0
        return (today_start, today_start.replace(hour=hour))

    # No time keywords -> default last 24 hours
    return (now - timedelta(hours=24), now)


# --- Entity Resolution ---


def _load_known_entities(session: Session) -> list[tuple[str, str, str]]:
    """Load all reference + face profile names. Returns (name, display_name, type)."""
    entities: list[tuple[str, str, str]] = []
    for ref in session.query(Reference).all():
        entities.append((ref.name, ref.display_name, "reference"))
    for fp in session.query(FaceProfile).all():
        entities.append((fp.name, fp.display_name, "face_profile"))
    return entities


def _strip_possessive(text: str) -> str:
    """Remove possessives: "cleaner's car" -> "cleaner car"."""
    return re.sub(r"['`]s\b", "", text)


def resolve_entity(question: str, session: Session) -> tuple[str | None, str | None, str | None]:
    """Fuzzy-match question text against known entity names.

    Returns (name, display_name, entity_type) or (None, None, None).
    """
    entities = _load_known_entities(session)
    if not entities:
        return None, None, None

    q = _strip_possessive(question.lower())

    # Exact substring match on display_name (case-insensitive)
    for name, display_name, etype in entities:
        if display_name.lower() in q or name.lower() in q:
            return name, display_name, etype

    # Fuzzy match against display names
    display_names = [e[1] for e in entities]
    words = re.findall(r"\b[a-z]{3,}\b", q)
    for word in words:
        matches = get_close_matches(word, [dn.lower() for dn in display_names], n=1, cutoff=0.7)
        if matches:
            idx = [dn.lower() for dn in display_names].index(matches[0])
            return entities[idx][0], entities[idx][1], entities[idx][2]

    return None, None, None


# --- Camera Extraction ---

_CAMERA_KEYWORDS = ["front door", "back door", "driveway", "garage", "porch", "patio", "garden"]


def _extract_camera(question: str) -> str | None:
    """Extract camera name from question if mentioned."""
    q = question.lower()
    for kw in _CAMERA_KEYWORDS:
        if kw in q:
            return kw.title()
    # "at the <name>" pattern
    m = re.search(r"at (?:the )?(\w[\w\s]{2,20}?)(?:\?|$|\.)", q)
    if m:
        candidate = m.group(1).strip().title()
        if any(kw.title() in candidate for kw in _CAMERA_KEYWORDS):
            return candidate
    return None


# --- Intent Classification ---


def classify_intent(question: str, session: Session) -> ChatIntent:
    """Classify a user question into a structured ChatIntent.

    Uses keyword matching (no LLM). Image intents checked first.
    """
    q = question.lower().strip()
    time_from, time_to = parse_time_range(question)
    entity_name, entity_display_name, entity_type = resolve_entity(question, session)
    camera = _extract_camera(question)

    # Match against intent patterns
    category = "recent_events"  # default
    for intent_cat, keywords in _INTENT_PATTERNS:
        for kw in keywords:
            if re.search(kw, q):
                category = intent_cat
                break
        else:
            continue
        break

    # If entity is a reference and category is generic, specialize to vehicle_query
    if entity_type == "reference" and category == "recent_events":
        category = "vehicle_query"
    # If entity is a face profile and category is generic, specialize to person_query
    if entity_type == "face_profile" and category == "recent_events":
        category = "person_query"

    return ChatIntent(
        category=category,
        time_from=time_from,
        time_to=time_to,
        entity_name=entity_name,
        entity_display_name=entity_display_name,
        entity_type=entity_type,
        camera_filter=camera,
        is_image_intent=category in IMAGE_INTENTS,
        raw_question=question,
    )


# --- Scope Check ---


def is_in_scope(question: str) -> bool:
    """Check if the question is about Ring camera / detection activity.

    Returns False for completely off-topic questions (weather, jokes, etc.).
    """
    q = question.lower()
    # Short questions (< 4 words) are always in-scope (likely commands)
    if len(q.split()) < 4:
        return True
    return any(kw in q for kw in DOMAIN_KEYWORDS)
