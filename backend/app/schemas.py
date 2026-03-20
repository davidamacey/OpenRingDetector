"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict


class DetectionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    class_name: str
    class_id: int
    confidence: float
    xcenter: float
    ycenter: float
    width: float
    height: float


class EventResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    event_type: str
    camera_name: str
    occurred_at: datetime
    snapshot_url: str | None
    detection_summary: str | None
    reference_name: str | None
    display_name: str | None
    caption: str | None
    detections: list[DetectionResponse] = []
    visit_id: int | None


class EventListResponse(BaseModel):
    total: int
    items: list[EventResponse]


class VisitResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    reference_name: str
    display_name: str
    camera_name: str
    arrived_at: datetime
    last_motion_at: datetime
    departed_at: datetime | None
    duration_minutes: int | None
    snapshot_url: str | None
    is_active: bool


class VisitListResponse(BaseModel):
    total: int
    items: list[VisitResponse]


class ReferenceResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uuid: str
    name: str
    display_name: str
    category: str
    visit_count: int
    last_seen: datetime | None
    sample_image_url: str | None


class FaceProfileResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    uuid: str
    name: str
    display_name: str
    created_at: datetime
    visit_count: int
    last_seen: datetime | None
    sample_image_url: str | None


class ComponentStatus(BaseModel):
    status: str  # ok | warn | fail | missing | off
    detail: str


class SystemStatus(BaseModel):
    database: ComponentStatus
    gpu: ComponentStatus
    yolo_model: ComponentStatus
    ring_token: ComponentStatus
    face_models: ComponentStatus
    ollama: ComponentStatus
    archive_dir: ComponentStatus
    watcher_running: bool
    uptime_seconds: int


class SettingsResponse(BaseModel):
    ring: dict
    model: dict
    captioner: dict
    notify: dict
    storage: dict


class UnmatchedVehicle(BaseModel):
    cluster_id: str
    sighting_count: int
    first_seen: datetime
    last_seen: datetime
    camera_name: str
    representative_crop_url: str | None
    all_crop_urls: list[str]


class UnmatchedFace(BaseModel):
    face_embedding_uuid: str
    sighting_count: int
    last_seen: datetime
    camera_name: str
    crop_url: str | None


class AnalyticsEventPerDay(BaseModel):
    date: str
    count: int


class AnalyticsHeatmap(BaseModel):
    hour: int
    count: int


class AnalyticsTopVisitor(BaseModel):
    display_name: str
    visit_count: int
    last_seen: datetime


class AnalyticsDetectionType(BaseModel):
    class_name: str
    count: int
    percentage: float


class AnalyticsVisitDuration(BaseModel):
    bucket: str
    count: int
