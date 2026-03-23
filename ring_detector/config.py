"""Centralized configuration loaded from environment and .env file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "9550"))
    name: str = os.getenv("DB_NAME", "ring")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "postgres")

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


@dataclass
class RingConfig:
    token_path: Path = Path(os.getenv("RING_TOKEN_PATH", "./tokens/token.cache"))
    fcm_credentials_path: Path = Path(
        os.getenv("RING_FCM_CREDENTIALS_PATH", "./tokens/fcm_credentials.json")
    )
    camera_name: str = os.getenv("RING_CAMERA_NAME", "")
    # Timer-fallback: silence before departure recorded (primary detection is FOV-presence-driven)
    departure_timeout: int = int(os.getenv("DEPARTURE_TIMEOUT", "7200"))
    # Consecutive motion events without vehicle in FOV before departure is declared
    departure_miss_threshold: int = int(os.getenv("DEPARTURE_MISS_THRESHOLD", "2"))
    # Delay before "Unknown Visitor" push (window for vehicle detection to cancel it)
    unknown_visitor_delay: int = int(os.getenv("UNKNOWN_VISITOR_DELAY", "120"))
    # Seconds to ignore duplicate motion events from same camera
    cooldown_seconds: int = int(os.getenv("MOTION_COOLDOWN", "30"))


@dataclass
class StorageConfig:
    archive_dir: Path = Path(os.getenv("ARCHIVE_DIR", "/mnt/nas/ring_archive"))
    video_subdir: str = "videos"
    snapshot_subdir: str = "snapshots"

    def video_dir(self) -> Path:
        d = self.archive_dir / self.video_subdir
        d.mkdir(parents=True, exist_ok=True)
        return d

    def snapshot_dir(self) -> Path:
        d = self.archive_dir / self.snapshot_subdir
        d.mkdir(parents=True, exist_ok=True)
        return d


@dataclass
class NotifyConfig:
    ntfy_url: str = os.getenv("NTFY_URL", "http://localhost:9552/ring_cam")
    attach_snapshot: bool = os.getenv("NTFY_ATTACH_SNAPSHOT", "true").lower() == "true"


@dataclass
class ModelConfig:
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "./models/yolo26m.pt")
    scrfd_model_path: str = os.getenv("SCRFD_MODEL_PATH", "./models/scrfd_10g_bnkps.onnx")
    arcface_model_path: str = os.getenv("ARCFACE_MODEL_PATH", "./models/arcface_w600k_r50.onnx")
    device: str = os.getenv("TORCH_DEVICE", "cuda:0")
    image_size: int = 640
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))


@dataclass
class CaptionerConfig:
    enabled: bool = os.getenv("CAPTIONER_ENABLED", "false").lower() == "true"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:9551")
    model: str = os.getenv("CAPTIONER_MODEL", "gemma3:4b")
    timeout: int = int(os.getenv("CAPTIONER_TIMEOUT", "60"))
    chat_timeout: int = int(os.getenv("CHAT_TIMEOUT", "90"))


@dataclass
class FaceDetectionConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("ENABLE_FACE_DETECTION", "true").lower() == "true"
    )
    match_threshold: float = field(
        default_factory=lambda: float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
    )
    min_face_size: int = field(default_factory=lambda: int(os.getenv("FACE_MIN_SIZE", "50")))
    # "local" = ONNX Runtime (default); "triton" = triton-api HTTP
    backend: str = field(default_factory=lambda: os.getenv("FACE_BACKEND", "local"))
    triton_http_url: str = field(
        default_factory=lambda: os.getenv("TRITON_HTTP_URL", "http://localhost:8000")
    )


@dataclass
class VideoAnalysisConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("VIDEO_ANALYSIS_ENABLED", "true").lower() == "true"
    )
    # Extract every Nth frame (~1fps at 30fps video)
    frame_interval: int = field(
        default_factory=lambda: int(os.getenv("VIDEO_FRAME_INTERVAL", "30"))
    )
    # Maximum frames to analyze per video
    max_frames: int = field(default_factory=lambda: int(os.getenv("VIDEO_MAX_FRAMES", "15")))
    # Seconds to wait for Ring to process the video
    wait_timeout: int = field(default_factory=lambda: int(os.getenv("VIDEO_WAIT_TIMEOUT", "60")))
    # Seconds between retry attempts
    retry_delay: int = field(default_factory=lambda: int(os.getenv("VIDEO_RETRY_DELAY", "5")))


@dataclass
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    captioner: CaptionerConfig = field(default_factory=CaptionerConfig)
    face: FaceDetectionConfig = field(default_factory=FaceDetectionConfig)
    video: VideoAnalysisConfig = field(default_factory=VideoAnalysisConfig)


settings = Settings()
