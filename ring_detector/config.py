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
    port: int = int(os.getenv("DB_PORT", "5433"))
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
    # Seconds without new motion to consider someone "departed"
    departure_timeout: int = int(os.getenv("DEPARTURE_TIMEOUT", "300"))
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
    ntfy_url: str = os.getenv("NTFY_URL", "http://ntfy.superdave.us/ring_cam")
    attach_snapshot: bool = os.getenv("NTFY_ATTACH_SNAPSHOT", "true").lower() == "true"


@dataclass
class ModelConfig:
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "./models/yolov8m.pt")
    device: str = os.getenv("TORCH_DEVICE", "cuda:0")
    image_size: int = 640
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))


@dataclass
class CaptionerConfig:
    enabled: bool = os.getenv("CAPTIONER_ENABLED", "false").lower() == "true"
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model: str = os.getenv("CAPTIONER_MODEL", "gemma3:4b")
    timeout: int = int(os.getenv("CAPTIONER_TIMEOUT", "60"))


@dataclass
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    captioner: CaptionerConfig = field(default_factory=CaptionerConfig)


settings = Settings()
