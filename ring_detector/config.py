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
class MilvusConfig:
    host: str = os.getenv("MILVUS_HOST", "localhost")
    port: str = os.getenv("MILVUS_PORT", "19530")


@dataclass
class RingConfig:
    token_path: Path = Path(os.getenv("RING_TOKEN_PATH", "./tokens/token.cache"))
    poll_interval_seconds: int = int(os.getenv("RING_POLL_INTERVAL", "60"))
    camera_name: str = os.getenv("RING_CAMERA_NAME", "")


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


@dataclass
class ModelConfig:
    yolo_model_path: str = os.getenv("YOLO_MODEL_PATH", "./models/yolov8m.pt")
    device: str = os.getenv("TORCH_DEVICE", "cuda:0")
    image_size: int = 640
    batch_size: int = int(os.getenv("BATCH_SIZE", "50"))


@dataclass
class Settings:
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    ring: RingConfig = field(default_factory=RingConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    notify: NotifyConfig = field(default_factory=NotifyConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


settings = Settings()
