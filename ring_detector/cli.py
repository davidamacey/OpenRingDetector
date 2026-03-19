"""CLI entry points for all ring-detector commands."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from math import ceil
from pathlib import Path
from uuid import uuid4

import numpy as np
from tqdm import tqdm

from ring_detector.config import settings
from ring_detector.database import (
    VisitEvent,
    create_tables,
    find_similar_embeddings,
    get_all_references,
    get_session,
    insert_embeddings_bulk,
    insert_metadata_bulk,
    upsert_reference,
)
from ring_detector.detector import (
    chunk,
    compute_yolo_embeddings,
    process_batch,
)
from ring_detector.image_utils import get_files, prepare_batch
from ring_detector.models import load_models

log = logging.getLogger(__name__)


def _setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )


def embed_main():
    """Batch embed images from a directory into the database."""
    parser = argparse.ArgumentParser(description="Batch process images for detection & embedding")
    parser.add_argument("dir", type=str, help="Directory containing images")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()
    _setup_logging()

    create_tables()
    session = get_session()
    models = load_models()

    image_files = get_files(args.dir)
    log.info("Found %d images to process", len(image_files))

    num_batches = ceil(len(image_files) / args.batch_size)
    for batch in tqdm(
        chunk(image_files, args.batch_size),
        desc="Processing",
        total=num_batches,
    ):
        paths, resized, padded = prepare_batch(batch)
        result = process_batch(
            models,
            paths,
            resized,
            padded,
            session=session,
            batch_size=args.batch_size,
        )
        log.info(
            "Batch: %d images, %d detections, %d faces",
            result["images"],
            result["detections"],
            result["faces"],
        )

    log.info("Done")


def ref_main():
    """Create a reference from a directory of images."""
    parser = argparse.ArgumentParser(description="Create a reference vector")
    parser.add_argument("dir", type=str, help="Directory of reference images")
    parser.add_argument("--name", type=str, required=True, help="Reference key")
    parser.add_argument("--display-name", type=str, default=None)
    parser.add_argument(
        "--category",
        type=str,
        default="vehicle",
        choices=["vehicle", "person", "other"],
    )
    args = parser.parse_args()
    _setup_logging()

    display_name = args.display_name or args.name.replace("_", " ").title()

    create_tables()
    session = get_session()
    models = load_models()

    image_files = get_files(args.dir)
    log.info("Found %d reference images", len(image_files))

    paths, resized, padded = prepare_batch(image_files)
    embeddings = compute_yolo_embeddings(models, padded)
    mean_vec = np.mean(embeddings, axis=0).tolist()

    upsert_reference(session, args.name, display_name, mean_vec, args.category)

    # Store individual reference embeddings
    meta_records = [
        {
            "file_uuid": uuid4().hex,
            "created_at": datetime.now(),
            "path": p,
            "file_name": Path(p).name,
            "height": 0,
            "width": 0,
        }
        for p in paths
    ]
    insert_metadata_bulk(session, meta_records)
    insert_embeddings_bulk(
        session,
        [m["file_uuid"] for m in meta_records],
        embeddings,
        embed_type="reference",
        labels=[args.name] * len(paths),
    )

    log.info("Reference '%s' created (%d images)", display_name, len(paths))

    results = find_similar_embeddings(session, mean_vec, embed_type="reference", limit=1)
    if results:
        log.info("Verification — closest distance: %.4f", results[0]["distance"])


def visits_main():
    """View recent visit history."""
    parser = argparse.ArgumentParser(description="View visit history")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--active", action="store_true")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    create_tables()
    session = get_session()

    query = session.query(VisitEvent).order_by(VisitEvent.arrived_at.desc())
    if args.active:
        query = query.filter(VisitEvent.departed_at.is_(None))
    if args.name:
        query = query.filter(VisitEvent.reference_name == args.name)

    visits = query.limit(args.limit).all()
    if not visits:
        print("No visits found.")
        return

    hdr = f"{'Name':<20} {'Camera':<15} {'Arrived':<20} {'Departed':<20} {'Duration'}"
    print(f"\n{hdr}\n{'-' * len(hdr)}")

    for v in visits:
        arrived = v.arrived_at.strftime("%Y-%m-%d %H:%M") if v.arrived_at else "?"
        if v.departed_at:
            departed = v.departed_at.strftime("%Y-%m-%d %H:%M")
            dur = int((v.departed_at - v.arrived_at).total_seconds() / 60)
            duration = f"{dur} min"
        else:
            departed = "-- active --"
            dur = int((datetime.now() - v.arrived_at).total_seconds() / 60)
            duration = f"~{dur} min (ongoing)"
        print(f"{v.display_name:<20} {v.camera_name:<15} {arrived:<20} {departed:<20} {duration}")
    print()


def refs_main():
    """List all configured references."""
    create_tables()
    session = get_session()
    refs = get_all_references(session)

    if not refs:
        print("No references configured. Use `ring-ref` to create one.")
        return

    print(f"\n{'Name':<25} {'Display Name':<25} {'Category':<12}")
    print("-" * 62)
    for ref in refs:
        print(f"{ref.name:<25} {ref.display_name:<25} {ref.category:<12}")
    print()


def status_main():
    """Check system health: database, Ring API, GPU, Ollama."""
    _setup_logging()

    checks = []

    # Database
    try:
        create_tables()
        session = get_session()
        session.execute(__import__("sqlalchemy").text("SELECT 1"))
        refs = get_all_references(session)
        checks.append(("PostgreSQL + pgvector", "OK", f"{len(refs)} references"))
    except Exception as e:
        checks.append(("PostgreSQL + pgvector", "FAIL", str(e)[:60]))

    # GPU
    try:
        import torch

        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            checks.append(("NVIDIA GPU", "OK", f"{name} ({mem:.0f} GB)"))
        else:
            checks.append(("NVIDIA GPU", "WARN", "CUDA not available, using CPU"))
    except Exception as e:
        checks.append(("NVIDIA GPU", "FAIL", str(e)[:60]))

    # YOLO model
    model_path = Path(settings.model.yolo_model_path)
    if model_path.is_file():
        size_mb = model_path.stat().st_size / 1e6
        checks.append(("YOLO Model", "OK", f"{model_path.name} ({size_mb:.0f} MB)"))
    else:
        checks.append(("YOLO Model", "MISSING", f"Expected at {model_path}"))

    # Ring API
    try:
        token_path = settings.ring.token_path
        if token_path.is_file():
            checks.append(("Ring Token", "OK", str(token_path)))
        else:
            checks.append(("Ring Token", "MISSING", "Run: python -m ring_doorbell.cli --auth"))
    except Exception as e:
        checks.append(("Ring Token", "FAIL", str(e)[:60]))

    # Ollama
    if settings.captioner.enabled:
        from ring_detector.captioner import is_available

        if is_available():
            checks.append(("Ollama Captioner", "OK", settings.captioner.model))
        else:
            checks.append(("Ollama Captioner", "FAIL", "Not reachable or model missing"))
    else:
        checks.append(("Ollama Captioner", "OFF", "Set CAPTIONER_ENABLED=true"))

    # Archive dir
    archive = settings.storage.archive_dir
    if archive.is_dir():
        checks.append(("Archive Dir", "OK", str(archive)))
    else:
        checks.append(("Archive Dir", "MISSING", str(archive)))

    # Print results
    print(f"\n{'Component':<25} {'Status':<10} {'Detail'}")
    print("-" * 75)
    for name, status, detail in checks:
        icon = {"OK": "+", "WARN": "~", "FAIL": "X", "MISSING": "!", "OFF": "-"}.get(status, "?")
        print(f"[{icon}] {name:<23} {status:<10} {detail}")
    print()


if __name__ == "__main__":
    embed_main()
