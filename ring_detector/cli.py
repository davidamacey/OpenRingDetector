"""CLI entry points for batch processing, reference management, and visit history."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from math import ceil
from uuid import uuid4

import numpy as np
from tqdm import tqdm

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


def embed_main():
    """Batch embed images from a directory into the database."""
    parser = argparse.ArgumentParser(description="Batch process images for detection & embedding")
    parser.add_argument("dir", type=str, help="Directory containing images")
    parser.add_argument("--batch-size", type=int, default=50)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    create_tables()
    session = get_session()
    models = load_models()

    image_files = get_files(args.dir)
    log.info("Found %d images to process", len(image_files))

    num_batches = ceil(len(image_files) / args.batch_size)

    for batch in tqdm(
        chunk(image_files, args.batch_size),
        desc="Processing batches",
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

    log.info("Completed processing all images")


def ref_main():
    """Create a reference from a directory of images."""
    parser = argparse.ArgumentParser(description="Create a reference vector from images")
    parser.add_argument("dir", type=str, help="Directory of reference images")
    parser.add_argument(
        "--name", type=str, required=True, help="Reference name (e.g. cleaners_car)"
    )
    parser.add_argument(
        "--display-name", type=str, default=None, help="Display name (e.g. Cleaner)"
    )
    parser.add_argument(
        "--category",
        type=str,
        default="vehicle",
        choices=["vehicle", "person", "other"],
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

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

    # Store individual embeddings
    meta_records = [
        {
            "file_uuid": uuid4().hex,
            "date": datetime.now(),
            "path": p,
            "file_name": p.split("/")[-1],
            "height": 0,
            "width": 0,
        }
        for p in paths
    ]
    insert_metadata_bulk(session, meta_records)
    file_uuids = [m["file_uuid"] for m in meta_records]
    insert_embeddings_bulk(
        session, file_uuids, embeddings, embed_type="reference", labels=[args.name] * len(paths)
    )

    log.info("Reference '%s' created with %d images", display_name, len(paths))

    results = find_similar_embeddings(session, mean_vec, embed_type="reference", limit=1)
    if results:
        log.info("Closest reference image (distance=%.4f)", results[0]["distance"])


def visits_main():
    """View recent visit history."""
    parser = argparse.ArgumentParser(description="View visit history")
    parser.add_argument("--limit", type=int, default=20, help="Number of visits to show")
    parser.add_argument(
        "--active", action="store_true", help="Show only active (not departed) visits"
    )
    parser.add_argument("--name", type=str, default=None, help="Filter by reference name")
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

    print(f"\n{'Name':<20} {'Camera':<15} {'Arrived':<20} {'Departed':<20} {'Duration'}")
    print("-" * 95)

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


if __name__ == "__main__":
    embed_main()
