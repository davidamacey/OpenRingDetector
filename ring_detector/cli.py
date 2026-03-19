"""CLI entry points for batch processing and reference management."""

from __future__ import annotations

import argparse
import logging
from math import ceil

import numpy as np
from tqdm import tqdm

from ring_detector.database import (
    create_tables,
    find_similar_embeddings,
    get_session,
    insert_embeddings_bulk,
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
        "--display-name", type=str, default=None, help="Display name (e.g. Cleaner's Car)"
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

    # Compute mean reference vector
    mean_vec = np.mean(embeddings, axis=0).tolist()

    # Store mean reference vector in PostgreSQL (pgvector)
    upsert_reference(session, args.name, display_name, mean_vec, args.category)

    # Also store individual embeddings for later search
    from datetime import datetime
    from uuid import uuid4

    # Create metadata entries for reference images
    meta_records = []
    for p in paths:
        meta_records.append(
            {
                "file_uuid": uuid4().hex,
                "date": datetime.now(),
                "path": p,
                "file_name": p.split("/")[-1],
                "height": 0,
                "width": 0,
            }
        )

    from ring_detector.database import insert_metadata_bulk

    insert_metadata_bulk(session, meta_records)
    file_uuids = [m["file_uuid"] for m in meta_records]
    insert_embeddings_bulk(
        session, file_uuids, embeddings, embed_type="reference", labels=[args.name] * len(paths)
    )

    log.info("Reference '%s' created with %d images", display_name, len(paths))

    # Verify by finding closest match
    results = find_similar_embeddings(session, mean_vec, embed_type="reference", limit=1)
    if results:
        log.info("Closest reference image (distance=%.4f)", results[0]["distance"])


if __name__ == "__main__":
    embed_main()
