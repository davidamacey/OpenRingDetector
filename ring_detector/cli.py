"""CLI entry points for all ring-detector commands."""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
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
    delete_face_profile,
    find_similar_embeddings,
    get_all_face_profiles,
    get_all_references,
    get_session,
    insert_embeddings_bulk,
    insert_metadata_bulk,
    run_migrations,
    upsert_face_profile,
    upsert_reference,
)
from ring_detector.detector import (
    chunk,
    compute_clip_embeddings,
    detect_faces_simple,
    process_batch,
)
from ring_detector.image_utils import get_files, imread_safe, pad_to_square, prepare_batch
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

    run_migrations()
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

    run_migrations()
    session = get_session()
    models = load_models()

    image_files = get_files(args.dir)
    log.info("Found %d reference images", len(image_files))

    paths, resized, padded = prepare_batch(image_files)
    embeddings = compute_clip_embeddings(models, padded)
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

    run_migrations()
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
    run_migrations()
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
        run_migrations()
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
            props = torch.cuda.get_device_properties(0)
            mem = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / 1e9
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
            checks.append(("Ring Token", "MISSING", "Run: ring-auth"))
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


def face_main():
    """Manage face profiles: add, list, or delete known persons for recognition."""
    parser = argparse.ArgumentParser(
        description="Manage face profiles for person recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ring-face add david photo.jpg --display-name David\n"
            "  ring-face list\n"
            "  ring-face delete david"
        ),
    )
    sub = parser.add_subparsers(dest="cmd", metavar="COMMAND")

    add_p = sub.add_parser("add", help="Extract face embedding from image and save as profile")
    add_p.add_argument("name", type=str, help="Unique profile key (e.g. david)")
    add_p.add_argument("image", type=str, help="Path to image containing the person's face")
    add_p.add_argument("--display-name", type=str, default=None, help="Human-readable name")

    sub.add_parser("list", help="List all known face profiles")

    del_p = sub.add_parser("delete", help="Remove a face profile")
    del_p.add_argument("name", type=str, help="Profile key to remove")

    args = parser.parse_args()
    _setup_logging()

    if args.cmd == "add":
        _face_add(args)
    elif args.cmd == "list":
        _face_list()
    elif args.cmd == "delete":
        _face_delete(args)
    else:
        parser.print_help()


def _face_add(args) -> None:
    img = imread_safe(args.image)
    if img is None:
        log.error("Cannot read image: %s", args.image)
        return

    run_migrations()
    session = get_session()
    models = load_models()

    if models.face_detector is None:
        log.error("Face models not available — check ENABLE_FACE_DETECTION and model files")
        return

    padded = pad_to_square(img)
    _crops, embeddings = detect_faces_simple(
        models, padded, min_face_size=settings.face.min_face_size
    )

    if not embeddings:
        log.error("No face detected in %s (try a clearer, front-facing photo)", args.image)
        return

    if len(embeddings) > 1:
        log.warning(
            "%d faces found in image — using the first one. "
            "Use a single-face photo for best results.",
            len(embeddings),
        )

    display_name = args.display_name or args.name.replace("_", " ").title()
    upsert_face_profile(session, args.name, display_name, embeddings[0])
    log.info("Face profile '%s' (%s) saved", args.name, display_name)


def _face_list() -> None:
    run_migrations()
    session = get_session()
    profiles = get_all_face_profiles(session)

    if not profiles:
        print("No face profiles configured. Use `ring-face add` to create one.")
        return

    print(f"\n{'Name':<25} {'Display Name':<25} {'Created'}")
    print("-" * 70)
    for p in profiles:
        created = p.created_at.strftime("%Y-%m-%d %H:%M") if p.created_at else "?"
        print(f"{p.name:<25} {p.display_name:<25} {created}")
    print()


def _face_delete(args) -> None:
    run_migrations()
    session = get_session()
    if delete_face_profile(session, args.name):
        log.info("Face profile '%s' deleted", args.name)
    else:
        log.error("Face profile '%s' not found", args.name)


def auth_main():
    """Authenticate with Ring and save OAuth token for ring-watch."""
    _setup_logging()

    token_path = settings.ring.token_path
    if token_path.is_file():
        print(f"Existing token found at {token_path}")
        resp = input("Re-authenticate? [y/N]: ").strip().lower()
        if resp != "y":
            print("Keeping existing token. Use ring-status to verify.")
            return

    username = input("Ring email: ")
    password = getpass.getpass("Ring password: ")

    from ring_doorbell import Auth
    from ring_doorbell.const import USER_AGENT

    def _save_token(token: dict) -> None:
        token_path.parent.mkdir(parents=True, exist_ok=True)
        token_path.write_text(json.dumps(token))

    async def _authenticate() -> None:
        auth = Auth(USER_AGENT, None, _save_token)
        try:
            await auth.async_fetch_token(username, password)
        except Exception:
            code = input("2FA Code: ")
            await auth.async_fetch_token(username, password, code)

    asyncio.run(_authenticate())
    log.info("Ring token saved to %s", token_path)
    print(f"\nAuthentication successful! Token saved to {token_path}")
    print("Run `ring-status` to verify, then `ring-watch` to start.")


def test_main():
    """Test the detection pipeline on local images or videos (no Ring account needed)."""
    parser = argparse.ArgumentParser(
        description="Test detection pipeline on local images/videos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  ring-test snapshot.jpg\n"
            "  ring-test ./videos/front_door.mp4\n"
            "  ring-test ./photos/ --save-crops\n"
            "  ring-test clip.mp4 --frame-interval 30\n"
            "  ring-test clip.mp4 --notify --camera 'Front Door'"
        ),
    )
    parser.add_argument("input", type=str, help="Image file, video file, or directory of images")
    parser.add_argument(
        "--frame-interval",
        type=int,
        default=30,
        help="For videos: process every Nth frame (default: 30)",
    )
    parser.add_argument(
        "--save-crops",
        action="store_true",
        help="Save detected object/face crops to ./test_output/",
    )
    parser.add_argument(
        "--no-faces",
        action="store_true",
        help="Skip face detection",
    )
    parser.add_argument(
        "--no-vehicles",
        action="store_true",
        help="Skip vehicle matching",
    )
    parser.add_argument(
        "--notify",
        action="store_true",
        help="Send real ntfy notifications (simulates full watcher pipeline)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="Test Camera",
        help="Camera name for notifications (default: 'Test Camera')",
    )
    args = parser.parse_args()
    _setup_logging()

    import cv2

    from ring_detector.captioner import caption_image
    from ring_detector.detector import (
        compute_clip_embeddings,
        detect_faces_simple,
        run_detection,
    )
    from ring_detector.image_utils import (
        VIDEO_EXTENSIONS,
        extract_key_frames,
        imread_safe,
        pad_to_square,
    )

    input_path = Path(args.input)
    if not input_path.exists():
        log.error("Input not found: %s", input_path)
        return

    # Collect frames to process
    frames: list[tuple[str, np.ndarray]] = []  # (label, bgr_image)

    if input_path.is_dir():
        image_files = get_files(str(input_path))
        for f in image_files:
            img = imread_safe(f)
            if img is not None:
                frames.append((f, img))
    elif input_path.suffix.lower() in VIDEO_EXTENSIONS:
        key_frames = extract_key_frames(
            input_path,
            frame_interval=args.frame_interval,
            max_frames=settings.video.max_frames,
        )
        for frame_num, frame in key_frames:
            frames.append((f"{input_path.name}:frame{frame_num}", frame))
    else:
        img = imread_safe(str(input_path))
        if img is not None:
            frames.append((str(input_path), img))

    if not frames:
        log.error("No images to process")
        return

    log.info("Processing %d frame(s)", len(frames))

    # Load models + DB
    run_migrations()
    session = get_session()
    models = load_models()

    output_dir = Path("test_output")
    if args.save_crops:
        output_dir.mkdir(exist_ok=True)

    for idx, (label, img_bgr) in enumerate(frames):
        print(f"\n{'=' * 70}")
        print(f"Frame {idx + 1}/{len(frames)}: {label}")
        print(f"  Size: {img_bgr.shape[1]}x{img_bgr.shape[0]}")

        # Prepare for detection
        padded = pad_to_square(img_bgr)
        resized = cv2.resize(img_bgr, (640, int(640 * img_bgr.shape[0] / img_bgr.shape[1])))
        paths_batch = [label]

        # YOLO detection
        _meta, df_dets, img_crops = run_detection(models, [resized], paths_batch)

        if len(df_dets) == 0:
            print("  Detections: none")
        else:
            counts = df_dets["class_name"].value_counts()
            summary = ", ".join(f"{cls} x{n}" if n > 1 else cls for cls, n in counts.items())
            print(f"  Detections: {summary}")
            for _, det in df_dets.iterrows():
                print(f"    - {det['class_name']} ({det['confidence']}% conf)")

        # Captioning
        is_image = input_path.is_file() and input_path.suffix.lower() not in VIDEO_EXTENSIONS
        if settings.captioner.enabled and is_image:
            cap_text = caption_image(str(input_path))
            if cap_text:
                print(f"  Caption: {cap_text}")

        # Vehicle matching
        vehicle_dets = df_dets[df_dets["class_id"].isin({2, 7})] if len(df_dets) > 0 else df_dets
        if not args.no_vehicles and len(vehicle_dets) > 0 and img_crops:
            from ring_detector.database import match_against_references

            indices = vehicle_dets.index.tolist()
            crops = [img_crops[i] for i in indices if i < len(img_crops)]
            if crops:
                padded_crops = [pad_to_square(c) for c in crops]
                embeddings = compute_clip_embeddings(models, padded_crops)
                matches = match_against_references(session, embeddings)
                if matches:
                    print("  Vehicle matches:")
                    for m in matches:
                        print(
                            f"    - {m['display_name']} ({m['reference_name']}, "
                            f"similarity={m['similarity']:.4f})"
                        )
                else:
                    print("  Vehicle matches: none (no matching references)")

                if args.save_crops:
                    for i, crop in enumerate(crops):
                        crop_path = output_dir / f"frame{idx}_vehicle{i}.jpg"
                        cv2.imwrite(str(crop_path), crop)

        # Face detection
        if not args.no_faces and settings.face.enabled and models.face_detector is not None:
            from ring_detector.database import match_against_face_profiles

            face_crops, face_embeddings = detect_faces_simple(
                models, padded, settings.face.min_face_size
            )
            if face_embeddings:
                face_matches = match_against_face_profiles(
                    session, face_embeddings, settings.face.match_threshold
                )
                matched_names = [m["display_name"] for m in face_matches]
                unmatched = len(face_embeddings) - len(face_matches)
                print(f"  Faces: {len(face_embeddings)} detected")
                if matched_names:
                    print(f"    Matched: {', '.join(matched_names)}")
                if unmatched > 0:
                    print(f"    Unknown: {unmatched}")

                if args.save_crops:
                    for i, crop in enumerate(face_crops):
                        crop_path = output_dir / f"frame{idx}_face{i}.jpg"
                        cv2.imwrite(str(crop_path), crop)
            else:
                print("  Faces: none detected")

    print(f"\n{'=' * 70}")
    print(f"Done. Processed {len(frames)} frame(s).")
    if args.save_crops:
        print(f"Crops saved to {output_dir}/")

    # --notify: run the full watcher analysis pipeline and send real notifications
    if args.notify:
        _test_notify(frames, models, session, args)


def _test_notify(
    frames: list[tuple[str, np.ndarray]],
    models,
    session,
    args,
) -> None:
    """Run the watcher's aggregated analysis and send real notifications."""
    import cv2

    from ring_detector.captioner import caption_frames
    from ring_detector.database import match_against_face_profiles, match_against_references
    from ring_detector.detector import (
        clear_gpu_memory,
        compute_clip_embeddings,
        detect_faces_simple,
        run_detection,
    )
    from ring_detector.image_utils import pad_to_square
    from ring_detector.notifications import (
        notify_arrival,
        notify_known_person,
        notify_motion,
        notify_unknown_visitor,
    )
    from ring_detector.watcher import _cluster_embeddings

    cam_name = args.camera
    frame_images = [f[1] for f in frames]
    frame_labels = [f[0] for f in frames]

    print(f"\n{'=' * 70}")
    print(f"Running full notification pipeline for '{cam_name}'...")

    # Batched YOLO detection
    resized_all = [cv2.resize(f, (640, int(640 * f.shape[0] / f.shape[1]))) for f in frame_images]
    padded_all = [pad_to_square(f) for f in frame_images]
    _meta, df_dets, img_crops = run_detection(models, resized_all, frame_labels)

    # Aggregate across frames
    all_vehicle_embs: list[list[float]] = []
    all_face_embs: list[list[float]] = []
    all_classes: dict[str, int] = {}
    has_persons = False
    best_frame_idx = 0
    best_det_count = 0

    if len(df_dets) > 0:
        for i, label in enumerate(frame_labels):
            frame_meta = _meta[_meta["path"] == label]
            if frame_meta.empty:
                continue
            fuuid = frame_meta["file_uuid"].iloc[0]
            fdets = df_dets[df_dets["file_uuid"] == fuuid]

            if len(fdets) > best_det_count:
                best_det_count = len(fdets)
                best_frame_idx = i

            for cls in fdets["class_name"]:
                all_classes[cls] = all_classes.get(cls, 0) + 1
            if "person" in fdets["class_name"].values:
                has_persons = True

            if not args.no_vehicles:
                vdets = fdets[fdets["class_id"].isin({2, 7})]
                if len(vdets) > 0 and img_crops:
                    indices = vdets.index.tolist()
                    crops = [img_crops[j] for j in indices if j < len(img_crops)]
                    if crops:
                        pcr = [pad_to_square(c) for c in crops]
                        all_vehicle_embs.extend(compute_clip_embeddings(models, pcr))

        if not args.no_faces and settings.face.enabled and models.face_detector is not None:
            for p in padded_all:
                _fc, fe = detect_faces_simple(models, p, settings.face.min_face_size)
                all_face_embs.extend(fe)

    clear_gpu_memory()

    # Build summary
    summary_parts = []
    for cls in sorted(all_classes, key=lambda c: -all_classes[c]):
        n = max(1, all_classes[cls] // max(1, len(frames)))
        summary_parts.append(f"{cls} x{n}" if n > 1 else cls)
    summary = ", ".join(summary_parts)

    # Save best frame
    best_frame = frame_images[best_frame_idx]
    best_path = Path("test_output") / f"best_frame_{cam_name.replace(' ', '_')}.jpg"
    best_path.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(best_path), best_frame)
    snap_str = str(best_path)

    # Multi-frame caption
    if settings.captioner.enabled:
        tmp_paths = []
        step = max(1, len(frame_images) // 5)
        for idx in range(0, len(frame_images), step):
            p = Path("test_output") / f"_caption_{idx}.jpg"
            cv2.imwrite(str(p), frame_images[idx])
            tmp_paths.append(str(p))
        cap = caption_frames(tmp_paths, max_frames=5)
        for p in tmp_paths:
            Path(p).unlink(missing_ok=True)
        if cap:
            summary = cap
            print(f"  Caption: {cap}")

    # Vehicle matching
    matched_refs = []
    if all_vehicle_embs:
        clustered = _cluster_embeddings(all_vehicle_embs, threshold=0.90)
        matches = match_against_references(session, clustered)
        seen = set()
        for m in matches:
            if m["reference_name"] not in seen:
                matched_refs.append(m)
                seen.add(m["reference_name"])

    # Face matching
    face_matches = []
    if all_face_embs:
        clustered_faces = _cluster_embeddings(all_face_embs, threshold=0.80)
        face_matches = match_against_face_profiles(
            session, clustered_faces, settings.face.match_threshold
        )

    # Build combined summary
    face_names = [m["display_name"] for m in face_matches]
    if face_names and summary:
        combined = f"{summary}, {', '.join(face_names)}"
    elif face_names:
        combined = ", ".join(face_names)
    else:
        combined = summary

    # Send notifications
    ts = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if matched_refs:
        for m in matched_refs:
            print(f"  NOTIFY: {m['display_name']} arrived at {cam_name} ({combined})")
            notify_arrival(m["display_name"], cam_name, combined, snap_str)
    elif face_matches:
        seen_f: set[str] = set()
        for fm in face_matches:
            if fm["profile_name"] not in seen_f:
                seen_f.add(fm["profile_name"])
                print(f"  NOTIFY: {fm['display_name']} arrived at {cam_name}")
                notify_known_person(fm["display_name"], cam_name, summary, snap_str)
    elif has_persons:
        print(f"  NOTIFY: Unknown visitor at {cam_name} ({summary})")
        notify_unknown_visitor(cam_name, summary, snap_str)
    else:
        print(f"  NOTIFY: Motion at {cam_name} ({summary})")
        notify_motion(cam_name, f"{ts} — {summary}" if summary else ts, snap_str)

    print("  Notification sent!")


if __name__ == "__main__":
    embed_main()
