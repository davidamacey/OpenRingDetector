#!/usr/bin/env bash
# Download YOLO11m detection weights from ultralytics asset releases.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/../models"
mkdir -p "$MODELS_DIR"

MODEL="yolo11m.pt"
DEST="$MODELS_DIR/$MODEL"

if [[ -f "$DEST" ]]; then
    echo "Model already exists: $DEST"
    exit 0
fi

URL="https://github.com/ultralytics/assets/releases/download/v8.3.0/$MODEL"

echo "Downloading $MODEL..."
curl -L --progress-bar -o "$DEST" "$URL"
echo "Saved: $DEST"
