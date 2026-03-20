#!/usr/bin/env bash
# Download SCRFD-10G face detection and ArcFace w600k_r50 recognition ONNX models.
# Run once before starting ring-watch with face detection enabled.
set -euo pipefail

MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)/models"
mkdir -p "$MODEL_DIR"

SCRFD_FILE="$MODEL_DIR/scrfd_10g_bnkps.onnx"
ARCFACE_FILE="$MODEL_DIR/arcface_w600k_r50.onnx"

download() {
    local url="$1"
    local dest="$2"
    local desc="$3"

    if [[ -f "$dest" ]]; then
        echo "$desc already present at $dest"
        return 0
    fi

    echo "Downloading $desc ..."
    if command -v wget &>/dev/null; then
        wget -q --show-progress -O "$dest" "$url" || { rm -f "$dest"; return 1; }
    elif command -v curl &>/dev/null; then
        curl -L --progress-bar -o "$dest" "$url" || { rm -f "$dest"; return 1; }
    else
        echo "Error: neither wget nor curl found. Install one and retry." >&2
        return 1
    fi
    echo "Saved: $dest"
}

# SCRFD-10G — face detection with 5-point landmarks (~17 MB)
download \
    "https://huggingface.co/LPDoctor/insightface/resolve/main/scrfd_10g_bnkps.onnx" \
    "$SCRFD_FILE" \
    "SCRFD-10G face detector" || \
download \
    "https://huggingface.co/Aitrepreneur/insightface/resolve/main/models/antelopev2/scrfd_10g_bnkps.onnx" \
    "$SCRFD_FILE" \
    "SCRFD-10G face detector (mirror)"

# ArcFace w600k_r50 — face recognition ResNet-50 (~166 MB)
download \
    "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l/w600k_r50.onnx" \
    "$ARCFACE_FILE" \
    "ArcFace w600k_r50 face recognizer"

echo ""
echo "Face models ready:"
echo "  SCRFD:   $SCRFD_FILE"
echo "  ArcFace: $ARCFACE_FILE"
echo ""
echo "Next: ring-face add <name> <photo.jpg> --display-name \"Full Name\""
