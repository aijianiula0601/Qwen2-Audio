#!/bin/bash

set -e

# Configuration
DATA_DIR="data"
CACHE_DIR="${DATA_DIR}/cache"
PRETRAIN_DIR="${DATA_DIR}/pretrain"
SFT_DIR="${DATA_DIR}/sft"
DPO_DIR="${DATA_DIR}/dpo"

# Create base directories
mkdir -p "${DATA_DIR}" "${CACHE_DIR}" "${PRETRAIN_DIR}" "${SFT_DIR}" "${DPO_DIR}"

echo "=== Qwen2-Audio Data Download Script ==="
echo "Data directory: ${DATA_DIR}"

# Check dependencies
command -v wget >/dev/null 2>&1 || { echo "wget is required but not installed. Aborting." >&2; exit 1; }
command -v python3 >/dev/null 2>&1 || { echo "python3 is required but not installed. Aborting." >&2; exit 1; }
command -v unzip >/dev/null 2>&1 || { echo "unzip is required but not installed. Aborting." >&2; exit 1; }

# Install required Python packages
echo "Installing required Python packages..."
python3 -c "import numpy, soundfile, torchaudio, librosa, tqdm" 2>/dev/null || {
    pip install numpy soundfile torchaudio librosa tqdm
}

# Download datasets
echo "Starting dataset downloads..."

# Pretrain datasets
echo "Downloading pretrain datasets..."
bash scripts/dataset/download_common_voice.sh
bash scripts/dataset/download_librispeech.sh
bash scripts/dataset/download_air_bench.sh
bash scripts/dataset/download_emotion_datasets.sh
bash scripts/dataset/download_classification_datasets.sh

# SFT datasets
echo "Downloading SFT datasets..."
bash scripts/dataset/download_air_bench.sh

# DPO datasets
echo "Downloading DPO datasets..."
bash scripts/dataset/download_air_bench.sh

# Process emotion datasets
echo "Processing emotion datasets..."
python3 scripts/dataset/process_emotion_datasets.py

# Process classification datasets
echo "Processing classification datasets..."
python3 scripts/dataset/process_classification_datasets.py

# Generate test audio for all datasets
echo "Generating test audio files..."
datasets=(
    "${PRETRAIN_DIR}/common_voice/train"
    "${PRETRAIN_DIR}/librispeech/audio"
    "${PRETRAIN_DIR}/air_bench/audio"
    "${PRETRAIN_DIR}/emotion/iemocap/processed"
    "${PRETRAIN_DIR}/emotion/ravdess/processed"
    "${PRETRAIN_DIR}/emotion/cremad/processed"
    "${PRETRAIN_DIR}/classification/esc50/processed"
    "${PRETRAIN_DIR}/classification/urbansound8k/processed"
    "${PRETRAIN_DIR}/classification/audioset/processed"
    "${SFT_DIR}/air_bench/audio"
    "${DPO_DIR}/air_bench/audio"
)

for dataset_dir in "${datasets[@]}"; do
    python3 scripts/dataset/generate_test_audio.py "${dataset_dir}"
done

echo "=== Data Download Complete ==="
echo "Data structure:"
find "${DATA_DIR}" -type f -name "*.jsonl" | head -10
echo ""
echo "Please review the downloaded data and replace placeholders with actual datasets."
echo "Remember to:"
echo "1. Download actual Common Voice dataset from Mozilla"
echo "2. Download actual LibriSpeech dataset from OpenSLR"
echo "3. Download actual AIR-Bench dataset from the official source"
echo "4. Download emotion datasets (IEMOCAP, RAVDESS, CREMA-D) from their respective sources"
echo "5. Download classification datasets (UrbanSound8K, AudioSet) from their respective sources"
echo "6. Verify audio file formats and paths"
echo "7. Process and convert audio files to the required format (16kHz, mono)" 