#!/bin/bash

set -e

# Configuration
DATA_DIR="data"
CACHE_DIR="${DATA_DIR}/cache"
PRETRAIN_DIR="${DATA_DIR}/pretrain"

# Create directories
mkdir -p "${CACHE_DIR}" "${PRETRAIN_DIR}/librispeech"

echo "=== Downloading LibriSpeech Dataset ==="

# Download LibriSpeech train-clean-100 subset
libri_url="http://www.openslr.org/resources/12/train-clean-100.tar.gz"
libri_archive="${CACHE_DIR}/train-clean-100.tar.gz"

# Download if not exists
if [ ! -f "${libri_archive}" ]; then
    echo "Downloading LibriSpeech train-clean-100..."
    wget -O "${libri_archive}" "${libri_url}" || {
        echo "Failed to download LibriSpeech"
        exit 1
    }
else
    echo "LibriSpeech archive already exists, skipping download..."
fi

# Extract if not already extracted
if [ ! -d "${PRETRAIN_DIR}/librispeech/LibriSpeech" ]; then
    echo "Extracting LibriSpeech..."
    tar -xzf "${libri_archive}" -C "${PRETRAIN_DIR}/librispeech"
fi

# Convert to training format
echo "Converting LibriSpeech to training format..."
python3 process_librispeech.py \
    "${PRETRAIN_DIR}/librispeech" \
    "${PRETRAIN_DIR}/librispeech/data.jsonl" 