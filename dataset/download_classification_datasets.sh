#!/bin/bash

set -e

# Configuration
DATA_DIR="data"
CACHE_DIR="${DATA_DIR}/cache"
PRETRAIN_DIR="${DATA_DIR}/pretrain"

# Create directories
mkdir -p "${CACHE_DIR}" "${PRETRAIN_DIR}/classification"

echo "=== Downloading Audio Classification Datasets ==="

# ESC-50 Dataset
echo "Setting up ESC-50 dataset..."
esc50_dir="${PRETRAIN_DIR}/classification/esc50"
mkdir -p "${esc50_dir}"

echo "Downloading ESC-50 dataset..."
wget -O "${CACHE_DIR}/ESC-50.zip" "https://github.com/karolpiczak/ESC-50/archive/refs/heads/master.zip" || {
    echo "Failed to download ESC-50 dataset"
    exit 1
}

# Extract ESC-50
if [ ! -d "${esc50_dir}/ESC-50-master" ]; then
    echo "Extracting ESC-50 dataset..."
    unzip -q "${CACHE_DIR}/ESC-50.zip" -d "${esc50_dir}"
fi

# UrbanSound8K Dataset
echo "Setting up UrbanSound8K dataset..."
urbansound_dir="${PRETRAIN_DIR}/classification/urbansound8k"
mkdir -p "${urbansound_dir}"

echo "Note: UrbanSound8K dataset requires registration. Please visit:"
echo "https://urbansounddataset.weebly.com/urbansound8k.html"
echo "After downloading, place the data in: ${urbansound_dir}"

# AudioSet Dataset
echo "Setting up AudioSet dataset..."
audioset_dir="${PRETRAIN_DIR}/classification/audioset"
mkdir -p "${audioset_dir}"

echo "Note: AudioSet dataset requires registration. Please visit:"
echo "https://research.google.com/audioset/"
echo "After downloading, place the data in: ${audioset_dir}"

# Create data.jsonl templates
for dataset in esc50 urbansound8k audioset; do
    cat > "${PRETRAIN_DIR}/classification/${dataset}/data.jsonl" << EOF
{"audio_path": "audio/sample1.wav", "text": "Sample text", "category": "dog", "instruction": "What sound is this?", "response": "This is the sound of a dog barking."}
{"audio_path": "audio/sample2.wav", "text": "Sample text", "category": "car", "instruction": "What sound is this?", "response": "This is the sound of a car engine."}
EOF
done

echo "=== Audio Classification Datasets Setup Complete ==="
echo "Please download the actual datasets and place them in their respective directories."
echo "After downloading, run the processing script to convert the audio files to the required format." 