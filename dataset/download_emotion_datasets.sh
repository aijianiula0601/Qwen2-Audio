#!/bin/bash

set -e

# Configuration
DATA_DIR="data"
CACHE_DIR="${DATA_DIR}/cache"
PRETRAIN_DIR="${DATA_DIR}/pretrain"

# Create directories
mkdir -p "${CACHE_DIR}" "${PRETRAIN_DIR}/emotion"

echo "=== Downloading Emotion Datasets ==="

# IEMOCAP Dataset
echo "Setting up IEMOCAP dataset..."
iemocap_dir="${PRETRAIN_DIR}/emotion/iemocap"
mkdir -p "${iemocap_dir}"

echo "Note: IEMOCAP dataset requires license agreement. Please visit:"
echo "https://sail.usc.edu/iemocap/iemocap_release.htm"
echo "After downloading, place the data in: ${iemocap_dir}"

# RAVDESS Dataset
echo "Setting up RAVDESS dataset..."
ravdess_dir="${PRETRAIN_DIR}/emotion/ravdess"
mkdir -p "${ravdess_dir}"

echo "Note: RAVDESS dataset requires registration. Please visit:"
echo "https://zenodo.org/record/1188976"
echo "After downloading, place the data in: ${ravdess_dir}"

# CREMA-D Dataset
echo "Setting up CREMA-D dataset..."
cremad_dir="${PRETRAIN_DIR}/emotion/cremad"
mkdir -p "${cremad_dir}"

echo "Note: CREMA-D dataset requires registration. Please visit:"
echo "https://github.com/CheyneyComputerScience/CREMA-D"
echo "After downloading, place the data in: ${cremad_dir}"

# Create data.jsonl templates
for dataset in iemocap ravdess cremad; do
    cat > "${PRETRAIN_DIR}/emotion/${dataset}/data.jsonl" << EOF
{"audio_path": "audio/sample1.wav", "text": "Sample text", "emotion": "neutral", "instruction": "What emotion is expressed in this audio?", "response": "The audio expresses a neutral emotion."}
{"audio_path": "audio/sample2.wav", "text": "Sample text", "emotion": "happy", "instruction": "What emotion is expressed in this audio?", "response": "The audio expresses a happy emotion."}
EOF
done

echo "=== Emotion Datasets Setup Complete ==="
echo "Please download the actual datasets and place them in their respective directories."
echo "After downloading, run the processing script to convert the audio files to the required format." 