#!/bin/bash

set -e

# Configuration
DATA_DIR="data"
PRETRAIN_DIR="${DATA_DIR}/pretrain"

# Create directories
mkdir -p "${PRETRAIN_DIR}/common_voice"

echo "=== Downloading Common Voice Dataset ==="

# Create placeholder structure
echo "Creating Common Voice placeholder structure..."
mkdir -p "${PRETRAIN_DIR}/common_voice/train" \
         "${PRETRAIN_DIR}/common_voice/dev" \
         "${PRETRAIN_DIR}/common_voice/test"

# Create sample data.jsonl
cat > "${PRETRAIN_DIR}/common_voice/data.jsonl" << 'EOF'
{"audio_path": "train/sample1.wav", "text": "Hello world", "instruction": "", "response": "Hello world"}
{"audio_path": "train/sample2.wav", "text": "This is a test", "instruction": "", "response": "This is a test"}
EOF

echo "Common Voice structure created. Please populate with actual data." 