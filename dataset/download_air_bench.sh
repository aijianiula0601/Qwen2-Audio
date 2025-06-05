#!/bin/bash

set -e

# Configuration
DATA_DIR="data"

echo "=== Downloading AIR-Bench Dataset ==="

# Create placeholder structures for all stages
for stage in pretrain sft dpo; do
    air_dir="${DATA_DIR}/${stage}/air_bench"
    mkdir -p "${air_dir}/audio"
    
    echo "Creating AIR-Bench ${stage} placeholder..."
    cat > "${air_dir}/data.jsonl" << EOF
{"audio_path": "audio/sample1.wav", "text": "Audio description", "instruction": "Describe the audio", "response": "Audio description"}
{"audio_path": "audio/sample2.wav", "text": "Music playing", "instruction": "What do you hear?", "response": "Music playing"}
EOF
done

echo "AIR-Bench placeholders created. Please download actual dataset." 