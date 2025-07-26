#!/bin/bash

# Stage 1: Audio-Text Alignment Pre-training Data Download
# This script downloads large-scale speech recognition datasets for audio-text alignment

set -e

# Configuration
STAGE1_DATA_DIR="data/stage1_alignment"
DOWNLOAD_DIR="downloads/stage1"

echo "========================================================"
echo "Stage 1: Audio-Text Alignment Pre-training Data Download"
echo "========================================================"

# Create directories
mkdir -p "$STAGE1_DATA_DIR"
mkdir -p "$DOWNLOAD_DIR"

# Function to check if file exists and skip download
check_and_download() {
    local url=$1
    local output_path=$2
    local description=$3
    
    if [ -f "$output_path" ]; then
        echo "✓ $description already exists, skipping download"
        return 0
    fi
    
    echo "📥 Downloading $description..."
    wget -c "$url" -O "$output_path"
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully downloaded $description"
    else
        echo "❌ Failed to download $description"
        return 1
    fi
}

# Function to extract tar files
extract_tar() {
    local tar_file=$1
    local extract_dir=$2
    local description=$3
    
    if [ -d "$extract_dir" ]; then
        echo "✓ $description already extracted, skipping"
        return 0
    fi
    
    echo "📦 Extracting $description..."
    mkdir -p "$extract_dir"
    tar -xf "$tar_file" -C "$extract_dir" --strip-components=1
    
    if [ $? -eq 0 ]; then
        echo "✅ Successfully extracted $description"
    else
        echo "❌ Failed to extract $description"
        return 1
    fi
}

echo ""
echo "🎯 Stage 1 focuses on learning basic audio-text alignment"
echo "📊 Datasets: LibriSpeech, CommonVoice, VoxPopuli"
echo ""

# 1. LibriSpeech Dataset
echo "1️⃣  LibriSpeech Dataset"
echo "----------------------------------------"

# LibriSpeech train-clean-100 (24GB)
LIBRISPEECH_URL_100="http://www.openslr.org/resources/12/train-clean-100.tar.gz"
LIBRISPEECH_TAR_100="$DOWNLOAD_DIR/librispeech-train-clean-100.tar.gz"
LIBRISPEECH_DIR_100="$STAGE1_DATA_DIR/librispeech/train-clean-100"

check_and_download "$LIBRISPEECH_URL_100" "$LIBRISPEECH_TAR_100" "LibriSpeech train-clean-100"
extract_tar "$LIBRISPEECH_TAR_100" "$LIBRISPEECH_DIR_100" "LibriSpeech train-clean-100"

# LibriSpeech train-clean-360 (23GB)
LIBRISPEECH_URL_360="http://www.openslr.org/resources/12/train-clean-360.tar.gz"
LIBRISPEECH_TAR_360="$DOWNLOAD_DIR/librispeech-train-clean-360.tar.gz"
LIBRISPEECH_DIR_360="$STAGE1_DATA_DIR/librispeech/train-clean-360"

check_and_download "$LIBRISPEECH_URL_360" "$LIBRISPEECH_TAR_360" "LibriSpeech train-clean-360"
extract_tar "$LIBRISPEECH_TAR_360" "$LIBRISPEECH_DIR_360" "LibriSpeech train-clean-360"

# LibriSpeech dev sets
LIBRISPEECH_DEV_CLEAN_URL="http://www.openslr.org/resources/12/dev-clean.tar.gz"
LIBRISPEECH_DEV_CLEAN_TAR="$DOWNLOAD_DIR/librispeech-dev-clean.tar.gz"
LIBRISPEECH_DEV_CLEAN_DIR="$STAGE1_DATA_DIR/librispeech/dev-clean"

check_and_download "$LIBRISPEECH_DEV_CLEAN_URL" "$LIBRISPEECH_DEV_CLEAN_TAR" "LibriSpeech dev-clean"
extract_tar "$LIBRISPEECH_DEV_CLEAN_TAR" "$LIBRISPEECH_DEV_CLEAN_DIR" "LibriSpeech dev-clean"

echo ""
echo "2️⃣  CommonVoice Dataset (English)"
echo "----------------------------------------"

# Note: CommonVoice requires manual download due to licensing
CV_INFO_FILE="$STAGE1_DATA_DIR/commonvoice_download_info.txt"
cat > "$CV_INFO_FILE" << 'EOF'
CommonVoice Dataset Download Instructions:

1. Go to: https://commonvoice.mozilla.org/en/datasets
2. Select English dataset
3. Download the latest version (requires email registration)
4. Extract to: data/stage1_alignment/commonvoice/
5. Rename folder to: en/

The dataset contains:
- Validated train/dev/test splits
- High-quality crowd-sourced speech data
- Multiple speakers and accents

Size: ~15GB for English
EOF

echo "📝 CommonVoice download instructions saved to: $CV_INFO_FILE"
echo "⚠️  Manual download required due to licensing terms"

echo ""
echo "3️⃣  VoxPopuli Dataset (English subset)"
echo "----------------------------------------"

# VoxPopuli requires manual processing, provide instructions
VOXPOPULI_INFO_FILE="$STAGE1_DATA_DIR/voxpopuli_download_info.txt"
cat > "$VOXPOPULI_INFO_FILE" << 'EOF'
VoxPopuli Dataset Download Instructions:

1. Install datasets library: pip install datasets
2. Use the following Python script to download:

```python
from datasets import load_dataset

# Load English subset
dataset = load_dataset("facebook/voxpopuli", "en", trust_remote_code=True)

# Save to local directory
dataset.save_to_disk("data/stage1_alignment/voxpopuli/en")
```

The dataset contains:
- European Parliament recordings
- Multilingual speech data
- High-quality transcriptions

Size: ~10GB for English subset
EOF

echo "📝 VoxPopuli download instructions saved to: $VOXPOPULI_INFO_FILE"
echo "⚠️  Requires Hugging Face datasets library"

echo ""
echo "4️⃣  Creating Stage 1 Training Data Lists"
echo "----------------------------------------"

# Create data preparation script
STAGE1_PREP_SCRIPT="$STAGE1_DATA_DIR/prepare_stage1_data.py"
cat > "$STAGE1_PREP_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Prepare Stage 1 training data from downloaded datasets.
Converts datasets to LlamaAudio format for audio-text alignment training.
"""

import os
import json
import glob
from pathlib import Path
import librosa
import soundfile as sf
from tqdm import tqdm

def process_librispeech(librispeech_dir, output_file):
    """Process LibriSpeech data."""
    data = []
    
    # Find all transcript files
    trans_files = glob.glob(f"{librispeech_dir}/*/*/*/*.trans.txt")
    
    for trans_file in tqdm(trans_files, desc="Processing LibriSpeech"):
        base_dir = os.path.dirname(trans_file)
        
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) != 2:
                    continue
                    
                audio_id, transcript = parts
                audio_path = os.path.join(base_dir, f"{audio_id}.flac")
                
                if os.path.exists(audio_path):
                    # Check audio duration (skip if too long/short)
                    try:
                        duration = librosa.get_duration(filename=audio_path)
                        if 1.0 <= duration <= 20.0:  # 1-20 seconds
                            data.append({
                                "audio_path": audio_path,
                                "transcript": transcript.upper(),
                                "duration": duration,
                                "dataset": "librispeech",
                                "task": "transcription"
                            })
                    except:
                        continue
    
    # Save to JSONL
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processed {len(data)} LibriSpeech samples -> {output_file}")

def process_commonvoice(cv_dir, output_file):
    """Process CommonVoice data."""
    data = []
    
    # Process train.tsv
    train_tsv = os.path.join(cv_dir, "train.tsv")
    if not os.path.exists(train_tsv):
        print(f"CommonVoice data not found at {cv_dir}")
        return
    
    clips_dir = os.path.join(cv_dir, "clips")
    
    with open(train_tsv, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        header = lines[0].strip().split('\t')
        
        for line in tqdm(lines[1:], desc="Processing CommonVoice"):
            parts = line.strip().split('\t')
            if len(parts) < len(header):
                continue
            
            row = dict(zip(header, parts))
            
            audio_file = row.get('path', '')
            sentence = row.get('sentence', '')
            
            if audio_file and sentence:
                audio_path = os.path.join(clips_dir, audio_file)
                
                if os.path.exists(audio_path):
                    try:
                        duration = librosa.get_duration(filename=audio_path)
                        if 1.0 <= duration <= 15.0:
                            data.append({
                                "audio_path": audio_path,
                                "transcript": sentence,
                                "duration": duration,
                                "dataset": "commonvoice",
                                "task": "transcription"
                            })
                    except:
                        continue
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Processed {len(data)} CommonVoice samples -> {output_file}")

if __name__ == "__main__":
    stage1_dir = Path("data/stage1_alignment")
    
    # Process LibriSpeech
    librispeech_dirs = [
        stage1_dir / "librispeech" / "train-clean-100",
        stage1_dir / "librispeech" / "train-clean-360"
    ]
    
    all_librispeech_data = []
    for lib_dir in librispeech_dirs:
        if lib_dir.exists():
            process_librispeech(str(lib_dir), f"{stage1_dir}/librispeech_temp.jsonl")
            
            # Load and combine
            with open(f"{stage1_dir}/librispeech_temp.jsonl", 'r') as f:
                for line in f:
                    all_librispeech_data.append(json.loads(line))
    
    # Save combined LibriSpeech data
    with open(f"{stage1_dir}/librispeech_train.jsonl", 'w') as f:
        for item in all_librispeech_data:
            f.write(json.dumps(item) + '\n')
    
    # Process CommonVoice if available
    cv_dir = stage1_dir / "commonvoice" / "en"
    if cv_dir.exists():
        process_commonvoice(str(cv_dir), f"{stage1_dir}/commonvoice_train.jsonl")
    
    print("Stage 1 data preparation completed!")
    print(f"LibriSpeech: {len(all_librispeech_data)} samples")
    
    # Create combined training file
    combined_data = []
    
    for file_path in [f"{stage1_dir}/librispeech_train.jsonl", f"{stage1_dir}/commonvoice_train.jsonl"]:
        if os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    combined_data.append(json.loads(line))
    
    # Shuffle and split
    import random
    random.shuffle(combined_data)
    
    train_size = int(0.95 * len(combined_data))
    train_data = combined_data[:train_size]
    val_data = combined_data[train_size:]
    
    # Save train/val splits
    with open(f"{stage1_dir}/train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item) + '\n')
    
    with open(f"{stage1_dir}/val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Final dataset: {len(train_data)} train, {len(val_data)} val samples")
EOF

chmod +x "$STAGE1_PREP_SCRIPT"
echo "📜 Data preparation script created: $STAGE1_PREP_SCRIPT"

echo ""
echo "5️⃣  Next Steps"
echo "----------------------------------------"
echo "1. Wait for LibriSpeech downloads to complete"
echo "2. Manually download CommonVoice (see instructions above)"
echo "3. Run data preparation script:"
echo "   cd data/stage1_alignment && python prepare_stage1_data.py"
echo "4. Start Stage 1 training with:"
echo "   bash scripts/train_stage1.sh"

echo ""
echo "📊 Expected Dataset Sizes:"
echo "   • LibriSpeech: ~50GB, ~280K samples"
echo "   • CommonVoice: ~15GB, ~150K samples"
echo "   • Total: ~65GB, ~430K samples"

echo ""
echo "🎯 Stage 1 Training Objective:"
echo "   Learn basic audio-text alignment through speech recognition"
echo "   Freeze LLM parameters, train audio encoder and connector"

echo ""
echo "========================================================"
echo "Stage 1 data download setup completed!"
echo "========================================================" 