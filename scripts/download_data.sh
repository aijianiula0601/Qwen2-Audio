#!/bin/bash

# Qwen2-Audio Data Download and Preparation Script
# Downloads common datasets for audio-language training

set -e

# Configuration
DATA_DIR="data"
CACHE_DIR="${DATA_DIR}/cache"
PRETRAIN_DIR="${DATA_DIR}/pretrain"
SFT_DIR="${DATA_DIR}/sft"
DPO_DIR="${DATA_DIR}/dpo"

# Create directories
mkdir -p "${DATA_DIR}" "${CACHE_DIR}" "${PRETRAIN_DIR}" "${SFT_DIR}" "${DPO_DIR}"

echo "=== Qwen2-Audio Data Download Script ==="
echo "Data directory: ${DATA_DIR}"

# Function to download file if not exists
download_if_not_exists() {
    local url=$1
    local output_path=$2
    local description=$3
    
    if [ ! -f "${output_path}" ]; then
        echo "Downloading ${description}..."
        wget -O "${output_path}" "${url}" || {
            echo "Failed to download ${description}"
            return 1
        }
    else
        echo "${description} already exists, skipping..."
    fi
}

# Function to extract archive
extract_archive() {
    local archive_path=$1
    local extract_dir=$2
    local description=$3
    
    echo "Extracting ${description}..."
    case "${archive_path}" in
        *.tar.gz|*.tgz)
            tar -xzf "${archive_path}" -C "${extract_dir}"
            ;;
        *.tar.bz2)
            tar -xjf "${archive_path}" -C "${extract_dir}"
            ;;
        *.zip)
            unzip -q "${archive_path}" -d "${extract_dir}"
            ;;
        *)
            echo "Unsupported archive format: ${archive_path}"
            return 1
            ;;
    esac
}

# Download Common Voice dataset (for pretraining)
download_common_voice() {
    echo "=== Downloading Common Voice Dataset ==="
    
    local cv_dir="${PRETRAIN_DIR}/common_voice"
    mkdir -p "${cv_dir}"
    
    # Note: This would normally require registration and agreement to Mozilla's terms
    # For demonstration, we'll create a placeholder structure
    echo "Creating Common Voice placeholder structure..."
    mkdir -p "${cv_dir}/train" "${cv_dir}/dev" "${cv_dir}/test"
    
    # Create sample data.jsonl
    cat > "${cv_dir}/data.jsonl" << 'EOF'
{"audio_path": "train/sample1.wav", "text": "Hello world", "instruction": "", "response": "Hello world"}
{"audio_path": "train/sample2.wav", "text": "This is a test", "instruction": "", "response": "This is a test"}
EOF
    
    echo "Common Voice structure created. Please populate with actual data."
}

# Download LibriSpeech dataset (for pretraining)
download_librispeech() {
    echo "=== Downloading LibriSpeech Dataset ==="
    
    local libri_dir="${PRETRAIN_DIR}/librispeech"
    mkdir -p "${libri_dir}"
    
    # Download LibriSpeech train-clean-100 subset
    local libri_url="http://www.openslr.org/resources/12/train-clean-100.tar.gz"
    local libri_archive="${CACHE_DIR}/train-clean-100.tar.gz"
    
    download_if_not_exists "${libri_url}" "${libri_archive}" "LibriSpeech train-clean-100"
    
    if [ ! -d "${libri_dir}/LibriSpeech" ]; then
        extract_archive "${libri_archive}" "${libri_dir}" "LibriSpeech"
    fi
    
    # Convert LibriSpeech to our format
    echo "Converting LibriSpeech to training format..."
    python3 << 'EOF'
import os
import json
import glob

def convert_librispeech(libri_dir, output_file):
    data = []
    
    # Find all .trans.txt files
    trans_files = glob.glob(os.path.join(libri_dir, "LibriSpeech", "train-clean-100", "*", "*", "*.trans.txt"))
    
    for trans_file in trans_files:
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    audio_path = os.path.join(os.path.dirname(trans_file), f"{file_id}.flac")
                    if os.path.exists(audio_path):
                        data.append({
                            "audio_path": audio_path,
                            "text": text.lower(),
                            "instruction": "",
                            "response": text.lower()
                        })
    
    # Write data.jsonl
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(data)} LibriSpeech samples")

if __name__ == "__main__":
    convert_librispeech("data/pretrain/librispeech", "data/pretrain/librispeech/data.jsonl")
EOF
}

# Download AIR-Bench dataset (for all stages)
download_air_bench() {
    echo "=== Downloading AIR-Bench Dataset ==="
    
    # Note: AIR-Bench would need to be downloaded from the official source
    # Creating placeholder structures for demonstration
    
    for stage in pretrain sft dpo; do
        local air_dir="${DATA_DIR}/${stage}/air_bench"
        mkdir -p "${air_dir}"
        
        echo "Creating AIR-Bench ${stage} placeholder..."
        cat > "${air_dir}/data.jsonl" << EOF
{"audio_path": "audio/sample1.wav", "text": "Audio description", "instruction": "Describe the audio", "response": "Audio description"}
{"audio_path": "audio/sample2.wav", "text": "Music playing", "instruction": "What do you hear?", "response": "Music playing"}
EOF
        
        mkdir -p "${air_dir}/audio"
    done
    
    echo "AIR-Bench placeholders created. Please download actual dataset."
}

# Create synthetic audio for testing
create_test_audio() {
    echo "=== Creating Test Audio Files ==="
    
    # Install required Python packages if not present
    python3 -c "import numpy, soundfile" 2>/dev/null || {
        echo "Installing audio processing dependencies..."
        pip install numpy soundfile
    }
    
    python3 << 'EOF'
import numpy as np
import soundfile as sf
import os

def create_test_audio(output_dir, duration=3.0, sample_rate=16000):
    """Create simple test audio files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sine wave audio
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Different frequencies for different samples
    frequencies = [440, 880, 1320]  # A4, A5, E6
    
    for i, freq in enumerate(frequencies):
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        output_path = os.path.join(output_dir, f"sample{i+1}.wav")
        sf.write(output_path, audio, sample_rate)
        print(f"Created {output_path}")

# Create test audio for all datasets
datasets = [
    "data/pretrain/common_voice/train",
    "data/pretrain/librispeech/audio",
    "data/pretrain/air_bench/audio",
    "data/sft/air_bench/audio",
    "data/dpo/air_bench/audio"
]

for dataset_dir in datasets:
    create_test_audio(dataset_dir)
EOF
}

# Main execution
main() {
    echo "Starting data download and preparation..."
    
    # Check dependencies
    command -v wget >/dev/null 2>&1 || { echo "wget is required but not installed. Aborting." >&2; exit 1; }
    command -v python3 >/dev/null 2>&1 || { echo "python3 is required but not installed. Aborting." >&2; exit 1; }
    
    # Download datasets
    download_common_voice
    download_librispeech
    download_air_bench
    
    # Create test audio
    create_test_audio
    
    echo "=== Data Download Complete ==="
    echo "Data structure:"
    find "${DATA_DIR}" -type f -name "*.jsonl" | head -10
    echo ""
    echo "Please review the downloaded data and replace placeholders with actual datasets."
    echo "Remember to:"
    echo "1. Download actual Common Voice dataset from Mozilla"
    echo "2. Download actual AIR-Bench dataset from the official source"
    echo "3. Verify audio file formats and paths"
}

# Run main function
main "$@" 