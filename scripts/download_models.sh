#!/bin/bash

# Qwen2-Audio Model Download Script
# This script downloads the required models for training

set -e

# Default configurations
MODEL_TYPE="qwen2_0.5b"
MODELS_DIR="Qwen"
SKIP_DOWNLOAD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_TYPE="$2"
            shift 2
            ;;
        --models-dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --model MODEL_TYPE         Model type (qwen2_0.5b, qwen2_7b, qwen2_70b, llama3_8b)"
            echo "  --models-dir DIR           Directory to store models (default: Qwen)"
            echo "  --skip-download            Skip download and only verify models"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Example:"
            echo "  bash scripts/download_models.sh --model qwen2_0.5b"
            echo "  bash scripts/download_models.sh --model qwen2_7b --models-dir models"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Model configurations
declare -A MODEL_CONFIGS
MODEL_CONFIGS["qwen2_0.5b"]="Qwen/Qwen2-0.5B"
MODEL_CONFIGS["qwen2_7b"]="Qwen/Qwen2-7B"
MODEL_CONFIGS["qwen2_70b"]="Qwen/Qwen2-70B"
MODEL_CONFIGS["llama3_8b"]="meta-llama/Meta-Llama-3-8B"

# Validate model type
if [[ ! ${MODEL_CONFIGS[$MODEL_TYPE]+_} ]]; then
    echo "Error: Invalid model type '$MODEL_TYPE'"
    echo "Supported types: ${!MODEL_CONFIGS[@]}"
    exit 1
fi

MODEL_NAME=${MODEL_CONFIGS[$MODEL_TYPE]}
MODEL_DIR="${MODELS_DIR}/${MODEL_NAME##*/}"

echo "=== Qwen2-Audio Model Download ==="
echo "Model type: $MODEL_TYPE"
echo "Model name: $MODEL_NAME"
echo "Model directory: $MODEL_DIR"
echo "Skip download: $SKIP_DOWNLOAD"

# Create models directory
mkdir -p "$MODELS_DIR"

# Function to check if model is properly downloaded
check_model_downloaded() {
    local model_path=$1
    local required_files=("config.json" "pytorch_model.bin" "tokenizer.json" "tokenizer_config.json")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "${model_path}/${file}" ]]; then
            return 1
        fi
    done
    return 0
}

# Function to download model using Python
download_model() {
    local model_name=$1
    local model_dir=$2
    
    echo "Downloading model: $model_name"
    echo "Target directory: $model_dir"
    
    # Create a Python script to download the model
    cat > /tmp/download_model.py << EOF
import os
import sys
import shutil
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

def download_model(model_name, model_dir):
    try:
        print(f"Downloading {model_name} to {model_dir}...")
        
        # Download model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # Download tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            local_files_only=False
        )
        
        # Save model and tokenizer
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        
        print(f"✅ Successfully downloaded {model_name} to {model_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False

def convert_hf_cache_to_standard(cache_dir, target_dir):
    """Convert Huggingface cache format to standard format"""
    try:
        print(f"Converting cache format from {cache_dir} to {target_dir}")
        
        # Find the snapshot directory
        snapshots_dir = os.path.join(cache_dir, "snapshots")
        if not os.path.exists(snapshots_dir):
            print(f"❌ Snapshots directory not found in {cache_dir}")
            return False
            
        # Get the first (and usually only) snapshot
        snapshots = os.listdir(snapshots_dir)
        if not snapshots:
            print(f"❌ No snapshots found in {snapshots_dir}")
            return False
            
        snapshot_dir = os.path.join(snapshots_dir, snapshots[0])
        print(f"Using snapshot: {snapshot_dir}")
        
        # Create target directory
        os.makedirs(target_dir, exist_ok=True)
        
        # Copy all files from snapshot to target
        for item in os.listdir(snapshot_dir):
            src = os.path.join(snapshot_dir, item)
            dst = os.path.join(target_dir, item)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
            elif os.path.isdir(src):
                shutil.copytree(src, dst, dirs_exist_ok=True)
        
        print(f"✅ Successfully converted cache to {target_dir}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to convert cache: {e}")
        return False

if __name__ == "__main__":
    model_name = sys.argv[1]
    model_dir = sys.argv[2]
    
    # Check if we have a cache directory
    cache_dir = os.path.join(model_dir, "models--" + model_name.replace("/", "--"))
    if os.path.exists(cache_dir):
        print(f"Found cache directory: {cache_dir}")
        success = convert_hf_cache_to_standard(cache_dir, model_dir)
    else:
        print(f"No cache directory found, downloading fresh...")
        success = download_model(model_name, model_dir)
    
    sys.exit(0 if success else 1)
EOF

    # Run the download script
    python /tmp/download_model.py "$model_name" "$model_dir"
    local exit_code=$?
    
    # Clean up
    rm -f /tmp/download_model.py
    
    return $exit_code
}

# Check if model is already downloaded
if check_model_downloaded "$MODEL_DIR"; then
    echo "✅ Model $MODEL_TYPE is already downloaded in $MODEL_DIR"
    echo "Model files:"
    ls -la "$MODEL_DIR"
else
    echo "❌ Model $MODEL_TYPE is not properly downloaded in $MODEL_DIR"
    
    if [ "$SKIP_DOWNLOAD" = true ]; then
        echo "Skipping download (--skip-download specified)"
        echo "Please manually download the model or run without --skip-download"
        exit 1
    fi
    
    # Download the model
    if download_model "$MODEL_NAME" "$MODEL_DIR"; then
        echo "✅ Model download completed successfully!"
        
        # Verify download
        if check_model_downloaded "$MODEL_DIR"; then
            echo "✅ Model verification passed!"
            echo "Model files:"
            ls -la "$MODEL_DIR"
        else
            echo "❌ Model verification failed!"
            echo "Some required files are missing:"
            ls -la "$MODEL_DIR" || true
            exit 1
        fi
    else
        echo "❌ Model download failed!"
        exit 1
    fi
fi

echo ""
echo "=== Model Download Summary ==="
echo "Model type: $MODEL_TYPE"
echo "Model directory: $MODEL_DIR"
echo "Status: ✅ Ready for training"
echo ""
echo "You can now run training with:"
echo "  bash scripts/train_pretrain_multinode.sh --model $MODEL_TYPE --nnodes 1 --node-rank 0 --master-addr localhost --master-port 29500 --gpus-per-node 8" 