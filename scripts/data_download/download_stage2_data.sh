#!/bin/bash

# Stage 2: Audio Understanding & Description Data Download
# This script downloads audio captioning and description datasets

set -e

# Configuration
STAGE2_DATA_DIR="data/stage2_understanding"
DOWNLOAD_DIR="downloads/stage2"

echo "========================================================"
echo "Stage 2: Audio Understanding & Description Data Download"
echo "========================================================"

# Create directories
mkdir -p "$STAGE2_DATA_DIR"
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

echo ""
echo "🎯 Stage 2 focuses on audio understanding and description"
echo "📊 Datasets: AudioCaps, Clotho, WavCaps, FSD50K"
echo ""

# 1. AudioCaps Dataset
echo "1️⃣  AudioCaps Dataset"
echo "----------------------------------------"

AUDIOCAPS_INFO_FILE="$STAGE2_DATA_DIR/audiocaps_download_info.txt"
cat > "$AUDIOCAPS_INFO_FILE" << 'EOF'
AudioCaps Dataset Download Instructions:

AudioCaps is built on AudioSet and requires downloading from YouTube.
We'll use the processed version available on Hugging Face.

1. Install required libraries:
   pip install datasets yt-dlp

2. Use the following Python script:

```python
from datasets import load_dataset
import os

# Download AudioCaps dataset
dataset = load_dataset("cakiki/audiocaps", trust_remote_code=True)

# Save to local directory
dataset.save_to_disk("data/stage2_understanding/audiocaps")

print("AudioCaps dataset downloaded successfully!")
```

Alternative: Download from official source
1. Go to: https://github.com/cdjkim/audiocaps
2. Follow their download instructions
3. Extract to: data/stage2_understanding/audiocaps/

Dataset contains:
- 46K audio clips with captions
- 5 captions per audio clip
- YouTube-sourced content

Size: ~8GB
EOF

echo "📝 AudioCaps download instructions saved to: $AUDIOCAPS_INFO_FILE"

# 2. Clotho Dataset  
echo ""
echo "2️⃣  Clotho Dataset"
echo "----------------------------------------"

CLOTHO_INFO_FILE="$STAGE2_DATA_DIR/clotho_download_info.txt"
cat > "$CLOTHO_INFO_FILE" << 'EOF'
Clotho Dataset Download Instructions:

1. Go to: https://zenodo.org/record/4783391
2. Download the following files:
   - clotho_audio_development.7z (development audio)
   - clotho_audio_validation.7z (validation audio) 
   - clotho_audio_evaluation.7z (evaluation audio)
   - clotho_captions_development.csv
   - clotho_captions_validation.csv
   - clotho_captions_evaluation.csv

3. Extract audio files to:
   - data/stage2_understanding/clotho/development/
   - data/stage2_understanding/clotho/validation/
   - data/stage2_understanding/clotho/evaluation/

4. Place CSV files in:
   - data/stage2_understanding/clotho/

Python download script (requires zenodo_get):
```bash
pip install zenodo_get
zenodo_get 4783391 -o data/stage2_understanding/clotho/
```

Dataset contains:
- 6K audio clips with natural language captions
- 5 captions per audio clip  
- High-quality recordings
- Focus on everyday sounds

Size: ~2GB
EOF

echo "📝 Clotho download instructions saved to: $CLOTHO_INFO_FILE"

# 3. WavCaps Dataset
echo ""
echo "3️⃣  WavCaps Dataset"
echo "----------------------------------------"

WAVCAPS_INFO_FILE="$STAGE2_DATA_DIR/wavcaps_download_info.txt"
cat > "$WAVCAPS_INFO_FILE" << 'EOF'
WavCaps Dataset Download Instructions:

WavCaps is a large-scale dataset with multiple subsets.

1. Install required libraries:
   pip install datasets

2. Download using Python script:

```python
from datasets import load_dataset

# Download WavCaps subsets
subsets = ["audioset", "bbc", "soundbible", "freeaudio", "freesound"]

for subset in subsets:
    print(f"Downloading WavCaps {subset}...")
    try:
        dataset = load_dataset("cakiki/wavcaps", subset, trust_remote_code=True)
        dataset.save_to_disk(f"data/stage2_understanding/wavcaps/{subset}")
        print(f"✅ Downloaded WavCaps {subset}")
    except Exception as e:
        print(f"❌ Failed to download {subset}: {e}")
```

Alternative: Official download
1. Go to: https://github.com/XinhaoMei/WavCaps
2. Follow their data preparation instructions

Dataset contains:
- 400K+ audio-caption pairs
- Multiple domains and sources
- Rich textual descriptions

Total size: ~50GB
EOF

echo "📝 WavCaps download instructions saved to: $WAVCAPS_INFO_FILE"

# 4. FSD50K Dataset (for sound event descriptions)
echo ""
echo "4️⃣  FSD50K Dataset"
echo "----------------------------------------"

FSD50K_INFO_FILE="$STAGE2_DATA_DIR/fsd50k_download_info.txt"
cat > "$FSD50K_INFO_FILE" << 'EOF'
FSD50K Dataset Download Instructions:

FSD50K provides high-quality sound events with labels that we can convert to descriptions.

1. Go to: https://zenodo.org/record/4060432
2. Download:
   - FSD50K.dev_audio.zip (development audio)
   - FSD50K.eval_audio.zip (evaluation audio)  
   - FSD50K.ground_truth.zip (metadata and labels)

3. Extract to: data/stage2_understanding/fsd50k/

Python download script:
```bash
pip install zenodo_get
zenodo_get 4060432 -o data/stage2_understanding/fsd50k/
```

We'll convert the labels to natural language descriptions using templates.

Dataset contains:
- 51K audio clips
- 200 sound event classes
- Professional annotations

Size: ~25GB
EOF

echo "📝 FSD50K download instructions saved to: $FSD50K_INFO_FILE"

echo ""
echo "5️⃣  Creating Stage 2 Training Data Lists"
echo "----------------------------------------"

# Create data preparation script
STAGE2_PREP_SCRIPT="$STAGE2_DATA_DIR/prepare_stage2_data.py"
cat > "$STAGE2_PREP_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Prepare Stage 2 training data from downloaded datasets.
Converts audio captioning datasets to LlamaAudio format for audio understanding training.
"""

import os
import json
import pandas as pd
from pathlib import Path
import librosa
from tqdm import tqdm
import random

def process_audiocaps(audiocaps_dir, output_file):
    """Process AudioCaps dataset."""
    data = []
    
    try:
        from datasets import load_from_disk
        dataset = load_from_disk(audiocaps_dir)
        
        for split in ["train", "test", "valid"]:
            if split in dataset:
                for item in tqdm(dataset[split], desc=f"Processing AudioCaps {split}"):
                    audio_path = item.get("audio", {}).get("path")
                    captions = item.get("captions", [])
                    
                    if audio_path and captions:
                        for caption in captions:
                            if caption.strip():
                                data.append({
                                    "audio_path": audio_path,
                                    "instruction": "请描述这段音频的内容。",
                                    "output": caption,
                                    "dataset": "audiocaps",
                                    "task": "audio_captioning"
                                })
    except Exception as e:
        print(f"Error processing AudioCaps: {e}")
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(data)} AudioCaps samples -> {output_file}")

def process_clotho(clotho_dir, output_file):
    """Process Clotho dataset."""
    data = []
    
    # Process each split
    for split in ["development", "validation", "evaluation"]:
        csv_file = os.path.join(clotho_dir, f"clotho_captions_{split}.csv")
        audio_dir = os.path.join(clotho_dir, split)
        
        if os.path.exists(csv_file) and os.path.exists(audio_dir):
            df = pd.read_csv(csv_file)
            
            for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing Clotho {split}"):
                filename = row["file_name"]
                audio_path = os.path.join(audio_dir, filename)
                
                if os.path.exists(audio_path):
                    # Clotho has 5 captions per audio (caption_1 to caption_5)
                    for i in range(1, 6):
                        caption_col = f"caption_{i}"
                        if caption_col in row and pd.notna(row[caption_col]):
                            caption = row[caption_col].strip()
                            if caption:
                                data.append({
                                    "audio_path": audio_path,
                                    "instruction": "描述这段音频中的声音。",
                                    "output": caption,
                                    "dataset": "clotho",
                                    "task": "audio_captioning"
                                })
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(data)} Clotho samples -> {output_file}")

def process_wavcaps(wavcaps_dir, output_file):
    """Process WavCaps dataset."""
    data = []
    
    subsets = ["audioset", "bbc", "soundbible", "freeaudio", "freesound"]
    
    for subset in subsets:
        subset_dir = os.path.join(wavcaps_dir, subset)
        if os.path.exists(subset_dir):
            try:
                from datasets import load_from_disk
                dataset = load_from_disk(subset_dir)
                
                for split in dataset.keys():
                    for item in tqdm(dataset[split], desc=f"Processing WavCaps {subset} {split}"):
                        audio_path = item.get("audio", {}).get("path")
                        caption = item.get("caption", "")
                        
                        if audio_path and caption.strip():
                            data.append({
                                "audio_path": audio_path,
                                "instruction": "请描述这段音频。",
                                "output": caption,
                                "dataset": f"wavcaps_{subset}",
                                "task": "audio_captioning"
                            })
            except Exception as e:
                print(f"Error processing WavCaps {subset}: {e}")
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(data)} WavCaps samples -> {output_file}")

def process_fsd50k(fsd50k_dir, output_file):
    """Process FSD50K dataset by converting labels to descriptions."""
    data = []
    
    # Load ground truth
    gt_file = os.path.join(fsd50k_dir, "FSD50K.ground_truth", "vocabulary.csv")
    if not os.path.exists(gt_file):
        print(f"FSD50K vocabulary not found at {gt_file}")
        return
    
    # Load vocabulary and create description templates
    vocab_df = pd.read_csv(gt_file)
    label_to_desc = {}
    
    for _, row in vocab_df.iterrows():
        label = row["display_name"]
        # Create natural language descriptions
        templates = [
            f"这段音频包含{label}的声音。",
            f"音频中可以听到{label}。", 
            f"这是{label}的录音。",
            f"音频片段中有{label}的声音。"
        ]
        label_to_desc[row["mid"]] = templates
    
    # Process development and evaluation sets
    for split in ["dev", "eval"]:
        audio_dir = os.path.join(fsd50k_dir, f"FSD50K.{split}_audio")
        labels_file = os.path.join(fsd50k_dir, "FSD50K.ground_truth", f"{split}.csv")
        
        if os.path.exists(audio_dir) and os.path.exists(labels_file):
            labels_df = pd.read_csv(labels_file)
            
            for _, row in tqdm(labels_df.iterrows(), total=len(labels_df), desc=f"Processing FSD50K {split}"):
                filename = row["fname"]
                audio_path = os.path.join(audio_dir, filename)
                labels = row["labels"].split(",")
                
                if os.path.exists(audio_path):
                    for label in labels:
                        label = label.strip()
                        if label in label_to_desc:
                            description = random.choice(label_to_desc[label])
                            data.append({
                                "audio_path": audio_path,
                                "instruction": "这段音频中有什么声音？",
                                "output": description,
                                "dataset": "fsd50k",
                                "task": "sound_classification"
                            })
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"Processed {len(data)} FSD50K samples -> {output_file}")

if __name__ == "__main__":
    stage2_dir = Path("data/stage2_understanding")
    
    # Process each dataset
    datasets = [
        ("audiocaps", process_audiocaps),
        ("clotho", process_clotho), 
        ("wavcaps", process_wavcaps),
        ("fsd50k", process_fsd50k)
    ]
    
    all_data = []
    
    for dataset_name, process_func in datasets:
        dataset_dir = stage2_dir / dataset_name
        output_file = stage2_dir / f"{dataset_name}_processed.jsonl"
        
        if dataset_dir.exists():
            print(f"\n Processing {dataset_name}...")
            process_func(str(dataset_dir), str(output_file))
            
            # Load processed data
            if output_file.exists():
                with open(output_file, 'r') as f:
                    for line in f:
                        all_data.append(json.loads(line))
    
    if all_data:
        # Shuffle and split data
        random.shuffle(all_data)
        
        train_size = int(0.95 * len(all_data))
        train_data = all_data[:train_size]
        val_data = all_data[train_size:]
        
        # Save train/val splits
        with open(stage2_dir / "train.jsonl", 'w') as f:
            for item in train_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(stage2_dir / "val.jsonl", 'w') as f:
            for item in val_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"\n Stage 2 data preparation completed!")
        print(f"Train samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        print(f"Total samples: {len(all_data)}")
        
        # Dataset statistics
        dataset_counts = {}
        for item in all_data:
            dataset = item["dataset"]
            dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        
        print("\n Dataset distribution:")
        for dataset, count in sorted(dataset_counts.items()):
            print(f"  {dataset}: {count:,} samples")
    else:
        print("No data processed. Please download datasets first.")
EOF

chmod +x "$STAGE2_PREP_SCRIPT"
echo "📜 Data preparation script created: $STAGE2_PREP_SCRIPT"

echo ""
echo "6️⃣  Automated Download Script"
echo "----------------------------------------"

# Create automated download script
AUTO_DOWNLOAD_SCRIPT="$STAGE2_DATA_DIR/auto_download.py"
cat > "$AUTO_DOWNLOAD_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Automated download script for Stage 2 datasets.
Downloads datasets that can be automatically downloaded.
"""

import os
import subprocess
from pathlib import Path

def download_audiocaps():
    """Download AudioCaps dataset."""
    try:
        from datasets import load_dataset
        print("📥 Downloading AudioCaps...")
        dataset = load_dataset("cakiki/audiocaps", trust_remote_code=True)
        dataset.save_to_disk("data/stage2_understanding/audiocaps")
        print("✅ AudioCaps downloaded successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to download AudioCaps: {e}")
        return False

def download_wavcaps():
    """Download WavCaps dataset."""
    try:
        from datasets import load_dataset
        subsets = ["audioset", "bbc", "soundbible"]  # Download smaller subsets first
        
        for subset in subsets:
            print(f"📥 Downloading WavCaps {subset}...")
            try:
                dataset = load_dataset("cakiki/wavcaps", subset, trust_remote_code=True)
                dataset.save_to_disk(f"data/stage2_understanding/wavcaps/{subset}")
                print(f"✅ WavCaps {subset} downloaded successfully!")
            except Exception as e:
                print(f"❌ Failed to download WavCaps {subset}: {e}")
        return True
    except Exception as e:
        print(f"❌ Failed to download WavCaps: {e}")
        return False

def download_with_zenodo(record_id, output_dir, description):
    """Download dataset using zenodo_get."""
    try:
        # Check if zenodo_get is installed
        subprocess.run(["zenodo_get", "--version"], check=True, capture_output=True)
        
        print(f"📥 Downloading {description} from Zenodo...")
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = ["zenodo_get", str(record_id), "-o", output_dir]
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        print(f"✅ {description} downloaded successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to download {description}: {e}")
        return False
    except FileNotFoundError:
        print(f"⚠️  zenodo_get not found. Install with: pip install zenodo_get")
        return False

if __name__ == "__main__":
    print("🚀 Starting Stage 2 automatic downloads...")
    
    # Create output directory
    os.makedirs("data/stage2_understanding", exist_ok=True)
    
    # Download datasets
    results = {}
    
    # AudioCaps
    results["AudioCaps"] = download_audiocaps()
    
    # WavCaps
    results["WavCaps"] = download_wavcaps()
    
    # Clotho (requires zenodo_get)
    results["Clotho"] = download_with_zenodo(
        4783391, 
        "data/stage2_understanding/clotho/",
        "Clotho dataset"
    )
    
    # FSD50K (requires zenodo_get)
    results["FSD50K"] = download_with_zenodo(
        4060432,
        "data/stage2_understanding/fsd50k/", 
        "FSD50K dataset"
    )
    
    # Print results
    print("\n" + "="*50)
    print("DOWNLOAD SUMMARY")
    print("="*50)
    
    for dataset, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"{dataset:<15} {status}")
    
    successful = sum(results.values())
    total = len(results)
    
    print(f"\nDownloaded {successful}/{total} datasets successfully")
    
    if successful > 0:
        print("\n📝 Next steps:")
        print("1. Run data preparation: python prepare_stage2_data.py")
        print("2. Start Stage 2 training: bash scripts/train_stage2.sh")
    else:
        print("\n⚠️  Please check the download instructions and try again")
EOF

chmod +x "$AUTO_DOWNLOAD_SCRIPT"
echo "📜 Automated download script created: $AUTO_DOWNLOAD_SCRIPT"

echo ""
echo "7️⃣  Next Steps"
echo "----------------------------------------"
echo "1. Try automated download first:"
echo "   cd data/stage2_understanding && python auto_download.py"
echo ""
echo "2. For datasets that require manual download, follow instructions in:"
echo "   - $AUDIOCAPS_INFO_FILE"
echo "   - $CLOTHO_INFO_FILE" 
echo "   - $WAVCAPS_INFO_FILE"
echo "   - $FSD50K_INFO_FILE"
echo ""
echo "3. Run data preparation:"
echo "   cd data/stage2_understanding && python prepare_stage2_data.py"
echo ""
echo "4. Start Stage 2 training:"
echo "   bash scripts/train_stage2.sh"

echo ""
echo "📊 Expected Dataset Sizes:"
echo "   • AudioCaps: ~8GB, ~230K samples"
echo "   • Clotho: ~2GB, ~24K samples"
echo "   • WavCaps: ~50GB, ~400K samples"
echo "   • FSD50K: ~25GB, ~51K samples"
echo "   • Total: ~85GB, ~705K samples"

echo ""
echo "🎯 Stage 2 Training Objective:"
echo "   Learn audio understanding and description capabilities"
echo "   Train audio encoder + connector, optionally LLM with LoRA"

echo ""
echo "========================================================"
echo "Stage 2 data download setup completed!"
echo "========================================================" 