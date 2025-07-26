#!/usr/bin/env python3
"""
Create Sample Data for LlamaAudio Training

This script creates sample training data for testing the LlamaAudio framework.
"""

import json
import os
from pathlib import Path
import argparse


def create_sample_data(output_dir: str, num_samples: int = 100):
    """Create sample training data."""
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Sample data templates
    templates = [
        {
            "instruction": "请描述这段音频的内容",
            "outputs": [
                "这是一段优美的钢琴音乐，旋律舒缓动人。",
                "这段音频包含了轻柔的小提琴演奏。",
                "这是一首节奏明快的流行歌曲。",
                "这段音频是自然环境的声音，包含鸟鸣和风声。",
                "这是一段人声对话，语调平和友好。"
            ]
        },
        {
            "instruction": "这段音频是什么类型的？",
            "outputs": [
                "这是古典音乐类型的音频。",
                "这属于环境音效类型。",
                "这是语音通话类型的音频。",
                "这是摇滚音乐类型。",
                "这是自然声音类型的音频。"
            ]
        },
        {
            "instruction": "分析音频中的情感色彩",
            "outputs": [
                "这段音频传达出平静和宁静的情感。",
                "音频中充满了快乐和活力的情感。",
                "这段音频给人带来忧郁和沉思的感觉。",
                "音频表现出紧张和刺激的情绪。",
                "这段音频营造了温暖和舒适的氛围。"
            ]
        }
    ]
    
    # Conversation templates
    conversation_templates = [
        [
            {"role": "user", "content": "请告诉我这段音频里有什么？"},
            {"role": "assistant", "content": "这段音频包含了悠扬的音乐声，听起来像是古典音乐作品。"}
        ],
        [
            {"role": "user", "content": "这个声音让你联想到什么？"},
            {"role": "assistant", "content": "这个声音让我联想到宁静的夜晚和满天的星星。"}
        ],
        [
            {"role": "user", "content": "音频中的主要元素是什么？"},
            {"role": "assistant", "content": "主要元素包括弦乐器的演奏和轻柔的背景音。"}
        ]
    ]
    
    # Generate training data
    train_data = []
    val_data = []
    
    for i in range(num_samples):
        # Create dummy audio path
        audio_path = f"audio/sample_{i:06d}.wav"
        
        if i % 3 == 0:
            # Conversation format
            sample = {
                "audio_path": audio_path,
                "conversation": conversation_templates[i % len(conversation_templates)]
            }
        else:
            # Instruction format
            template = templates[i % len(templates)]
            output = template["outputs"][i % len(template["outputs"])]
            
            sample = {
                "audio_path": audio_path,
                "instruction": template["instruction"],
                "output": output
            }
        
        # Split into train/val (90/10)
        if i < int(num_samples * 0.9):
            train_data.append(sample)
        else:
            val_data.append(sample)
    
    # Save training data
    train_file = output_path / "train.jsonl"
    with open(train_file, 'w', encoding='utf-8') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save validation data
    val_file = output_path / "val.jsonl"
    with open(val_file, 'w', encoding='utf-8') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Create README for the dataset
    readme_content = f"""# Sample Dataset for LlamaAudio

This is a synthetic dataset created for testing the LlamaAudio framework.

## Dataset Statistics
- Total samples: {num_samples}
- Training samples: {len(train_data)}
- Validation samples: {len(val_data)}

## Data Format
Each line in the JSONL files contains:
- `audio_path`: Path to audio file (dummy paths)
- `instruction`: Instruction for the model (for instruction-following format)
- `output`: Expected output (for instruction-following format)
- `conversation`: Multi-turn conversation (for conversation format)

## Note
This is synthetic data for testing purposes. In real usage, you would need:
1. Actual audio files in the specified paths
2. Real human-annotated audio descriptions
3. Proper audio-text alignment

## Usage
Update the `train_data_path` and `val_data_path` in your training config to point to these files:

```yaml
training:
  train_data_path: "{train_file}"
  val_data_path: "{val_file}"
```
"""
    
    readme_file = output_path / "README.md"
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"Sample data created successfully!")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Files saved to: {output_path}")
    print(f"  - {train_file}")
    print(f"  - {val_file}")
    print(f"  - {readme_file}")
    print()
    print("Note: This is synthetic data for testing. You'll need real audio files and descriptions for actual training.")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="Create sample data for LlamaAudio")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data",
        help="Output directory for sample data"
    )
    parser.add_argument(
        "--num_samples", 
        type=int, 
        default=100,
        help="Number of samples to create"
    )
    
    args = parser.parse_args()
    
    create_sample_data(args.output_dir, args.num_samples)


if __name__ == "__main__":
    main() 