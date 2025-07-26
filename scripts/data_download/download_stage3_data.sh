#!/bin/bash

# Stage 3: Instruction Following & Conversation Data Download
# This script downloads audio instruction-following and conversation datasets

set -e

# Configuration
STAGE3_DATA_DIR="data/stage3_conversation"
DOWNLOAD_DIR="downloads/stage3"

echo "========================================================"
echo "Stage 3: Instruction Following & Conversation Data Download"
echo "========================================================"

# Create directories
mkdir -p "$STAGE3_DATA_DIR"
mkdir -p "$DOWNLOAD_DIR"

echo ""
echo "🎯 Stage 3 focuses on instruction following and conversation"
echo "📊 Datasets: Alpaca-Audio, ShareGPT-Audio, Multi-turn Audio QA"
echo ""

# 1. Create Alpaca-Style Audio Instructions
echo "1️⃣  Alpaca-Style Audio Instructions"
echo "----------------------------------------"

ALPACA_AUDIO_INFO="$STAGE3_DATA_DIR/alpaca_audio_info.txt"
cat > "$ALPACA_AUDIO_INFO" << 'EOF'
Alpaca-Style Audio Instructions:

Since there's limited publicly available audio instruction-following data,
we'll create it by combining existing datasets with instruction templates.

Sources:
1. Use AudioCaps/Clotho data with instruction templates
2. Convert speech recognition data to instruction format
3. Create audio Q&A from sound classification data

This will be handled by the data preparation script.
EOF

echo "📝 Alpaca-Audio info saved to: $ALPACA_AUDIO_INFO"

# 2. ShareGPT-Style Audio Conversations
echo ""
echo "2️⃣  ShareGPT-Style Audio Conversations"
echo "----------------------------------------"

SHAREGPT_AUDIO_INFO="$STAGE3_DATA_DIR/sharegpt_audio_info.txt"
cat > "$SHAREGPT_AUDIO_INFO" << 'EOF'
ShareGPT-Style Audio Conversations:

We'll create multi-turn conversations involving audio content.

Approach:
1. Take single-turn audio descriptions
2. Generate follow-up questions using LLM
3. Create multi-turn conversations about audio content
4. Include various instruction types:
   - Audio description
   - Sound identification  
   - Audio comparison
   - Content analysis
   - Emotional interpretation

This will be synthetically generated using the preparation script.
EOF

echo "📝 ShareGPT-Audio info saved to: $SHAREGPT_AUDIO_INFO"

# 3. Audio Question Answering Dataset
echo ""
echo "3️⃣  Audio Question Answering (MUSIC-AVQA)"
echo "----------------------------------------"

MUSIC_AVQA_INFO="$STAGE3_DATA_DIR/music_avqa_info.txt"
cat > "$MUSIC_AVQA_INFO" << 'EOF'
MUSIC-AVQA Dataset Download Instructions:

1. Go to: https://github.com/GeWu-Lab/MUSIC-AVQA
2. Download the dataset following their instructions:
   - Audio files
   - Question-answer pairs
   - Metadata

3. Extract to: data/stage3_conversation/music_avqa/

Dataset contains:
- Music audio with visual and audio questions
- Multiple choice and open-ended questions
- Rich audio understanding tasks

Size: ~5GB

Alternative: Use existing audio datasets and generate Q&A pairs
EOF

echo "📝 MUSIC-AVQA info saved to: $MUSIC_AVQA_INFO"

# 4. Create synthetic instruction-following data
echo ""
echo "4️⃣  Creating Stage 3 Training Data"
echo "----------------------------------------"

# Create data preparation script
STAGE3_PREP_SCRIPT="$STAGE3_DATA_DIR/prepare_stage3_data.py"
cat > "$STAGE3_PREP_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Prepare Stage 3 training data for instruction following and conversation.
Creates multi-turn conversations and instruction-following examples.
"""

import os
import json
import random
from pathlib import Path
from typing import List, Dict
import copy

class InstructionGenerator:
    """Generate instruction-following examples from audio data."""
    
    def __init__(self):
        # Instruction templates for different tasks
        self.description_templates = [
            "请详细描述这段音频的内容。",
            "听这段音频，告诉我你听到了什么。",
            "分析这段音频并描述其特征。",
            "这段音频包含什么？请详细说明。",
            "描述一下这段音频中的声音。"
        ]
        
        self.identification_templates = [
            "这段音频是什么类型的声音？",
            "识别这段音频中的主要声音。",
            "这段音频中有什么声音？",
            "请分类这段音频的内容。",
            "判断这段音频的声音类型。"
        ]
        
        self.analysis_templates = [
            "分析这段音频的情感色彩。",
            "这段音频给你什么感受？",
            "评价这段音频的音质特点。",
            "这段音频的节奏和旋律如何？",
            "描述这段音频的氛围和情调。"
        ]
        
        self.comparison_templates = [
            "比较这段音频与典型的{}有何不同。",
            "这段音频与{}类声音的相似之处是什么？",
            "分析这段音频相对于普通{}的特点。"
        ]
    
    def create_instruction_example(self, audio_path: str, base_description: str, task_type: str = "description"):
        """Create an instruction-following example."""
        
        if task_type == "description":
            instruction = random.choice(self.description_templates)
            output = base_description
        elif task_type == "identification":
            instruction = random.choice(self.identification_templates)
            # Extract key sounds from description
            output = self._extract_sound_types(base_description)
        elif task_type == "analysis":
            instruction = random.choice(self.analysis_templates)
            output = self._generate_analysis(base_description)
        else:
            instruction = random.choice(self.description_templates)
            output = base_description
        
        return {
            "audio_path": audio_path,
            "instruction": instruction,
            "output": output,
            "task": "instruction_following"
        }
    
    def create_conversation_example(self, audio_path: str, base_description: str):
        """Create a multi-turn conversation example."""
        
        conversations = [
            self._create_description_conversation(base_description),
            self._create_analysis_conversation(base_description),
            self._create_detailed_conversation(base_description)
        ]
        
        conversation = random.choice(conversations)
        
        return {
            "audio_path": audio_path,
            "conversation": conversation,
            "task": "multi_turn_conversation"
        }
    
    def _create_description_conversation(self, description: str):
        """Create a conversation focused on description."""
        return [
            {"role": "user", "content": "请描述这段音频。"},
            {"role": "assistant", "content": description},
            {"role": "user", "content": "能更详细一些吗？"},
            {"role": "assistant", "content": self._expand_description(description)},
            {"role": "user", "content": "这个声音给你什么感觉？"},
            {"role": "assistant", "content": self._generate_emotional_response(description)}
        ]
    
    def _create_analysis_conversation(self, description: str):
        """Create a conversation focused on analysis."""
        return [
            {"role": "user", "content": "分析一下这段音频。"},
            {"role": "assistant", "content": f"这段音频{description}"},
            {"role": "user", "content": "音质如何？"},
            {"role": "assistant", "content": self._analyze_quality(description)},
            {"role": "user", "content": "适合在什么场景播放？"},
            {"role": "assistant", "content": self._suggest_scenarios(description)}
        ]
    
    def _create_detailed_conversation(self, description: str):
        """Create a detailed conversation."""
        return [
            {"role": "user", "content": "这是什么音频？"},
            {"role": "assistant", "content": description},
            {"role": "user", "content": "有什么特别之处？"},
            {"role": "assistant", "content": self._highlight_features(description)},
            {"role": "user", "content": "持续时间如何？"},
            {"role": "assistant", "content": "这段音频长度适中，包含了完整的音频内容。"},
            {"role": "user", "content": "总体评价如何？"},
            {"role": "assistant", "content": self._provide_overall_assessment(description)}
        ]
    
    def _extract_sound_types(self, description: str):
        """Extract sound types from description."""
        sound_keywords = ["音乐", "语音", "自然声", "机械声", "动物声", "环境音"]
        for keyword in sound_keywords:
            if keyword in description:
                return f"这段音频主要包含{keyword}。"
        return "这段音频包含多种声音元素。"
    
    def _generate_analysis(self, description: str):
        """Generate analysis based on description."""
        if "音乐" in description:
            return "这段音频具有优美的旋律和和谐的节奏，给人带来愉悦的听觉体验。"
        elif "语音" in description or "对话" in description:
            return "这段音频中的语音清晰自然，语调平和，便于理解。"
        else:
            return "这段音频具有独特的声音特征，富有表现力。"
    
    def _expand_description(self, description: str):
        """Expand the original description."""
        expansions = [
            f"{description}音频质量清晰，层次分明。",
            f"{description}整体音效自然流畅。",
            f"{description}声音细节丰富，具有很强的表现力。"
        ]
        return random.choice(expansions)
    
    def _generate_emotional_response(self, description: str):
        """Generate emotional response."""
        emotions = [
            "这个声音给人一种平静祥和的感觉。",
            "听起来很有活力和生气。",
            "这个声音很温暖，让人感到舒适。",
            "具有一种神秘而吸引人的氛围。"
        ]
        return random.choice(emotions)
    
    def _analyze_quality(self, description: str):
        """Analyze audio quality."""
        return "音频质量良好，清晰度高，没有明显的噪音干扰。"
    
    def _suggest_scenarios(self, description: str):
        """Suggest usage scenarios."""
        if "音乐" in description:
            return "适合在休闲时间播放，也可以作为背景音乐使用。"
        elif "自然" in description:
            return "适合冥想、放松或作为环境音效使用。"
        else:
            return "可以用于多种场景，具有很好的实用性。"
    
    def _highlight_features(self, description: str):
        """Highlight special features."""
        return f"这段音频的特别之处在于其独特的声音质感和丰富的音频层次。"
    
    def _provide_overall_assessment(self, description: str):
        """Provide overall assessment."""
        return "总体来说，这是一段高质量的音频，具有良好的听觉效果和表现力。"

def load_stage2_data(stage2_dir: Path):
    """Load processed data from Stage 2."""
    data = []
    stage2_train_file = stage2_dir / "train.jsonl"
    
    if stage2_train_file.exists():
        with open(stage2_train_file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        print(f"Loaded {len(data)} samples from Stage 2")
    else:
        print("Stage 2 data not found. Please run Stage 2 preparation first.")
    
    return data

def load_stage1_data(stage1_dir: Path):
    """Load some data from Stage 1 for instruction creation."""
    data = []
    stage1_train_file = stage1_dir / "train.jsonl"
    
    if stage1_train_file.exists():
        with open(stage1_train_file, 'r') as f:
            lines = f.readlines()
            # Use only a subset (10%) of Stage 1 data
            selected_lines = random.sample(lines, min(len(lines), len(lines) // 10))
            
            for line in selected_lines:
                item = json.loads(line)
                # Convert transcription to description format
                item["output"] = f"这段音频是语音内容：{item.get('transcript', '')}"
                data.append(item)
        
        print(f"Loaded {len(data)} samples from Stage 1")
    
    return data

def create_stage3_data(input_data: List[Dict], generator: InstructionGenerator):
    """Create Stage 3 instruction and conversation data."""
    instruction_data = []
    conversation_data = []
    
    for item in input_data:
        audio_path = item["audio_path"]
        base_description = item.get("output", "")
        
        if not base_description:
            continue
        
        # Create multiple instruction examples per audio
        for task_type in ["description", "identification", "analysis"]:
            inst_example = generator.create_instruction_example(
                audio_path, base_description, task_type
            )
            instruction_data.append(inst_example)
        
        # Create conversation example
        conv_example = generator.create_conversation_example(
            audio_path, base_description
        )
        conversation_data.append(conv_example)
    
    return instruction_data, conversation_data

if __name__ == "__main__":
    stage3_dir = Path("data/stage3_conversation")
    stage2_dir = Path("data/stage2_understanding")
    stage1_dir = Path("data/stage1_alignment")
    
    # Initialize generator
    generator = InstructionGenerator()
    
    # Load data from previous stages
    print("Loading data from previous stages...")
    stage2_data = load_stage2_data(stage2_dir)
    stage1_data = load_stage1_data(stage1_dir)
    
    all_input_data = stage2_data + stage1_data
    
    if not all_input_data:
        print("No input data found. Please complete Stage 1 and Stage 2 first.")
        exit(1)
    
    # Create Stage 3 data
    print("Creating Stage 3 instruction and conversation data...")
    instruction_data, conversation_data = create_stage3_data(all_input_data, generator)
    
    # Combine all data
    all_stage3_data = instruction_data + conversation_data
    
    # Shuffle data
    random.shuffle(all_stage3_data)
    
    # Split into train/val
    train_size = int(0.95 * len(all_stage3_data))
    train_data = all_stage3_data[:train_size]
    val_data = all_stage3_data[train_size:]
    
    # Save data
    with open(stage3_dir / "train.jsonl", 'w') as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(stage3_dir / "val.jsonl", 'w') as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # Save by task type for analysis
    instruction_only = [item for item in all_stage3_data if item.get("task") == "instruction_following"]
    conversation_only = [item for item in all_stage3_data if item.get("task") == "multi_turn_conversation"]
    
    with open(stage3_dir / "instructions.jsonl", 'w') as f:
        for item in instruction_only:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    with open(stage3_dir / "conversations.jsonl", 'w') as f:
        for item in conversation_only:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("\n" + "="*50)
    print("STAGE 3 DATA PREPARATION COMPLETED")
    print("="*50)
    print(f"Total samples: {len(all_stage3_data):,}")
    print(f"Train samples: {len(train_data):,}")
    print(f"Validation samples: {len(val_data):,}")
    print(f"Instruction samples: {len(instruction_only):,}")
    print(f"Conversation samples: {len(conversation_only):,}")
    print("="*50)
    
    # Create dataset info
    info = {
        "stage": 3,
        "task": "instruction_following_and_conversation",
        "total_samples": len(all_stage3_data),
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "instruction_samples": len(instruction_only),
        "conversation_samples": len(conversation_only),
        "data_sources": {
            "stage2_samples": len(stage2_data),
            "stage1_samples": len(stage1_data)
        }
    }
    
    with open(stage3_dir / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    print(f"Dataset info saved to: {stage3_dir}/dataset_info.json")
    print("\nReady for Stage 3 training!")
EOF

chmod +x "$STAGE3_PREP_SCRIPT"
echo "📜 Data preparation script created: $STAGE3_PREP_SCRIPT"

echo ""
echo "5️⃣  Additional High-Quality Instruction Data"
echo "----------------------------------------"

# Create script to download additional instruction data
ADDITIONAL_DATA_SCRIPT="$STAGE3_DATA_DIR/download_additional_instruction_data.py"
cat > "$ADDITIONAL_DATA_SCRIPT" << 'EOF'
#!/usr/bin/env python3
"""
Download additional high-quality instruction-following datasets.
"""

import json
import os
from pathlib import Path

def download_alpaca_gpt4():
    """Download Alpaca-GPT4 data and adapt for audio context."""
    try:
        import requests
        
        print("📥 Downloading Alpaca-GPT4 instruction data...")
        url = "https://raw.githubusercontent.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/main/data/alpaca_gpt4_data.json"
        
        response = requests.get(url)
        response.raise_for_status()
        
        data = response.json()
        
        # Adapt some instructions for audio context
        audio_adapted = []
        audio_keywords = ["sound", "audio", "music", "voice", "hear", "listen"]
        
        for item in data:
            instruction = item.get("instruction", "").lower()
            if any(keyword in instruction for keyword in audio_keywords):
                # These can be adapted for audio context
                audio_adapted.append({
                    "instruction": item["instruction"],
                    "input": item.get("input", ""),
                    "output": item["output"],
                    "source": "alpaca_gpt4_adapted"
                })
        
        # Save adapted instructions
        output_file = "data/stage3_conversation/alpaca_audio_adapted.jsonl"
        with open(output_file, 'w') as f:
            for item in audio_adapted:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        print(f"✅ Saved {len(audio_adapted)} audio-adapted instructions")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download Alpaca-GPT4 data: {e}")
        return False

def create_audio_specific_instructions():
    """Create audio-specific instruction templates."""
    
    audio_instructions = [
        {
            "instruction": "解释如何识别音频中的不同乐器",
            "input": "",
            "output": "识别音频中的不同乐器需要注意以下几个方面：1. 音色特征：每种乐器都有独特的音色；2. 频率范围：不同乐器的音频频率范围不同；3. 演奏技巧：注意乐器特有的演奏技巧和效果；4. 音量动态：观察音量的变化模式；5. 时间特征：注意音符的起始和衰减特征。通过训练和经验积累，可以逐渐提高识别准确性。"
        },
        {
            "instruction": "描述音频质量评估的标准",
            "input": "",
            "output": "音频质量评估主要包括以下标准：1. 清晰度：声音是否清晰，有无失真；2. 噪音水平：背景噪音的多少；3. 动态范围：音量变化的范围；4. 频率响应：各频段的平衡性；5. 立体声效果：左右声道的分离度；6. 压缩质量：数字音频的压缩损失程度。这些标准帮助我们客观评价音频的技术质量。"
        },
        {
            "instruction": "如何分析音频的情感内容",
            "input": "",
            "output": "分析音频情感内容可以从多个维度入手：1. 音调变化：上升音调通常表示兴奋或疑问，下降音调可能表示悲伤或确定；2. 语速节奏：快速语音可能表示紧张或兴奋，慢速可能表示平静或忧郁；3. 音量动态：大声可能表示愤怒或激动，小声可能表示秘密或悲伤；4. 停顿模式：停顿的长短和位置反映说话者的情绪状态；5. 声音质感：颤抖、哽咽等声音特征。"
        }
    ]
    
    output_file = "data/stage3_conversation/audio_specific_instructions.jsonl"
    with open(output_file, 'w') as f:
        for item in audio_instructions:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ Created {len(audio_instructions)} audio-specific instructions")

if __name__ == "__main__":
    os.makedirs("data/stage3_conversation", exist_ok=True)
    
    print("🚀 Downloading additional instruction data...")
    
    # Download adapted Alpaca data
    download_alpaca_gpt4()
    
    # Create audio-specific instructions
    create_audio_specific_instructions()
    
    print("✅ Additional instruction data download completed!")
EOF

chmod +x "$ADDITIONAL_DATA_SCRIPT"
echo "📜 Additional data download script created: $ADDITIONAL_DATA_SCRIPT"

echo ""
echo "6️⃣  Next Steps"
echo "----------------------------------------"
echo "1. Ensure Stage 1 and Stage 2 data are prepared"
echo ""
echo "2. Download additional instruction data:"
echo "   cd data/stage3_conversation && python download_additional_instruction_data.py"
echo ""
echo "3. Generate Stage 3 training data:"
echo "   cd data/stage3_conversation && python prepare_stage3_data.py"
echo ""
echo "4. Start Stage 3 training:"
echo "   bash scripts/train_stage3.sh"

echo ""
echo "📊 Expected Dataset Sizes:"
echo "   • Generated Instructions: ~300K samples"
echo "   • Multi-turn Conversations: ~100K samples"
echo "   • Additional Instructions: ~5K samples"
echo "   • Total: ~405K samples"

echo ""
echo "🎯 Stage 3 Training Objective:"
echo "   Learn instruction following and multi-turn conversation"
echo "   Fine-tune entire model or use LoRA for LLM layers"

echo ""
echo "⚠️  Note: Stage 3 requires data from Stage 1 and Stage 2"
echo "   Please complete previous stages first"

echo ""
echo "========================================================"
echo "Stage 3 data download setup completed!"
echo "========================================================" 