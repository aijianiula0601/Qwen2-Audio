#!/bin/bash

# Qwen2-Audio 完整训练流水线
# 包含环境设置、数据下载、处理和三阶段训练

set -e

echo "=== Qwen2-Audio 完整训练流水线 ==="

# =============================================================================
# 1. 环境准备
# =============================================================================

echo "1. 检查并安装依赖..."

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 需要Python 3.8+"
    exit 1
fi

# 检查CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "检测到CUDA环境:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "警告: 未检测到CUDA，将使用CPU训练（不推荐）"
fi

# 安装依赖
echo "安装Python依赖..."
pip3 install -r requirements.txt

# 检查Hugging Face Hub登录状态
if ! python3 -c "from huggingface_hub import HfApi; HfApi().whoami()" 2>/dev/null; then
    echo "请先登录Hugging Face Hub:"
    echo "huggingface-cli login"
    read -p "按回车键继续，或Ctrl+C退出..."
fi

# =============================================================================
# 2. 数据下载
# =============================================================================

echo "2. 下载训练数据..."

# 检查是否已存在数据
if [ ! -d "data/raw" ]; then
    echo "开始下载数据集..."
    bash scripts/download_training_data.sh
else
    echo "数据目录已存在，跳过下载"
    read -p "是否重新下载数据? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/raw
        bash scripts/download_training_data.sh
    fi
fi

# =============================================================================
# 3. 数据处理
# =============================================================================

echo "3. 处理训练数据..."

if [ ! -f "data/stage1_pretraining/train.jsonl" ]; then
    echo "开始数据处理..."
    python3 scripts/process_training_data.py --data_root data
else
    echo "处理后的数据已存在"
    read -p "是否重新处理数据? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf data/stage1_pretraining data/stage2_sft data/stage3_dpo
        python3 scripts/process_training_data.py --data_root data
    fi
fi

# 检查数据处理结果
echo "数据处理完成，统计信息:"
for stage in stage1_pretraining stage2_sft stage3_dpo; do
    if [ -f "data/$stage/train.jsonl" ]; then
        count=$(wc -l < "data/$stage/train.jsonl")
        echo "- $stage: $count 条样本"
    fi
done

# =============================================================================
# 4. 模型选择和下载
# =============================================================================

echo "4. 选择和下载基础模型..."

# 让用户选择基础模型
echo "请选择要使用的Qwen2基础模型:"
echo "1) Qwen/Qwen2-0.5B (最小模型, 适合测试)"
echo "2) Qwen/Qwen2-1.5B (小模型, 适合开发)"
echo "3) Qwen/Qwen2-7B (推荐, 平衡性能和资源)"
echo "4) Qwen/Qwen2-72B (大模型, 需要更多GPU)"
echo "5) Qwen/Qwen2-57B-A14B (MoE模型, 高效大模型)"
echo "6) 自定义模型路径"

read -p "请输入选择 (1-6): " model_choice

case $model_choice in
    1)
        BASE_MODEL="Qwen/Qwen2-0.5B"
        MODEL_SIZE="0.5B"
        MIN_GPU_MEMORY=4
        RECOMMENDED_GPUS=1
        ;;
    2)
        BASE_MODEL="Qwen/Qwen2-1.5B"
        MODEL_SIZE="1.5B"
        MIN_GPU_MEMORY=8
        RECOMMENDED_GPUS=1
        ;;
    3)
        BASE_MODEL="Qwen/Qwen2-7B"
        MODEL_SIZE="7B"
        MIN_GPU_MEMORY=16
        RECOMMENDED_GPUS=2
        ;;
    4)
        BASE_MODEL="Qwen/Qwen2-72B"
        MODEL_SIZE="72B"
        MIN_GPU_MEMORY=80
        RECOMMENDED_GPUS=8
        ;;
    5)
        BASE_MODEL="Qwen/Qwen2-57B-A14B"
        MODEL_SIZE="57B-A14B"
        MIN_GPU_MEMORY=48
        RECOMMENDED_GPUS=4
        ;;
    6)
        read -p "请输入自定义模型路径 (如 Qwen/Qwen2-14B): " BASE_MODEL
        MODEL_SIZE="custom"
        MIN_GPU_MEMORY=24
        RECOMMENDED_GPUS=2
        ;;
    *)
        echo "无效选择，使用默认模型 Qwen/Qwen2-7B"
        BASE_MODEL="Qwen/Qwen2-7B"
        MODEL_SIZE="7B"
        MIN_GPU_MEMORY=16
        RECOMMENDED_GPUS=2
        ;;
esac

echo "已选择模型: $BASE_MODEL ($MODEL_SIZE)"
echo "推荐GPU配置: ${RECOMMENDED_GPUS}卡, 每卡至少${MIN_GPU_MEMORY}GB显存"

# 检查硬件是否满足要求
if command -v nvidia-smi &> /dev/null; then
    TOTAL_GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum}')
    TOTAL_GPU_MEMORY_GB=$((TOTAL_GPU_MEMORY / 1024))
    REQUIRED_MEMORY=$((MIN_GPU_MEMORY * RECOMMENDED_GPUS))
    
    echo "检测到总GPU显存: ${TOTAL_GPU_MEMORY_GB}GB"
    echo "模型推荐显存: ${REQUIRED_MEMORY}GB"
    
    if [ $TOTAL_GPU_MEMORY_GB -lt $REQUIRED_MEMORY ]; then
        echo "⚠️  警告: 显存可能不足，建议使用更小的模型或启用模型分片"
        read -p "是否继续训练? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            echo "训练已取消"
            exit 1
        fi
    fi
fi

# 构建模型目录名
MODEL_DIR_NAME=$(echo $BASE_MODEL | tr '/' '_')

# 下载选择的基础模型
if [ ! -d "models/$MODEL_DIR_NAME" ]; then
    echo "下载 $BASE_MODEL 模型..."
    mkdir -p models
    cd models
    
    # 检查是否是Hugging Face模型
    if [[ $BASE_MODEL == *"/"* ]]; then
        git clone https://huggingface.co/$BASE_MODEL $MODEL_DIR_NAME
    else
        echo "错误: 无效的模型路径格式"
        exit 1
    fi
    
    cd ..
    echo "✅ 模型下载完成"
else
    echo "✅ $BASE_MODEL 模型已存在"
fi

# 下载Whisper-large-v3
if [ ! -d "models/whisper-large-v3" ]; then
    echo "下载Whisper-large-v3模型..."
    cd models
    git clone https://huggingface.co/openai/whisper-large-v3
    cd ..
    echo "✅ Whisper模型下载完成"
else
    echo "✅ Whisper-large-v3模型已存在"
fi

# =============================================================================
# 5. 训练配置
# =============================================================================

echo "5. 配置训练参数..."

# 检测GPU数量
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "检测到 $GPU_COUNT 个GPU"
else
    GPU_COUNT=0
    echo "未检测到GPU，使用CPU训练"
fi

# 根据模型大小和GPU数量调整训练参数
case $MODEL_SIZE in
    "0.5B")
        if [ $GPU_COUNT -ge 1 ]; then
            WORLD_SIZE=1
            BATCH_SIZE=8
            GRADIENT_ACCUMULATION=2
        else
            WORLD_SIZE=1
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION=4
        fi
        ;;
    "1.5B")
        if [ $GPU_COUNT -ge 2 ]; then
            WORLD_SIZE=2
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION=4
        elif [ $GPU_COUNT -eq 1 ]; then
            WORLD_SIZE=1
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION=8
        else
            WORLD_SIZE=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=16
        fi
        ;;
    "7B")
        if [ $GPU_COUNT -ge 4 ]; then
            WORLD_SIZE=4
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION=4
        elif [ $GPU_COUNT -ge 2 ]; then
            WORLD_SIZE=2
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION=8
        elif [ $GPU_COUNT -eq 1 ]; then
            WORLD_SIZE=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=16
        else
            echo "❌ 错误: 7B模型至少需要1个GPU"
            exit 1
        fi
        ;;
    "72B")
        if [ $GPU_COUNT -ge 8 ]; then
            WORLD_SIZE=8
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=8
        elif [ $GPU_COUNT -ge 4 ]; then
            WORLD_SIZE=4
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=16
            echo "⚠️  警告: 72B模型推荐8卡训练，当前配置可能需要模型分片"
        else
            echo "❌ 错误: 72B模型至少需要4个GPU (推荐8个)"
            echo "建议选择更小的模型或增加GPU数量"
            exit 1
        fi
        ;;
    "57B-A14B")
        if [ $GPU_COUNT -ge 6 ]; then
            WORLD_SIZE=6
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION=4
        elif [ $GPU_COUNT -ge 4 ]; then
            WORLD_SIZE=4
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=8
        else
            echo "❌ 错误: 57B-A14B MoE模型至少需要4个GPU (推荐6个)"
            exit 1
        fi
        ;;
    "custom")
        # 自定义模型使用保守配置
        if [ $GPU_COUNT -ge 4 ]; then
            WORLD_SIZE=4
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION=8
        elif [ $GPU_COUNT -ge 2 ]; then
            WORLD_SIZE=2
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=16
        elif [ $GPU_COUNT -eq 1 ]; then
            WORLD_SIZE=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=32
        else
            WORLD_SIZE=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=64
            echo "警告: CPU训练将非常缓慢"
        fi
        ;;
    *)
        # 默认配置 (7B模型)
        if [ $GPU_COUNT -ge 4 ]; then
            WORLD_SIZE=4
            BATCH_SIZE=4
            GRADIENT_ACCUMULATION=4
        elif [ $GPU_COUNT -ge 2 ]; then
            WORLD_SIZE=2
            BATCH_SIZE=2
            GRADIENT_ACCUMULATION=8
        elif [ $GPU_COUNT -eq 1 ]; then
            WORLD_SIZE=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=16
        else
            WORLD_SIZE=1
            BATCH_SIZE=1
            GRADIENT_ACCUMULATION=32
            echo "警告: CPU训练将非常缓慢"
        fi
        ;;
esac

# 根据模型大小调整学习率
case $MODEL_SIZE in
    "0.5B"|"1.5B")
        LEARNING_RATE_STAGE1=2e-4
        LEARNING_RATE_STAGE2=1e-4
        LEARNING_RATE_STAGE3=1e-5
        ;;
    "7B")
        LEARNING_RATE_STAGE1=1e-4
        LEARNING_RATE_STAGE2=5e-5
        LEARNING_RATE_STAGE3=5e-6
        ;;
    "72B"|"57B-A14B")
        LEARNING_RATE_STAGE1=5e-5
        LEARNING_RATE_STAGE2=2e-5
        LEARNING_RATE_STAGE3=2e-6
        ;;
    *)
        LEARNING_RATE_STAGE1=1e-4
        LEARNING_RATE_STAGE2=5e-5
        LEARNING_RATE_STAGE3=5e-6
        ;;
esac

echo "训练配置:"
echo "- 选择模型: $BASE_MODEL ($MODEL_SIZE)"
echo "- GPU数量: $GPU_COUNT"
echo "- World Size: $WORLD_SIZE"
echo "- Batch Size: $BATCH_SIZE"
echo "- Gradient Accumulation: $GRADIENT_ACCUMULATION"
echo "- 有效Batch Size: $((BATCH_SIZE * GRADIENT_ACCUMULATION * WORLD_SIZE))"
echo "- 学习率 (Stage1/2/3): $LEARNING_RATE_STAGE1 / $LEARNING_RATE_STAGE2 / $LEARNING_RATE_STAGE3"

# 检查是否需要启用模型分片或其他优化
if [[ $MODEL_SIZE == "72B" ]] || [[ $MODEL_SIZE == "57B-A14B" ]]; then
    echo ""
    echo "🔧 大模型训练优化建议:"
    echo "- 启用梯度检查点 (gradient checkpointing)"
    echo "- 使用DeepSpeed ZeRO-3分片"
    echo "- 启用混合精度训练 (fp16/bf16)"
    
    # 设置大模型训练标志
    USE_DEEPSPEED=true
    USE_GRADIENT_CHECKPOINTING=true
else
    USE_DEEPSPEED=false
    USE_GRADIENT_CHECKPOINTING=false
fi

# =============================================================================
# 6. 三阶段训练
# =============================================================================

echo "6. 开始三阶段训练..."

# 询问用户要执行哪些阶段
echo "请选择要执行的训练阶段:"
echo "1) 仅Stage 1 (预训练)"
echo "2) 仅Stage 2 (监督微调)"
echo "3) 仅Stage 3 (DPO)"
echo "4) Stage 1 + 2"
echo "5) Stage 2 + 3"
echo "6) 全部三个阶段"
echo "7) 跳过训练"

read -p "请输入选择 (1-7): " stage_choice

case $stage_choice in
    1|4|6)
        echo "开始Stage 1: 预训练..."
        export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((GPU_COUNT-1)))
        export WORLD_SIZE=$WORLD_SIZE
        
        # 将配置参数写入临时配置文件
        cat > .training_config << EOF
BASE_MODEL="$BASE_MODEL"
MODEL_DIR_NAME="$MODEL_DIR_NAME"
WORLD_SIZE=$WORLD_SIZE
BATCH_SIZE=$BATCH_SIZE
GRADIENT_ACCUMULATION=$GRADIENT_ACCUMULATION
LEARNING_RATE_STAGE1=$LEARNING_RATE_STAGE1
USE_DEEPSPEED=$USE_DEEPSPEED
USE_GRADIENT_CHECKPOINTING=$USE_GRADIENT_CHECKPOINTING
EOF
        
        # 修改训练脚本参数
        sed -i "s|MODEL_NAME_OR_PATH=\".*\"|MODEL_NAME_OR_PATH=\"models/$MODEL_DIR_NAME\"|g" scripts/train_stage1.sh
        sed -i "s/BATCH_SIZE=.*/BATCH_SIZE=$BATCH_SIZE/g" scripts/train_stage1.sh
        sed -i "s/GRADIENT_ACCUMULATION_STEPS=.*/GRADIENT_ACCUMULATION_STEPS=$GRADIENT_ACCUMULATION/g" scripts/train_stage1.sh
        sed -i "s/LEARNING_RATE=.*/LEARNING_RATE=$LEARNING_RATE_STAGE1/g" scripts/train_stage1.sh
        sed -i "s/--nproc_per_node=.*/--nproc_per_node=$WORLD_SIZE/g" scripts/train_stage1.sh
        
        # 为大模型添加优化参数
        if [ "$USE_DEEPSPEED" = true ]; then
            # 创建DeepSpeed配置文件
            cat > deepspeed_config_stage1.json << 'EOF'
{
    "fp16": {
        "enabled": true,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu"
        },
        "offload_param": {
            "device": "cpu"
        },
        "overlap_comm": true,
        "contiguous_gradients": true,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e6,
        "stage3_param_persistence_threshold": 1e4
    },
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_PLACEHOLDER,
    "gradient_clipping": 1.0,
    "train_batch_size": EFFECTIVE_BATCH_SIZE_PLACEHOLDER
}
EOF
            # 替换占位符
            EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRADIENT_ACCUMULATION * WORLD_SIZE))
            sed -i "s/GRADIENT_ACCUMULATION_PLACEHOLDER/$GRADIENT_ACCUMULATION/g" deepspeed_config_stage1.json
            sed -i "s/EFFECTIVE_BATCH_SIZE_PLACEHOLDER/$EFFECTIVE_BATCH_SIZE/g" deepspeed_config_stage1.json
            
            # 修改训练脚本使用DeepSpeed
            sed -i '/torchrun/i # 使用DeepSpeed训练大模型' scripts/train_stage1.sh
            sed -i 's/torchrun/deepspeed --num_gpus='$WORLD_SIZE'/g' scripts/train_stage1.sh
            sed -i '/--report_to/a \    --deepspeed deepspeed_config_stage1.json \\' scripts/train_stage1.sh
        fi
        
        bash scripts/train_stage1.sh
        
        if [ $? -ne 0 ]; then
            echo "Stage 1 训练失败"
            exit 1
        fi
        echo "Stage 1 训练完成"
        ;;
esac

case $stage_choice in
    2|4|5|6)
        echo "开始Stage 2: 监督微调..."
        
        # 检查Stage 1输出
        if [ $stage_choice -eq 2 ] || [ $stage_choice -eq 5 ]; then
            if [ ! -d "checkpoints/qwen2-audio-stage1" ]; then
                echo "错误: 未找到Stage 1检查点，请先运行Stage 1"
                exit 1
            fi
        fi
        
        # 加载配置
        source .training_config 2>/dev/null || true
        
        # 为Stage 2调整batch size (通常更小)
        STAGE2_BATCH_SIZE=$((BATCH_SIZE < 2 ? 1 : BATCH_SIZE/2))
        STAGE2_GRADIENT_ACCUMULATION=$((GRADIENT_ACCUMULATION * 2))
        
        # 修改训练脚本参数
        sed -i "s|MODEL_NAME_OR_PATH=\".*\"|MODEL_NAME_OR_PATH=\"checkpoints/qwen2-audio-stage1\"|g" scripts/train_stage2.sh
        sed -i "s/BATCH_SIZE=.*/BATCH_SIZE=$STAGE2_BATCH_SIZE/g" scripts/train_stage2.sh
        sed -i "s/GRADIENT_ACCUMULATION_STEPS=.*/GRADIENT_ACCUMULATION_STEPS=$STAGE2_GRADIENT_ACCUMULATION/g" scripts/train_stage2.sh
        sed -i "s/LEARNING_RATE=.*/LEARNING_RATE=$LEARNING_RATE_STAGE2/g" scripts/train_stage2.sh
        sed -i "s/--nproc_per_node=.*/--nproc_per_node=$WORLD_SIZE/g" scripts/train_stage2.sh
        
        # 为大模型配置DeepSpeed
        if [ "$USE_DEEPSPEED" = true ]; then
            cp deepspeed_config_stage1.json deepspeed_config_stage2.json
            EFFECTIVE_BATCH_SIZE=$((STAGE2_BATCH_SIZE * STAGE2_GRADIENT_ACCUMULATION * WORLD_SIZE))
            sed -i "s/\"gradient_accumulation_steps\": .*/\"gradient_accumulation_steps\": $STAGE2_GRADIENT_ACCUMULATION,/g" deepspeed_config_stage2.json
            sed -i "s/\"train_batch_size\": .*/\"train_batch_size\": $EFFECTIVE_BATCH_SIZE/g" deepspeed_config_stage2.json
            
            sed -i 's/torchrun/deepspeed --num_gpus='$WORLD_SIZE'/g' scripts/train_stage2.sh
            sed -i '/--report_to/a \    --deepspeed deepspeed_config_stage2.json \\' scripts/train_stage2.sh
        fi
        
        bash scripts/train_stage2.sh
        
        if [ $? -ne 0 ]; then
            echo "Stage 2 训练失败"
            exit 1
        fi
        echo "Stage 2 训练完成"
        ;;
esac

case $stage_choice in
    3|5|6)
        echo "开始Stage 3: DPO..."
        
        # 检查Stage 2输出
        if [ $stage_choice -eq 3 ]; then
            if [ ! -d "checkpoints/qwen2-audio-stage2" ]; then
                echo "错误: 未找到Stage 2检查点，请先运行Stage 2"
                exit 1
            fi
        fi
        
        # 加载配置
        source .training_config 2>/dev/null || true
        
        # DPO通常使用更小的batch size
        STAGE3_BATCH_SIZE=1
        STAGE3_GRADIENT_ACCUMULATION=$((GRADIENT_ACCUMULATION * 4))
        
        # 修改训练脚本参数
        sed -i "s|MODEL_NAME_OR_PATH=\".*\"|MODEL_NAME_OR_PATH=\"checkpoints/qwen2-audio-stage2\"|g" scripts/train_stage3.sh
        sed -i "s/BATCH_SIZE=.*/BATCH_SIZE=$STAGE3_BATCH_SIZE/g" scripts/train_stage3.sh
        sed -i "s/GRADIENT_ACCUMULATION_STEPS=.*/GRADIENT_ACCUMULATION_STEPS=$STAGE3_GRADIENT_ACCUMULATION/g" scripts/train_stage3.sh
        sed -i "s/LEARNING_RATE=.*/LEARNING_RATE=$LEARNING_RATE_STAGE3/g" scripts/train_stage3.sh
        sed -i "s/--nproc_per_node=.*/--nproc_per_node=$WORLD_SIZE/g" scripts/train_stage3.sh
        
        # 为大模型配置DeepSpeed
        if [ "$USE_DEEPSPEED" = true ]; then
            cp deepspeed_config_stage1.json deepspeed_config_stage3.json
            EFFECTIVE_BATCH_SIZE=$((STAGE3_BATCH_SIZE * STAGE3_GRADIENT_ACCUMULATION * WORLD_SIZE))
            sed -i "s/\"gradient_accumulation_steps\": .*/\"gradient_accumulation_steps\": $STAGE3_GRADIENT_ACCUMULATION,/g" deepspeed_config_stage3.json
            sed -i "s/\"train_batch_size\": .*/\"train_batch_size\": $EFFECTIVE_BATCH_SIZE/g" deepspeed_config_stage3.json
            
            sed -i 's/torchrun/deepspeed --num_gpus='$WORLD_SIZE'/g' scripts/train_stage3.sh
            sed -i '/--report_to/a \    --deepspeed deepspeed_config_stage3.json \\' scripts/train_stage3.sh
        fi
        
        bash scripts/train_stage3.sh
        
        if [ $? -ne 0 ]; then
            echo "Stage 3 训练失败"
            exit 1
        fi
        echo "Stage 3 训练完成"
        ;;
esac

# =============================================================================
# 7. 训练后处理
# =============================================================================

if [ $stage_choice -ne 7 ]; then
    echo "7. 训练完成后处理..."
    
    # 保存训练日志
    mkdir -p logs
    mv *.log logs/ 2>/dev/null || true
    
    # 创建模型使用示例
    cat > test_model.py << EOF
#!/usr/bin/env python3
"""
测试训练好的Qwen2-Audio模型
"""

import os
import torch
from src import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

def test_model():
    # 加载配置
    try:
        with open('.training_config', 'r') as f:
            config_content = f.read()
        config = {}
        for line in config_content.strip().split('\n'):
            key, value = line.split('=', 1)
            config[key] = value.strip('"')
    except:
        config = {
            'BASE_MODEL': 'Qwen/Qwen2-7B',
            'MODEL_SIZE': '7B'
        }
    
    # 确定模型路径
    model_paths = [
        "checkpoints/qwen2-audio-stage3",
        "checkpoints/qwen2-audio-stage2", 
        "checkpoints/qwen2-audio-stage1"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("错误: 未找到训练好的模型检查点")
        print("请先运行训练脚本: bash scripts/setup_and_train.sh")
        return
    
    print(f"使用模型: {config.get('BASE_MODEL', 'Unknown')} ({config.get('MODEL_SIZE', 'Unknown')})")
    print(f"加载检查点: {model_path}")
    
    # 检查是否有GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 加载模型和处理器
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        processor = Qwen2AudioProcessor.from_pretrained(model_path)
        
        if device == "cpu":
            model = model.to(device)
        
        print("✅ 模型加载成功")
        
        # 测试音频文件
        test_audio_files = [
            "test_audio.wav",
            "demo/test_audio.wav", 
            "data/raw/test.wav"
        ]
        
        audio_path = None
        for path in test_audio_files:
            if os.path.exists(path):
                audio_path = path
                break
        
        if not audio_path:
            print("⚠️  未找到测试音频文件")
            print("请提供以下任一测试音频:")
            for path in test_audio_files:
                print(f"  - {path}")
            return
        
        print(f"使用测试音频: {audio_path}")
        
        # 测试不同任务
        test_prompts = [
            "请转录这段音频",
            "请描述这段音频的内容",
            "这段音频表达了什么情感？",
            "请总结音频中的主要信息"
        ]
        
        for prompt in test_prompts:
            print(f"\n🔍 测试提示: {prompt}")
            
            try:
                # 处理输入
                inputs = processor(
                    audio_path=audio_path,
                    text=prompt,
                    return_tensors="pt"
                )
                
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 生成回复
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_length=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码输出
                response = processor.decode(outputs[0], skip_special_tokens=True)
                print(f"💬 模型回复: {response}")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
        
        print(f"\n✅ 模型测试完成")
        print(f"模型规模: {config.get('MODEL_SIZE', 'Unknown')}")
        print(f"基础模型: {config.get('BASE_MODEL', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("可能原因:")
        print("- 显存不足 (尝试使用更小的模型)")
        print("- 检查点损坏 (重新训练)")
        print("- 依赖缺失 (检查requirements.txt)")

if __name__ == "__main__":
    test_model()
EOF
    
    chmod +x test_model.py
    echo "已创建模型测试脚本: test_model.py"
    
    # 显示训练总结
    echo ""
    echo "=== 训练完成总结 ==="
    echo "检查点保存位置:"
    ls -la checkpoints/ 2>/dev/null || echo "无检查点目录"
    
    echo ""
    echo "要测试模型，请运行:"
    echo "python3 test_model.py"
    
    echo ""
    echo "要启动Web演示，请运行:"
    echo "python3 demo/web_demo_audio.py"
fi

echo ""
echo "=== Qwen2-Audio 训练流水线完成 ===" 