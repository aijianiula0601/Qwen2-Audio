#!/bin/bash

# Qwen2-Audio 快速开始脚本
# 提供预设配置快速开始训练

set -e

echo "=== Qwen2-Audio 快速开始 ==="
echo ""

# 检查基本依赖
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 需要Python 3.8+"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ 错误: 需要Git"
    exit 1
fi

# 显示预设配置
echo "🚀 选择预设配置:"
echo ""
echo "1) 🔬 开发测试 (Qwen2-0.5B + 小数据集)"
echo "   - 模型: 0.5B参数"
echo "   - 显存: 4GB"
echo "   - 时间: 2-4小时"
echo "   - 适合: 代码测试、功能验证"
echo ""
echo "2) 🎯 标准训练 (Qwen2-7B + 完整数据集)"
echo "   - 模型: 7B参数"  
echo "   - 显存: 16GB×2"
echo "   - 时间: 1-3天"
echo "   - 适合: 实际应用、论文复现"
echo ""
echo "3) 🏆 高性能训练 (Qwen2-72B + 完整数据集)"
echo "   - 模型: 72B参数"
echo "   - 显存: 80GB×8"
echo "   - 时间: 1-2周"  
echo "   - 适合: 最佳性能、生产部署"
echo ""
echo "4) ⚡ MoE高效训练 (Qwen2-57B-A14B + 完整数据集)"
echo "   - 模型: 57B参数(MoE)"
echo "   - 显存: 48GB×4"
echo "   - 时间: 5-10天"
echo "   - 适合: 平衡性能和效率"
echo ""
echo "5) 🛠️  自定义配置"
echo ""

read -p "请选择配置 (1-5): " config_choice

case $config_choice in
    1)
        echo "🔬 开发测试配置"
        export QUICK_START_MODEL="Qwen/Qwen2-0.5B"
        export QUICK_START_DATASETS="librispeech fleurs"
        export QUICK_START_STAGES="1 2"
        ;;
    2)
        echo "🎯 标准训练配置"
        export QUICK_START_MODEL="Qwen/Qwen2-7B"
        export QUICK_START_DATASETS="all"
        export QUICK_START_STAGES="1 2 3"
        ;;
    3)
        echo "🏆 高性能训练配置"
        export QUICK_START_MODEL="Qwen/Qwen2-72B"
        export QUICK_START_DATASETS="all"
        export QUICK_START_STAGES="1 2 3"
        ;;
    4)
        echo "⚡ MoE高效训练配置"
        export QUICK_START_MODEL="Qwen/Qwen2-57B-A14B"
        export QUICK_START_DATASETS="all"
        export QUICK_START_STAGES="1 2 3"
        ;;
    5)
        echo "🛠️  启动自定义配置..."
        bash scripts/setup_and_train.sh
        exit 0
        ;;
    *)
        echo "❌ 无效选择"
        exit 1
        ;;
esac

echo "配置选择完成，开始自动化训练..."
echo ""

# 检查GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "检测到 $GPU_COUNT 个GPU"
else
    echo "⚠️  未检测到GPU，训练将很慢"
fi

# 确认开始
echo ""
echo "⚠️  重要提示:"
echo "- 完整训练需要大量时间和资源"
echo "- 确保有足够的存储空间 (至少2TB)"
echo "- 训练过程中请保持网络连接稳定"
echo ""

read -p "确认开始训练? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ 训练已取消"
    exit 0
fi

# 创建快速启动配置文件
cat > .quick_start_config << EOF
# Quick Start Configuration
QUICK_START_MODEL="$QUICK_START_MODEL"
QUICK_START_DATASETS="$QUICK_START_DATASETS"
QUICK_START_STAGES="$QUICK_START_STAGES"
QUICK_START_MODE=true
EOF

echo "✅ 开始快速训练流程..."

# 添加项目根目录到 Python 路径
export PYTHONPATH=$PYTHONPATH:$(pwd)

# 修改setup_and_train.sh以支持快速启动
export QUICK_START_MODE=true
bash scripts/setup_and_train.sh

echo ""
echo "🎉 快速开始流程完成!"
echo ""
echo "📝 查看训练结果:"
echo "- 检查点: checkpoints/"
echo "- 日志: logs/"
echo "- 测试模型: python3 test_model.py"
echo ""
echo "📚 更多信息:"
echo "- 训练指南: TRAINING_GUIDE.md"
echo "- 数据集说明: README_TRAINING.md" 