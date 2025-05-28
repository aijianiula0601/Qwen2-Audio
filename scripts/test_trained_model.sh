#!/bin/bash

# Qwen2-Audio 训练后模型测试脚本
# 自动化测试训练好的模型性能

set -e

echo "=== Qwen2-Audio 模型测试脚本 ==="

# =============================================================================
# 1. 参数和配置
# =============================================================================

# 默认参数
MODEL_PATH=""
DEVICE="auto"
TEST_MODE="comprehensive"
OUTPUT_DIR="test_results"
CREATE_DEMO=false
BENCHMARK=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path)
            MODEL_PATH="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --test_mode)
            TEST_MODE="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --benchmark)
            BENCHMARK=true
            shift
            ;;
        --demo)
            CREATE_DEMO=true
            shift
            ;;
        -h|--help)
            echo "用法: $0 [选项]"
            echo ""
            echo "选项:"
            echo "  --model_path PATH    模型路径 (默认自动查找)"
            echo "  --device DEVICE      设备 (auto|cuda|cpu, 默认auto)"
            echo "  --test_mode MODE     测试模式 (quick|comprehensive|benchmark, 默认comprehensive)"
            echo "  --output_dir DIR     输出目录 (默认test_results)"
            echo "  --benchmark          运行性能基准测试"
            echo "  --demo               启动演示界面"
            echo "  -h, --help           显示帮助信息"
            echo ""
            echo "测试模式说明:"
            echo "  quick        快速测试 (基本功能验证)"
            echo "  comprehensive 综合测试 (完整功能测试)"
            echo "  benchmark    性能基准测试"
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            echo "使用 --help 查看帮助"
            exit 1
            ;;
    esac
done

# =============================================================================
# 2. 环境检查
# =============================================================================

echo "1. 检查环境和依赖..."

# 检查Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误: 需要Python 3.8+"
    exit 1
fi

# 检查必要的Python包
echo "检查Python依赖..."
python3 -c "
import sys
required_packages = ['torch', 'torchaudio', 'transformers', 'numpy']
missing_packages = []

for pkg in required_packages:
    try:
        __import__(pkg)
        print(f'✅ {pkg}')
    except ImportError:
        missing_packages.append(pkg)
        print(f'❌ {pkg}')

if missing_packages:
    print(f'缺少依赖: {missing_packages}')
    print('请运行: pip install -r requirements.txt')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ 依赖检查失败"
    exit 1
fi

# 检查CUDA
if [ "$DEVICE" = "auto" ] || [ "$DEVICE" = "cuda" ]; then
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ 检测到CUDA环境"
        nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
    else
        echo "⚠️  未检测到CUDA，将使用CPU"
        DEVICE="cpu"
    fi
fi

# =============================================================================
# 3. 查找模型
# =============================================================================

echo "2. 查找训练好的模型..."

if [ -z "$MODEL_PATH" ]; then
    # 自动查找模型
    CANDIDATE_PATHS=(
        "checkpoints/qwen2-audio-stage3"
        "checkpoints/qwen2-audio-stage2" 
        "checkpoints/qwen2-audio-stage1"
        "./qwen2-audio-7b"
        "../checkpoints/qwen2-audio-stage3"
        "../checkpoints/qwen2-audio-stage2"
        "../checkpoints/qwen2-audio-stage1"
    )
    
    for path in "${CANDIDATE_PATHS[@]}"; do
        if [ -d "$path" ]; then
            MODEL_PATH="$path"
            break
        fi
    done
    
    if [ -z "$MODEL_PATH" ]; then
        echo "❌ 错误: 未找到训练好的模型"
        echo "请指定 --model_path 或先运行训练脚本"
        exit 1
    fi
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "❌ 错误: 模型路径不存在: $MODEL_PATH"
    exit 1
fi

echo "✅ 找到模型: $MODEL_PATH"

# 检查模型文件
REQUIRED_FILES=("config.json" "pytorch_model.bin")
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$MODEL_PATH/$file" ] && [ ! -f "$MODEL_PATH/pytorch_model-00001-of-*.bin" ]; then
        echo "⚠️  警告: 可能缺少模型文件: $file"
    fi
done

# =============================================================================
# 4. 创建输出目录
# =============================================================================

echo "3. 准备测试环境..."

mkdir -p "$OUTPUT_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TEST_SESSION_DIR="$OUTPUT_DIR/test_session_$TIMESTAMP"
mkdir -p "$TEST_SESSION_DIR"

echo "测试会话目录: $TEST_SESSION_DIR"

# 记录测试配置
cat > "$TEST_SESSION_DIR/test_config.txt" << EOF
测试配置
========
时间: $(date)
模型路径: $MODEL_PATH
设备: $DEVICE
测试模式: $TEST_MODE
输出目录: $TEST_SESSION_DIR
系统信息: $(uname -a)
Python版本: $(python3 --version)
EOF

# =============================================================================
# 5. 创建测试音频
# =============================================================================

echo "4. 准备测试音频..."

# 查找现有的测试音频
TEST_AUDIO_FILES=()
AUDIO_SEARCH_PATHS=(
    "test_audio.wav"
    "demo/test_audio.wav"
    "data/test_audio.wav"
    "examples/*.wav"
    "samples/*.wav"
    "demo/*.wav"
)

for path in "${AUDIO_SEARCH_PATHS[@]}"; do
    if [[ "$path" == *"*"* ]]; then
        # 处理通配符
        for file in $path; do
            if [ -f "$file" ]; then
                TEST_AUDIO_FILES+=("$file")
            fi
        done
    elif [ -f "$path" ]; then
        TEST_AUDIO_FILES+=("$path")
    fi
done

# 如果没有找到测试音频，创建一些
if [ ${#TEST_AUDIO_FILES[@]} -eq 0 ]; then
    echo "未找到测试音频，创建示例音频..."
    python3 scripts/test_inference.py --create_test_audio --model_path "$MODEL_PATH" --device "$DEVICE" > /dev/null 2>&1 || true
    
    # 重新查找
    for path in test_samples/*.wav; do
        if [ -f "$path" ]; then
            TEST_AUDIO_FILES+=("$path")
        fi
    done
fi

echo "找到 ${#TEST_AUDIO_FILES[@]} 个测试音频文件"

# =============================================================================
# 6. 运行测试
# =============================================================================

echo "5. 开始模型测试..."

case $TEST_MODE in
    "quick")
        echo "🚀 快速测试模式"
        
        # 基本功能测试
        echo "运行基本推理测试..."
        python3 scripts/test_inference.py \
            --model_path "$MODEL_PATH" \
            --device "$DEVICE" \
            --audio_files "${TEST_AUDIO_FILES[@]:0:2}" \
            --output "$TEST_SESSION_DIR/quick_test_results.json" \
            2>&1 | tee "$TEST_SESSION_DIR/quick_test.log"
        
        if [ $? -eq 0 ]; then
            echo "✅ 快速测试完成"
        else
            echo "❌ 快速测试失败"
            exit 1
        fi
        ;;
        
    "comprehensive")
        echo "🎯 综合测试模式"
        
        # 完整推理测试
        echo "运行完整推理测试..."
        python3 scripts/test_inference.py \
            --model_path "$MODEL_PATH" \
            --device "$DEVICE" \
            --audio_files "${TEST_AUDIO_FILES[@]}" \
            --output "$TEST_SESSION_DIR/comprehensive_test_results.json" \
            2>&1 | tee "$TEST_SESSION_DIR/comprehensive_test.log"
        
        # 创建性能报告
        echo "生成测试报告..."
        python3 -c "
import json
import os

result_file = '$TEST_SESSION_DIR/comprehensive_test_results.json'
if os.path.exists(result_file):
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    report = []
    report.append('Qwen2-Audio 综合测试报告')
    report.append('=' * 50)
    report.append(f'模型路径: {results.get(\"model_path\", \"未知\")}')
    report.append(f'测试时间: {results.get(\"timestamp\", \"未知\")}')
    report.append(f'设备: {results.get(\"device\", \"未知\")}')
    report.append('')
    
    stats = results.get('statistics', {})
    report.append(f'测试统计:')
    report.append(f'  总测试数: {stats.get(\"total_tests\", 0)}')
    report.append(f'  总耗时: {stats.get(\"total_time\", 0):.2f}秒')
    report.append(f'  平均耗时: {stats.get(\"average_time\", 0):.2f}秒/测试')
    report.append('')
    
    for test in results.get('tests', []):
        task_name = test.get('task', 'unknown')
        test_results = test.get('results', [])
        report.append(f'{task_name.upper()} 任务:')
        report.append(f'  测试样本数: {len(test_results)}')
        
        if test_results:
            avg_time = sum(r.get('time_cost', 0) for r in test_results) / len(test_results)
            report.append(f'  平均推理时间: {avg_time:.2f}秒')
        report.append('')
    
    # 保存报告
    with open('$TEST_SESSION_DIR/test_report.txt', 'w', encoding='utf-8') as f:
        f.write('\\n'.join(report))
    
    print('\\n'.join(report))
else:
    print('未找到测试结果文件')
" 2>&1 | tee "$TEST_SESSION_DIR/report_generation.log"
        
        if [ $? -eq 0 ]; then
            echo "✅ 综合测试完成"
        else
            echo "❌ 综合测试失败"
            exit 1
        fi
        ;;
        
    "benchmark")
        echo "📊 性能基准测试模式"
        BENCHMARK=true
        ;;
esac

# =============================================================================
# 7. 性能基准测试
# =============================================================================

if [ "$BENCHMARK" = true ]; then
    echo "6. 运行性能基准测试..."
    
    # 检查是否有基准测试依赖
    echo "检查基准测试依赖..."
    python3 -c "
try:
    import jiwer, rouge_score, sentence_transformers
    print('✅ 基准测试依赖完整')
except ImportError as e:
    print('⚠️  基准测试依赖不完整:', e)
    print('安装命令: pip install jiwer rouge-score sentence-transformers')
" 2>&1 | tee "$TEST_SESSION_DIR/benchmark_deps_check.log"
    
    # 运行基准测试
    echo "运行性能基准测试..."
    python3 scripts/benchmark.py \
        --model_path "$MODEL_PATH" \
        --device "$DEVICE" \
        --create_sample_data \
        --output "$TEST_SESSION_DIR/benchmark_results.json" \
        2>&1 | tee "$TEST_SESSION_DIR/benchmark.log"
    
    if [ $? -eq 0 ]; then
        echo "✅ 基准测试完成"
    else
        echo "❌ 基准测试失败，但继续执行其他任务"
    fi
fi

# =============================================================================
# 8. 启动演示界面
# =============================================================================

if [ "$CREATE_DEMO" = true ]; then
    echo "7. 启动演示界面..."
    
    # 检查Gradio依赖
    python3 -c "
try:
    import gradio
    print('✅ Gradio已安装')
except ImportError:
    print('❌ Gradio未安装')
    print('安装命令: pip install gradio')
    exit(1)
"
    
    if [ $? -eq 0 ]; then
        echo "启动Gradio演示界面..."
        echo "访问 http://localhost:7860 查看演示"
        
        # 在后台启动演示
        nohup python3 demo/gradio_demo.py \
            --model_path "$MODEL_PATH" \
            --device "$DEVICE" \
            --port 7860 > "$TEST_SESSION_DIR/gradio_demo.log" 2>&1 &
        
        DEMO_PID=$!
        echo "演示界面进程ID: $DEMO_PID"
        echo "$DEMO_PID" > "$TEST_SESSION_DIR/demo_pid.txt"
        
        echo "演示界面已启动，日志文件: $TEST_SESSION_DIR/gradio_demo.log"
        echo "要停止演示，运行: kill $DEMO_PID"
    else
        echo "⚠️  无法启动演示界面，Gradio未安装"
    fi
fi

# =============================================================================
# 9. 测试结果总结
# =============================================================================

echo ""
echo "=== 测试完成总结 ==="
echo "📁 测试会话目录: $TEST_SESSION_DIR"
echo ""

# 列出生成的文件
echo "📄 生成的文件:"
for file in "$TEST_SESSION_DIR"/*; do
    if [ -f "$file" ]; then
        size=$(du -h "$file" | cut -f1)
        echo "  - $(basename "$file") ($size)"
    fi
done

echo ""
echo "🎯 测试结果:"

# 检查测试结果
if [ -f "$TEST_SESSION_DIR/quick_test_results.json" ]; then
    echo "✅ 快速测试: 已完成"
fi

if [ -f "$TEST_SESSION_DIR/comprehensive_test_results.json" ]; then
    echo "✅ 综合测试: 已完成"
fi

if [ -f "$TEST_SESSION_DIR/benchmark_results.json" ]; then
    echo "✅ 基准测试: 已完成"
fi

if [ -f "$TEST_SESSION_DIR/demo_pid.txt" ]; then
    DEMO_PID=$(cat "$TEST_SESSION_DIR/demo_pid.txt")
    if kill -0 $DEMO_PID 2>/dev/null; then
        echo "✅ 演示界面: 运行中 (PID: $DEMO_PID)"
        echo "   访问: http://localhost:7860"
    else
        echo "❌ 演示界面: 启动失败"
    fi
fi

echo ""
echo "📖 使用建议:"
echo "1. 查看测试报告: cat $TEST_SESSION_DIR/test_report.txt"
echo "2. 分析测试结果: 检查 *_results.json 文件"
echo "3. 查看详细日志: 检查 *.log 文件"

if [ "$CREATE_DEMO" = true ] && [ -f "$TEST_SESSION_DIR/demo_pid.txt" ]; then
    echo "4. 访问演示界面: http://localhost:7860"
    echo "5. 停止演示: kill \$(cat $TEST_SESSION_DIR/demo_pid.txt)"
fi

echo ""
echo "🎉 测试脚本执行完成！" 