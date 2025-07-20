#!/bin/bash

# Qwen2-Audio Multi-Node Full Training Pipeline
# This script runs the complete training pipeline (pretrain -> sft -> dpo) across multiple nodes

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 默认配置
DEFAULT_NODES=("v100_f178" "v100_f165" "v100")
DEFAULT_MASTER_PORT=29500
DEFAULT_GPUS_PER_NODE=8
DEFAULT_MODEL_TYPE="qwen2_0.5b"
DEFAULT_CONDA_ENV="qwen2-audio-multinode"

# 显示帮助信息
show_help() {
    cat << EOF
Qwen2-Audio Multi-Node Full Training Pipeline

用法: $0 [选项]

选项:
    --nodes NODES             节点列表，用逗号分隔 (默认: ${DEFAULT_NODES[*]})
    --master-port PORT        主节点端口 (默认: $DEFAULT_MASTER_PORT)
    --gpus-per-node GPUS      每节点GPU数量 (默认: $DEFAULT_GPUS_PER_NODE)
    --model MODEL_TYPE        模型类型 (默认: $DEFAULT_MODEL_TYPE)
    --conda-env ENV_NAME      Conda环境名称 (默认: $DEFAULT_CONDA_ENV)
    --config CONFIG_FILE      配置文件路径
    --resume-stage STAGE      从指定阶段恢复 (pretrain, sft, dpo)
    --resume-checkpoint PATH  从指定检查点恢复
    --skip-pretrain           跳过预训练阶段
    --skip-sft                跳过SFT阶段
    --skip-dpo                跳过DPO阶段
    --dry-run                 只显示命令，不执行
    --verbose                 详细输出
    --help, -h                显示此帮助信息

训练阶段:
    1. Pretrain (预训练) - 在大规模音频-文本数据上训练
    2. SFT (监督微调) - 在高质量指令数据上微调
    3. DPO (直接偏好优化) - 基于人类偏好优化模型

示例:
    # 完整训练流程
    $0 --model qwen2_7b --nodes "node1,node2,node3"
    
    # 从SFT阶段开始
    $0 --model qwen2_7b --resume-stage sft --resume-checkpoint outputs/pretrain_qwen2_7b_xxx/
    
    # 跳过预训练，只做SFT和DPO
    $0 --model qwen2_7b --skip-pretrain

EOF
}

# 解析命令行参数
parse_args() {
    NODES=("${DEFAULT_NODES[@]}")
    MASTER_PORT=$DEFAULT_MASTER_PORT
    GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE
    MODEL_TYPE=$DEFAULT_MODEL_TYPE
    CONDA_ENV=$DEFAULT_CONDA_ENV
    CONFIG_FILE=""
    RESUME_STAGE=""
    RESUME_CHECKPOINT=""
    SKIP_PRETRAIN=false
    SKIP_SFT=false
    SKIP_DPO=false
    DRY_RUN=false
    VERBOSE=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --nodes)
                IFS=',' read -ra NODES <<< "$2"
                shift 2
                ;;
            --master-port)
                MASTER_PORT="$2"
                shift 2
                ;;
            --gpus-per-node)
                GPUS_PER_NODE="$2"
                shift 2
                ;;
            --model)
                MODEL_TYPE="$2"
                shift 2
                ;;
            --conda-env)
                CONDA_ENV="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --resume-stage)
                RESUME_STAGE="$2"
                shift 2
                ;;
            --resume-checkpoint)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --skip-pretrain)
                SKIP_PRETRAIN=true
                shift
                ;;
            --skip-sft)
                SKIP_SFT=true
                shift
                ;;
            --skip-dpo)
                SKIP_DPO=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 设置全局变量
    NNODES=${#NODES[@]}
    MASTER_ADDR=${NODES[0]}
    WORLD_SIZE=$((NNODES * GPUS_PER_NODE))
    
    if [[ "$VERBOSE" == "true" ]]; then
        log_debug "配置信息:"
        log_debug "  节点: ${NODES[*]}"
        log_debug "  节点数: $NNODES"
        log_debug "  主节点: $MASTER_ADDR"
        log_debug "  每节点GPU: $GPUS_PER_NODE"
        log_debug "  总GPU数: $WORLD_SIZE"
        log_debug "  模型: $MODEL_TYPE"
        log_debug "  端口: $MASTER_PORT"
        log_debug "  跳过预训练: $SKIP_PRETRAIN"
        log_debug "  跳过SFT: $SKIP_SFT"
        log_debug "  跳过DPO: $SKIP_DPO"
    fi
}

# 检查环境
check_environment() {
    log_info "检查训练环境..."
    
    # 检查数据目录
    if [[ "$SKIP_PRETRAIN" == "false" ]] && [[ ! -d "data/pretrain" ]]; then
        log_error "预训练数据不存在: data/pretrain"
        exit 1
    fi
    
    if [[ "$SKIP_SFT" == "false" ]] && [[ ! -d "data/sft" ]]; then
        log_error "SFT数据不存在: data/sft"
        exit 1
    fi
    
    if [[ "$SKIP_DPO" == "false" ]] && [[ ! -d "data/dpo" ]]; then
        log_error "DPO数据不存在: data/dpo"
        exit 1
    fi
    
    # 检查配置文件
    if [[ -n "$CONFIG_FILE" ]] && [[ ! -f "$CONFIG_FILE" ]]; then
        log_error "配置文件不存在: $CONFIG_FILE"
        exit 1
    fi
    
    # 检查训练脚本
    if [[ ! -f "scripts/train_pretrain_multinode.sh" ]]; then
        log_error "预训练脚本不存在: scripts/train_pretrain_multinode.sh"
        exit 1
    fi
    
    if [[ ! -f "scripts/train_sft_multinode.sh" ]]; then
        log_error "SFT脚本不存在: scripts/train_sft_multinode.sh"
        exit 1
    fi
    
    if [[ ! -f "scripts/train_dpo_multinode.sh" ]]; then
        log_error "DPO脚本不存在: scripts/train_dpo_multinode.sh"
        exit 1
    fi
    
    log_info "✅ 环境检查通过"
}

# 创建hostfile
create_hostfile() {
    HOSTFILE="configs/hostfile_${NNODES}nodes_${GPUS_PER_NODE}gpu.txt"
    log_info "创建hostfile: $HOSTFILE"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_debug "创建hostfile内容:"
        for node in "${NODES[@]}"; do
            log_debug "  $node slots=$GPUS_PER_NODE"
        done
    else
        > "$HOSTFILE"
        for node in "${NODES[@]}"; do
            echo "$node slots=$GPUS_PER_NODE" >> "$HOSTFILE"
        done
        log_info "Hostfile已创建:"
        cat "$HOSTFILE"
    fi
}

# 运行训练阶段
run_training_stage() {
    local stage=$1
    local resume_checkpoint=$2
    
    log_info "=== 开始 $stage 阶段训练 ==="
    
    # 构建训练命令
    local train_script="scripts/train_${stage}_multinode.sh"
    local cmd_args=(
        "--model" "$MODEL_TYPE"
        "--hostfile" "$HOSTFILE"
        "--master-addr" "$MASTER_ADDR"
        "--master-port" "$MASTER_PORT"
        "--gpus-per-node" "$GPUS_PER_NODE"
        "--conda-env" "$CONDA_ENV"
    )
    
    if [[ -n "$CONFIG_FILE" ]]; then
        cmd_args+=("--config" "$CONFIG_FILE")
    fi
    
    if [[ -n "$resume_checkpoint" ]]; then
        cmd_args+=("--resume" "$resume_checkpoint")
    fi
    
    # 显示完整命令
    local full_cmd="bash $train_script ${cmd_args[*]}"
    log_info "执行命令: $full_cmd"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_debug "DRY RUN: 不会实际执行命令"
        return
    fi
    
    # 创建日志目录
    local log_dir="logs/$(date +%Y%m%d_%H%M%S)_${stage}_${MODEL_TYPE}"
    mkdir -p "$log_dir"
    
    # 启动训练
    log_info "训练日志将保存到: $log_dir"
    nohup bash "$train_script" "${cmd_args[@]}" > "$log_dir/training.log" 2>&1 &
    
    local training_pid=$!
    echo "$training_pid" > "$log_dir/training.pid"
    
    log_info "训练已启动，PID: $training_pid"
    
    # 等待训练完成
    log_info "等待训练完成..."
    wait $training_pid
    
    # 检查训练结果
    if [[ $? -eq 0 ]]; then
        log_info "✅ $stage 阶段训练完成"
        
        # 查找最新的输出目录
        local latest_output=$(find outputs -name "${stage}_${MODEL_TYPE}_*" -type d | sort | tail -1)
        if [[ -n "$latest_output" ]]; then
            log_info "模型保存在: $latest_output"
            echo "$latest_output" > "/tmp/qwen2_audio_${stage}_output"
        fi
    else
        log_error "❌ $stage 阶段训练失败"
        log_info "查看日志: tail -f $log_dir/training.log"
        exit 1
    fi
}

# 主训练流程
main_training_pipeline() {
    log_info "=== Qwen2-Audio 多机训练流程 ==="
    log_info "模型: $MODEL_TYPE"
    log_info "节点数: $NNODES"
    log_info "总GPU数: $WORLD_SIZE"
    log_info "节点列表: ${NODES[*]}"
    
    # 检查环境
    check_environment
    
    # 创建hostfile
    create_hostfile
    
    # 确定开始阶段
    local current_stage="pretrain"
    local resume_checkpoint="$RESUME_CHECKPOINT"
    
    if [[ -n "$RESUME_STAGE" ]]; then
        case "$RESUME_STAGE" in
            pretrain)
                current_stage="pretrain"
                ;;
            sft)
                current_stage="sft"
                if [[ -z "$resume_checkpoint" ]]; then
                    log_error "从SFT阶段恢复需要指定预训练检查点"
                    exit 1
                fi
                ;;
            dpo)
                current_stage="dpo"
                if [[ -z "$resume_checkpoint" ]]; then
                    log_error "从DPO阶段恢复需要指定SFT检查点"
                    exit 1
                fi
                ;;
            *)
                log_error "无效的恢复阶段: $RESUME_STAGE"
                exit 1
                ;;
        esac
    fi
    
    # 执行训练阶段
    if [[ "$current_stage" == "pretrain" ]] && [[ "$SKIP_PRETRAIN" == "false" ]]; then
        run_training_stage "pretrain" "$resume_checkpoint"
        current_stage="sft"
        resume_checkpoint=$(cat "/tmp/qwen2_audio_pretrain_output" 2>/dev/null || echo "")
    fi
    
    if [[ "$current_stage" == "sft" ]] && [[ "$SKIP_SFT" == "false" ]]; then
        if [[ -z "$resume_checkpoint" ]]; then
            # 查找最新的预训练检查点
            resume_checkpoint=$(find outputs -name "pretrain_${MODEL_TYPE}_*" -type d | sort | tail -1)
            if [[ -z "$resume_checkpoint" ]]; then
                log_error "未找到预训练检查点，无法开始SFT训练"
                exit 1
            fi
        fi
        run_training_stage "sft" "$resume_checkpoint"
        current_stage="dpo"
        resume_checkpoint=$(cat "/tmp/qwen2_audio_sft_output" 2>/dev/null || echo "")
    fi
    
    if [[ "$current_stage" == "dpo" ]] && [[ "$SKIP_DPO" == "false" ]]; then
        if [[ -z "$resume_checkpoint" ]]; then
            # 查找最新的SFT检查点
            resume_checkpoint=$(find outputs -name "sft_${MODEL_TYPE}_*" -type d | sort | tail -1)
            if [[ -z "$resume_checkpoint" ]]; then
                log_error "未找到SFT检查点，无法开始DPO训练"
                exit 1
            fi
        fi
        run_training_stage "dpo" "$resume_checkpoint"
    fi
    
    log_info "=== 训练流程完成 ==="
    
    # 显示最终结果
    local final_output=$(find outputs -name "dpo_${MODEL_TYPE}_*" -type d | sort | tail -1)
    if [[ -n "$final_output" ]]; then
        log_info "🎉 训练完成！最终模型保存在: $final_output"
        log_info "模型已准备好用于推理！"
    else
        log_warn "未找到最终模型输出目录"
    fi
}

# 清理临时文件
cleanup() {
    rm -f /tmp/qwen2_audio_*_output
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 设置清理函数
    trap cleanup EXIT
    
    # 执行主训练流程
    main_training_pipeline
}

# 执行主函数
main "$@" 