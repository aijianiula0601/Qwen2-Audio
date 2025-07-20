#!/bin/bash

# Qwen2-Audio Multi-Node Training Launcher
# 统一的多机训练启动脚本，支持多种启动方式

set -e

# 默认配置
DEFAULT_NODES=("v100_f178" "v100_f165" "v100")
DEFAULT_MASTER_PORT=29500
DEFAULT_GPUS_PER_NODE=8
DEFAULT_MODEL_TYPE="qwen2_0.5b"
DEFAULT_STAGE="pretrain"
DEFAULT_CONDA_ENV="qwen2-audio-multinode"

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

# 显示帮助信息
show_help() {
    cat << EOF
Qwen2-Audio Multi-Node Training Launcher

用法: $0 [选项] [命令]

选项:
    --nodes NODES             节点列表，用逗号分隔 (默认: ${DEFAULT_NODES[*]})
    --master-port PORT        主节点端口 (默认: $DEFAULT_MASTER_PORT)
    --gpus-per-node GPUS      每节点GPU数量 (默认: $DEFAULT_GPUS_PER_NODE)
    --model MODEL_TYPE        模型类型 (默认: $DEFAULT_MODEL_TYPE)
    --stage STAGE             训练阶段: pretrain, sft, dpo (默认: $DEFAULT_STAGE)
    --conda-env ENV_NAME      Conda环境名称 (默认: $DEFAULT_CONDA_ENV)
    --config CONFIG_FILE      配置文件路径
    --resume CHECKPOINT       从检查点恢复
    --dry-run                 只显示命令，不执行
    --verbose                 详细输出
    --help, -h                显示此帮助信息

命令:
    check-env                 检查所有节点的环境
    check-network             检查节点间网络连通性
    check-gpu                 检查所有节点的GPU状态
    launch                    启动训练
    stop                      停止训练
    status                    查看训练状态
    logs                      查看训练日志

示例:
    # 检查环境
    $0 check-env --nodes "node1,node2,node3"
    
    # 启动训练
    $0 launch --model qwen2_7b --stage pretrain --nodes "node1,node2,node3"
    
    # 使用hostfile方式启动
    $0 launch --model qwen2_7b --stage pretrain --hostfile configs/hostfile.txt
    
    # 检查训练状态
    $0 status
    
    # 查看日志
    $0 logs

EOF
}

# 解析命令行参数
parse_args() {
    NODES=("${DEFAULT_NODES[@]}")
    MASTER_PORT=$DEFAULT_MASTER_PORT
    GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE
    MODEL_TYPE=$DEFAULT_MODEL_TYPE
    STAGE=$DEFAULT_STAGE
    CONDA_ENV=$DEFAULT_CONDA_ENV
    CONFIG_FILE=""
    RESUME_CHECKPOINT=""
    HOSTFILE=""
    DRY_RUN=false
    VERBOSE=false
    COMMAND=""
    
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
            --stage)
                STAGE="$2"
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
            --resume)
                RESUME_CHECKPOINT="$2"
                shift 2
                ;;
            --hostfile)
                HOSTFILE="$2"
                shift 2
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
            check-env|check-network|check-gpu|launch|stop|status|logs)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "未知选项: $1"
                show_help
                exit 1
                ;;
        esac
    done
    
    # 验证必需参数
    if [[ -z "$COMMAND" ]]; then
        log_error "请指定要执行的命令"
        show_help
        exit 1
    fi
    
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
        log_debug "  阶段: $STAGE"
        log_debug "  端口: $MASTER_PORT"
    fi
}

# 检查环境
check_environment() {
    log_info "检查所有节点的环境..."
    
    for i in "${!NODES[@]}"; do
        node=${NODES[$i]}
        log_info "检查节点 $node (rank $i)..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "ssh $node '检查环境命令'"
            continue
        fi
        
        # 检查基本环境
        ssh -o ConnectTimeout=10 "$node" "
            echo '=== 节点: $node ==='
            echo 'Python版本:'
            python --version 2>/dev/null || echo 'Python未安装'
            
            echo 'CUDA版本:'
            nvidia-smi --query-gpu=cuda_version --format=csv,noheader 2>/dev/null || echo 'CUDA未安装'
            
            echo 'GPU数量:'
            nvidia-smi --list-gpus 2>/dev/null | wc -l || echo '0'
            
            echo '项目目录:'
            ls -la /mnt/cephfs/hjh/pycharm_projects/my_github/nlp/Qwen2-Audio 2>/dev/null || echo '项目目录不存在'
            
            echo 'Conda环境:'
            conda env list 2>/dev/null | grep $CONDA_ENV || echo 'Conda环境不存在'
        " || {
            log_error "无法连接到节点 $node"
        }
        echo
    done
}

# 检查网络连通性
check_network() {
    log_info "检查节点间网络连通性..."
    
    for i in "${!NODES[@]}"; do
        for j in "${!NODES[@]}"; do
            if [[ $i -ne $j ]]; then
                source_node=${NODES[$i]}
                target_node=${NODES[$j]}
                
                log_info "测试 $source_node -> $target_node"
                
                if [[ "$DRY_RUN" == "true" ]]; then
                    log_debug "ssh $source_node 'ping -c 3 $target_node'"
                else
                    ssh "$source_node" "ping -c 3 $target_node" 2>/dev/null || {
                        log_warn "无法从 $source_node 连接到 $target_node"
                    }
                fi
            fi
        done
    done
    
    # 测试端口连通性
    log_info "测试端口连通性..."
    for node in "${NODES[@]}"; do
        log_info "测试 $node:$MASTER_PORT"
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "nc -zv $node $MASTER_PORT"
        else
            nc -zv "$node" "$MASTER_PORT" 2>/dev/null || {
                log_warn "端口 $MASTER_PORT 在 $node 上不可用"
            }
        fi
    done
}

# 检查GPU状态
check_gpu() {
    log_info "检查所有节点的GPU状态..."
    
    for node in "${NODES[@]}"; do
        log_info "检查节点 $node 的GPU..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "ssh $node 'nvidia-smi'"
        else
            ssh "$node" "
                echo '=== $node GPU状态 ==='
                nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits
            " 2>/dev/null || {
                log_error "无法获取 $node 的GPU信息"
            }
        fi
        echo
    done
}

# 创建hostfile
create_hostfile() {
    if [[ -n "$HOSTFILE" ]]; then
        log_info "使用指定的hostfile: $HOSTFILE"
        return
    fi
    
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

# 启动训练
launch_training() {
    log_info "启动 $STAGE 阶段训练..."
    log_info "模型: $MODEL_TYPE"
    log_info "节点数: $NNODES"
    log_info "总GPU数: $WORLD_SIZE"
    
    # 创建hostfile
    create_hostfile
    
    # 构建训练命令
    TRAIN_SCRIPT="scripts/train_${STAGE}_multinode.sh"
    
    if [[ ! -f "$TRAIN_SCRIPT" ]]; then
        log_error "训练脚本不存在: $TRAIN_SCRIPT"
        exit 1
    fi
    
    # 构建命令参数
    CMD_ARGS=(
        "--model" "$MODEL_TYPE"
        "--hostfile" "$HOSTFILE"
        "--master-addr" "$MASTER_ADDR"
        "--master-port" "$MASTER_PORT"
        "--gpus-per-node" "$GPUS_PER_NODE"
    )
    
    if [[ -n "$CONFIG_FILE" ]]; then
        CMD_ARGS+=("--config" "$CONFIG_FILE")
    fi
    
    if [[ -n "$RESUME_CHECKPOINT" ]]; then
        CMD_ARGS+=("--resume" "$RESUME_CHECKPOINT")
    fi
    
    if [[ -n "$CONDA_ENV" ]]; then
        CMD_ARGS+=("--conda-env" "$CONDA_ENV")
    fi
    
    # 显示完整命令
    FULL_CMD="bash $TRAIN_SCRIPT ${CMD_ARGS[*]}"
    log_info "执行命令: $FULL_CMD"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_debug "DRY RUN: 不会实际执行命令"
        return
    fi
    
    # 创建日志目录
    LOG_DIR="logs/$(date +%Y%m%d_%H%M%S)_${STAGE}_${MODEL_TYPE}"
    mkdir -p "$LOG_DIR"
    
    # 启动训练
    log_info "训练日志将保存到: $LOG_DIR"
    nohup bash "$TRAIN_SCRIPT" "${CMD_ARGS[@]}" > "$LOG_DIR/training.log" 2>&1 &
    
    TRAINING_PID=$!
    echo "$TRAINING_PID" > "$LOG_DIR/training.pid"
    
    log_info "训练已启动，PID: $TRAINING_PID"
    log_info "查看日志: tail -f $LOG_DIR/training.log"
}

# 停止训练
stop_training() {
    log_info "停止训练..."
    
    # 查找训练进程
    PID_FILES=(logs/*/training.pid)
    
    if [[ ${#PID_FILES[@]} -eq 0 ]]; then
        log_warn "未找到训练进程"
        return
    fi
    
    for pid_file in "${PID_FILES[@]}"; do
        if [[ -f "$pid_file" ]]; then
            PID=$(cat "$pid_file")
            log_info "停止进程 $PID..."
            
            if [[ "$DRY_RUN" == "true" ]]; then
                log_debug "kill $PID"
            else
                kill "$PID" 2>/dev/null || log_warn "进程 $PID 不存在或已停止"
                rm -f "$pid_file"
            fi
        fi
    done
    
    # 停止所有节点上的训练
    for node in "${NODES[@]}"; do
        log_info "停止节点 $node 上的训练进程..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "ssh $node 'pkill -f train_.*_multinode.sh'"
        else
            ssh "$node" "pkill -f train_.*_multinode.sh" 2>/dev/null || true
        fi
    done
}

# 查看训练状态
check_status() {
    log_info "检查训练状态..."
    
    # 检查本地进程
    PID_FILES=(logs/*/training.pid)
    
    if [[ ${#PID_FILES[@]} -eq 0 ]]; then
        log_info "本地无训练进程"
    else
        for pid_file in "${PID_FILES[@]}"; do
            if [[ -f "$pid_file" ]]; then
                PID=$(cat "$pid_file")
                if kill -0 "$PID" 2>/dev/null; then
                    log_info "本地训练进程运行中，PID: $PID"
                else
                    log_warn "本地训练进程已停止，PID: $PID"
                fi
            fi
        done
    fi
    
    # 检查所有节点的进程
    for node in "${NODES[@]}"; do
        log_info "检查节点 $node 的训练进程..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "ssh $node 'ps aux | grep train_.*_multinode.sh'"
        else
            ssh "$node" "ps aux | grep train_.*_multinode.sh | grep -v grep" 2>/dev/null || {
                log_info "节点 $node 无训练进程"
            }
        fi
    done
    
    # 检查GPU使用情况
    log_info "检查GPU使用情况..."
    for node in "${NODES[@]}"; do
        log_info "节点 $node 的GPU使用情况:"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "ssh $node 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
        else
            ssh "$node" "nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv" 2>/dev/null || {
                log_error "无法获取 $node 的GPU信息"
            }
        fi
        echo
    done
}

# 查看训练日志
show_logs() {
    log_info "查看训练日志..."
    
    # 查找最新的日志文件
    LATEST_LOG=$(find logs -name "training.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 | cut -d' ' -f2-)
    
    if [[ -n "$LATEST_LOG" ]]; then
        log_info "最新日志文件: $LATEST_LOG"
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_debug "tail -f $LATEST_LOG"
        else
            tail -f "$LATEST_LOG"
        fi
    else
        log_warn "未找到训练日志文件"
        
        # 显示所有日志文件
        LOG_FILES=(logs/*/training.log)
        if [[ ${#LOG_FILES[@]} -gt 0 ]]; then
            log_info "可用的日志文件:"
            for log_file in "${LOG_FILES[@]}"; do
                echo "  $log_file"
            done
        fi
    fi
}

# 主函数
main() {
    # 解析参数
    parse_args "$@"
    
    # 执行命令
    case "$COMMAND" in
        check-env)
            check_environment
            ;;
        check-network)
            check_network
            ;;
        check-gpu)
            check_gpu
            ;;
        launch)
            launch_training
            ;;
        stop)
            stop_training
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs
            ;;
        *)
            log_error "未知命令: $COMMAND"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@" 