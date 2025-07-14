#!/bin/bash

# Qwen2-Audio Multi-Node Training Quick Launch Example
# This script demonstrates how to launch multi-node training

set -e

echo "=== Qwen2-Audio Multi-Node Training Quick Launch ==="

# Configuration
NODES=("v100_f178" "v100_f165" "v100")
NNODES=${#NODES[@]}
MASTER_ADDR=${NODES[0]}
MASTER_PORT=10050
GPUS_PER_NODE=8
MODEL_TYPE="qwen2_0.5b"

echo "Cluster Configuration:"
echo "  Nodes: ${NODES[*]}"
echo "  Total nodes: $NNODES"  
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  GPUs per node: $GPUS_PER_NODE"
echo "  Model: $MODEL_TYPE"
echo

# Method 1: Using hostfile (recommended)
echo "=== Method 1: Using DeepSpeed Hostfile ==="

# Create hostfile
HOSTFILE="configs/hostfile_auto.txt"
echo "Creating hostfile: $HOSTFILE"
> $HOSTFILE
for node in "${NODES[@]}"; do
    echo "$node slots=$GPUS_PER_NODE" >> $HOSTFILE
done

echo "Hostfile content:"
cat $HOSTFILE
echo

# Launch using hostfile (only run on master node)
echo "Launch command (run on master node only):"
echo "bash scripts/train_pretrain_multinode.sh \\"
echo "    --model $MODEL_TYPE \\"
echo "    --hostfile $HOSTFILE \\"
echo "    --master-addr $MASTER_ADDR \\"
echo "    --master-port $MASTER_PORT"
echo

# Method 2: Manual launch on each node
echo "=== Method 2: Manual Launch Commands ==="
echo "✅ FIXED: DeepSpeed launcher issue resolved!"
echo "   - Added --no_ssh flag to avoid pdsh dependency"
echo "   - Automatic temporary hostfile creation"
echo ""
echo "⚠️  IMPORTANT: Before using Method 2, you need to update the ALL_NODES array"
echo "   in scripts/train_pretrain_multinode.sh with your actual node IPs/hostnames!"
echo ""
echo "Example: Edit scripts/train_pretrain_multinode.sh and update this line:"
echo "   ALL_NODES=(\"202.168.100.178\" \"202.168.100.251\" \"202.168.100.165\")"
echo ""
echo "Run these commands on respective nodes:"
echo

for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    echo "# On $node (node rank $i):"
    echo "bash scripts/train_pretrain_multinode.sh \\"
    echo "    --model $MODEL_TYPE \\"
    echo "    --nnodes $NNODES \\"
    echo "    --node-rank $i \\"
    echo "    --master-addr $MASTER_ADDR \\"
    echo "    --master-port $MASTER_PORT \\"
    echo "    --gpus-per-node $GPUS_PER_NODE"
    echo
done

# Method 3: Automated SSH launch (if SSH keys are configured)
echo "=== Method 3: Automated SSH Launch ==="
echo "# Configure SSH settings and uncomment to enable automated launch"
echo

# SSH Configuration - 修改这些变量
SSH_USER="huangjiahong.dracu"  # 替换为你的SSH用户名
PROJECT_DIR="/mnt/cephfs/hjh/pycharm_projects/my_github/nlp/Qwen2-Audio"  # 项目目录

echo "SSH User: $SSH_USER"
echo "Project Directory: $PROJECT_DIR"
echo
echo "# Uncomment the following section to enable automated launch"

cat << 'EOF'
echo "Launching training on all nodes automatically..."

# Launch on all nodes in parallel
for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    echo "Starting training on $node (rank $i)..."
    
    ssh -n $SSH_USER@$node "
        cd $PROJECT_DIR
        nohup bash scripts/train_pretrain_multinode.sh \
            --model $MODEL_TYPE \
            --nnodes $NNODES \
            --node-rank $i \
            --master-addr $MASTER_ADDR \
            --master-port $MASTER_PORT \
            --gpus-per-node $GPUS_PER_NODE \
            > logs/node_${i}_$(date +%Y%m%d_%H%M%S).log 2>&1 &
    " &
done

echo "Training launched on all nodes. Check logs for progress."
echo "Monitor with: tail -f logs/node_*.log"
EOF

echo
echo "=== Pre-launch Checklist ==="
echo "□ All nodes have the same code and environment"
echo "□ All nodes can access training data"
echo "□ SSH passwordless login configured (for Method 1 and 3)"
echo "□ Firewall allows communication on port $MASTER_PORT"
echo "□ All nodes have GPU resources available"
echo "□ Network bandwidth sufficient for multi-node communication"
echo
echo "=== Monitoring Commands ==="
echo "# Check GPU usage on all nodes:"
echo "for node in ${NODES[*]}; do"
echo "    echo \"=== \$node ===\""
echo "    ssh \$node 'nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv'"
echo "done"
echo
echo "# Monitor training logs:"
echo "tail -f outputs/pretrain_*/pretrain_node*.log"
echo
echo "Ready to launch multi-node training!" 