#!/bin/bash

# Quick test for multi-node training fix
# This script performs a minimal test to verify the fix works

set -e

echo "=== Quick Multi-Node Training Test ==="
echo "This script tests if the DeepSpeed launcher fix works with your configuration."
echo

# Check if we're in the right environment
if [[ "$CONDA_DEFAULT_ENV" != "qwen2-audio-multinode" ]]; then
    echo "⚠️  Please activate the correct conda environment first:"
    echo "   conda activate qwen2-audio-multinode"
    exit 1
fi

# Configuration
NODES=("202.168.100.178" "202.168.100.251" "202.168.100.165")
MASTER_ADDR=${NODES[0]}
MASTER_PORT=10021  # Use different port to avoid conflicts
GPUS_PER_NODE=8
MODEL_TYPE="qwen2_0.5b"

echo "Test configuration:"
echo "  Nodes: ${NODES[*]}"
echo "  Master: $MASTER_ADDR"
echo "  Port: $MASTER_PORT"
echo

# Test 1: Check if training script exists and is updated
echo "=== Test 1: Training Script Check ==="
SCRIPT_FILE="scripts/train_pretrain_multinode.sh"

if [ ! -f "$SCRIPT_FILE" ]; then
    echo "❌ Training script not found: $SCRIPT_FILE"
    exit 1
fi

# Check if the script has the fix (contains --no_ssh)
if grep -q "\-\-no_ssh" "$SCRIPT_FILE"; then
    echo "✅ Training script contains --no_ssh fix"
else
    echo "❌ Training script missing --no_ssh fix"
    echo "   Please run the fix script: bash scripts/fix_multinode_training.sh"
    exit 1
fi

# Check if ALL_NODES is configured
if grep -q 'ALL_NODES=("202.168.100.178"' "$SCRIPT_FILE"; then
    echo "✅ ALL_NODES array is configured with your nodes"
else
    echo "⚠️  ALL_NODES array may not be configured correctly"
    echo "   Current configuration in script:"
    grep "ALL_NODES=" "$SCRIPT_FILE" | head -1
fi

# Test 2: Dry run of training command
echo ""
echo "=== Test 2: Dry Run Test ==="

# Create a minimal test to see if the command would work
echo "Testing training script help (dry run)..."
if bash "$SCRIPT_FILE" --help >/dev/null 2>&1; then
    echo "✅ Training script help works"
else
    echo "❌ Training script has issues"
    exit 1
fi

# Test 3: DeepSpeed command validation
echo ""
echo "=== Test 3: DeepSpeed Command Validation ==="

# Create temporary hostfile for testing
TEMP_HOSTFILE="/tmp/qwen2_audio_test_${MASTER_PORT}.txt"
> $TEMP_HOSTFILE
for node in "${NODES[@]}"; do
    echo "$node slots=$GPUS_PER_NODE" >> $TEMP_HOSTFILE
done

echo "Created test hostfile: $TEMP_HOSTFILE"
echo "Content:"
cat $TEMP_HOSTFILE

# Test DeepSpeed command
echo ""
echo "Testing DeepSpeed command syntax..."
TEST_CMD="deepspeed --hostfile=$TEMP_HOSTFILE --master_port=$MASTER_PORT --node_rank=0 --no_ssh --help"

if eval "$TEST_CMD" >/dev/null 2>&1; then
    echo "✅ DeepSpeed command syntax is valid"
else
    echo "❌ DeepSpeed command syntax failed"
    rm -f $TEMP_HOSTFILE
    exit 1
fi

# Clean up
rm -f $TEMP_HOSTFILE

# Test 4: Generate actual launch commands
echo ""
echo "=== Test 4: Launch Commands ==="
echo "If all tests pass, use these commands to start training:"
echo

for i in "${!NODES[@]}"; do
    node=${NODES[$i]}
    echo "# On $node (node rank $i):"
    echo "conda activate qwen2-audio-multinode"
    echo "cd $(pwd)"
    echo "bash scripts/train_pretrain_multinode.sh \\"
    echo "    --model $MODEL_TYPE \\"
    echo "    --nnodes ${#NODES[@]} \\"
    echo "    --node-rank $i \\"
    echo "    --master-addr $MASTER_ADDR \\"
    echo "    --master-port $MASTER_PORT \\"
    echo "    --gpus-per-node $GPUS_PER_NODE"
    echo
done

echo "🎉 All tests passed! The multi-node training fix should work."
echo ""
echo "Next steps:"
echo "1. Copy the commands above to each respective node"
echo "2. Run them simultaneously (start with node 0 first)"
echo "3. Monitor the logs for successful initialization"
echo "4. Check GPU utilization: nvidia-smi"
echo ""
echo "Troubleshooting:"
echo "- If you get connection errors, check firewall settings"
echo "- If you get CUDA errors, verify GPU availability"
echo "- If you get import errors, check the conda environment" 