#!/bin/bash

# Qwen2-Audio Multi-Node Training Fix Script
# This script helps configure and test the fix for the DeepSpeed hostfile issue

set -e

echo "=== Qwen2-Audio Multi-Node Training Fix ==="
echo "This script helps you configure and test the fix for the DeepSpeed hostfile issue."
echo

# Configuration
NODES=("202.168.100.178" "202.168.100.251" "202.168.100.165")
MASTER_ADDR=${NODES[0]}
MASTER_PORT=10020
GPUS_PER_NODE=8

echo "Current node configuration:"
echo "  Nodes: ${NODES[*]}"
echo "  Master: $MASTER_ADDR"
echo "  Port: $MASTER_PORT"
echo "  GPUs per node: $GPUS_PER_NODE"
echo

# Function to update the training script
update_training_script() {
    local script_file="scripts/train_pretrain_multinode.sh"
    
    if [ ! -f "$script_file" ]; then
        echo "❌ Training script not found: $script_file"
        exit 1
    fi
    
    echo "Updating $script_file with your node configuration..."
    
    # Create backup
    cp "$script_file" "${script_file}.backup.$(date +%Y%m%d_%H%M%S)"
    echo "✅ Backup created: ${script_file}.backup.$(date +%Y%m%d_%H%M%S)"
    
    # Build the nodes array string for the script
    nodes_array_str="("
    for i in "${!NODES[@]}"; do
        if [ $i -eq 0 ]; then
            nodes_array_str+="\"${NODES[$i]}\""
        else
            nodes_array_str+=" \"${NODES[$i]}\""
        fi
    done
    nodes_array_str+=")"
    
    # Update the ALL_NODES array in the script
    sed -i "s/ALL_NODES=(\"\$MASTER_ADDR\")/ALL_NODES=$nodes_array_str/" "$script_file"
    
    echo "✅ Updated ALL_NODES array in $script_file"
    echo "   New configuration: ALL_NODES=$nodes_array_str"
}

# Function to test the configuration
test_configuration() {
    echo ""
    echo "=== Testing Configuration ==="
    
    # Test SSH connectivity
    echo "Testing SSH connectivity to all nodes..."
    for node in "${NODES[@]}"; do
        echo -n "  Testing $node... "
        if ssh -o ConnectTimeout=5 -o BatchMode=yes "$node" "echo 'SSH OK'" 2>/dev/null; then
            echo "✅ OK"
        else
            echo "❌ FAILED"
            echo "    Please ensure SSH passwordless login is configured for $node"
        fi
    done
    
    # Test GPU availability
    echo ""
    echo "Testing GPU availability on all nodes..."
    for node in "${NODES[@]}"; do
        echo "  Node $node:"
        ssh "$node" "nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null || echo '    ❌ nvidia-smi failed'" | sed 's/^/    /'
    done
    
    # Test port availability
    echo ""
    echo "Testing port availability..."
    echo -n "  Testing port $MASTER_PORT on $MASTER_ADDR... "
    if nc -z "$MASTER_ADDR" "$MASTER_PORT" 2>/dev/null; then
        echo "❌ Port is already in use"
        echo "    Consider using a different port with --master-port"
    else
        echo "✅ Port is available"
    fi
}

# Function to generate launch commands
generate_launch_commands() {
    echo ""
    echo "=== Launch Commands ==="
    echo "After updating the configuration, use these commands:"
    echo
    
    for i in "${!NODES[@]}"; do
        node=${NODES[$i]}
        echo "# On $node (node rank $i):"
        echo "bash scripts/train_pretrain_multinode.sh \\"
        echo "    --model qwen2_0.5b \\"
        echo "    --nnodes ${#NODES[@]} \\"
        echo "    --node-rank $i \\"
        echo "    --master-addr $MASTER_ADDR \\"
        echo "    --master-port $MASTER_PORT \\"
        echo "    --gpus-per-node $GPUS_PER_NODE"
        echo
    done
}

# Function to show fix details
show_fix_details() {
    echo ""
    echo "=== Fix Details ==="
    echo "The issue was: DeepSpeed requires a hostfile for multi-node training (nnodes > 1)"
    echo "The fix: Automatically create a temporary hostfile with all node information"
    echo ""
    echo "What was changed in scripts/train_pretrain_multinode.sh:"
    echo "  1. Added detection for multi-node training without hostfile"
    echo "  2. Create temporary hostfile with all nodes when needed"
    echo "  3. Use the temporary hostfile for DeepSpeed launcher"
    echo ""
    echo "Manual alternative: You can always use Method 1 with explicit hostfile:"
    echo "  bash scripts/multi_node_launch.sh  # and follow Method 1 instructions"
}

# Main menu
while true; do
    echo ""
    echo "Choose an option:"
    echo "  1) Update training script with current node configuration"
    echo "  2) Test current configuration"
    echo "  3) Generate launch commands"
    echo "  4) Show fix details"
    echo "  5) Edit node configuration"
    echo "  q) Quit"
    echo
    read -p "Enter your choice: " choice
    
    case $choice in
        1)
            update_training_script
            ;;
        2)
            test_configuration
            ;;
        3)
            generate_launch_commands
            ;;
        4)
            show_fix_details
            ;;
        5)
            echo ""
            echo "Current nodes: ${NODES[*]}"
            echo "Enter new node list (space-separated):"
            read -a NEW_NODES
            if [ ${#NEW_NODES[@]} -gt 0 ]; then
                NODES=("${NEW_NODES[@]}")
                MASTER_ADDR=${NODES[0]}
                echo "✅ Updated nodes: ${NODES[*]}"
                echo "✅ Updated master: $MASTER_ADDR"
            fi
            ;;
        q|Q)
            echo "Goodbye!"
            exit 0
            ;;
        *)
            echo "Invalid choice. Please try again."
            ;;
    esac
done 