#!/bin/bash

# LlamaAudio Quick Start Script
# This script helps users quickly get started with LlamaAudio training

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_header() {
    echo -e "${PURPLE}🎵 $1${NC}"
}

# Welcome message
clear
echo "========================================================"
print_header "Welcome to LlamaAudio Quick Start!"
echo "========================================================"
echo ""
echo "This script will help you set up and start training your"
echo "LlamaAudio model using the three-stage training approach."
echo ""

# Check if running from correct directory
if [ ! -f "requirements.txt" ] || [ ! -d "llama_audio" ]; then
    print_error "Please run this script from the LlamaAudio project root directory"
    exit 1
fi

# Function to check prerequisites
check_prerequisites() {
    print_info "Checking prerequisites..."
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1-2)
    print_info "Python version: $python_version"
    
    # Check if CUDA is available
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        print_success "Found $gpu_count GPU(s)"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | head -3
    else
        print_warning "NVIDIA GPUs not detected. Training will be very slow on CPU."
    fi
    
    # Check disk space
    available_space=$(df -h . | tail -1 | awk '{print $4}')
    print_info "Available disk space: $available_space"
    
    echo ""
}

# Function to install dependencies
install_dependencies() {
    print_info "Installing dependencies..."
    
    if [ ! -d "venv" ]; then
        print_info "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    print_info "Activating virtual environment..."
    source venv/bin/activate
    
    print_info "Installing Python packages..."
    pip install --upgrade pip
    pip install -r requirements.txt
    
    print_success "Dependencies installed successfully!"
    echo ""
}

# Function to show training options
show_training_options() {
    echo "========================================================"
    print_header "Training Options"
    echo "========================================================"
    echo ""
    echo "Please choose your training approach:"
    echo ""
    echo "1) 🚀 Full Auto Training (Recommended)"
    echo "   - Automatically run all three stages"
    echo "   - Download all required datasets"
    echo "   - Estimated time: 26-52 hours"
    echo ""
    echo "2) 📚 Stage-by-Stage Training"
    echo "   - Manual control over each stage"
    echo "   - Review progress between stages"
    echo "   - More flexibility"
    echo ""
    echo "3) 🔧 Advanced Configuration"
    echo "   - Custom model settings"
    echo "   - Multi-node distributed training"
    echo "   - Expert users only"
    echo ""
    echo "4) 🎮 Demo Only"
    echo "   - Skip training, use pre-trained model"
    echo "   - Launch interactive demo"
    echo ""
    echo "5) ❓ Help & Documentation"
    echo "   - View detailed guides"
    echo "   - System requirements"
    echo ""
    
    read -p "Enter your choice (1-5): " choice
    echo ""
    
    case $choice in
        1) run_full_auto_training ;;
        2) run_stage_by_stage_training ;;
        3) run_advanced_configuration ;;
        4) run_demo_only ;;
        5) show_help ;;
        *) print_error "Invalid choice. Please run the script again."; exit 1 ;;
    esac
}

# Function for full auto training
run_full_auto_training() {
    print_header "Full Auto Training Selected"
    echo ""
    
    print_warning "This will:"
    echo "  • Download ~150GB of training data"
    echo "  • Train for 26-52 hours (depending on hardware)"
    echo "  • Use significant compute resources"
    echo ""
    
    read -p "Do you want to continue? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        exit 0
    fi
    
    # Get GPU configuration
    if command -v nvidia-smi &> /dev/null; then
        gpu_count=$(nvidia-smi --list-gpus | wc -l)
        echo ""
        print_info "Detected $gpu_count GPU(s)"
        read -p "How many GPUs to use for training? ($gpu_count): " num_gpus
        num_gpus=${num_gpus:-$gpu_count}
    else
        num_gpus=0
        print_warning "No GPUs detected. Using CPU (very slow)."
    fi
    
    # Choose model size
    echo ""
    echo "Choose Llama model size:"
    echo "1) Llama-3.3-70B-Instruct (Recommended, requires 8+ GPUs)"
    echo "2) Llama-3.1-8B-Instruct (Smaller, works with 2+ GPUs)"
    echo ""
    read -p "Enter choice (1-2): " model_choice
    
    case $model_choice in
        1) model_name="meta-llama/Llama-3.3-70B-Instruct" ;;
        2) model_name="meta-llama/Llama-3.1-8B-Instruct" ;;
        *) model_name="meta-llama/Llama-3.3-70B-Instruct" ;;
    esac
    
    print_info "Selected model: $model_name"
    print_info "Using $num_gpus GPU(s)"
    
    # Start training
    echo ""
    print_header "Starting Full Auto Training..."
    print_info "This will take 26-52 hours. You can monitor progress with TensorBoard."
    print_info "TensorBoard: tensorboard --logdir outputs/ --port 6006"
    echo ""
    
    if [ $num_gpus -gt 0 ]; then
        bash scripts/train_all_stages.sh \
            --num_gpus $num_gpus \
            --model_name "$model_name" \
            --auto_continue
    else
        print_error "CPU training not recommended for large models"
        exit 1
    fi
}

# Function for stage-by-stage training
run_stage_by_stage_training() {
    print_header "Stage-by-Stage Training Selected"
    echo ""
    
    echo "This approach allows you to:"
    echo "  • Review each stage results"
    echo "  • Adjust configurations between stages"
    echo "  • Stop and resume at any point"
    echo ""
    
    echo "Training stages:"
    echo "  Stage 1: Audio-Text Alignment (6-12 hours)"
    echo "  Stage 2: Audio Understanding (12-24 hours)"
    echo "  Stage 3: Instruction Following (8-16 hours)"
    echo ""
    
    read -p "Start with Stage 1? (y/N): " confirm
    if [[ ! $confirm =~ ^[Yy]$ ]]; then
        echo "Training cancelled."
        exit 0
    fi
    
    print_info "Starting Stage 1: Audio-Text Alignment"
    print_info "Downloading Stage 1 data..."
    bash scripts/data_download/download_stage1_data.sh
    
    print_info "Starting Stage 1 training..."
    bash scripts/train_stage1.sh
    
    print_success "Stage 1 completed!"
    echo ""
    
    read -p "Continue to Stage 2? (y/N): " confirm
    if [[ $confirm =~ ^[Yy]$ ]]; then
        print_info "Starting Stage 2: Audio Understanding"
        bash scripts/data_download/download_stage2_data.sh
        bash scripts/train_stage2.sh
        
        print_success "Stage 2 completed!"
        echo ""
        
        read -p "Continue to Stage 3? (y/N): " confirm
        if [[ $confirm =~ ^[Yy]$ ]]; then
            print_info "Starting Stage 3: Instruction Following"
            bash scripts/data_download/download_stage3_data.sh
            bash scripts/train_stage3.sh
            
            print_success "All stages completed!"
            echo ""
            show_completion_message
        fi
    fi
}

# Function for advanced configuration
run_advanced_configuration() {
    print_header "Advanced Configuration"
    echo ""
    
    print_warning "This is for expert users who want to customize training settings."
    echo ""
    
    echo "Available options:"
    echo "  • Multi-node distributed training"
    echo "  • Custom model configurations"
    echo "  • Advanced DeepSpeed settings"
    echo "  • Custom data paths"
    echo ""
    
    echo "For advanced configuration, please:"
    echo "1. Edit configuration files in configs/"
    echo "2. Use command-line arguments with training scripts"
    echo "3. Refer to docs/MULTI_STAGE_TRAINING.md"
    echo ""
    
    print_info "Example multi-node training:"
    echo "bash scripts/train_all_stages.sh \\"
    echo "  --num_nodes 4 \\"
    echo "  --node_rank 0 \\"
    echo "  --master_addr '192.168.1.100' \\"
    echo "  --auto_continue"
    echo ""
}

# Function for demo only
run_demo_only() {
    print_header "Demo Mode"
    echo ""
    
    print_info "Demo mode allows you to test LlamaAudio without training."
    echo ""
    
    # Check if trained model exists
    if [ -d "outputs/stage3_conversation/final_model" ]; then
        print_success "Found trained model!"
        model_path="outputs/stage3_conversation/final_model"
    else
        print_warning "No trained model found."
        echo ""
        echo "Options:"
        echo "1) Use a pre-trained model (download required)"
        echo "2) Train a model first"
        echo ""
        read -p "Enter choice (1-2): " demo_choice
        
        case $demo_choice in
            1) 
                print_info "Pre-trained model download not implemented yet."
                print_info "Please train a model first or provide a model path."
                exit 1
                ;;
            2)
                print_info "Please run the training first."
                exit 1
                ;;
            *)
                print_error "Invalid choice."
                exit 1
                ;;
        esac
    fi
    
    print_info "Starting interactive demo..."
    python scripts/interactive_demo.py \
        --model_path "$model_path" \
        --port 7860 \
        --share
}

# Function to show help
show_help() {
    print_header "Help & Documentation"
    echo ""
    
    echo "📚 Available Documentation:"
    echo ""
    echo "• README.md - Project overview and quick start"
    echo "• docs/MULTI_STAGE_TRAINING.md - Detailed training guide"
    echo "• IMPLEMENTATION_SUMMARY.md - Complete feature overview"
    echo ""
    
    echo "📋 System Requirements:"
    echo ""
    echo "Minimum:"
    echo "• 8 GPUs (A100 40GB recommended)"
    echo "• 512GB RAM"
    echo "• 250GB storage"
    echo ""
    
    echo "Recommended:"
    echo "• 2-4 nodes with 8 GPUs each"
    echo "• 1TB+ RAM per node"  
    echo "• 500GB+ SSD storage"
    echo ""
    
    echo "🔗 Useful Commands:"
    echo ""
    echo "# Monitor training progress"
    echo "tensorboard --logdir outputs/ --port 6006"
    echo ""
    echo "# Check GPU usage"
    echo "watch -n 1 nvidia-smi"
    echo ""
    echo "# Evaluate trained model"
    echo "python scripts/evaluate.py --model_path outputs/stage3_conversation/final_model"
    echo ""
    
    read -p "Press Enter to return to main menu..."
    show_training_options
}

# Function to show completion message
show_completion_message() {
    echo "========================================================"
    print_header "🎉 Training Completed Successfully! 🎉"
    echo "========================================================"
    echo ""
    
    print_success "Your LlamaAudio model is ready!"
    echo ""
    
    echo "Next steps:"
    echo ""
    echo "1. 🎮 Test your model:"
    echo "   python scripts/interactive_demo.py \\"
    echo "     --model_path outputs/stage3_conversation/final_model \\"
    echo "     --port 7860 --share"
    echo ""
    
    echo "2. 📊 Evaluate performance:"
    echo "   python scripts/evaluate.py \\"
    echo "     --model_path outputs/stage3_conversation/final_model"
    echo ""
    
    echo "3. 🚀 Deploy for production:"
    echo "   See deployment section in README.md"
    echo ""
    
    print_info "Model location: outputs/stage3_conversation/final_model/"
    print_info "Training logs: outputs/*/training.log"
    print_info "TensorBoard logs: outputs/*/tensorboard/"
    echo ""
}

# Main execution
main() {
    check_prerequisites
    
    # Ask about dependency installation
    read -p "Do you want to install/update dependencies? (Y/n): " install_deps
    if [[ ! $install_deps =~ ^[Nn]$ ]]; then
        install_dependencies
    fi
    
    show_training_options
}

# Run main function
main 