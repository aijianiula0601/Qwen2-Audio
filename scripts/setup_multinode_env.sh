#!/bin/bash

# Qwen2-Audio Multi-Node Environment Setup Script
# 多机训练环境配置脚本

set -e

# 配置选项
SETUP_METHOD="conda"  # conda, docker, virtualenv
PYTHON_VERSION="3.10"
PROJECT_NAME="qwen2-audio"
CONDA_ENV_NAME="qwen2-audio-multinode"
REQUIREMENTS_FILE="requirements.txt"

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --method)
            SETUP_METHOD="$2"
            shift 2
            ;;
        --python-version)
            PYTHON_VERSION="$2"
            shift 2
            ;;
        --env-name)
            CONDA_ENV_NAME="$2"
            shift 2
            ;;
        --requirements)
            REQUIREMENTS_FILE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --method METHOD            Setup method (conda, docker, virtualenv)"
            echo "  --python-version VERSION   Python version (default: 3.10)"
            echo "  --env-name NAME            Environment name (default: qwen2-audio-multinode)"
            echo "  --requirements FILE        Requirements file (default: requirements.txt)"
            echo "  -h, --help                 Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --method conda --python-version 3.10"
            echo "  $0 --method docker"
            echo "  $0 --method virtualenv --env-name my-qwen2-env"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

print_header "Qwen2-Audio Multi-Node Environment Setup"
print_status "Setup method: $SETUP_METHOD"
print_status "Python version: $PYTHON_VERSION"
print_status "Environment name: $CONDA_ENV_NAME"

# 检查基础依赖
check_dependencies() {
    print_header "Checking Dependencies"
    
    # 检查CUDA
    if command -v nvidia-smi &> /dev/null; then
        print_status "NVIDIA driver found: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"
    else
        print_error "NVIDIA driver not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    # 检查Git
    if ! command -v git &> /dev/null; then
        print_error "Git not found. Please install Git first."
        exit 1
    fi
    
    print_status "Basic dependencies check passed"
}

# 方法1: Conda环境配置
setup_conda_env() {
    print_header "Setting up Conda Environment"
    
    # 检查conda是否安装
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install Miniconda or Anaconda first."
        print_status "Download from: https://docs.conda.io/en/latest/miniconda.html"
        exit 1
    fi
    
    # 创建conda环境
    print_status "Creating conda environment: $CONDA_ENV_NAME"
    conda create -n "$CONDA_ENV_NAME" python="$PYTHON_VERSION" -y
    
    # 激活环境
    print_status "Activating conda environment"
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV_NAME"
    
    # 安装PyTorch (CUDA版本) - 通过conda安装，避免版本冲突
    print_status "Installing PyTorch with CUDA support via conda"
    conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y
    
    # 验证PyTorch安装
    print_status "Verifying PyTorch installation"
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"
    
    # 安装其他依赖 (排除PyTorch相关包)
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_status "Installing other requirements from $REQUIREMENTS_FILE (excluding PyTorch packages)"
        
        # 创建临时requirements文件，排除torch相关包
        TEMP_REQUIREMENTS="/tmp/requirements_filtered.txt"
        grep -v -E "^torch==|^torchaudio==|^torchvision==" "$REQUIREMENTS_FILE" > "$TEMP_REQUIREMENTS"
        
        print_status "Filtered requirements content:"
        cat "$TEMP_REQUIREMENTS"
        
        pip install -r "$TEMP_REQUIREMENTS"
        
        # 清理临时文件
        rm -f "$TEMP_REQUIREMENTS"
    else
        print_warning "Requirements file $REQUIREMENTS_FILE not found"
        print_status "Installing basic requirements"
        pip install deepspeed transformers librosa soundfile gradio numpy pandas
    fi
    
    # 最终验证安装
    print_status "Final verification of key packages"
    python -c "
import sys
print(f'Python executable: {sys.executable}')

packages = {
    'torch': 'PyTorch',
    'torchaudio': 'TorchAudio', 
    'transformers': 'Transformers',
    'deepspeed': 'DeepSpeed',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'gradio': 'Gradio',
    'numpy': 'NumPy',
    'pandas': 'Pandas'
}

for pkg, name in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError as e:
        print(f'❌ {name}: not installed - {e}')
"
    
    print_status "Conda environment setup completed"
    print_status "To activate: conda activate $CONDA_ENV_NAME"
}

# 方法2: Docker环境配置
setup_docker_env() {
    print_header "Setting up Docker Environment"
    
    # 检查Docker是否安装
    if ! command -v docker &> /dev/null; then
        print_error "Docker not found. Please install Docker first."
        exit 1
    fi
    
    # 检查NVIDIA Docker Runtime
    if ! docker info | grep -q nvidia; then
        print_error "NVIDIA Docker runtime not found. Please install nvidia-docker2."
        exit 1
    fi
    
    # 创建Dockerfile
    cat > Dockerfile << 'EOF'
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

# 设置工作目录
WORKDIR /workspace

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements并安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 安装DeepSpeed
RUN pip install deepspeed

# 设置环境变量
ENV PYTHONPATH=/workspace:$PYTHONPATH
ENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 创建非root用户
RUN useradd -m -s /bin/bash qwen2user
USER qwen2user

CMD ["/bin/bash"]
EOF

    # 构建Docker镜像
    print_status "Building Docker image: $PROJECT_NAME"
    docker build -t "$PROJECT_NAME:latest" .
    
    # 创建Docker运行脚本
    cat > run_docker.sh << EOF
#!/bin/bash
docker run -it --rm \\
    --gpus all \\
    --shm-size=32g \\
    --ulimit memlock=-1 \\
    --ulimit stack=67108864 \\
    -v \$(pwd):/workspace \\
    -v /etc/passwd:/etc/passwd:ro \\
    -v /etc/group:/etc/group:ro \\
    --network host \\
    $PROJECT_NAME:latest
EOF
    chmod +x run_docker.sh
    
    print_status "Docker environment setup completed"
    print_status "To run: ./run_docker.sh"
}

# 方法3: Virtualenv环境配置
setup_virtualenv() {
    print_header "Setting up Virtual Environment"
    
    # 检查Python版本
    if ! python$PYTHON_VERSION --version &> /dev/null; then
        print_error "Python $PYTHON_VERSION not found. Please install Python $PYTHON_VERSION first."
        exit 1
    fi
    
    # 安装virtualenv
    python$PYTHON_VERSION -m pip install --user virtualenv
    
    # 创建虚拟环境
    print_status "Creating virtual environment: $CONDA_ENV_NAME"
    python$PYTHON_VERSION -m venv "$CONDA_ENV_NAME"
    
    # 激活环境
    source "$CONDA_ENV_NAME/bin/activate"
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装PyTorch (使用正确的CUDA索引)
    print_status "Installing PyTorch with CUDA support via pip"
    pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    
    # 验证PyTorch安装
    print_status "Verifying PyTorch installation"
    python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
"
    
    # 安装其他依赖 (排除PyTorch相关包)
    if [ -f "$REQUIREMENTS_FILE" ]; then
        print_status "Installing other requirements from $REQUIREMENTS_FILE (excluding PyTorch packages)"
        
        # 创建临时requirements文件，排除torch相关包
        TEMP_REQUIREMENTS="/tmp/requirements_filtered.txt"
        grep -v -E "^torch==|^torchaudio==|^torchvision==" "$REQUIREMENTS_FILE" > "$TEMP_REQUIREMENTS"
        
        pip install -r "$TEMP_REQUIREMENTS"
        
        # 清理临时文件
        rm -f "$TEMP_REQUIREMENTS"
    else
        print_warning "Requirements file $REQUIREMENTS_FILE not found"
        print_status "Installing basic requirements"
        pip install deepspeed transformers librosa soundfile gradio numpy pandas
    fi
    
    # 最终验证安装
    print_status "Final verification of key packages"
    python -c "
import sys
print(f'Python executable: {sys.executable}')

packages = {
    'torch': 'PyTorch',
    'torchaudio': 'TorchAudio', 
    'transformers': 'Transformers',
    'deepspeed': 'DeepSpeed',
    'librosa': 'Librosa',
    'soundfile': 'SoundFile',
    'gradio': 'Gradio',
    'numpy': 'NumPy',
    'pandas': 'Pandas'
}

for pkg, name in packages.items():
    try:
        module = __import__(pkg)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {name}: {version}')
    except ImportError as e:
        print(f'❌ {name}: not installed - {e}')
"
    
    print_status "Virtual environment setup completed"
    print_status "To activate: source $CONDA_ENV_NAME/bin/activate"
}

# 创建环境配置文件
create_env_config() {
    print_header "Creating Environment Configuration"
    
    # 创建激活脚本
    cat > activate_env.sh << EOF
#!/bin/bash
# Qwen2-Audio Environment Activation Script

# 设置项目路径
export PROJECT_ROOT="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="\$PROJECT_ROOT:\$PYTHONPATH"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# NCCL设置（根据网络环境调整）
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口修改
# export NCCL_IB_DISABLE=1  # 如果InfiniBand有问题则取消注释

# DeepSpeed设置
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1

echo "Environment variables set for Qwen2-Audio multi-node training"
echo "Project root: \$PROJECT_ROOT"
echo "CUDA devices: \$CUDA_VISIBLE_DEVICES"
EOF

    chmod +x activate_env.sh
    
    # 创建环境检查脚本
    cp MULTI_NODE_ENVIRONMENT_SETUP.md env_check.py 2>/dev/null || true
    
    print_status "Environment configuration files created"
    print_status "  - activate_env.sh: Environment activation script"
    print_status "  - env_check.py: Environment consistency checker"
}

# 同步环境到其他节点
sync_to_nodes() {
    print_header "Environment Synchronization Guide"
    
    cat << 'EOF'
要将环境同步到其他节点，请执行以下步骤：

1. 使用rsync同步代码和环境:
   rsync -avz --exclude='.git' /path/to/qwen2-audio/ user@node1:/path/to/qwen2-audio/
   rsync -avz --exclude='.git' /path/to/qwen2-audio/ user@node2:/path/to/qwen2-audio/

2. 在每个节点上运行相同的环境配置:
   ssh user@node1 "cd /path/to/qwen2-audio && bash scripts/setup_multinode_env.sh --method conda"
   ssh user@node2 "cd /path/to/qwen2-audio && bash scripts/setup_multinode_env.sh --method conda"

3. 验证所有节点环境一致性:
   python env_check.py  # 在每个节点上运行

4. 对比环境检查结果，确保一致性

EOF
}

# 主函数
main() {
    check_dependencies
    
    case $SETUP_METHOD in
        conda)
            setup_conda_env
            ;;
        docker)
            setup_docker_env
            ;;
        virtualenv)
            setup_virtualenv
            ;;
        *)
            print_error "Unknown setup method: $SETUP_METHOD"
            print_status "Supported methods: conda, docker, virtualenv"
            exit 1
            ;;
    esac
    
    create_env_config
    sync_to_nodes
    
    print_header "Setup Complete"
    print_status "Environment setup completed successfully!"
    print_status "Next steps:"
    print_status "1. Sync environment to all nodes"
    print_status "2. Run env_check.py on each node to verify consistency"
    print_status "3. Start multi-node training with scripts/train_pretrain_multinode.sh"
}

# 运行主函数
main "$@" 