# Qwen2-Audio 多机训练环境配置指南

本文档详细介绍如何为Qwen2-Audio多机训练配置一致的Python环境。

## 🎯 环境一致性的重要性

多机训练要求所有节点具备：
- **相同的Python版本和依赖包版本**
- **一致的CUDA和PyTorch版本**
- **相同的环境变量和路径配置**
- **统一的项目代码和数据访问方式**

不一致的环境可能导致：
- 分布式训练初始化失败
- 模型参数同步错误
- 训练过程中的随机崩溃
- 性能不一致

## 🛠️ 三种环境配置方案

### 方案1：Conda环境（推荐）

**优点**：依赖管理简单，环境隔离良好，易于版本控制  
**适用场景**：大多数情况，特别是开发和小规模部署

#### 快速配置
```bash
# 在所有节点上执行
bash scripts/setup_multinode_env.sh --method conda --python-version 3.10
```

#### 手动配置步骤
```bash
# 1. 安装Miniconda（如果未安装）
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/bin/activate

# 2. 创建专用环境
conda create -n qwen2-audio-multinode python=3.10 -y
conda activate qwen2-audio-multinode

# 3. 安装PyTorch (CUDA 12.4)
conda install pytorch==2.6.0 torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia -y

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 方案2：Docker容器（推荐用于生产）

**优点**：完全隔离，一致性最好，易于部署和扩展  
**适用场景**：生产环境，云部署，大规模集群

#### 快速配置
```bash
# 在所有节点上执行
bash scripts/setup_multinode_env.sh --method docker
```

#### 手动配置步骤
```bash
# 1. 安装Docker和NVIDIA Container Toolkit
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# 安装NVIDIA Container Toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker

# 2. 构建项目镜像
docker build -t qwen2-audio:multinode .

# 3. 创建运行脚本
cat > run_container.sh << 'EOF'
#!/bin/bash
docker run -it --rm \
    --gpus all \
    --shm-size=32g \
    --network host \
    -v $(pwd):/workspace \
    -v /data:/data \
    qwen2-audio:multinode
EOF
chmod +x run_container.sh

# 4. 在容器中验证
./run_container.sh
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

### 方案3：Virtual Environment

**优点**：轻量级，不需要额外工具  
**适用场景**：简单环境，临时测试

#### 快速配置
```bash
bash scripts/setup_multinode_env.sh --method virtualenv --python-version 3.10
```

#### 手动配置步骤
```bash
# 1. 创建虚拟环境
python3.10 -m venv qwen2-audio-multinode
source qwen2-audio-multinode/bin/activate

# 2. 升级pip
pip install --upgrade pip

# 3. 安装PyTorch
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. 安装项目依赖
pip install -r requirements.txt

# 5. 验证安装
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## 📋 依赖包版本要求

基于`requirements.txt`的关键依赖：

```txt
# 核心框架
torch==2.6.0
torchaudio==2.6.0+cu124
transformers==4.52.4
deepspeed==0.16.9

# 音频处理
librosa==0.11.0
soundfile==0.13.1

# 数据处理
numpy==2.2.6
pandas==2.2.3

# 其他工具
gradio==5.33.0
PyYAML==6.0.2
tqdm==4.67.1
```

**注意**：
- CUDA版本必须与PyTorch兼容
- DeepSpeed版本需与PyTorch版本匹配
- 所有节点必须使用完全相同的版本

## 🔧 环境变量配置

### 必需的环境变量

```bash
# 项目路径
export PROJECT_ROOT="/path/to/Qwen2-Audio"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# CUDA设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 分布式训练
export MASTER_ADDR="node0.cluster.com"
export MASTER_PORT=29500
export WORLD_SIZE=32  # 总GPU数
export RANK=0         # 当前节点全局排名
export LOCAL_RANK=0   # 当前节点本地排名

# NCCL通信优化
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口修改
export NCCL_IB_DISABLE=1        # 如果没有InfiniBand

# DeepSpeed优化
export DS_BUILD_CPU_ADAM=1
export DS_BUILD_UTILS=1
```

### 自动环境设置脚本

```bash
# 使用项目提供的环境激活脚本
source activate_env.sh
```

## 🚀 环境同步策略

### 策略1：代码和环境分离同步

```bash
# 1. 同步代码（排除环境文件）
rsync -avz --exclude='.git' --exclude='*/miniconda3/*' \
    /path/to/qwen2-audio/ user@node1:/path/to/qwen2-audio/

# 2. 在每个节点独立配置环境
ssh user@node1 "cd /path/to/qwen2-audio && bash scripts/setup_multinode_env.sh --method conda"
ssh user@node2 "cd /path/to/qwen2-audio && bash scripts/setup_multinode_env.sh --method conda"
```

### 策略2：使用共享存储

```bash
# 1. 在共享存储上配置环境（仅需一次）
cd /shared/qwen2-audio
bash scripts/setup_multinode_env.sh --method conda

# 2. 所有节点使用相同路径
export PROJECT_ROOT="/shared/qwen2-audio"
export CONDA_PREFIX="/shared/qwen2-audio/miniconda3/envs/qwen2-audio-multinode"
```

### 策略3：Docker镜像分发

```bash
# 1. 在主节点构建镜像
docker build -t qwen2-audio:v1.0 .

# 2. 保存和分发镜像
docker save qwen2-audio:v1.0 | gzip > qwen2-audio-v1.0.tar.gz
scp qwen2-audio-v1.0.tar.gz user@node1:/tmp/
scp qwen2-audio-v1.0.tar.gz user@node2:/tmp/

# 3. 在其他节点加载镜像
ssh user@node1 "gunzip -c /tmp/qwen2-audio-v1.0.tar.gz | docker load"
ssh user@node2 "gunzip -c /tmp/qwen2-audio-v1.0.tar.gz | docker load"
```

## ✅ 环境一致性验证

### 1. 使用环境检查脚本

```bash
# 在每个节点运行环境检查
python MULTI_NODE_ENVIRONMENT_SETUP.md  # 实际上是env_check.py

# 收集所有节点的检查结果
for node in node0 node1 node2 node3; do
    scp user@$node:/path/to/qwen2-audio/env_check_*.json ./
done

# 比较结果
python -c "
import json
import glob

files = glob.glob('env_check_*.json')
for f in files:
    with open(f) as file:
        data = json.load(file)
        print(f'{f}: PyTorch {data["packages"]["torch"]}, CUDA {data["cuda"]["cuda_version"]}')
"
```

### 2. 快速验证命令

```bash
# 在所有节点运行以下命令，输出应该一致
python -c "
import torch, transformers, deepspeed
print(f'PyTorch: {torch.__version__}')
print(f'Transformers: {transformers.__version__}')
print(f'DeepSpeed: {deepspeed.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU Count: {torch.cuda.device_count()}')
print(f'NCCL Available: {torch.distributed.is_nccl_available()}')
"
```

### 3. 分布式通信测试

```bash
# 在主节点（node0）运行
python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='tcp://node0:29500', world_size=4, rank=0)
print('Rank 0 initialized successfully')
dist.destroy_process_group()
"

# 在其他节点运行（修改rank）
python -c "
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl', init_method='tcp://node0:29500', world_size=4, rank=1)
print('Rank 1 initialized successfully')
dist.destroy_process_group()
"
```

## 🐛 常见问题解决

### 问题1：版本不一致

**现象**：不同节点报告不同的包版本
```bash
# 解决方案：强制重新安装
pip uninstall torch torchaudio transformers deepspeed -y
pip install torch==2.6.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers==4.52.4 deepspeed==0.16.9
```

### 问题2：CUDA版本冲突

**现象**：`RuntimeError: CUDA error: no kernel image is available for execution`
```bash
# 检查CUDA兼容性
nvidia-smi  # 查看驱动版本
python -c "import torch; print(torch.version.cuda)"  # 查看PyTorch CUDA版本

# 重新安装匹配的PyTorch版本
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121  # 根据实际CUDA版本
```

### 问题3：DeepSpeed编译失败

**现象**：`No module named 'deepspeed.ops.adam'`
```bash
# 重新编译DeepSpeed
pip uninstall deepspeed -y
DS_BUILD_CPU_ADAM=1 DS_BUILD_UTILS=1 pip install deepspeed --no-cache-dir

# 或使用预编译版本
pip install deepspeed --no-deps --force-reinstall
```

### 问题4：路径不一致

**现象**：`ModuleNotFoundError: No module named 'models'`
```bash
# 确保所有节点使用相同的PROJECT_ROOT
export PROJECT_ROOT="/consistent/path/to/qwen2-audio"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# 或在Python代码中添加路径
import sys
sys.path.append('/path/to/qwen2-audio')
```

## 📚 最佳实践总结

### 1. 环境管理
- ✅ 使用Conda进行包管理，Docker进行完整环境隔离
- ✅ 固定所有依赖包版本，避免自动更新
- ✅ 定期备份工作环境，便于快速恢复

### 2. 同步策略
- ✅ 优先使用共享存储，减少同步复杂度
- ✅ 使用自动化脚本，避免手动配置差异
- ✅ 建立环境验证流程，训练前必须检查

### 3. 监控维护
- ✅ 定期检查环境一致性
- ✅ 监控各节点资源使用情况
- ✅ 建立问题排查清单和解决方案库

通过以上方案，你可以确保多机训练环境的一致性和稳定性。建议首先在2个节点上测试验证，确认无误后再扩展到更多节点。 