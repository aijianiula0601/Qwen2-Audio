#!/usr/bin/env python3
"""
Qwen2-Audio Multi-Node Environment Consistency Checker
检查多机训练环境的一致性
"""

import sys
import json
import subprocess
import socket
import torch
import platform
from importlib import import_module
from packaging import version

def get_system_info():
    """获取系统信息"""
    return {
        'hostname': socket.gethostname(),
        'python_version': sys.version,
        'platform': platform.platform(),
        'architecture': platform.architecture(),
    }

def get_cuda_info():
    """获取CUDA信息"""
    cuda_info = {
        'cuda_available': torch.cuda.is_available(),
        'cuda_version': None,
        'cudnn_version': None,
        'gpu_count': 0,
        'gpu_names': []
    }
    
    if torch.cuda.is_available():
        cuda_info['cuda_version'] = torch.version.cuda
        cuda_info['cudnn_version'] = torch.backends.cudnn.version()
        cuda_info['gpu_count'] = torch.cuda.device_count()
        cuda_info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return cuda_info

def get_package_versions():
    """获取关键包版本"""
    key_packages = [
        'torch', 'torchaudio', 'transformers', 'deepspeed', 
        'numpy', 'pandas', 'librosa', 'soundfile', 'gradio'
    ]
    
    package_versions = {}
    for package in key_packages:
        try:
            module = import_module(package)
            package_versions[package] = getattr(module, '__version__', 'unknown')
        except ImportError:
            package_versions[package] = 'not_installed'
    
    return package_versions

def check_pytorch_distributed():
    """检查PyTorch分布式功能"""
    dist_info = {
        'backend_available': {
            'nccl': torch.distributed.is_nccl_available(),
            'gloo': torch.distributed.is_gloo_available(),
            'mpi': torch.distributed.is_mpi_available()
        }
    }
    return dist_info

def get_environment_variables():
    """获取关键环境变量"""
    import os
    env_vars = {}
    key_vars = [
        'CUDA_VISIBLE_DEVICES', 'PYTHONPATH', 'PATH',
        'MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK'
    ]
    
    for var in key_vars:
        env_vars[var] = os.environ.get(var, 'not_set')
    
    return env_vars

def main():
    """主函数"""
    print("=== Qwen2-Audio Multi-Node Environment Check ===")
    
    # 收集所有信息
    info = {
        'system': get_system_info(),
        'cuda': get_cuda_info(),
        'packages': get_package_versions(),
        'distributed': check_pytorch_distributed(),
        'environment': get_environment_variables()
    }
    
    # 输出详细信息
    print(f"\n🖥️  节点信息:")
    print(f"   主机名: {info['system']['hostname']}")
    print(f"   Python版本: {info['system']['python_version'].split()[0]}")
    print(f"   平台: {info['system']['platform']}")
    
    print(f"\n🚀 CUDA信息:")
    print(f"   CUDA可用: {info['cuda']['cuda_available']}")
    if info['cuda']['cuda_available']:
        print(f"   CUDA版本: {info['cuda']['cuda_version']}")
        print(f"   cuDNN版本: {info['cuda']['cudnn_version']}")
        print(f"   GPU数量: {info['cuda']['gpu_count']}")
        for i, gpu_name in enumerate(info['cuda']['gpu_names']):
            print(f"   GPU {i}: {gpu_name}")
    
    print(f"\n📦 关键包版本:")
    for pkg, ver in info['packages'].items():
        status = "✅" if ver != 'not_installed' else "❌"
        print(f"   {status} {pkg}: {ver}")
    
    print(f"\n🌐 分布式后端:")
    for backend, available in info['distributed']['backend_available'].items():
        status = "✅" if available else "❌"
        print(f"   {status} {backend}: {available}")
    
    print(f"\n🔧 环境变量:")
    for var, value in info['environment'].items():
        print(f"   {var}: {value}")
    
    # 保存完整信息到JSON文件
    hostname = info['system']['hostname']
    output_file = f"env_check_{hostname}.json"
    
    with open(output_file, 'w') as f:
        json.dump(info, f, indent=2, default=str)
    
    print(f"\n💾 详细信息已保存到: {output_file}")
    print(f"\n✅ 环境检查完成！请将此文件发送给集群管理员进行一致性比较。")

if __name__ == "__main__":
    main() 