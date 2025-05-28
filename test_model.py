#!/usr/bin/env python3
"""
测试训练好的Qwen2-Audio模型
"""

import os
import torch
from src import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor

def test_model():
    # 加载配置
    try:
        with open('.training_config', 'r') as f:
            config_content = f.read()
        config = {}
        for line in config_content.strip().split('\n'):
            key, value = line.split('=', 1)
            config[key] = value.strip('"')
    except:
        config = {
            'BASE_MODEL': 'Qwen/Qwen2-7B',
            'MODEL_SIZE': '7B'
        }
    
    # 确定模型路径
    model_paths = [
        "checkpoints/qwen2-audio-stage3",
        "checkpoints/qwen2-audio-stage2", 
        "checkpoints/qwen2-audio-stage1"
    ]
    
    model_path = None
    for path in model_paths:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        print("错误: 未找到训练好的模型检查点")
        print("请先运行训练脚本: bash scripts/setup_and_train.sh")
        return
    
    print(f"使用模型: {config.get('BASE_MODEL', 'Unknown')} ({config.get('MODEL_SIZE', 'Unknown')})")
    print(f"加载检查点: {model_path}")
    
    # 检查是否有GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    
    try:
        # 加载模型和处理器
        model = Qwen2AudioForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
        processor = Qwen2AudioProcessor.from_pretrained(model_path)
        
        if device == "cpu":
            model = model.to(device)
        
        print("✅ 模型加载成功")
        
        # 测试音频文件
        test_audio_files = [
            "test_audio.wav",
            "demo/test_audio.wav", 
            "data/raw/test.wav"
        ]
        
        audio_path = None
        for path in test_audio_files:
            if os.path.exists(path):
                audio_path = path
                break
        
        if not audio_path:
            print("⚠️  未找到测试音频文件")
            print("请提供以下任一测试音频:")
            for path in test_audio_files:
                print(f"  - {path}")
            return
        
        print(f"使用测试音频: {audio_path}")
        
        # 测试不同任务
        test_prompts = [
            "请转录这段音频",
            "请描述这段音频的内容",
            "这段音频表达了什么情感？",
            "请总结音频中的主要信息"
        ]
        
        for prompt in test_prompts:
            print(f"\n🔍 测试提示: {prompt}")
            
            try:
                # 处理输入
                inputs = processor(
                    audio_path=audio_path,
                    text=prompt,
                    return_tensors="pt"
                )
                
                if device == "cuda":
                    inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # 生成回复
                with torch.no_grad():
                    outputs = model.generate(
                        **inputs, 
                        max_length=512,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=processor.tokenizer.eos_token_id
                    )
                
                # 解码输出
                response = processor.decode(outputs[0], skip_special_tokens=True)
                print(f"💬 模型回复: {response}")
                
            except Exception as e:
                print(f"❌ 生成失败: {e}")
        
        print(f"\n✅ 模型测试完成")
        print(f"模型规模: {config.get('MODEL_SIZE', 'Unknown')}")
        print(f"基础模型: {config.get('BASE_MODEL', 'Unknown')}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("可能原因:")
        print("- 显存不足 (尝试使用更小的模型)")
        print("- 检查点损坏 (重新训练)")
        print("- 依赖缺失 (检查requirements.txt)")

if __name__ == "__main__":
    test_model()
