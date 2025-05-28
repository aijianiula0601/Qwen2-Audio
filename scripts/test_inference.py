#!/usr/bin/env python3
"""
Qwen2-Audio 推理测试脚本
测试训练好的模型在各种音频任务上的性能
"""

import os
import sys
import json
import time
import argparse
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
except ImportError:
    print("❌ 错误: 无法导入Qwen2Audio模块")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)


class Qwen2AudioTester:
    def __init__(self, model_path: str, device: str = "auto"):
        """初始化测试器"""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        self.load_model()
        
    def _setup_device(self, device: str) -> str:
        """设置设备"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                print(f"🔧 使用GPU: {torch.cuda.get_device_name()}")
                print(f"   显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            else:
                device = "cpu"
                print("🔧 使用CPU (推理速度较慢)")
        return device
        
    def load_model(self):
        """加载模型和处理器"""
        print(f"📥 加载模型: {self.model_path}")
        
        try:
            # 加载处理器
            self.processor = Qwen2AudioProcessor.from_pretrained(self.model_path)
            print("✅ 处理器加载成功")
            
            # 加载模型
            if self.device == "cuda":
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                self.model = Qwen2AudioForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float32,
                    trust_remote_code=True
                ).to(self.device)
                
            self.model.eval()
            print("✅ 模型加载成功")
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise
            
    def generate_response(self, audio_path: str, prompt: str, **kwargs) -> str:
        """生成模型回复"""
        try:
            # 处理输入
            inputs = self.processor(
                audio_path=audio_path,
                text=prompt,
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成参数
            generation_kwargs = {
                "max_length": kwargs.get("max_length", 512),
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "top_p": kwargs.get("top_p", 0.9),
                "pad_token_id": self.processor.tokenizer.eos_token_id,
                **kwargs
            }
            
            # 生成回复
            start_time = time.time()
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            inference_time = time.time() - start_time
            
            # 解码输出
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            return response, inference_time
            
        except Exception as e:
            return f"生成失败: {str(e)}", 0.0
            
    def test_asr(self, audio_files: List[str]) -> Dict:
        """测试语音识别任务"""
        print("\n🎯 测试语音识别 (ASR)")
        results = []
        
        asr_prompts = [
            "请转录这段音频",
            "将这段语音转换为文字",
            "请识别音频中的语音内容",
            "听录音并写出对应的文本"
        ]
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                continue
                
            print(f"\n📁 测试文件: {audio_file}")
            
            for prompt in asr_prompts[:2]:  # 只测试前两个prompt
                print(f"💬 提示: {prompt}")
                
                response, time_cost = self.generate_response(
                    audio_file, prompt, 
                    max_length=256,
                    temperature=0.1,
                    do_sample=False
                )
                
                print(f"🤖 回复: {response}")
                print(f"⏱️  耗时: {time_cost:.2f}秒")
                
                results.append({
                    "task": "asr",
                    "audio_file": audio_file,
                    "prompt": prompt,
                    "response": response,
                    "time_cost": time_cost
                })
                
        return {"task": "asr", "results": results}
        
    def test_audio_understanding(self, audio_files: List[str]) -> Dict:
        """测试音频理解任务"""
        print("\n🎯 测试音频理解")
        results = []
        
        understanding_prompts = [
            "请描述这段音频的内容",
            "这段音频中发生了什么？",
            "分析并描述音频中的声音",
            "请总结音频的主要信息"
        ]
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                continue
                
            print(f"\n📁 测试文件: {audio_file}")
            
            for prompt in understanding_prompts[:2]:
                print(f"💬 提示: {prompt}")
                
                response, time_cost = self.generate_response(
                    audio_file, prompt,
                    max_length=512,
                    temperature=0.7
                )
                
                print(f"🤖 回复: {response}")
                print(f"⏱️  耗时: {time_cost:.2f}秒")
                
                results.append({
                    "task": "understanding",
                    "audio_file": audio_file,
                    "prompt": prompt,
                    "response": response,
                    "time_cost": time_cost
                })
                
        return {"task": "understanding", "results": results}
        
    def test_emotion_recognition(self, audio_files: List[str]) -> Dict:
        """测试情感识别任务"""
        print("\n🎯 测试情感识别")
        results = []
        
        emotion_prompts = [
            "请识别这段音频中的情感",
            "分析说话者的情感状态",
            "这段语音表达了什么情感？",
            "判断音频中的情绪倾向"
        ]
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                continue
                
            print(f"\n📁 测试文件: {audio_file}")
            
            for prompt in emotion_prompts[:2]:
                print(f"💬 提示: {prompt}")
                
                response, time_cost = self.generate_response(
                    audio_file, prompt,
                    max_length=128,
                    temperature=0.3
                )
                
                print(f"🤖 回复: {response}")
                print(f"⏱️  耗时: {time_cost:.2f}秒")
                
                results.append({
                    "task": "emotion",
                    "audio_file": audio_file,
                    "prompt": prompt,
                    "response": response,
                    "time_cost": time_cost
                })
                
        return {"task": "emotion", "results": results}
        
    def test_translation(self, audio_files: List[str]) -> Dict:
        """测试语音翻译任务"""
        print("\n🎯 测试语音翻译")
        results = []
        
        translation_prompts = [
            "请将这段英文音频翻译为中文",
            "请将这段中文音频翻译为英文",
            "翻译这段语音的内容",
            "请提供这段音频的翻译"
        ]
        
        for audio_file in audio_files:
            if not os.path.exists(audio_file):
                continue
                
            print(f"\n📁 测试文件: {audio_file}")
            
            for prompt in translation_prompts[:2]:
                print(f"💬 提示: {prompt}")
                
                response, time_cost = self.generate_response(
                    audio_file, prompt,
                    max_length=256,
                    temperature=0.5
                )
                
                print(f"🤖 回复: {response}")
                print(f"⏱️  耗时: {time_cost:.2f}秒")
                
                results.append({
                    "task": "translation",
                    "audio_file": audio_file,
                    "prompt": prompt,
                    "response": response,
                    "time_cost": time_cost
                })
                
        return {"task": "translation", "results": results}
        
    def run_comprehensive_test(self, audio_files: List[str], output_file: str = None) -> Dict:
        """运行综合测试"""
        print("🚀 开始综合推理测试")
        print(f"📊 测试文件数量: {len([f for f in audio_files if os.path.exists(f)])}")
        
        all_results = {
            "model_path": self.model_path,
            "device": self.device,
            "test_files": audio_files,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "tests": []
        }
        
        # 运行各种测试
        test_functions = [
            self.test_asr,
            self.test_audio_understanding,
            self.test_emotion_recognition,
            self.test_translation
        ]
        
        for test_func in test_functions:
            try:
                result = test_func(audio_files)
                all_results["tests"].append(result)
            except Exception as e:
                print(f"❌ 测试 {test_func.__name__} 失败: {e}")
        
        # 计算统计信息
        total_tests = sum(len(test["results"]) for test in all_results["tests"])
        total_time = sum(
            result["time_cost"] 
            for test in all_results["tests"] 
            for result in test["results"]
        )
        avg_time = total_time / total_tests if total_tests > 0 else 0
        
        all_results["statistics"] = {
            "total_tests": total_tests,
            "total_time": total_time,
            "average_time": avg_time
        }
        
        print(f"\n📊 测试统计:")
        print(f"   总测试数: {total_tests}")
        print(f"   总耗时: {total_time:.2f}秒")
        print(f"   平均耗时: {avg_time:.2f}秒/测试")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, ensure_ascii=False, indent=2)
            print(f"📝 结果已保存到: {output_file}")
        
        return all_results


def find_test_audio_files() -> List[str]:
    """查找测试音频文件"""
    potential_paths = [
        "test_audio.wav",
        "demo/test_audio.wav",
        "data/test_audio.wav",
        "examples/test_audio.wav",
        "samples/*.wav",
        "demo/*.wav"
    ]
    
    audio_files = []
    for path in potential_paths:
        if '*' in path:
            # 处理通配符
            from glob import glob
            audio_files.extend(glob(path))
        elif os.path.exists(path):
            audio_files.append(path)
    
    return audio_files


def create_test_audio():
    """创建测试音频文件"""
    print("🔧 创建测试音频文件...")
    
    # 创建一个简单的正弦波音频作为测试
    sample_rate = 16000
    duration = 3  # 3秒
    frequency = 440  # A4音符
    
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    wave = np.sin(2 * np.pi * frequency * t)
    
    # 添加一些变化使其更有趣
    wave = wave * np.exp(-t / 2)  # 衰减
    
    # 保存为wav文件
    os.makedirs("test_samples", exist_ok=True)
    test_file = "test_samples/test_tone.wav"
    
    torchaudio.save(test_file, torch.tensor(wave).unsqueeze(0), sample_rate)
    print(f"✅ 创建测试音频: {test_file}")
    
    return [test_file]


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio推理测试")
    parser.add_argument("--model_path", type=str, 
                       help="模型路径 (默认自动查找)")
    parser.add_argument("--audio_files", nargs="+", 
                       help="测试音频文件列表")
    parser.add_argument("--output", type=str, default="inference_test_results.json",
                       help="结果输出文件")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"], help="推理设备")
    parser.add_argument("--tasks", nargs="+", 
                       choices=["asr", "understanding", "emotion", "translation"],
                       default=["asr", "understanding", "emotion", "translation"],
                       help="要测试的任务")
    parser.add_argument("--create_test_audio", action="store_true",
                       help="创建测试音频文件")
    
    args = parser.parse_args()
    
    # 查找模型路径
    if not args.model_path:
        model_paths = [
            "checkpoints/qwen2-audio-stage3",
            "checkpoints/qwen2-audio-stage2", 
            "checkpoints/qwen2-audio-stage1",
            "./qwen2-audio-7b"  # 预训练模型
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = path
                break
        
        if not model_path:
            print("❌ 错误: 未找到模型检查点")
            print("请指定 --model_path 或先运行训练脚本")
            return
    else:
        model_path = args.model_path
    
    # 查找音频文件
    if args.audio_files:
        audio_files = args.audio_files
    else:
        audio_files = find_test_audio_files()
        
    if not audio_files and not args.create_test_audio:
        print("⚠️  未找到测试音频文件")
        print("将创建测试音频文件...")
        audio_files = create_test_audio()
    elif args.create_test_audio:
        test_audio = create_test_audio()
        audio_files.extend(test_audio)
    
    if not audio_files:
        print("❌ 错误: 没有可用的测试音频文件")
        return
    
    print(f"📁 找到 {len(audio_files)} 个测试音频文件:")
    for f in audio_files:
        print(f"   - {f}")
    
    # 运行测试
    try:
        tester = Qwen2AudioTester(model_path, args.device)
        results = tester.run_comprehensive_test(audio_files, args.output)
        
        print("\n🎉 推理测试完成!")
        print(f"📊 测试结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 