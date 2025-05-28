#!/usr/bin/env python3
"""
Qwen2-Audio 性能基准测试脚本
测试模型在标准数据集上的性能指标
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
    from test_inference import Qwen2AudioTester
except ImportError:
    print("❌ 错误: 无法导入必要模块")
    print("请确保在项目根目录运行此脚本")
    sys.exit(1)

# 评估指标计算
try:
    import jiwer  # 用于WER计算
    from rouge_score import rouge_scorer  # 用于ROUGE计算
    from sentence_transformers import SentenceTransformer  # 用于语义相似度
    EVALUATION_DEPS_AVAILABLE = True
except ImportError:
    print("⚠️  警告: 评估依赖未安装，将跳过详细指标计算")
    print("安装命令: pip install jiwer rouge-score sentence-transformers")
    EVALUATION_DEPS_AVAILABLE = False


class PerformanceBenchmark:
    def __init__(self, model_path: str, device: str = "auto"):
        """初始化性能基准测试"""
        self.tester = Qwen2AudioTester(model_path, device)
        self.device = self.tester.device
        
        # 初始化评估器
        if EVALUATION_DEPS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
            try:
                self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            except:
                self.sentence_model = None
                print("⚠️  无法加载语义相似度模型")
        else:
            self.rouge_scorer = None
            self.sentence_model = None
    
    def calculate_wer(self, reference: str, hypothesis: str) -> float:
        """计算词错误率 (WER)"""
        if not EVALUATION_DEPS_AVAILABLE:
            return 0.0
        
        try:
            return jiwer.wer(reference.lower(), hypothesis.lower())
        except:
            return 1.0
    
    def calculate_rouge(self, reference: str, hypothesis: str) -> Dict[str, float]:
        """计算ROUGE分数"""
        if not self.rouge_scorer:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
        
        try:
            scores = self.rouge_scorer.score(reference, hypothesis)
            return {
                "rouge1": scores['rouge1'].fmeasure,
                "rouge2": scores['rouge2'].fmeasure,
                "rougeL": scores['rougeL'].fmeasure
            }
        except:
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def calculate_semantic_similarity(self, reference: str, hypothesis: str) -> float:
        """计算语义相似度"""
        if not self.sentence_model:
            return 0.0
        
        try:
            embeddings = self.sentence_model.encode([reference, hypothesis])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return float(similarity)
        except:
            return 0.0
    
    def benchmark_asr(self, test_data: List[Dict]) -> Dict:
        """ASR任务基准测试"""
        print("\n🎯 ASR任务基准测试")
        
        results = []
        total_wer = 0.0
        total_time = 0.0
        
        for i, item in enumerate(test_data):
            audio_path = item["audio_path"]
            reference_text = item["text"]
            
            if not os.path.exists(audio_path):
                continue
            
            print(f"📁 测试 {i+1}/{len(test_data)}: {os.path.basename(audio_path)}")
            
            # 生成转录
            response, inference_time = self.tester.generate_response(
                audio_path, 
                "请转录这段音频",
                max_length=256,
                temperature=0.1,
                do_sample=False
            )
            
            # 计算指标
            wer = self.calculate_wer(reference_text, response)
            rouge_scores = self.calculate_rouge(reference_text, response)
            semantic_sim = self.calculate_semantic_similarity(reference_text, response)
            
            result = {
                "audio_file": audio_path,
                "reference": reference_text,
                "hypothesis": response,
                "wer": wer,
                "rouge_scores": rouge_scores,
                "semantic_similarity": semantic_sim,
                "inference_time": inference_time
            }
            
            results.append(result)
            total_wer += wer
            total_time += inference_time
            
            print(f"   WER: {wer:.3f}")
            print(f"   ROUGE-L: {rouge_scores['rougeL']:.3f}")
            print(f"   推理时间: {inference_time:.2f}s")
        
        # 计算平均指标
        avg_metrics = {
            "average_wer": total_wer / len(results) if results else 1.0,
            "average_rouge1": np.mean([r["rouge_scores"]["rouge1"] for r in results]),
            "average_rouge2": np.mean([r["rouge_scores"]["rouge2"] for r in results]),
            "average_rougeL": np.mean([r["rouge_scores"]["rougeL"] for r in results]),
            "average_semantic_similarity": np.mean([r["semantic_similarity"] for r in results]),
            "average_inference_time": total_time / len(results) if results else 0.0,
            "total_samples": len(results)
        }
        
        return {
            "task": "asr_benchmark",
            "metrics": avg_metrics,
            "detailed_results": results
        }
    
    def benchmark_audio_understanding(self, test_data: List[Dict]) -> Dict:
        """音频理解任务基准测试"""
        print("\n🎯 音频理解任务基准测试")
        
        results = []
        total_time = 0.0
        
        for i, item in enumerate(test_data):
            audio_path = item["audio_path"]
            reference_description = item["description"]
            
            if not os.path.exists(audio_path):
                continue
            
            print(f"📁 测试 {i+1}/{len(test_data)}: {os.path.basename(audio_path)}")
            
            # 生成描述
            response, inference_time = self.tester.generate_response(
                audio_path,
                "请描述这段音频的内容",
                max_length=512,
                temperature=0.7
            )
            
            # 计算指标
            rouge_scores = self.calculate_rouge(reference_description, response)
            semantic_sim = self.calculate_semantic_similarity(reference_description, response)
            
            result = {
                "audio_file": audio_path,
                "reference": reference_description,
                "hypothesis": response,
                "rouge_scores": rouge_scores,
                "semantic_similarity": semantic_sim,
                "inference_time": inference_time
            }
            
            results.append(result)
            total_time += inference_time
            
            print(f"   ROUGE-L: {rouge_scores['rougeL']:.3f}")
            print(f"   语义相似度: {semantic_sim:.3f}")
            print(f"   推理时间: {inference_time:.2f}s")
        
        # 计算平均指标
        avg_metrics = {
            "average_rouge1": np.mean([r["rouge_scores"]["rouge1"] for r in results]),
            "average_rouge2": np.mean([r["rouge_scores"]["rouge2"] for r in results]),
            "average_rougeL": np.mean([r["rouge_scores"]["rougeL"] for r in results]),
            "average_semantic_similarity": np.mean([r["semantic_similarity"] for r in results]),
            "average_inference_time": total_time / len(results) if results else 0.0,
            "total_samples": len(results)
        }
        
        return {
            "task": "understanding_benchmark",
            "metrics": avg_metrics,
            "detailed_results": results
        }
    
    def benchmark_emotion_recognition(self, test_data: List[Dict]) -> Dict:
        """情感识别任务基准测试"""
        print("\n🎯 情感识别任务基准测试")
        
        results = []
        correct_predictions = 0
        total_time = 0.0
        
        # 定义情感标签映射
        emotion_labels = {
            "happy": ["开心", "快乐", "高兴", "愉快"],
            "sad": ["悲伤", "难过", "伤心", "沮丧"],
            "angry": ["愤怒", "生气", "恼怒", "火大"],
            "neutral": ["中性", "平静", "正常", "平常"],
            "surprise": ["惊讶", "惊奇", "意外", "震惊"],
            "fear": ["恐惧", "害怕", "担心", "紧张"]
        }
        
        for i, item in enumerate(test_data):
            audio_path = item["audio_path"]
            true_emotion = item["emotion"]
            
            if not os.path.exists(audio_path):
                continue
            
            print(f"📁 测试 {i+1}/{len(test_data)}: {os.path.basename(audio_path)}")
            
            # 生成情感识别
            response, inference_time = self.tester.generate_response(
                audio_path,
                "请识别这段音频中的情感",
                max_length=128,
                temperature=0.3
            )
            
            # 检查预测是否正确
            predicted_correct = False
            if true_emotion in emotion_labels:
                for label in emotion_labels[true_emotion]:
                    if label in response.lower():
                        predicted_correct = True
                        break
            
            if predicted_correct:
                correct_predictions += 1
            
            result = {
                "audio_file": audio_path,
                "true_emotion": true_emotion,
                "predicted_response": response,
                "correct": predicted_correct,
                "inference_time": inference_time
            }
            
            results.append(result)
            total_time += inference_time
            
            print(f"   真实情感: {true_emotion}")
            print(f"   预测结果: {response}")
            print(f"   正确性: {'✅' if predicted_correct else '❌'}")
            print(f"   推理时间: {inference_time:.2f}s")
        
        # 计算准确率
        accuracy = correct_predictions / len(results) if results else 0.0
        
        avg_metrics = {
            "accuracy": accuracy,
            "correct_predictions": correct_predictions,
            "total_samples": len(results),
            "average_inference_time": total_time / len(results) if results else 0.0
        }
        
        return {
            "task": "emotion_benchmark",
            "metrics": avg_metrics,
            "detailed_results": results
        }
    
    def memory_usage_benchmark(self) -> Dict:
        """内存使用基准测试"""
        print("\n🎯 内存使用基准测试")
        
        if self.device == "cuda":
            # GPU内存使用
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
            
            # 创建测试音频
            test_audio = torch.randn(1, 16000 * 10)  # 10秒音频
            test_audio_path = "temp_test_audio.wav"
            torchaudio.save(test_audio_path, test_audio, 16000)
            
            try:
                # 测试推理内存使用
                response, inference_time = self.tester.generate_response(
                    test_audio_path,
                    "请转录这段音频",
                    max_length=256
                )
                
                peak_memory = torch.cuda.max_memory_allocated()
                inference_memory = peak_memory - initial_memory
                
                # 清理
                torch.cuda.empty_cache()
                os.remove(test_audio_path)
                
                return {
                    "device": "cuda",
                    "initial_memory_mb": initial_memory / 1e6,
                    "peak_memory_mb": peak_memory / 1e6,
                    "inference_memory_mb": inference_memory / 1e6,
                    "inference_time": inference_time
                }
                
            except Exception as e:
                os.remove(test_audio_path) if os.path.exists(test_audio_path) else None
                return {"device": "cuda", "error": str(e)}
                
        else:
            # CPU内存使用 (简化版)
            import psutil
            import os
            
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss
            
            # 创建测试音频
            test_audio = torch.randn(1, 16000 * 5)  # 5秒音频
            test_audio_path = "temp_test_audio.wav"
            torchaudio.save(test_audio_path, test_audio, 16000)
            
            try:
                # 测试推理
                response, inference_time = self.tester.generate_response(
                    test_audio_path,
                    "请转录这段音频",
                    max_length=256
                )
                
                peak_memory = process.memory_info().rss
                inference_memory = peak_memory - initial_memory
                
                # 清理
                os.remove(test_audio_path)
                
                return {
                    "device": "cpu",
                    "initial_memory_mb": initial_memory / 1e6,
                    "peak_memory_mb": peak_memory / 1e6,
                    "inference_memory_mb": inference_memory / 1e6,
                    "inference_time": inference_time
                }
                
            except Exception as e:
                os.remove(test_audio_path) if os.path.exists(test_audio_path) else None
                return {"device": "cpu", "error": str(e)}
    
    def latency_benchmark(self, audio_lengths: List[int] = [1, 3, 5, 10, 15, 30]) -> Dict:
        """延迟基准测试"""
        print("\n🎯 延迟基准测试")
        
        results = []
        
        for length in audio_lengths:
            print(f"📏 测试音频长度: {length}秒")
            
            # 创建测试音频
            test_audio = torch.randn(1, 16000 * length)
            test_audio_path = f"temp_test_audio_{length}s.wav"
            torchaudio.save(test_audio_path, test_audio, 16000)
            
            try:
                # 多次测试取平均
                times = []
                for _ in range(3):
                    _, inference_time = self.tester.generate_response(
                        test_audio_path,
                        "请转录这段音频",
                        max_length=256,
                        temperature=0.1,
                        do_sample=False
                    )
                    times.append(inference_time)
                
                avg_time = np.mean(times)
                std_time = np.std(times)
                
                result = {
                    "audio_length_seconds": length,
                    "average_inference_time": avg_time,
                    "std_inference_time": std_time,
                    "real_time_factor": avg_time / length,  # 实时因子
                    "throughput_ratio": length / avg_time   # 吞吐率
                }
                
                results.append(result)
                
                print(f"   平均推理时间: {avg_time:.2f}s")
                print(f"   实时因子: {avg_time/length:.2f}x")
                print(f"   吞吐率: {length/avg_time:.2f}x")
                
                # 清理
                os.remove(test_audio_path)
                
            except Exception as e:
                print(f"   错误: {e}")
                os.remove(test_audio_path) if os.path.exists(test_audio_path) else None
        
        return {
            "task": "latency_benchmark",
            "results": results
        }
    
    def run_full_benchmark(self, test_data_config: Dict, output_file: str = None) -> Dict:
        """运行完整基准测试"""
        print("🚀 开始完整性能基准测试")
        
        benchmark_results = {
            "model_path": self.tester.model_path,
            "device": self.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_deps_available": EVALUATION_DEPS_AVAILABLE,
            "benchmarks": []
        }
        
        # 1. ASR基准测试
        if "asr" in test_data_config:
            try:
                asr_result = self.benchmark_asr(test_data_config["asr"])
                benchmark_results["benchmarks"].append(asr_result)
            except Exception as e:
                print(f"❌ ASR基准测试失败: {e}")
        
        # 2. 音频理解基准测试
        if "understanding" in test_data_config:
            try:
                understanding_result = self.benchmark_audio_understanding(test_data_config["understanding"])
                benchmark_results["benchmarks"].append(understanding_result)
            except Exception as e:
                print(f"❌ 音频理解基准测试失败: {e}")
        
        # 3. 情感识别基准测试
        if "emotion" in test_data_config:
            try:
                emotion_result = self.benchmark_emotion_recognition(test_data_config["emotion"])
                benchmark_results["benchmarks"].append(emotion_result)
            except Exception as e:
                print(f"❌ 情感识别基准测试失败: {e}")
        
        # 4. 内存使用基准测试
        try:
            memory_result = self.memory_usage_benchmark()
            benchmark_results["memory_benchmark"] = memory_result
        except Exception as e:
            print(f"❌ 内存基准测试失败: {e}")
        
        # 5. 延迟基准测试
        try:
            latency_result = self.latency_benchmark()
            benchmark_results["latency_benchmark"] = latency_result
        except Exception as e:
            print(f"❌ 延迟基准测试失败: {e}")
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(benchmark_results, f, ensure_ascii=False, indent=2)
            print(f"📝 基准测试结果已保存到: {output_file}")
        
        return benchmark_results


def create_sample_test_data():
    """创建示例测试数据"""
    print("🔧 创建示例测试数据...")
    
    os.makedirs("benchmark_data", exist_ok=True)
    
    # 创建示例音频文件和标注
    sample_data = {
        "asr": [],
        "understanding": [],
        "emotion": []
    }
    
    # 创建几个测试音频
    for i in range(3):
        # 创建不同频率的音频
        duration = 3
        sample_rate = 16000
        frequency = 440 * (i + 1)  # 不同频率
        
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = np.sin(2 * np.pi * frequency * t) * np.exp(-t / 2)
        
        audio_path = f"benchmark_data/test_audio_{i+1}.wav"
        torchaudio.save(audio_path, torch.tensor(wave).unsqueeze(0), sample_rate)
        
        # ASR数据
        sample_data["asr"].append({
            "audio_path": audio_path,
            "text": f"这是第{i+1}个测试音频，频率为{frequency}赫兹"
        })
        
        # 理解数据
        sample_data["understanding"].append({
            "audio_path": audio_path,
            "description": f"一段{frequency}赫兹的正弦波信号，持续{duration}秒，音量逐渐衰减"
        })
        
        # 情感数据
        emotions = ["happy", "neutral", "sad"]
        sample_data["emotion"].append({
            "audio_path": audio_path,
            "emotion": emotions[i]
        })
    
    # 保存测试数据配置
    with open("benchmark_data/test_config.json", 'w', encoding='utf-8') as f:
        json.dump(sample_data, f, ensure_ascii=False, indent=2)
    
    print("✅ 示例测试数据创建完成")
    return sample_data


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio性能基准测试")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--test_config", type=str, default="benchmark_data/test_config.json",
                       help="测试数据配置文件")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                       help="结果输出文件")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"], help="推理设备")
    parser.add_argument("--create_sample_data", action="store_true",
                       help="创建示例测试数据")
    
    args = parser.parse_args()
    
    # 创建示例数据
    if args.create_sample_data:
        test_data_config = create_sample_test_data()
    else:
        # 加载测试数据配置
        if os.path.exists(args.test_config):
            with open(args.test_config, 'r', encoding='utf-8') as f:
                test_data_config = json.load(f)
        else:
            print(f"⚠️  测试配置文件不存在: {args.test_config}")
            print("创建示例测试数据...")
            test_data_config = create_sample_test_data()
    
    # 查找模型路径
    if not args.model_path:
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
            print("❌ 错误: 未找到模型检查点")
            return
    else:
        model_path = args.model_path
    
    # 运行基准测试
    try:
        benchmark = PerformanceBenchmark(model_path, args.device)
        results = benchmark.run_full_benchmark(test_data_config, args.output)
        
        print("\n🎉 基准测试完成!")
        print(f"📊 详细结果已保存到: {args.output}")
        
    except Exception as e:
        print(f"❌ 基准测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 