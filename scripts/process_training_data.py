#!/usr/bin/env python3
"""
Qwen2-Audio Training Data Processing Script
将原始数据集转换为训练所需的JSONL格式

数据格式:
{
    "audio_path": "path/to/audio.wav",
    "instruction": "请转录这段音频",
    "input": "",
    "output": "转录文本内容",
    "task_type": "asr|translation|classification|generation",
    "language": "zh|en|fr|de|...",
    "conversation": [
        {"role": "user", "content": "用户输入"},
        {"role": "assistant", "content": "助手回复"}
    ]
}
"""

import os
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import librosa
import soundfile as sf
from datasets import load_from_disk, Dataset
from tqdm import tqdm
import pandas as pd
import multiprocessing as mp
from functools import partial


class DataProcessor:
    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw"
        self.processed_dir = self.data_root / "processed"
        self.processed_dir.mkdir(exist_ok=True)
        
        # 音频参数
        self.target_sr = 16000
        self.max_duration = 30.0  # 最大30秒
        
    def process_audio_file(self, audio_info: Tuple[Path, str, str, str]) -> Optional[Dict]:
        """处理单个音频文件
        
        Args:
            audio_info: (音频文件路径, 转录文本, 子集名称, 指令)
            
        Returns:
            处理后的数据样本或None（如果处理失败）
        """
        audio_file, transcription, subset, instruction = audio_info
        
        try:
            # 检查音频时长
            duration = librosa.get_duration(path=str(audio_file))
            if duration > self.max_duration:
                return None
                
            # 处理音频文件
            output_dir = self.processed_dir / "librispeech" / subset
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{audio_file.stem}.wav"
            
            # 如果输出文件已存在，跳过处理
            if output_path.exists():
                return None
                
            # 加载并重采样音频
            audio, sr = librosa.load(str(audio_file), sr=self.target_sr)
            
            # 保存处理后的音频
            sf.write(str(output_path), audio, self.target_sr)
            
            # 创建数据样本
            return {
                "audio_path": str(output_path.relative_to(self.data_root)),
                "instruction": instruction,
                "input": "",
                "output": transcription,
                "task_type": "asr",
                "language": "en"
            }
            
        except Exception as e:
            print(f"处理音频 {audio_file} 时出错: {e}")
            return None

    def process_librispeech(self) -> List[Dict]:
        """处理LibriSpeech ASR数据"""
        print("处理LibriSpeech数据...")
        data_samples = []
        
        librispeech_dir = self.raw_dir / "librispeech"
        if not librispeech_dir.exists():
            print("LibriSpeech目录不存在，跳过")
            
            return data_samples
            
        # 处理所有子集
        subsets = ["train-clean-100", "train-clean-360", "train-other-500"]
        instructions = [
            "请转录这段英文音频",
            "将这段音频转换为文字",
            "听录音并写出对应的文本",
            "请识别音频中的语音内容"
        ]
        
        # 收集所有需要处理的音频文件信息
        audio_files_to_process = []
        
        for subset in subsets:
            subset_dir = librispeech_dir / subset
            if not subset_dir.exists():
                print(f"子集 {subset} 不存在，跳过")
                continue
                
            print(f"收集子集 {subset} 的音频文件信息..., sub_dir:{subset_dir}")
            
            # 遍历所有说话人目录
            for speaker_dir in tqdm(list(subset_dir.iterdir()), desc=f"扫描 {subset} 说话人"):
                if not speaker_dir.is_dir():
                    continue
                
                # 遍历所有章节目录
                for chapter_dir in speaker_dir.iterdir():
                    if not chapter_dir.is_dir():
                        continue
                    # 读取转录文件
                    trans_files = list(chapter_dir.glob("*.trans.txt"))
                    if len(trans_files) != 1:
                        continue
                    trans_file = trans_files[0]
                    if not trans_file.exists():
                        continue
                        
                    # 解析转录文件
                    with open(trans_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            
                            parts = line.strip().split(' ', 1)
                            if len(parts) != 2:
                                continue
                                
                            audio_id, transcription = parts
                            audio_file = chapter_dir / f"{audio_id}.flac"
                            
                            if not audio_file.exists():
                                continue
                                
                            audio_files_to_process.append((
                                audio_file,
                                transcription,
                                subset,
                                random.choice(instructions)
                            ))
        
        print(f"共收集到 {len(audio_files_to_process)} 个音频文件待处理")
        
        # 使用多进程处理音频文件
        num_processes = max(1, mp.cpu_count() - 1)  # 保留一个CPU核心
        print(f"使用 {num_processes} 个进程进行处理")
        
        with mp.Pool(num_processes) as pool:
            # 使用tqdm显示进度
            results = list(tqdm(
                pool.imap(self.process_audio_file, audio_files_to_process),
                total=len(audio_files_to_process),
                desc="处理音频文件"
            ))
            
        # 收集处理结果
        data_samples = [r for r in results if r is not None]
        
        print(f"LibriSpeech处理完成，共处理 {len(data_samples)} 条数据")
        return data_samples
    
    def process_fleurs(self) -> List[Dict]:
        """处理FLEURS多语言ASR数据"""
        print("处理FLEURS数据...")
        data_samples = []
        
        fleurs_dir = self.raw_dir / "fleurs"
        if not fleurs_dir.exists():
            print("FLEURS目录不存在，跳过")
            return data_samples
            
        # 加载FLEURS数据集
        for lang in ["fleurs_zh", "fleurs_en"]:
            dataset_path = fleurs_dir / lang
            if dataset_path.exists():
                try:
                    dataset = load_from_disk(str(dataset_path))
                    
                    instructions = {
                        "zh": ["请转录这段中文音频", "将音频转换为中文文字"],
                        "en": ["Please transcribe this English audio", "Convert audio to English text"]
                    }
                    
                    lang_code = "zh" if "zh" in lang else "en"
                    
                    for i, item in enumerate(dataset["train"]):
                        if i >= 1000:  # 限制数量
                            break
                            
                        sample = {
                            "audio_path": f"data/processed/fleurs/{lang}/audio_{i:06d}.wav",
                            "instruction": random.choice(instructions[lang_code]),
                            "input": "",
                            "output": item["transcription"],
                            "task_type": "asr",
                            "language": lang_code
                        }
                        data_samples.append(sample)
                        
                except Exception as e:
                    print(f"处理{lang}时出错: {e}")
                    
        return data_samples
    
    def process_covost2(self) -> List[Dict]:
        """处理CoVoST2语音翻译数据"""
        print("处理CoVoST2数据...")
        data_samples = []
        
        covost2_dir = self.raw_dir / "covost2"
        if not covost2_dir.exists():
            print("CoVoST2目录不存在，跳过")
            return data_samples
            
        translation_instructions = {
            "en_zh": "请将这段英文音频翻译为中文",
            "zh_en": "请将这段中文音频翻译为英文",
            "en_de": "Please translate this English audio to German",
            "de_en": "Please translate this German audio to English"
        }
        
        # 处理各种语言方向
        for direction in ["en_zh", "zh_en", "en_de", "de_en"]:
            dataset_path = covost2_dir / f"covost2_{direction}"
            if dataset_path.exists():
                try:
                    dataset = load_from_disk(str(dataset_path))
                    src_lang, tgt_lang = direction.split("_")
                    
                    for i, item in enumerate(dataset):
                        if i >= 500:  # 限制数量
                            break
                            
                        sample = {
                            "audio_path": f"data/processed/covost2/{direction}/audio_{i:06d}.wav",
                            "instruction": translation_instructions.get(direction, f"Translate from {src_lang} to {tgt_lang}"),
                            "input": "",
                            "output": item["translation"],
                            "task_type": "translation",
                            "language": f"{src_lang}_{tgt_lang}"
                        }
                        data_samples.append(sample)
                        
                except Exception as e:
                    print(f"处理{direction}时出错: {e}")
                    
        return data_samples
    
    def process_meld(self) -> List[Dict]:
        """处理MELD情感识别数据"""
        print("处理MELD数据...")
        data_samples = []
        
        meld_dir = self.raw_dir / "meld"
        if not meld_dir.exists():
            print("MELD目录不存在，跳过")
            return data_samples
            
        dataset_path = meld_dir / "meld_raw"
        if dataset_path.exists():
            try:
                dataset = load_from_disk(str(dataset_path))
                
                emotion_instructions = [
                    "请识别这段音频中的情感",
                    "分析说话者的情感状态",
                    "这段语音表达了什么情感？"
                ]
                
                for i, item in enumerate(dataset["train"]):
                    if i >= 1000:
                        break
                        
                    sample = {
                        "audio_path": f"data/processed/meld/audio_{i:06d}.wav",
                        "instruction": random.choice(emotion_instructions),
                        "input": "",
                        "output": item["emotion"],
                        "task_type": "classification",
                        "language": "en"
                    }
                    data_samples.append(sample)
                    
            except Exception as e:
                print(f"处理MELD时出错: {e}")
                
        return data_samples
    
    def process_audiocaps(self) -> List[Dict]:
        """处理AudioCaps音频描述数据"""
        print("处理AudioCaps数据...")
        data_samples = []
        
        audiocaps_dir = self.raw_dir / "audiocaps"
        if not audiocaps_dir.exists():
            print("AudioCaps目录不存在，跳过")
            return data_samples
            
        dataset_path = audiocaps_dir / "audiocaps"
        if dataset_path.exists():
            try:
                dataset = load_from_disk(str(dataset_path))
                
                description_instructions = [
                    "请描述这段音频的内容",
                    "这段音频中发生了什么？",
                    "分析并描述音频中的声音"
                ]
                
                for i, item in enumerate(dataset["train"]):
                    if i >= 1000:
                        break
                        
                    sample = {
                        "audio_path": f"data/processed/audiocaps/audio_{i:06d}.wav",
                        "instruction": random.choice(description_instructions),
                        "input": "",
                        "output": item["caption"],
                        "task_type": "generation",
                        "language": "en"
                    }
                    data_samples.append(sample)
                    
            except Exception as e:
                print(f"处理AudioCaps时出错: {e}")
                
        return data_samples
    
    def process_musiccaps(self) -> List[Dict]:
        """处理MusicCaps音乐描述数据"""
        print("处理MusicCaps数据...")
        data_samples = []
        
        musiccaps_dir = self.raw_dir / "musiccaps"
        if not musiccaps_dir.exists():
            print("MusicCaps目录不存在，跳过")
            return data_samples
            
        dataset_path = musiccaps_dir / "musiccaps"
        if dataset_path.exists():
            try:
                dataset = load_from_disk(str(dataset_path))
                
                music_instructions = [
                    "请描述这段音乐",
                    "分析这首音乐的风格和特点",
                    "这是什么类型的音乐？"
                ]
                
                for i, item in enumerate(dataset["train"]):
                    if i >= 500:
                        break
                        
                    sample = {
                        "audio_path": f"data/processed/musiccaps/audio_{i:06d}.wav",
                        "instruction": random.choice(music_instructions),
                        "input": "",
                        "output": item["caption"],
                        "task_type": "generation",
                        "language": "en"
                    }
                    data_samples.append(sample)
                    
            except Exception as e:
                print(f"处理MusicCaps时出错: {e}")
                
        return data_samples
    
    def create_conversation_samples(self, samples: List[Dict]) -> List[Dict]:
        """将单轮样本转换为对话格式"""
        conversation_samples = []
        
        for sample in samples:
            conversation = [
                {
                    "role": "user",
                    "content": f"<|audio|>{sample['instruction']}"
                },
                {
                    "role": "assistant", 
                    "content": sample["output"]
                }
            ]
            
            conv_sample = {
                "audio_path": sample["audio_path"],
                "conversation": conversation,
                "task_type": sample["task_type"],
                "language": sample["language"]
            }
            conversation_samples.append(conv_sample)
            
        return conversation_samples
    
    def save_jsonl(self, data: List[Dict], output_path: str):
        """保存数据为JSONL格式"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
                
        print(f"已保存 {len(data)} 条数据到 {output_path}")
    
    def process_all_datasets(self):
        """处理所有数据集"""
        print("开始处理所有数据集...")
        
        # 第一阶段: 预训练数据 (主要是ASR任务)
        stage1_data = []
        stage1_data.extend(self.process_librispeech())
        stage1_data.extend(self.process_fleurs())
        stage1_data.extend(self.process_audiocaps())
        stage1_data.extend(self.process_musiccaps())
        
        # 随机打乱
        random.shuffle(stage1_data)
        self.save_jsonl(stage1_data, "data/stage1_pretraining/train.jsonl")
        
        # 第二阶段: 监督微调数据 (对话格式)
        stage2_data = []
        stage2_data.extend(self.process_covost2())
        stage2_data.extend(self.process_meld())
        
        # 转换为对话格式
        conversation_data = self.create_conversation_samples(stage2_data)
        random.shuffle(conversation_data)
        self.save_jsonl(conversation_data, "data/stage2_sft/train.jsonl")
        
        # 第三阶段: DPO数据 (需要人工标注的偏好数据)
        # 这里创建示例DPO数据格式
        dpo_samples = []
        for i in range(100):
            dpo_sample = {
                "audio_path": f"data/processed/dpo/audio_{i:06d}.wav",
                "prompt": "请描述这段音频",
                "chosen": f"这是一段清晰的语音描述 {i}",
                "rejected": f"这是一段模糊的描述 {i}",
                "task_type": "generation"
            }
            dpo_samples.append(dpo_sample)
            
        self.save_jsonl(dpo_samples, "data/stage3_dpo/train.jsonl")
        
        print("数据处理完成！")
        print(f"Stage 1 (预训练): {len(stage1_data)} 条样本")
        print(f"Stage 2 (SFT): {len(conversation_data)} 条样本")
        print(f"Stage 3 (DPO): {len(dpo_samples)} 条样本")


def main():
    parser = argparse.ArgumentParser(description="处理Qwen2-Audio训练数据")
    parser.add_argument("--data_root", default="data", help="数据根目录")
    parser.add_argument("--datasets", nargs="+", 
                       choices=["librispeech", "fleurs", "covost2", "meld", "audiocaps", "musiccaps", "all"],
                       default=["all"], help="要处理的数据集")
    
    args = parser.parse_args()
    
    processor = DataProcessor(args.data_root)
    
    if "all" in args.datasets:
        processor.process_all_datasets()
    else:
        # 处理指定数据集
        all_data = []
        for dataset in args.datasets:
            if dataset == "librispeech":
                all_data.extend(processor.process_librispeech())
            elif dataset == "fleurs":
                all_data.extend(processor.process_fleurs())
            elif dataset == "covost2":
                all_data.extend(processor.process_covost2())
            elif dataset == "meld":
                all_data.extend(processor.process_meld())
            elif dataset == "audiocaps":
                all_data.extend(processor.process_audiocaps())
            elif dataset == "musiccaps":
                all_data.extend(processor.process_musiccaps())
        
        processor.save_jsonl(all_data, f"data/processed/{'_'.join(args.datasets)}.jsonl")


if __name__ == "__main__":
    main() 