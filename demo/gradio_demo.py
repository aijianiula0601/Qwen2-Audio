#!/usr/bin/env python3
"""
Qwen2-Audio Gradio 演示界面
提供Web界面测试训练好的模型
"""

import os
import sys
import gradio as gr
import torch
import torchaudio
import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor
except ImportError:
    print("❌ 错误: 无法导入Qwen2Audio模块")
    sys.exit(1)


class Qwen2AudioDemo:
    def __init__(self, model_path: str = None, device: str = "auto"):
        """初始化演示应用"""
        self.model_path = self._find_model_path(model_path)
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        
        # 预设提示词
        self.prompts = {
            "asr": [
                "请转录这段音频",
                "将这段语音转换为文字", 
                "请识别音频中的语音内容",
                "听录音并写出对应的文本"
            ],
            "understanding": [
                "请描述这段音频的内容",
                "这段音频中发生了什么？",
                "分析并描述音频中的声音",
                "请总结音频的主要信息"
            ],
            "emotion": [
                "请识别这段音频中的情感",
                "分析说话者的情感状态",
                "这段语音表达了什么情感？",
                "判断音频中的情绪倾向"
            ],
            "translation": [
                "请将这段英文音频翻译为中文",
                "请将这段中文音频翻译为英文", 
                "翻译这段语音的内容",
                "请提供这段音频的翻译"
            ]
        }
        
    def _find_model_path(self, model_path: str) -> str:
        """查找模型路径"""
        if model_path and os.path.exists(model_path):
            return model_path
        
        # 自动查找
        candidate_paths = [
            "checkpoints/qwen2-audio-stage3",
            "checkpoints/qwen2-audio-stage2",
            "checkpoints/qwen2-audio-stage1",
            "../checkpoints/qwen2-audio-stage3",
            "../checkpoints/qwen2-audio-stage2", 
            "../checkpoints/qwen2-audio-stage1"
        ]
        
        for path in candidate_paths:
            if os.path.exists(path):
                return path
        
        raise ValueError("未找到模型检查点，请指定正确的模型路径")
    
    def _setup_device(self, device: str) -> str:
        """设置计算设备"""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self) -> str:
        """加载模型"""
        try:
            print(f"🔄 加载模型: {self.model_path}")
            
            # 加载处理器
            self.processor = Qwen2AudioProcessor.from_pretrained(self.model_path)
            
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
            
            status = f"✅ 模型加载成功!\n"
            status += f"📍 模型路径: {self.model_path}\n"
            status += f"🔧 设备: {self.device}\n"
            
            if self.device == "cuda":
                status += f"💾 GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
            
            return status
            
        except Exception as e:
            error_msg = f"❌ 模型加载失败: {str(e)}"
            print(error_msg)
            return error_msg
    
    def generate_response(self, audio_file, prompt: str, max_length: int = 512, 
                         temperature: float = 0.7, do_sample: bool = True) -> Tuple[str, float]:
        """生成模型回复"""
        if self.model is None or self.processor is None:
            return "❌ 模型未加载，请先加载模型", 0.0
        
        if audio_file is None:
            return "❌ 请上传音频文件", 0.0
        
        try:
            start_time = time.time()
            
            # 处理输入
            inputs = self.processor(
                audio_path=audio_file,
                text=prompt,
                return_tensors="pt"
            )
            
            if self.device == "cuda":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 生成参数
            generation_kwargs = {
                "max_length": max_length,
                "temperature": temperature,
                "do_sample": do_sample,
                "top_p": 0.9,
                "pad_token_id": self.processor.tokenizer.eos_token_id
            }
            
            # 生成回复
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            
            # 解码输出
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            inference_time = time.time() - start_time
            
            return response, inference_time
            
        except Exception as e:
            return f"❌ 生成失败: {str(e)}", 0.0
    
    def process_audio_task(self, audio_file, task_type: str, custom_prompt: str = "", 
                          max_length: int = 512, temperature: float = 0.7) -> Tuple[str, str]:
        """处理音频任务"""
        if task_type == "custom" and custom_prompt.strip():
            prompt = custom_prompt.strip()
        elif task_type in self.prompts:
            prompt = self.prompts[task_type][0]  # 使用第一个预设提示
        else:
            return "❌ 请选择任务类型或输入自定义提示", ""
        
        response, inference_time = self.generate_response(
            audio_file, prompt, max_length, temperature
        )
        
        # 格式化输出
        result = f"💬 提示: {prompt}\n\n"
        result += f"🤖 模型回复: {response}\n\n"
        result += f"⏱️ 推理耗时: {inference_time:.2f}秒"
        
        # 返回音频信息
        audio_info = ""
        if audio_file:
            try:
                # 获取音频信息
                waveform, sample_rate = torchaudio.load(audio_file)
                duration = waveform.shape[1] / sample_rate
                audio_info = f"📊 音频信息:\n"
                audio_info += f"   时长: {duration:.2f}秒\n"
                audio_info += f"   采样率: {sample_rate}Hz\n"
                audio_info += f"   通道数: {waveform.shape[0]}"
            except:
                audio_info = "📊 无法获取音频信息"
        
        return result, audio_info
    
    def batch_test(self, audio_files: List, task_type: str) -> str:
        """批量测试"""
        if not audio_files:
            return "❌ 请上传音频文件"
        
        if task_type not in self.prompts:
            return "❌ 请选择有效的任务类型"
        
        results = []
        total_time = 0
        
        for i, audio_file in enumerate(audio_files):
            if audio_file is None:
                continue
                
            prompt = self.prompts[task_type][0]
            response, inference_time = self.generate_response(
                audio_file, prompt, max_length=256, temperature=0.5
            )
            
            total_time += inference_time
            
            result_text = f"📁 文件 {i+1}: {os.path.basename(audio_file)}\n"
            result_text += f"💬 提示: {prompt}\n"
            result_text += f"🤖 回复: {response}\n"
            result_text += f"⏱️ 耗时: {inference_time:.2f}秒\n"
            result_text += "-" * 50
            
            results.append(result_text)
        
        # 汇总结果
        summary = f"📊 批量测试完成!\n"
        summary += f"   测试文件数: {len(audio_files)}\n"
        summary += f"   总耗时: {total_time:.2f}秒\n"
        summary += f"   平均耗时: {total_time/len(audio_files):.2f}秒/文件\n\n"
        
        return summary + "\n\n".join(results)
    
    def create_interface(self):
        """创建Gradio界面"""
        with gr.Blocks(title="Qwen2-Audio 演示", theme=gr.themes.Soft()) as interface:
            gr.Markdown("# 🎵 Qwen2-Audio 演示界面")
            gr.Markdown("上传音频文件，选择任务类型，体验多模态音频理解能力")
            
            # 模型状态显示
            model_status = gr.Textbox(
                label="🔧 模型状态",
                value="🔄 点击下方按钮加载模型...",
                interactive=False,
                lines=4
            )
            
            load_btn = gr.Button("🚀 加载模型", variant="primary")
            load_btn.click(
                fn=self.load_model,
                outputs=[model_status]
            )
            
            with gr.Tab("🎯 单文件测试"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 音频输入
                        audio_input = gr.Audio(
                            label="📤 上传音频文件",
                            type="filepath"
                        )
                        
                        # 任务选择
                        task_type = gr.Radio(
                            choices=[
                                ("🎤 语音识别 (ASR)", "asr"),
                                ("🧠 音频理解", "understanding"), 
                                ("😊 情感识别", "emotion"),
                                ("🌍 语音翻译", "translation"),
                                ("✍️ 自定义提示", "custom")
                            ],
                            label="📋 选择任务类型",
                            value="asr"
                        )
                        
                        # 自定义提示
                        custom_prompt = gr.Textbox(
                            label="✍️ 自定义提示词",
                            placeholder="选择'自定义提示'时，在此输入您的提示词...",
                            lines=2
                        )
                        
                        # 生成参数
                        with gr.Accordion("⚙️ 生成参数", open=False):
                            max_length = gr.Slider(
                                minimum=64, maximum=1024, value=512, step=32,
                                label="最大长度"
                            )
                            temperature = gr.Slider(
                                minimum=0.1, maximum=1.0, value=0.7, step=0.1,
                                label="温度"
                            )
                        
                        # 处理按钮
                        process_btn = gr.Button("🚀 开始处理", variant="primary")
                    
                    with gr.Column(scale=1):
                        # 结果显示
                        result_output = gr.Textbox(
                            label="📄 处理结果",
                            lines=15,
                            interactive=False
                        )
                        
                        # 音频信息
                        audio_info = gr.Textbox(
                            label="📊 音频信息", 
                            lines=4,
                            interactive=False
                        )
                
                # 处理事件
                process_btn.click(
                    fn=self.process_audio_task,
                    inputs=[audio_input, task_type, custom_prompt, max_length, temperature],
                    outputs=[result_output, audio_info]
                )
            
            with gr.Tab("📦 批量测试"):
                with gr.Row():
                    with gr.Column(scale=1):
                        # 批量音频输入
                        batch_audio = gr.File(
                            label="📤 上传多个音频文件",
                            file_count="multiple",
                            file_types=["audio"]
                        )
                        
                        # 批量任务选择
                        batch_task = gr.Radio(
                            choices=[
                                ("🎤 语音识别", "asr"),
                                ("🧠 音频理解", "understanding"),
                                ("😊 情感识别", "emotion"),
                                ("🌍 语音翻译", "translation")
                            ],
                            label="📋 批量任务类型",
                            value="asr"
                        )
                        
                        batch_btn = gr.Button("🚀 开始批量处理", variant="primary")
                    
                    with gr.Column(scale=1):
                        # 批量结果
                        batch_results = gr.Textbox(
                            label="📄 批量处理结果",
                            lines=20,
                            interactive=False
                        )
                
                # 批量处理事件
                batch_btn.click(
                    fn=self.batch_test,
                    inputs=[batch_audio, batch_task],
                    outputs=[batch_results]
                )
            
            with gr.Tab("🔧 预设提示词"):
                gr.Markdown("## 📝 各任务类型的预设提示词")
                
                for task, prompts in self.prompts.items():
                    task_names = {
                        "asr": "🎤 语音识别 (ASR)",
                        "understanding": "🧠 音频理解", 
                        "emotion": "😊 情感识别",
                        "translation": "🌍 语音翻译"
                    }
                    
                    with gr.Accordion(f"{task_names.get(task, task)}", open=False):
                        for i, prompt in enumerate(prompts):
                            gr.Textbox(
                                label=f"提示词 {i+1}",
                                value=prompt,
                                interactive=False
                            )
            
            with gr.Tab("ℹ️ 使用说明"):
                gr.Markdown("""
                ## 📖 使用指南
                
                ### 🚀 快速开始
                1. **加载模型**: 点击"加载模型"按钮
                2. **上传音频**: 支持 WAV, MP3, FLAC 等格式
                3. **选择任务**: 选择合适的任务类型
                4. **开始处理**: 点击"开始处理"获取结果
                
                ### 🎯 任务类型说明
                - **🎤 语音识别**: 将语音转换为文字
                - **🧠 音频理解**: 描述音频内容和场景
                - **😊 情感识别**: 识别说话者的情感状态
                - **🌍 语音翻译**: 跨语言语音翻译
                - **✍️ 自定义**: 使用您自己的提示词
                
                ### ⚙️ 参数调节
                - **最大长度**: 控制生成文本的最大长度
                - **温度**: 控制生成的随机性 (0.1=确定性, 1.0=创造性)
                
                ### 📦 批量处理
                - 支持同时处理多个音频文件
                - 适合批量测试和评估
                
                ### 💡 使用技巧
                - 音频质量越好，识别效果越佳
                - 建议音频长度在30秒以内
                - 清晰的语音效果最佳
                """)
        
        return interface


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Qwen2-Audio Gradio演示")
    parser.add_argument("--model_path", type=str, help="模型路径")
    parser.add_argument("--device", type=str, default="auto", 
                       choices=["auto", "cuda", "cpu"], help="计算设备")
    parser.add_argument("--port", type=int, default=7860, help="端口号")
    parser.add_argument("--share", action="store_true", help="创建公开链接")
    
    args = parser.parse_args()
    
    try:
        # 创建演示应用
        demo_app = Qwen2AudioDemo(args.model_path, args.device)
        interface = demo_app.create_interface()
        
        # 启动服务
        print(f"🚀 启动Gradio演示界面...")
        print(f"🔧 设备: {demo_app.device}")
        print(f"📍 模型路径: {demo_app.model_path}")
        print(f"🌐 端口: {args.port}")
        
        interface.launch(
            server_port=args.port,
            share=args.share,
            inbrowser=True
        )
        
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 