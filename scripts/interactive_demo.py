#!/usr/bin/env python3
"""
LlamaAudio Interactive Demo
This script provides an interactive interface to test the trained LlamaAudio model.
"""

import os
import sys
import argparse
import torch
import gradio as gr
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json
import librosa
import tempfile
import warnings

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_audio.models.llama_audio_model import LlamaAudioModel
from llama_audio.training.data_loader import AudioDataset

warnings.filterwarnings("ignore")

class LlamaAudioDemo:
    """Interactive demo for LlamaAudio model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the demo with a trained model."""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.load_model()
        
        # Demo configuration
        self.max_audio_length = 30.0  # seconds
        self.sample_rate = 16000
        
    def _setup_device(self, device: str) -> str:
        """Setup device for inference."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the trained LlamaAudio model."""
        print(f"🔄 Loading LlamaAudio model from: {self.model_path}")
        
        try:
            # Load model configuration
            config_path = os.path.join(self.model_path, "config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    model_config = json.load(f)
            else:
                # Use default configuration
                model_config = {
                    "llama_model_name": "meta-llama/Llama-3.3-70B-Instruct",
                    "whisper_model_name": "openai/whisper-large-v3",
                    "use_lora": True,
                    "lora_rank": 64,
                    "lora_alpha": 16,
                    "lora_dropout": 0.05
                }
            
            # Initialize model
            self.model = LlamaAudioModel(
                llama_model_name=model_config.get("llama_model_name", "meta-llama/Llama-3.3-70B-Instruct"),
                whisper_model_name=model_config.get("whisper_model_name", "openai/whisper-large-v3"),
                use_lora=model_config.get("use_lora", True),
                lora_rank=model_config.get("lora_rank", 64),
                lora_alpha=model_config.get("lora_alpha", 16),
                lora_dropout=model_config.get("lora_dropout", 0.05)
            )
            
            # Load model weights
            model_file = os.path.join(self.model_path, "pytorch_model.bin")
            if not os.path.exists(model_file):
                model_file = os.path.join(self.model_path, "model.safetensors")
            
            if os.path.exists(model_file):
                self.model.load_state_dict(torch.load(model_file, map_location=self.device))
            else:
                print("⚠️  Model weights not found, using base model")
            
            self.model.to(self.device)
            self.model.eval()
            print("✅ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            sys.exit(1)
    
    def preprocess_audio(self, audio_path: str) -> np.ndarray:
        """Preprocess audio file for the model."""
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Trim silence
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Limit length
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio
            
        except Exception as e:
            raise ValueError(f"Error processing audio: {e}")
    
    def generate_response(self, audio_path: str, instruction: str, max_length: int = 512) -> str:
        """Generate response for audio input with instruction."""
        try:
            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            
            # Prepare input
            with torch.no_grad():
                # Convert audio to tensor
                audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(self.device)
                
                # Generate response
                response = self.model.generate(
                    audio=audio_tensor,
                    instruction=instruction,
                    max_length=max_length,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.9
                )
                
                return response[0] if isinstance(response, list) else response
                
        except Exception as e:
            return f"❌ Error generating response: {e}"
    
    def create_gradio_interface(self) -> gr.Interface:
        """Create Gradio interface for the demo."""
        
        def audio_chat(audio_file, instruction, max_length):
            if audio_file is None:
                return "请先上传音频文件。"
            
            if not instruction.strip():
                instruction = "请描述这段音频的内容。"
            
            try:
                response = self.generate_response(audio_file, instruction, max_length)
                return response
            except Exception as e:
                return f"处理出错: {e}"
        
        def load_example_audio(example_name):
            """Load example audio files."""
            examples_dir = project_root / "examples" / "audio"
            example_path = examples_dir / f"{example_name}.wav"
            
            if example_path.exists():
                return str(example_path)
            return None
        
        # Gradio interface
        with gr.Blocks(title="LlamaAudio Demo", theme=gr.themes.Soft()) as interface:
            gr.Markdown("""
            # 🎵 LlamaAudio Interactive Demo
            
            Welcome to the LlamaAudio demo! This multimodal AI model can understand and discuss audio content.
            Upload an audio file and provide instructions to interact with the model.
            
            ## 🎯 What can LlamaAudio do?
            - **Audio Description**: Describe what you hear in the audio
            - **Content Analysis**: Analyze music, speech, or environmental sounds  
            - **Q&A**: Answer questions about audio content
            - **Transcription**: Convert speech to text
            - **Conversation**: Have multi-turn conversations about audio
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Audio input
                    audio_input = gr.Audio(
                        label="📤 Upload Audio File",
                        type="filepath",
                        format="wav"
                    )
                    
                    # Instruction input
                    instruction_input = gr.Textbox(
                        label="💬 Instruction (指令)",
                        placeholder="请描述这段音频的内容... (或其他指令)",
                        lines=2,
                        value="请描述这段音频的内容。"
                    )
                    
                    # Generation parameters
                    max_length_slider = gr.Slider(
                        minimum=50,
                        maximum=1024,
                        value=512,
                        step=50,
                        label="📏 Max Response Length"
                    )
                    
                    # Submit button
                    submit_btn = gr.Button("🚀 Generate Response", variant="primary")
                
                with gr.Column(scale=1):
                    # Output
                    output_text = gr.Textbox(
                        label="🤖 LlamaAudio Response",
                        lines=10,
                        max_lines=20
                    )
            
            # Example instructions
            gr.Markdown("### 💡 Example Instructions")
            example_instructions = [
                "请描述这段音频的内容。",
                "这段音频中有什么乐器？",
                "这段音频的情感色彩如何？",
                "转录这段音频中的语音内容。",
                "这段音频适合在什么场景播放？",
                "分析这段音频的音质特点。",
                "这段音频给你什么感受？"
            ]
            
            example_buttons = []
            with gr.Row():
                for instruction in example_instructions[:4]:
                    btn = gr.Button(instruction, size="sm")
                    example_buttons.append(btn)
            
            with gr.Row():
                for instruction in example_instructions[4:]:
                    btn = gr.Button(instruction, size="sm")
                    example_buttons.append(btn)
            
            # Set up event handlers
            submit_btn.click(
                fn=audio_chat,
                inputs=[audio_input, instruction_input, max_length_slider],
                outputs=output_text
            )
            
            # Example instruction buttons
            for btn, instruction in zip(example_buttons, example_instructions):
                btn.click(
                    fn=lambda x: x,
                    inputs=gr.State(instruction),
                    outputs=instruction_input
                )
            
            # Model information
            gr.Markdown(f"""
            ### 🔧 Model Information
            - **Model Path**: `{self.model_path}`
            - **Device**: `{self.device}`
            - **Max Audio Length**: `{self.max_audio_length}s`
            - **Sample Rate**: `{self.sample_rate}Hz`
            
            ### 📝 Usage Tips
            1. **Audio Format**: Supports most audio formats (WAV, MP3, FLAC, etc.)
            2. **Audio Length**: Keep audio under 30 seconds for best performance
            3. **Instructions**: Be specific about what you want to know about the audio
            4. **Language**: Supports both English and Chinese instructions
            """)
        
        return interface

def main():
    parser = argparse.ArgumentParser(description="LlamaAudio Interactive Demo")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained LlamaAudio model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run the Gradio interface on"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public link for the demo"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    print("🚀 Starting LlamaAudio Interactive Demo...")
    print(f"📂 Model Path: {args.model_path}")
    print(f"🖥️  Device: {args.device}")
    print(f"🌐 Port: {args.port}")
    
    # Initialize demo
    demo = LlamaAudioDemo(
        model_path=args.model_path,
        device=args.device
    )
    
    # Create and launch interface
    interface = demo.create_gradio_interface()
    
    print("\n🎉 Demo ready! Open your browser and start chatting with LlamaAudio!")
    print(f"🔗 Local URL: http://localhost:{args.port}")
    
    if args.share:
        print("🌐 Creating public link...")
    
    interface.launch(
        server_port=args.port,
        share=args.share,
        debug=args.debug,
        show_error=True
    )

if __name__ == "__main__":
    main() 