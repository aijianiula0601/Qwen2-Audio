#!/usr/bin/env python3
import torch
import argparse
import os
import sys
import librosa
import soundfile as sf
from transformers import WhisperFeatureExtractor, AutoTokenizer
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.model import Qwen2AudioModel, Qwen2AudioConfig


class Qwen2AudioInference:
    """
    Inference class for Qwen2-Audio model
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_path = model_path
        
        print(f"Loading model from: {model_path}")
        print(f"Using device: {self.device}")
        
        # Load model and tokenizer
        self.model, self.tokenizer, self.feature_extractor = self._load_model()
        
    def _load_model(self):
        """Load model, tokenizer, and feature extractor"""
        # Try to load from saved model directory
        if os.path.exists(os.path.join(self.model_path, "config.json")):
            # Load from HuggingFace format
            config = Qwen2AudioConfig.from_pretrained(self.model_path)
            model = Qwen2AudioModel.from_pretrained(self.model_path, config=config)
            tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            feature_extractor = WhisperFeatureExtractor.from_pretrained(
                config.audio_encoder_name if hasattr(config, 'audio_encoder_name') else "openai/whisper-large-v3"
            )
        else:
            # Load from checkpoint file
            print("Loading from checkpoint file...")
            # This would need to be implemented based on how checkpoints are saved
            raise NotImplementedError("Checkpoint loading not yet implemented")
        
        model.to(self.device)
        model.eval()
        
        return model, tokenizer, feature_extractor
    
    def load_audio(self, audio_path: str, sample_rate: int = 16000) -> torch.Tensor:
        """Load and preprocess audio file"""
        audio, sr = librosa.load(audio_path, sr=sample_rate)
        
        # Process with feature extractor
        audio_features = self.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt"
        )
        
        return audio_features['input_features'].to(self.device)
    
    def transcribe(self, audio_path: str, max_new_tokens: int = 100) -> str:
        """Transcribe audio to text"""
        # Load audio
        audio_features = self.load_audio(audio_path)
        
        # Prepare input prompt for transcription
        prompt = f"{self.model.config.audio_start_token}{self.model.config.audio_end_token} "
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                audio_values=audio_features,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Remove the prompt part
        response = response.replace(prompt.strip(), "").strip()
        
        return response
    
    def chat(self, audio_path: str, instruction: str, max_new_tokens: int = 100) -> str:
        """Chat with audio input"""
        # Load audio
        audio_features = self.load_audio(audio_path)
        
        # Prepare input prompt for chat
        prompt = f"User: {instruction}\n{self.model.config.audio_start_token}{self.model.config.audio_end_token}\nAssistant: "
        input_ids = self.tokenizer(prompt, return_tensors="pt")['input_ids'].to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids,
                audio_values=audio_features,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract assistant response
        if "Assistant: " in response:
            response = response.split("Assistant: ")[-1].strip()
        
        return response
    
    def analyze_audio(self, audio_path: str, max_new_tokens: int = 150) -> str:
        """Analyze audio content"""
        return self.chat(
            audio_path,
            "Please analyze this audio and describe what you hear.",
            max_new_tokens
        )


def main():
    parser = argparse.ArgumentParser(description="Qwen2-Audio Inference Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--audio_path", type=str, help="Path to audio file for inference")
    parser.add_argument("--mode", type=str, choices=["transcribe", "chat", "analyze"], default="transcribe",
                        help="Inference mode")
    parser.add_argument("--instruction", type=str, help="Instruction for chat mode")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (auto, cpu, cuda)")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = Qwen2AudioInference(args.model_path, args.device)
    
    if args.interactive:
        print("=== Qwen2-Audio Interactive Demo ===")
        print("Commands:")
        print("  transcribe <audio_path>")
        print("  chat <audio_path> <instruction>")
        print("  analyze <audio_path>")
        print("  quit")
        print()
        
        while True:
            try:
                command = input(">> ").strip().split()
                if not command:
                    continue
                    
                if command[0] == "quit":
                    break
                elif command[0] == "transcribe" and len(command) >= 2:
                    audio_path = command[1]
                    result = inference.transcribe(audio_path, args.max_tokens)
                    print(f"Transcription: {result}")
                elif command[0] == "chat" and len(command) >= 3:
                    audio_path = command[1]
                    instruction = " ".join(command[2:])
                    result = inference.chat(audio_path, instruction, args.max_tokens)
                    print(f"Response: {result}")
                elif command[0] == "analyze" and len(command) >= 2:
                    audio_path = command[1]
                    result = inference.analyze_audio(audio_path, args.max_tokens)
                    print(f"Analysis: {result}")
                else:
                    print("Invalid command. Use: transcribe/chat/analyze <audio_path> [instruction]")
                    
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    elif args.audio_path:
        print(f"Processing audio: {args.audio_path}")
        
        if args.mode == "transcribe":
            result = inference.transcribe(args.audio_path, args.max_tokens)
            print(f"Transcription: {result}")
        elif args.mode == "chat":
            instruction = args.instruction or "Please describe what you hear in this audio."
            result = inference.chat(args.audio_path, instruction, args.max_tokens)
            print(f"Response: {result}")
        elif args.mode == "analyze":
            result = inference.analyze_audio(args.audio_path, args.max_tokens)
            print(f"Analysis: {result}")
    
    else:
        print("Please provide --audio_path or use --interactive mode")


if __name__ == "__main__":
    main() 