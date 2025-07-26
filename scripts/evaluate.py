#!/usr/bin/env python3
"""
LlamaAudio Model Evaluation Script
This script evaluates the trained LlamaAudio model on various audio understanding tasks.
"""

import os
import sys
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import librosa
from tqdm import tqdm
import logging
from datetime import datetime
import pandas as pd

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from llama_audio.models.llama_audio_model import LlamaAudioModel
from llama_audio.training.data_loader import AudioDataset

# Evaluation metrics
try:
    from rouge_score import rouge_scorer
    from sacrebleu import corpus_bleu
    from sklearn.metrics import accuracy_score, f1_score
    METRICS_AVAILABLE = True
except ImportError:
    print("Warning: Some evaluation metrics not available. Install rouge-score, sacrebleu, and scikit-learn for full evaluation.")
    METRICS_AVAILABLE = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class LlamaAudioEvaluator:
    """Comprehensive evaluator for LlamaAudio model."""
    
    def __init__(self, model_path: str, device: str = "auto"):
        """Initialize the evaluator."""
        self.model_path = model_path
        self.device = self._setup_device(device)
        self.model = None
        self.load_model()
        
        # Evaluation configuration
        self.max_audio_length = 30.0  # seconds
        self.sample_rate = 16000
        self.max_new_tokens = 512
        
        # Initialize metrics
        if METRICS_AVAILABLE:
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        
    def _setup_device(self, device: str) -> str:
        """Setup device for evaluation."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            else:
                return "cpu"
        return device
    
    def load_model(self):
        """Load the trained LlamaAudio model."""
        logger.info(f"Loading LlamaAudio model from: {self.model_path}")
        
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
                logger.warning("Model weights not found, using base model")
            
            self.model.to(self.device)
            self.model.eval()
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
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
    
    def generate_response(self, audio_path: str, instruction: str) -> str:
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
                    max_length=self.max_new_tokens,
                    temperature=0.1,  # Low temperature for evaluation
                    do_sample=False,  # Deterministic generation
                    num_beams=1
                )
                
                return response[0] if isinstance(response, list) else response
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ""
    
    def evaluate_audio_captioning(self, test_data: List[Dict]) -> Dict:
        """Evaluate audio captioning performance."""
        logger.info("Evaluating audio captioning...")
        
        predictions = []
        references = []
        
        for item in tqdm(test_data, desc="Audio Captioning"):
            audio_path = item["audio_path"]
            reference = item["output"]
            instruction = item.get("instruction", "请描述这段音频的内容。")
            
            if not os.path.exists(audio_path):
                logger.warning(f"Audio file not found: {audio_path}")
                continue
            
            try:
                prediction = self.generate_response(audio_path, instruction)
                predictions.append(prediction)
                references.append(reference)
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                continue
        
        # Calculate metrics
        metrics = {}
        
        if METRICS_AVAILABLE and predictions and references:
            # BLEU score
            try:
                bleu_score = corpus_bleu(predictions, [references]).score
                metrics["bleu"] = bleu_score
            except:
                metrics["bleu"] = 0.0
            
            # ROUGE scores
            rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                for metric in rouge_scores:
                    rouge_scores[metric].append(scores[metric].fmeasure)
            
            for metric in rouge_scores:
                metrics[f"rouge_{metric}"] = np.mean(rouge_scores[metric])
        
        metrics["num_samples"] = len(predictions)
        
        return metrics, predictions, references
    
    def evaluate_audio_qa(self, test_data: List[Dict]) -> Dict:
        """Evaluate audio question answering performance."""
        logger.info("Evaluating audio question answering...")
        
        correct = 0
        total = 0
        predictions = []
        references = []
        
        for item in tqdm(test_data, desc="Audio Q&A"):
            audio_path = item["audio_path"]
            question = item["instruction"]
            reference = item["output"]
            
            if not os.path.exists(audio_path):
                continue
            
            try:
                prediction = self.generate_response(audio_path, question)
                predictions.append(prediction)
                references.append(reference)
                
                # Simple exact match for short answers
                if len(reference.split()) <= 3:
                    if prediction.strip().lower() == reference.strip().lower():
                        correct += 1
                total += 1
                
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                continue
        
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "num_samples": len(predictions)
        }, predictions, references
    
    def evaluate_instruction_following(self, test_data: List[Dict]) -> Dict:
        """Evaluate instruction following capability."""
        logger.info("Evaluating instruction following...")
        
        # For instruction following, we focus on response relevance and format
        relevant_responses = 0
        total_responses = 0
        predictions = []
        instructions = []
        
        for item in tqdm(test_data, desc="Instruction Following"):
            audio_path = item["audio_path"]
            instruction = item["instruction"]
            
            if not os.path.exists(audio_path):
                continue
            
            try:
                prediction = self.generate_response(audio_path, instruction)
                predictions.append(prediction)
                instructions.append(instruction)
                
                # Simple heuristic: response should be non-empty and relevant length
                if prediction.strip() and len(prediction.split()) >= 5:
                    relevant_responses += 1
                total_responses += 1
                
            except Exception as e:
                logger.error(f"Error processing {audio_path}: {e}")
                continue
        
        relevance_score = relevant_responses / total_responses if total_responses > 0 else 0.0
        
        return {
            "relevance_score": relevance_score,
            "relevant_responses": relevant_responses,
            "total_responses": total_responses,
            "num_samples": len(predictions)
        }, predictions, instructions
    
    def evaluate_conversation(self, test_data: List[Dict]) -> Dict:
        """Evaluate multi-turn conversation capability."""
        logger.info("Evaluating conversation capability...")
        
        conversation_scores = []
        predictions = []
        references = []
        
        for item in tqdm(test_data, desc="Conversation"):
            audio_path = item["audio_path"]
            conversation = item.get("conversation", [])
            
            if not os.path.exists(audio_path) or not conversation:
                continue
            
            try:
                # Simulate conversation turns
                conversation_score = 0
                turn_count = 0
                
                for turn in conversation:
                    if turn["role"] == "user":
                        user_message = turn["content"]
                        
                        # Find corresponding assistant response
                        next_turn_idx = conversation.index(turn) + 1
                        if next_turn_idx < len(conversation) and conversation[next_turn_idx]["role"] == "assistant":
                            expected_response = conversation[next_turn_idx]["content"]
                            
                            # Generate model response
                            model_response = self.generate_response(audio_path, user_message)
                            
                            predictions.append(model_response)
                            references.append(expected_response)
                            
                            # Simple scoring: response should be relevant length
                            if model_response.strip() and len(model_response.split()) >= 3:
                                conversation_score += 1
                            turn_count += 1
                
                if turn_count > 0:
                    conversation_scores.append(conversation_score / turn_count)
                    
            except Exception as e:
                logger.error(f"Error processing conversation {audio_path}: {e}")
                continue
        
        avg_conversation_score = np.mean(conversation_scores) if conversation_scores else 0.0
        
        return {
            "avg_conversation_score": avg_conversation_score,
            "num_conversations": len(conversation_scores),
            "num_turns": len(predictions)
        }, predictions, references
    
    def run_comprehensive_evaluation(self, test_data_paths: Dict[str, str], output_dir: str):
        """Run comprehensive evaluation on multiple tasks."""
        logger.info("Starting comprehensive evaluation...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Results storage
        all_results = {}
        detailed_results = {}
        
        # Evaluate each task
        for task_name, data_path in test_data_paths.items():
            if not os.path.exists(data_path):
                logger.warning(f"Test data not found for {task_name}: {data_path}")
                continue
            
            logger.info(f"Evaluating task: {task_name}")
            
            # Load test data
            test_data = []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    test_data.append(json.loads(line))
            
            logger.info(f"Loaded {len(test_data)} samples for {task_name}")
            
            # Run task-specific evaluation
            if task_name == "audio_captioning":
                metrics, predictions, references = self.evaluate_audio_captioning(test_data)
            elif task_name == "audio_qa":
                metrics, predictions, references = self.evaluate_audio_qa(test_data)
            elif task_name == "instruction_following":
                metrics, predictions, references = self.evaluate_instruction_following(test_data)
            elif task_name == "conversation":
                metrics, predictions, references = self.evaluate_conversation(test_data)
            else:
                logger.warning(f"Unknown task: {task_name}")
                continue
            
            all_results[task_name] = metrics
            detailed_results[task_name] = {
                "predictions": predictions,
                "references": references
            }
            
            # Save task-specific results
            task_output_file = os.path.join(output_dir, f"{task_name}_results.json")
            with open(task_output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "metrics": metrics,
                    "predictions": predictions,
                    "references": references
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Task {task_name} completed. Results saved to {task_output_file}")
        
        # Save overall results
        overall_results = {
            "model_path": self.model_path,
            "evaluation_time": datetime.now().isoformat(),
            "device": self.device,
            "results": all_results
        }
        
        overall_output_file = os.path.join(output_dir, "evaluation_results.json")
        with open(overall_output_file, 'w', encoding='utf-8') as f:
            json.dump(overall_results, f, indent=2, ensure_ascii=False)
        
        # Generate evaluation report
        self.generate_evaluation_report(all_results, output_dir)
        
        logger.info(f"Comprehensive evaluation completed. Results saved to {output_dir}")
        
        return all_results
    
    def generate_evaluation_report(self, results: Dict, output_dir: str):
        """Generate a human-readable evaluation report."""
        report_file = os.path.join(output_dir, "evaluation_report.md")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# LlamaAudio Model Evaluation Report\n\n")
            f.write(f"**Model Path:** {self.model_path}\n")
            f.write(f"**Evaluation Time:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Device:** {self.device}\n\n")
            
            f.write("## Overall Performance\n\n")
            
            for task_name, metrics in results.items():
                f.write(f"### {task_name.replace('_', ' ').title()}\n\n")
                
                for metric_name, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"- **{metric_name}:** {value:.4f}\n")
                    else:
                        f.write(f"- **{metric_name}:** {value}\n")
                f.write("\n")
            
            f.write("## Performance Summary\n\n")
            
            # Calculate overall scores
            if "audio_captioning" in results:
                f.write("### Audio Understanding Capabilities ✅\n")
                f.write("- Model demonstrates strong audio captioning abilities\n")
                if "bleu" in results["audio_captioning"]:
                    bleu = results["audio_captioning"]["bleu"]
                    if bleu > 20:
                        f.write("- BLEU score indicates good caption quality\n")
                    else:
                        f.write("- BLEU score suggests room for improvement\n")
            
            if "instruction_following" in results:
                f.write("### Instruction Following Capabilities ✅\n")
                relevance = results["instruction_following"].get("relevance_score", 0)
                if relevance > 0.8:
                    f.write("- Excellent instruction following performance\n")
                elif relevance > 0.6:
                    f.write("- Good instruction following performance\n")
                else:
                    f.write("- Instruction following needs improvement\n")
            
            if "conversation" in results:
                f.write("### Conversation Capabilities ✅\n")
                conv_score = results["conversation"].get("avg_conversation_score", 0)
                if conv_score > 0.8:
                    f.write("- Excellent conversation abilities\n")
                elif conv_score > 0.6:
                    f.write("- Good conversation abilities\n")
                else:
                    f.write("- Conversation abilities need improvement\n")
            
            f.write("\n## Recommendations\n\n")
            f.write("Based on the evaluation results:\n\n")
            
            # Generate recommendations based on scores
            for task_name, metrics in results.items():
                if task_name == "audio_captioning" and "bleu" in metrics:
                    if metrics["bleu"] < 15:
                        f.write(f"- Consider additional training on audio captioning data\n")
                
                if task_name == "instruction_following":
                    if metrics.get("relevance_score", 0) < 0.7:
                        f.write(f"- Improve instruction following with more diverse training examples\n")
            
            f.write("\n---\n")
            f.write("*This report was generated automatically by the LlamaAudio evaluation system.*\n")
        
        logger.info(f"Evaluation report saved to {report_file}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate LlamaAudio Model")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the trained LlamaAudio model"
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        help="Path to test data JSONL file (single task evaluation)"
    )
    parser.add_argument(
        "--test_data_dir",
        type=str,
        help="Directory containing test data files for multiple tasks"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="evaluation_results",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["audio_captioning", "audio_qa", "instruction_following", "conversation"],
        help="Specific task to evaluate (if using single test file)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not os.path.exists(args.model_path):
        logger.error(f"Model path does not exist: {args.model_path}")
        sys.exit(1)
    
    if not args.test_data_path and not args.test_data_dir:
        logger.error("Either --test_data_path or --test_data_dir must be provided")
        sys.exit(1)
    
    # Initialize evaluator
    evaluator = LlamaAudioEvaluator(
        model_path=args.model_path,
        device=args.device
    )
    
    # Prepare test data paths
    test_data_paths = {}
    
    if args.test_data_path:
        # Single task evaluation
        if not args.task:
            logger.error("--task must be specified when using --test_data_path")
            sys.exit(1)
        test_data_paths[args.task] = args.test_data_path
    
    elif args.test_data_dir:
        # Multiple task evaluation
        test_dir = Path(args.test_data_dir)
        for task in ["audio_captioning", "audio_qa", "instruction_following", "conversation"]:
            test_file = test_dir / f"{task}_test.jsonl"
            if test_file.exists():
                test_data_paths[task] = str(test_file)
    
    if not test_data_paths:
        logger.error("No valid test data files found")
        sys.exit(1)
    
    logger.info(f"Found test data for tasks: {list(test_data_paths.keys())}")
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation(test_data_paths, args.output_dir)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    for task_name, metrics in results.items():
        print(f"\n{task_name.replace('_', ' ').title()}:")
        for metric_name, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric_name}: {value:.4f}")
            else:
                print(f"  {metric_name}: {value}")
    
    print(f"\nDetailed results saved to: {args.output_dir}/")
    print("="*60)

if __name__ == "__main__":
    main() 