#!/usr/bin/env python3
import torch
import argparse
import os
import sys
import json
import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from demo.inference import Qwen2AudioInference

# Evaluation metrics
import jiwer
from typing import List, Dict, Any


class AudioEvaluator:
    """
    Comprehensive evaluator for Qwen2-Audio model
    """
    
    def __init__(self, model_path: str, device: str = "auto"):
        self.inference = Qwen2AudioInference(model_path, device)
        self.results = {}
        
    def evaluate_transcription(self, test_data: List[Dict], save_predictions: bool = True) -> Dict:
        """
        Evaluate speech transcription task
        test_data format: [{"audio_path": "...", "reference": "..."}, ...]
        """
        predictions = []
        references = []
        
        print("Evaluating transcription task...")
        for item in tqdm(test_data):
            try:
                # Get prediction
                pred = self.inference.transcribe(item['audio_path'], max_new_tokens=200)
                predictions.append(pred)
                references.append(item['reference'])
                
            except Exception as e:
                print(f"Error processing {item['audio_path']}: {e}")
                predictions.append("")
                references.append(item['reference'])
        
        # Calculate metrics
        wer = jiwer.wer(references, predictions)
        cer = jiwer.cer(references, predictions)
        
        results = {
            "task": "transcription",
            "num_samples": len(test_data),
            "wer": wer,
            "cer": cer,
            "predictions": predictions if save_predictions else None,
            "references": references if save_predictions else None
        }
        
        print(f"Transcription Results:")
        print(f"  WER: {wer:.4f}")
        print(f"  CER: {cer:.4f}")
        
        return results
    
    def evaluate_qa(self, test_data: List[Dict], save_predictions: bool = True) -> Dict:
        """
        Evaluate audio question answering task
        test_data format: [{"audio_path": "...", "question": "...", "answer": "..."}, ...]
        """
        predictions = []
        references = []
        
        print("Evaluating audio QA task...")
        for item in tqdm(test_data):
            try:
                # Get prediction
                pred = self.inference.chat(
                    item['audio_path'], 
                    item['question'], 
                    max_new_tokens=150
                )
                predictions.append(pred)
                references.append(item['answer'])
                
            except Exception as e:
                print(f"Error processing {item['audio_path']}: {e}")
                predictions.append("")
                references.append(item['answer'])
        
        # Calculate BLEU-like metrics (simple word overlap)
        exact_matches = sum(1 for p, r in zip(predictions, references) if p.strip().lower() == r.strip().lower())
        exact_match_rate = exact_matches / len(predictions)
        
        # Calculate word-level F1 (simple approximation)
        f1_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = set(pred.lower().split())
            ref_words = set(ref.lower().split())
            
            if len(pred_words) == 0 and len(ref_words) == 0:
                f1_scores.append(1.0)
            elif len(pred_words) == 0 or len(ref_words) == 0:
                f1_scores.append(0.0)
            else:
                intersection = len(pred_words & ref_words)
                precision = intersection / len(pred_words)
                recall = intersection / len(ref_words)
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
        
        avg_f1 = np.mean(f1_scores)
        
        results = {
            "task": "audio_qa",
            "num_samples": len(test_data),
            "exact_match": exact_match_rate,
            "avg_f1": avg_f1,
            "predictions": predictions if save_predictions else None,
            "references": references if save_predictions else None
        }
        
        print(f"Audio QA Results:")
        print(f"  Exact Match: {exact_match_rate:.4f}")
        print(f"  Average F1: {avg_f1:.4f}")
        
        return results
    
    def evaluate_classification(self, test_data: List[Dict], classes: List[str], save_predictions: bool = True) -> Dict:
        """
        Evaluate audio classification task
        test_data format: [{"audio_path": "...", "label": "..."}, ...]
        classes: list of possible class names
        """
        predictions = []
        references = []
        
        print("Evaluating audio classification task...")
        for item in tqdm(test_data):
            try:
                # Create classification prompt
                prompt = f"What type of audio is this? Choose from: {', '.join(classes)}"
                pred = self.inference.chat(item['audio_path'], prompt, max_new_tokens=50)
                
                # Find the most likely class in prediction
                pred_class = None
                for cls in classes:
                    if cls.lower() in pred.lower():
                        pred_class = cls
                        break
                
                predictions.append(pred_class or "unknown")
                references.append(item['label'])
                
            except Exception as e:
                print(f"Error processing {item['audio_path']}: {e}")
                predictions.append("unknown")
                references.append(item['label'])
        
        # Calculate accuracy
        correct = sum(1 for p, r in zip(predictions, references) if p == r)
        accuracy = correct / len(predictions)
        
        # Calculate per-class metrics
        class_metrics = {}
        for cls in classes:
            tp = sum(1 for p, r in zip(predictions, references) if p == cls and r == cls)
            fp = sum(1 for p, r in zip(predictions, references) if p == cls and r != cls)
            fn = sum(1 for p, r in zip(predictions, references) if p != cls and r == cls)
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            class_metrics[cls] = {"precision": precision, "recall": recall, "f1": f1}
        
        avg_f1 = np.mean([metrics["f1"] for metrics in class_metrics.values()])
        
        results = {
            "task": "classification",
            "num_samples": len(test_data),
            "accuracy": accuracy,
            "avg_f1": avg_f1,
            "class_metrics": class_metrics,
            "predictions": predictions if save_predictions else None,
            "references": references if save_predictions else None
        }
        
        print(f"Classification Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Average F1: {avg_f1:.4f}")
        
        return results
    
    def evaluate_generation(self, test_data: List[Dict], save_predictions: bool = True) -> Dict:
        """
        Evaluate audio captioning/description task
        test_data format: [{"audio_path": "...", "caption": "..."}, ...]
        """
        predictions = []
        references = []
        
        print("Evaluating audio caption generation...")
        for item in tqdm(test_data):
            try:
                # Get prediction
                pred = self.inference.analyze_audio(item['audio_path'], max_new_tokens=200)
                predictions.append(pred)
                references.append(item['caption'])
                
            except Exception as e:
                print(f"Error processing {item['audio_path']}: {e}")
                predictions.append("")
                references.append(item['caption'])
        
        # Calculate BLEU-like score (simplified)
        bleu_scores = []
        for pred, ref in zip(predictions, references):
            pred_words = pred.lower().split()
            ref_words = ref.lower().split()
            
            # Simple BLEU-1 approximation
            matches = sum(1 for word in pred_words if word in ref_words)
            bleu = matches / len(pred_words) if len(pred_words) > 0 else 0
            bleu_scores.append(bleu)
        
        avg_bleu = np.mean(bleu_scores)
        
        results = {
            "task": "caption_generation",
            "num_samples": len(test_data),
            "avg_bleu": avg_bleu,
            "predictions": predictions if save_predictions else None,
            "references": references if save_predictions else None
        }
        
        print(f"Caption Generation Results:")
        print(f"  Average BLEU: {avg_bleu:.4f}")
        
        return results


def load_test_data(data_path: str, task_type: str) -> List[Dict]:
    """Load test data from JSON file"""
    with open(data_path, 'r') as f:
        data = json.load(f)
    
    if task_type not in data:
        raise ValueError(f"Task type '{task_type}' not found in data file")
    
    return data[task_type]


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2-Audio Model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained model")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test data JSON file")
    parser.add_argument("--tasks", type=str, nargs="+", 
                        choices=["transcription", "qa", "classification", "generation"], 
                        default=["transcription"], help="Tasks to evaluate")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", 
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default="auto", help="Device to use")
    parser.add_argument("--save_predictions", action="store_true", 
                        help="Save predictions along with metrics")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = AudioEvaluator(args.model_path, args.device)
    
    # Load test data
    print(f"Loading test data from: {args.test_data}")
    
    all_results = {}
    
    for task in args.tasks:
        print(f"\n=== Evaluating {task} ===")
        
        try:
            test_data = load_test_data(args.test_data, task)
            print(f"Loaded {len(test_data)} samples for {task}")
            
            if task == "transcription":
                results = evaluator.evaluate_transcription(test_data, args.save_predictions)
            elif task == "qa":
                results = evaluator.evaluate_qa(test_data, args.save_predictions)
            elif task == "classification":
                # Load classes from data file if available
                with open(args.test_data, 'r') as f:
                    data_config = json.load(f)
                classes = data_config.get("classes", ["speech", "music", "sound", "noise"])
                results = evaluator.evaluate_classification(test_data, classes, args.save_predictions)
            elif task == "generation":
                results = evaluator.evaluate_generation(test_data, args.save_predictions)
            
            all_results[task] = results
            
            # Save individual task results
            task_output_file = os.path.join(args.output_dir, f"{task}_results.json")
            with open(task_output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to: {task_output_file}")
            
        except Exception as e:
            print(f"Error evaluating {task}: {e}")
            continue
    
    # Save overall results
    overall_output_file = os.path.join(args.output_dir, "evaluation_summary.json")
    summary = {
        "model_path": args.model_path,
        "tasks_evaluated": list(all_results.keys()),
        "results": all_results
    }
    
    with open(overall_output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n=== Evaluation Complete ===")
    print(f"Overall results saved to: {overall_output_file}")
    
    # Print summary
    print("\nSummary:")
    for task, results in all_results.items():
        print(f"  {task}:")
        if task == "transcription":
            print(f"    WER: {results['wer']:.4f}")
            print(f"    CER: {results['cer']:.4f}")
        elif task == "qa":
            print(f"    Exact Match: {results['exact_match']:.4f}")
            print(f"    Average F1: {results['avg_f1']:.4f}")
        elif task == "classification":
            print(f"    Accuracy: {results['accuracy']:.4f}")
            print(f"    Average F1: {results['avg_f1']:.4f}")
        elif task == "generation":
            print(f"    Average BLEU: {results['avg_bleu']:.4f}")


if __name__ == "__main__":
    main() 