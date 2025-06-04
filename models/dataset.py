import torch
import torchaudio
import json
import os
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from transformers import WhisperFeatureExtractor
import librosa
import soundfile as sf


class AudioTextDataset(Dataset):
    """
    Base dataset class for audio-text pairs
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        feature_extractor: WhisperFeatureExtractor,
        max_audio_length: float = 30.0,
        max_text_length: int = 2048,
        sample_rate: int = 16000,
        stage: str = "pretrain"
    ):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sample_rate = sample_rate
        self.stage = stage
        
        # Load dataset
        self.data = self.load_data()
        
    def load_data(self) -> List[Dict]:
        """Load dataset from disk"""
        if os.path.exists(os.path.join(self.data_path, "data.json")):
            with open(os.path.join(self.data_path, "data.json"), 'r', encoding='utf-8') as f:
                return json.load(f)
        elif os.path.exists(os.path.join(self.data_path, "data.jsonl")):
            data = []
            with open(os.path.join(self.data_path, "data.jsonl"), 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line.strip()))
            return data
        else:
            # Fallback: scan directory for audio files
            return self.scan_audio_directory()
    
    def scan_audio_directory(self) -> List[Dict]:
        """Scan directory for audio files and create dataset"""
        data = []
        audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
        
        for root, dirs, files in os.walk(self.data_path):
            for file in files:
                if any(file.lower().endswith(ext) for ext in audio_extensions):
                    audio_path = os.path.join(root, file)
                    # Look for corresponding text file
                    text_path = audio_path.rsplit('.', 1)[0] + '.txt'
                    if os.path.exists(text_path):
                        with open(text_path, 'r', encoding='utf-8') as f:
                            text = f.read().strip()
                        data.append({
                            'audio_path': audio_path,
                            'text': text,
                            'instruction': '',
                            'response': text
                        })
        return data
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """Load and preprocess audio file"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Truncate if too long
            max_samples = int(self.max_audio_length * self.sample_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            # Return silence as fallback
            return np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx) -> Dict[str, Any]:
        sample = self.data[idx]
        
        # Load audio
        audio_path = sample.get('audio_path', '')
        if audio_path and os.path.exists(audio_path):
            audio = self.load_audio(audio_path)
        else:
            # Generate dummy audio if path doesn't exist
            audio = np.zeros(int(self.sample_rate * 1.0), dtype=np.float32)
        
        # Process audio with Whisper feature extractor
        audio_features = self.feature_extractor(
            audio,
            sampling_rate=self.sample_rate,
            return_tensors="pt"
        )
        
        # Prepare text based on training stage
        if self.stage == "pretrain":
            # For pretraining, use simple audio transcription format
            text = sample.get('text', '')
            prompt = f"<|audio_bos|><|audio_eos|> {text}"
        elif self.stage == "sft":
            # For SFT, use instruction-response format
            instruction = sample.get('instruction', 'Please transcribe the audio.')
            response = sample.get('response', sample.get('text', ''))
            prompt = f"User: {instruction}\n<|audio_bos|><|audio_eos|>\nAssistant: {response}"
        elif self.stage == "dpo":
            # For DPO, need both chosen and rejected responses
            instruction = sample.get('instruction', 'Please transcribe the audio.')
            chosen = sample.get('chosen', sample.get('response', ''))
            rejected = sample.get('rejected', '')
            prompt = f"User: {instruction}\n<|audio_bos|><|audio_eos|>\nAssistant: {chosen}"
        else:
            text = sample.get('text', '')
            prompt = f"<|audio_bos|><|audio_eos|> {text}"
        
        # Tokenize text
        encoding = self.tokenizer(
            prompt,
            max_length=self.max_text_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        result = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'audio_values': audio_features['input_features'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0).clone()
        }
        
        # For DPO, also include rejected response
        if self.stage == "dpo" and 'rejected' in sample:
            rejected_prompt = f"User: {instruction}\n<|audio_bos|><|audio_eos|>\nAssistant: {sample['rejected']}"
            rejected_encoding = self.tokenizer(
                rejected_prompt,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            result['rejected_input_ids'] = rejected_encoding['input_ids'].squeeze(0)
            result['rejected_attention_mask'] = rejected_encoding['attention_mask'].squeeze(0)
            result['rejected_labels'] = rejected_encoding['input_ids'].squeeze(0).clone()
        
        return result


class PretrainDataset(AudioTextDataset):
    """Dataset for pretraining stage"""
    
    def __init__(self, data_path: str, tokenizer, feature_extractor: WhisperFeatureExtractor, **kwargs):
        super().__init__(data_path, tokenizer, feature_extractor, stage="pretrain", **kwargs)


class SFTDataset(AudioTextDataset):
    """Dataset for supervised fine-tuning stage"""
    
    def __init__(self, data_path: str, tokenizer, feature_extractor: WhisperFeatureExtractor, **kwargs):
        super().__init__(data_path, tokenizer, feature_extractor, stage="sft", **kwargs)


class DPODataset(AudioTextDataset):
    """Dataset for Direct Preference Optimization stage"""
    
    def __init__(self, data_path: str, tokenizer, feature_extractor: WhisperFeatureExtractor, **kwargs):
        super().__init__(data_path, tokenizer, feature_extractor, stage="dpo", **kwargs)


class MultiDatasetLoader:
    """
    Loader for multiple datasets with weighted sampling
    """
    
    def __init__(
        self,
        datasets_config: List[Dict[str, Any]],
        tokenizer,
        feature_extractor: WhisperFeatureExtractor,
        stage: str = "pretrain"
    ):
        self.datasets = []
        self.weights = []
        
        for config in datasets_config:
            dataset_class = {
                'pretrain': PretrainDataset,
                'sft': SFTDataset,
                'dpo': DPODataset
            }[stage]
            
            dataset = dataset_class(
                data_path=config['path'],
                tokenizer=tokenizer,
                feature_extractor=feature_extractor,
                **config.get('kwargs', {})
            )
            
            self.datasets.append(dataset)
            self.weights.append(config.get('weight', 1.0))
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        # Calculate total length
        self.total_length = sum(len(dataset) for dataset in self.datasets)
    
    def __len__(self):
        return self.total_length
    
    def get_dataset_and_index(self, idx):
        """Get which dataset and local index for global index"""
        # Simple round-robin for now
        # In practice, you might want weighted random sampling
        dataset_idx = idx % len(self.datasets)
        local_idx = (idx // len(self.datasets)) % len(self.datasets[dataset_idx])
        return self.datasets[dataset_idx], local_idx
    
    def __getitem__(self, idx):
        dataset, local_idx = self.get_dataset_and_index(idx)
        return dataset[local_idx]


def create_dataset(
    config: Dict[str, Any],
    tokenizer,
    feature_extractor: WhisperFeatureExtractor,
    stage: str = "pretrain"
) -> AudioTextDataset:
    """
    Create dataset from configuration
    """
    data_config = config.get('data_paths', {}).get(stage, {})
    datasets_config = data_config.get('datasets', [])
    
    if len(datasets_config) == 1:
        # Single dataset
        dataset_config = datasets_config[0]
        dataset_class = {
            'pretrain': PretrainDataset,
            'sft': SFTDataset,
            'dpo': DPODataset
        }[stage]
        
        return dataset_class(
            data_path=dataset_config['path'],
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            max_audio_length=config.get('training', {}).get('data', {}).get('max_audio_length', 30.0),
            max_text_length=config.get('training', {}).get('data', {}).get('max_text_length', 2048),
            sample_rate=config.get('training', {}).get('data', {}).get('sample_rate', 16000)
        )
    else:
        # Multiple datasets
        return MultiDatasetLoader(
            datasets_config=datasets_config,
            tokenizer=tokenizer,
            feature_extractor=feature_extractor,
            stage=stage
        )


def collate_fn(batch):
    """
    Custom collate function for batching audio-text data
    """
    # Standard keys that should be present in all samples
    keys = ['input_ids', 'attention_mask', 'audio_values', 'labels']
    
    # Check if any sample has DPO-specific keys
    has_rejected = any('rejected_input_ids' in sample for sample in batch)
    if has_rejected:
        keys.extend(['rejected_input_ids', 'rejected_attention_mask', 'rejected_labels'])
    
    result = {}
    
    for key in keys:
        if key in batch[0]:
            # Stack tensors
            values = [sample[key] for sample in batch if key in sample]
            if values:
                result[key] = torch.stack(values, dim=0)
    
    return result 