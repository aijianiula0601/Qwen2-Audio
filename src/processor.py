"""
Qwen2-Audio Processor Implementation

This module implements the processor for handling audio and text inputs
for the Qwen2-Audio model.
"""

import torch
import torchaudio
import librosa
import numpy as np
from typing import List, Union, Optional, Dict, Any
from transformers import AutoTokenizer, WhisperFeatureExtractor
import warnings


class Qwen2AudioProcessor:
    """
    Processor for Qwen2-Audio model that handles both audio and text inputs
    """
    
    def __init__(
        self,
        tokenizer_name: str = "Qwen/Qwen2-7B",
        feature_extractor_name: str = "openai/whisper-large-v3",
        sampling_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 160,
        win_length: int = 400,
        max_audio_length: int = 30  # seconds
    ):
        """
        Initialize Qwen2AudioProcessor
        
        Args:
            tokenizer_name: Name of the tokenizer model
            feature_extractor_name: Name of the audio feature extractor
            sampling_rate: Audio sampling rate (16kHz)
            n_mels: Number of mel-scale features
            hop_length: Hop length for STFT (10ms)
            win_length: Window length for STFT (25ms)
            max_audio_length: Maximum audio length in seconds
        """
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
        self.max_audio_length = max_audio_length
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Initialize audio feature extractor
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(
            feature_extractor_name
        )
        
        # Add special tokens
        self._add_special_tokens()
        
    def _add_special_tokens(self):
        """Add special tokens for audio processing"""
        special_tokens = {
            "additional_special_tokens": [
                "<|audio_bos|>",
                "<|audio_eos|>", 
                "<|AUDIO|>"
            ]
        }
        
        # Add tokens if they don't exist
        num_added = self.tokenizer.add_special_tokens(special_tokens)
        if num_added > 0:
            print(f"Added {num_added} special tokens")
    
    def load_audio(self, audio_path: str) -> np.ndarray:
        """
        Load audio file and resample to target sampling rate
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            audio: Audio waveform as numpy array
        """
        try:
            # Use librosa to load audio
            audio, sr = librosa.load(audio_path, sr=self.sampling_rate)
            
            # Limit audio length
            max_samples = int(self.max_audio_length * self.sampling_rate)
            if len(audio) > max_samples:
                audio = audio[:max_samples]
                warnings.warn(f"Audio truncated to {self.max_audio_length} seconds")
                
            return audio
            
        except Exception as e:
            raise ValueError(f"Error loading audio file {audio_path}: {e}")
    
    def process_audio(self, audio: Union[str, np.ndarray, List[float]]) -> torch.Tensor:
        """
        Process audio input into mel-spectrogram features
        
        Args:
            audio: Audio input (file path, numpy array, or list)
            
        Returns:
            audio_features: Mel-spectrogram features [n_mels, time_steps]
        """
        # Load audio if it's a file path
        if isinstance(audio, str):
            audio = self.load_audio(audio)
        elif isinstance(audio, list):
            audio = np.array(audio)
            
        # Ensure audio is 1D numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.numpy()
        if audio.ndim > 1:
            audio = audio.flatten()
            
        # Use Whisper feature extractor
        features = self.feature_extractor(
            audio,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        )
        
        return features.input_features.squeeze(0)  # Remove batch dimension
    
    def process_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Process text input using tokenizer
        
        Args:
            text: Input text string
            
        Returns:
            tokenized: Dictionary with input_ids and attention_mask
        """
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
    
    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]],
        add_generation_prompt: bool = True,
        tokenize: bool = True
    ) -> Union[str, Dict[str, torch.Tensor]]:
        """
        Apply chat template to conversation
        
        Args:
            conversation: List of conversation turns
            add_generation_prompt: Whether to add generation prompt
            tokenize: Whether to tokenize the result
            
        Returns:
            Formatted conversation string or tokenized result
        """
        # Build conversation string
        formatted_conversation = ""
        
        for turn in conversation:
            role = turn["role"]
            content = turn["content"]
            
            if role == "system":
                formatted_conversation += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted_conversation += f"<|im_start|>user\n"
                
                if isinstance(content, list):
                    # Handle multimodal content
                    for item in content:
                        if item["type"] == "text":
                            formatted_conversation += item["text"]
                        elif item["type"] == "audio":
                            # Add audio placeholder tokens
                            formatted_conversation += "<|audio_bos|><|AUDIO|><|audio_eos|>"
                else:
                    # Handle text-only content
                    formatted_conversation += content
                    
                formatted_conversation += "<|im_end|>\n"
                
            elif role == "assistant":
                formatted_conversation += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        if add_generation_prompt:
            formatted_conversation += "<|im_start|>assistant\n"
        
        if tokenize:
            return self.process_text(formatted_conversation)
        else:
            return formatted_conversation
    
    def __call__(
        self,
        text: Optional[Union[str, List[str]]] = None,
        audios: Optional[Union[str, np.ndarray, List[Union[str, np.ndarray]]]] = None,
        return_tensors: str = "pt",
        padding: bool = True,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Main processing function
        
        Args:
            text: Text input(s)
            audios: Audio input(s)
            return_tensors: Type of tensors to return
            padding: Whether to pad sequences
            
        Returns:
            Dictionary with processed inputs
        """
        outputs = {}
        
        # Process text
        if text is not None:
            if isinstance(text, str):
                text = [text]
                
            text_outputs = self.tokenizer(
                text,
                return_tensors=return_tensors,
                padding=padding,
                truncation=True,
                **kwargs
            )
            outputs.update(text_outputs)
        
        # Process audio
        if audios is not None:
            if not isinstance(audios, list):
                audios = [audios]
                
            audio_features = []
            for audio in audios:
                features = self.process_audio(audio)
                audio_features.append(features)
            
            # Stack audio features
            if audio_features:
                outputs["audio_features"] = torch.stack(audio_features)
        
        return outputs
    
    def batch_decode(
        self,
        sequences: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False
    ) -> List[str]:
        """
        Decode token sequences to text
        
        Args:
            sequences: Token sequences to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            List of decoded strings
        """
        return self.tokenizer.batch_decode(
            sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )
    
    def decode(
        self,
        token_ids: torch.Tensor,
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False
    ) -> str:
        """
        Decode single token sequence to text
        
        Args:
            token_ids: Token sequence to decode
            skip_special_tokens: Whether to skip special tokens
            clean_up_tokenization_spaces: Whether to clean up spaces
            
        Returns:
            Decoded string
        """
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces
        )


class AudioFeatureExtractor:
    """
    Standalone audio feature extractor for preprocessing
    """
    
    def __init__(
        self,
        sampling_rate: int = 16000,
        n_mels: int = 128,
        hop_length: int = 160,
        win_length: int = 400
    ):
        self.sampling_rate = sampling_rate
        self.n_mels = n_mels
        self.hop_length = hop_length
        self.win_length = win_length
    
    def extract_features(self, audio: np.ndarray) -> torch.Tensor:
        """Extract mel-spectrogram features from audio"""
        
        # Compute mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sampling_rate,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window='hann',
            center=True,
            pad_mode='reflect'
        )
        
        # Convert to log scale
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [-1, 1] range
        log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-8)
        
        return torch.tensor(log_mel_spec, dtype=torch.float32)


def create_processor(
    tokenizer_name: str = "Qwen/Qwen2-7B",
    feature_extractor_name: str = "openai/whisper-large-v3"
) -> Qwen2AudioProcessor:
    """Create a Qwen2AudioProcessor with default settings"""
    
    return Qwen2AudioProcessor(
        tokenizer_name=tokenizer_name,
        feature_extractor_name=feature_extractor_name
    ) 