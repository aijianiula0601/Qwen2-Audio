"""
Qwen2-Audio: A Large-Scale Audio-Language Model

This package implements the Qwen2-Audio model described in the paper:
"Qwen2-Audio Technical Report" (arXiv:2407.10759)
"""

from .model import (
    Qwen2AudioConfig,
    Qwen2AudioForConditionalGeneration,
    AudioEncoder,
    create_qwen2_audio_model,
    load_pretrained_qwen2_audio
)

from .processor import (
    Qwen2AudioProcessor,
    AudioFeatureExtractor,
    create_processor
)

from .trainer import (
    Qwen2AudioTrainer,
    Qwen2AudioDataset,
    Qwen2AudioTrainingArguments,
    train_qwen2_audio,
    create_data_collator
)

__version__ = "0.1.0"
__all__ = [
    "Qwen2AudioConfig",
    "Qwen2AudioForConditionalGeneration", 
    "AudioEncoder",
    "Qwen2AudioProcessor",
    "AudioFeatureExtractor",
    "Qwen2AudioTrainer",
    "Qwen2AudioDataset",
    "Qwen2AudioTrainingArguments",
    "create_qwen2_audio_model",
    "load_pretrained_qwen2_audio",
    "create_processor",
    "train_qwen2_audio",
    "create_data_collator"
] 