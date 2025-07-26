"""
LlamaAudio: A multimodal audio-language model based on Llama LLMs.

This framework integrates audio encoders with Llama-series language models
to enable audio understanding and generation capabilities.
"""

__version__ = "0.1.0"

from .models.llama_audio_model import LlamaAudioModel
from .models.audio_encoder import AudioEncoder
from .models.multimodal_connector import MultiModalConnector

__all__ = [
    "LlamaAudioModel",
    "AudioEncoder", 
    "MultiModalConnector",
] 