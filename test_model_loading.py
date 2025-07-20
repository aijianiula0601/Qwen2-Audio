#!/usr/bin/env python3
"""
Simple test script to verify model loading works correctly
"""

import os
import sys
import torch
import logging

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import create_model_from_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_model_loading():
    """Test if the model can be loaded without errors"""
    
    # Test configuration for Qwen2-0.5B
    test_config = {
        'model': {
            'audio_encoder': {
                'model_name': 'openai/whisper-large-v3',
                'freeze_encoder': False
            },
            'llm_backbone': {
                'model_name': 'Qwen/Qwen2-0.5B',
                'model_type': 'qwen2',
                'freeze_llm': False
            },
            'audio_projector': {
                'input_size': 1280,
                'hidden_size': 896,
                'intermediate_size': 3584,
                'num_layers': 2
            },
            'special_tokens': {
                'audio_start_token': '<|audio_bos|>',
                'audio_end_token': '<|audio_eos|>'
            }
        }
    }
    
    try:
        logger.info("Testing model loading...")
        logger.info(f"Config: {test_config}")
        
        # Create model
        model = create_model_from_config(test_config)
        
        logger.info("✅ Model loaded successfully!")
        logger.info(f"Model type: {type(model)}")
        logger.info(f"LLM type: {type(model.llm)}")
        logger.info(f"Audio encoder type: {type(model.audio_encoder)}")
        logger.info(f"Audio projector type: {type(model.audio_projector)}")
        
        # Test a simple forward pass
        logger.info("Testing forward pass...")
        
        # Create dummy inputs
        batch_size = 1
        seq_len = 10
        hidden_size = 896
        
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        audio_values = torch.randn(batch_size, 80, 100)  # mel-spectrogram shape
        
        # Forward pass
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                audio_values=audio_values
            )
        
        logger.info("✅ Forward pass successful!")
        logger.info(f"Output shape: {outputs.logits.shape}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_model_loading()
    if success:
        print("\n🎉 All tests passed! The model loading fix is working correctly.")
    else:
        print("\n💥 Tests failed. Please check the error messages above.")
        sys.exit(1) 