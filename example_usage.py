"""
Example usage of Qwen2AudioProcessor with proper special token handling
"""

from src.processor import Qwen2AudioProcessor
import torch

def main():
    # Initialize processor (special tokens are added automatically)
    print("Initializing Qwen2AudioProcessor...")
    processor = Qwen2AudioProcessor()
    
    # Check what special tokens were added
    special_tokens = processor.get_special_token_ids()
    print(f"Special token IDs: {special_tokens}")
    
    # Example: If you have a model, resize its embeddings
    # model = YourModel.from_pretrained("model_name")
    # processor.resize_model_embeddings(model)
    
    # Test tokenization with special tokens
    test_text = "Hello <|audio_bos|><|AUDIO|><|audio_eos|> world"
    tokenized = processor.process_text(test_text)
    print(f"Tokenized text: {tokenized}")
    
    # Decode to verify special tokens are preserved
    decoded = processor.decode(tokenized['input_ids'].squeeze())
    print(f"Decoded text: {decoded}")
    
    # Test with conversation template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What's in this audio?"},
                {"type": "audio", "audio": "path/to/audio.wav"}
            ]
        }
    ]
    
    formatted = processor.apply_chat_template(conversation, tokenize=False)
    print(f"Formatted conversation: {formatted}")

if __name__ == "__main__":
    main() 