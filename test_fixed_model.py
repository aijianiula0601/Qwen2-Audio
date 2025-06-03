#!/usr/bin/env python3

import torch
from src import Qwen2AudioForConditionalGeneration, Qwen2AudioProcessor, Qwen2AudioConfig

def test_model():
    # Test the model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using device: {}'.format(device))

    # Load config and model
    model_dir='/data1/hjh/huggingface/Qwen-7B'
    config = Qwen2AudioConfig.from_pretrained(model_dir)
    config.audio_hidden_size = 1280
    config.llm_hidden_size = 3584

    print('Loading model...')
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=torch.float16 if device == 'cuda' else torch.float32,
        device_map='auto' if device == 'cuda' else None
    )

    processor = Qwen2AudioProcessor()

    if device == 'cpu':
        model = model.to(device)

    print('Model loaded successfully')

    # Create test input
    audio_input = torch.randn(1, 128, 3000).to(device)  # [batch, n_mels, seq_len]

    conversation = [
        {'role': 'user', 'content': [
            {'type': 'audio', 'audio': audio_input},
            {'type': 'text', 'text': '请描述这段音频的内容'}
        ]}
    ]

    # Process inputs
    text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
    inputs = processor(text=text, audio_features=audio_input, return_tensors='pt')

    # Move to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    print('Input shapes:')
    print('input_ids: {}'.format(inputs["input_ids"].shape))
    print('attention_mask: {}'.format(inputs["attention_mask"].shape))
    print('audio_features: {}'.format(inputs["audio_features"].shape))

    # Test generation
    try:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=100,
                temperature=0.7,
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode output
        response = processor.decode(outputs[0], skip_special_tokens=True)
        print('Generation successful!')
        print('Response: {}'.format(response))
        
    except Exception as e:
        print('Error during generation: {}'.format(e))
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_model() 