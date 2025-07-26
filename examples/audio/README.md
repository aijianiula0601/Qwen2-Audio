# Examples Audio Directory

This directory should contain sample audio files for testing the LlamaAudio model.

## Required Audio Files

To test the model with the provided sample data (`examples/sample_test_data.jsonl`), please add the following audio files:

### 1. speech_sample.wav
- **Type**: Speech recording
- **Content**: Clear English speech
- **Duration**: 5-15 seconds
- **Format**: WAV, 16kHz sample rate
- **Purpose**: Test speech recognition capabilities

### 2. music_sample.wav
- **Type**: Music recording
- **Content**: Piano music (slow, relaxing)
- **Duration**: 10-20 seconds
- **Format**: WAV, 16kHz sample rate
- **Purpose**: Test music description capabilities

### 3. environment_sample.wav
- **Type**: Environmental audio
- **Content**: Nature sounds (birds, wind)
- **Duration**: 8-15 seconds
- **Format**: WAV, 16kHz sample rate
- **Purpose**: Test environmental sound classification

### 4. conversation_sample.wav
- **Type**: Conversation recording
- **Content**: Two people talking about casual topics
- **Duration**: 15-30 seconds
- **Format**: WAV, 16kHz sample rate
- **Purpose**: Test conversation analysis

### 5. music_emotion.wav
- **Type**: Emotional music
- **Content**: Calm, peaceful instrumental music
- **Duration**: 10-20 seconds
- **Format**: WAV, 16kHz sample rate
- **Purpose**: Test emotion analysis capabilities

## How to Add Audio Files

1. **Create your own**: Record or find suitable audio files
2. **Convert format**: Ensure all files are WAV format with 16kHz sample rate
3. **Use ffmpeg to convert** (if needed):
   ```bash
   ffmpeg -i input_audio.mp3 -ar 16000 -ac 1 output_audio.wav
   ```

## Alternative: Use Your Own Audio

You can also:
1. Replace the `audio_path` values in `sample_test_data.jsonl` with paths to your own audio files
2. Update the corresponding `instruction` and `output` fields to match your audio content

## Legal Note

Ensure any audio files you add comply with copyright laws and are properly licensed for your use case.

## Testing Without Audio Files

If you don't have audio files, you can still test other parts of the system:
- Training pipeline (will download datasets automatically)
- Configuration validation
- Model loading (will use dummy data)

## Quick Test Command

Once you have audio files in place:

```bash
# Test evaluation with sample data
python scripts/evaluate.py \
    --model_path outputs/stage3_conversation/final_model \
    --test_data_path examples/sample_test_data.jsonl \
    --task audio_captioning \
    --output_dir test_evaluation

# Test interactive demo
python scripts/interactive_demo.py \
    --model_path outputs/stage3_conversation/final_model \
    --port 7860
``` 