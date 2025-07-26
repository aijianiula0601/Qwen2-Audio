# Sample Dataset for LlamaAudio

This is a synthetic dataset created for testing the LlamaAudio framework.

## Dataset Statistics
- Total samples: 100
- Training samples: 90
- Validation samples: 10

## Data Format
Each line in the JSONL files contains:
- `audio_path`: Path to audio file (dummy paths)
- `instruction`: Instruction for the model (for instruction-following format)
- `output`: Expected output (for instruction-following format)
- `conversation`: Multi-turn conversation (for conversation format)

## Note
This is synthetic data for testing purposes. In real usage, you would need:
1. Actual audio files in the specified paths
2. Real human-annotated audio descriptions
3. Proper audio-text alignment

## Usage
Update the `train_data_path` and `val_data_path` in your training config to point to these files:

```yaml
training:
  train_data_path: "data/train.jsonl"
  val_data_path: "data/val.jsonl"
```
