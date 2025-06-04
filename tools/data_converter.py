#!/usr/bin/env python3
import argparse
import json
import os
import pandas as pd
import librosa
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")


class DataConverter:
    """
    Convert various audio datasets to Qwen2-Audio training format
    """
    
    def __init__(self, output_dir: str = "data/converted"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def convert_common_voice(self, data_dir: str, split: str = "train") -> str:
        """
        Convert Common Voice dataset to training format
        Expected structure: data_dir/validated.tsv, data_dir/clips/
        """
        print(f"Converting Common Voice {split} split...")
        
        # Read metadata
        tsv_path = Path(data_dir) / "validated.tsv"
        if not tsv_path.exists():
            raise FileNotFoundError(f"Common Voice metadata not found: {tsv_path}")
        
        df = pd.read_csv(tsv_path, sep='\t')
        clips_dir = Path(data_dir) / "clips"
        
        converted_data = []
        
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            audio_path = clips_dir / row['path']
            if not audio_path.exists():
                continue
                
            # Convert audio to standard format if needed
            output_audio_path = self.output_dir / "common_voice" / split / f"{idx:06d}.wav"
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Load and save audio in standard format
                audio, sr = librosa.load(audio_path, sr=16000)
                sf.write(output_audio_path, audio, 16000)
                
                # Create training sample
                sample = {
                    "audio_path": str(output_audio_path.relative_to(self.output_dir.parent)),
                    "text": row['sentence'],
                    "speaker_id": row.get('client_id', 'unknown'),
                    "age": row.get('age', 'unknown'),
                    "gender": row.get('gender', 'unknown'),
                    "accent": row.get('accent', 'unknown'),
                    "source": "common_voice"
                }
                converted_data.append(sample)
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
        
        # Save converted data
        output_file = self.output_dir / f"common_voice_{split}.json"
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"Converted {len(converted_data)} samples to {output_file}")
        return str(output_file)
    
    def convert_librispeech(self, data_dir: str, split: str = "train-clean-100") -> str:
        """
        Convert LibriSpeech dataset to training format
        Expected structure: data_dir/speaker_id/chapter_id/speaker-chapter-utterance.flac
        """
        print(f"Converting LibriSpeech {split} split...")
        
        data_path = Path(data_dir) / split
        if not data_path.exists():
            raise FileNotFoundError(f"LibriSpeech data not found: {data_path}")
        
        converted_data = []
        file_count = 0
        
        # Walk through all audio files
        for flac_file in tqdm(list(data_path.rglob("*.flac"))):
            # Find corresponding transcript
            transcript_file = flac_file.parent / f"{flac_file.stem}.trans.txt"
            if not transcript_file.exists():
                # Look for chapter-level transcript
                chapter_transcript = flac_file.parent / f"{flac_file.parent.name}.trans.txt"
                if chapter_transcript.exists():
                    transcript_file = chapter_transcript
                else:
                    continue
            
            # Read transcript
            with open(transcript_file, 'r') as f:
                transcripts = {}
                for line in f:
                    parts = line.strip().split(' ', 1)
                    if len(parts) == 2:
                        transcripts[parts[0]] = parts[1]
            
            utterance_id = flac_file.stem
            if utterance_id not in transcripts:
                continue
            
            # Convert audio
            output_audio_path = self.output_dir / "librispeech" / split / f"{file_count:08d}.wav"
            output_audio_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Load and save audio
                audio, sr = librosa.load(flac_file, sr=16000)
                sf.write(output_audio_path, audio, 16000)
                
                # Extract speaker and chapter info
                parts = flac_file.stem.split('-')
                speaker_id = parts[0] if len(parts) > 0 else 'unknown'
                chapter_id = parts[1] if len(parts) > 1 else 'unknown'
                
                sample = {
                    "audio_path": str(output_audio_path.relative_to(self.output_dir.parent)),
                    "text": transcripts[utterance_id],
                    "speaker_id": speaker_id,
                    "chapter_id": chapter_id,
                    "utterance_id": utterance_id,
                    "source": "librispeech"
                }
                converted_data.append(sample)
                file_count += 1
                
            except Exception as e:
                print(f"Error processing {flac_file}: {e}")
                continue
        
        # Save converted data
        output_file = self.output_dir / f"librispeech_{split.replace('-', '_')}.json"
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"Converted {len(converted_data)} samples to {output_file}")
        return str(output_file)
    
    def convert_custom_dataset(self, data_dir: str, metadata_file: str = None) -> str:
        """
        Convert custom dataset to training format
        Metadata file should contain: audio_path, text, and optional fields
        """
        print(f"Converting custom dataset from {data_dir}...")
        
        data_path = Path(data_dir)
        converted_data = []
        
        if metadata_file and Path(metadata_file).exists():
            # Use provided metadata file
            with open(metadata_file, 'r') as f:
                if metadata_file.endswith('.json'):
                    metadata = json.load(f)
                elif metadata_file.endswith('.csv'):
                    metadata = pd.read_csv(metadata_file).to_dict('records')
                else:
                    raise ValueError("Metadata file must be JSON or CSV")
            
            for idx, item in tqdm(enumerate(metadata)):
                audio_path = data_path / item['audio_path']
                if not audio_path.exists():
                    continue
                
                # Convert audio
                output_audio_path = self.output_dir / "custom" / f"{idx:06d}.wav"
                output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    audio, sr = librosa.load(audio_path, sr=16000)
                    sf.write(output_audio_path, audio, 16000)
                    
                    sample = {
                        "audio_path": str(output_audio_path.relative_to(self.output_dir.parent)),
                        "text": item['text'],
                        "source": "custom"
                    }
                    
                    # Add any additional fields
                    for key, value in item.items():
                        if key not in ['audio_path', 'text']:
                            sample[key] = value
                    
                    converted_data.append(sample)
                    
                except Exception as e:
                    print(f"Error processing {audio_path}: {e}")
                    continue
        
        else:
            # Auto-discover audio files and try to find transcripts
            audio_extensions = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
            audio_files = []
            
            for ext in audio_extensions:
                audio_files.extend(data_path.rglob(f"*{ext}"))
            
            for idx, audio_file in tqdm(enumerate(audio_files)):
                # Look for corresponding text file
                text_file = audio_file.with_suffix('.txt')
                if not text_file.exists():
                    continue
                
                with open(text_file, 'r') as f:
                    text = f.read().strip()
                
                # Convert audio
                output_audio_path = self.output_dir / "custom" / f"{idx:06d}.wav"
                output_audio_path.parent.mkdir(parents=True, exist_ok=True)
                
                try:
                    audio, sr = librosa.load(audio_file, sr=16000)
                    sf.write(output_audio_path, audio, 16000)
                    
                    sample = {
                        "audio_path": str(output_audio_path.relative_to(self.output_dir.parent)),
                        "text": text,
                        "original_path": str(audio_file),
                        "source": "custom"
                    }
                    converted_data.append(sample)
                    
                except Exception as e:
                    print(f"Error processing {audio_file}: {e}")
                    continue
        
        # Save converted data
        output_file = self.output_dir / "custom_dataset.json"
        with open(output_file, 'w') as f:
            json.dump(converted_data, f, indent=2)
        
        print(f"Converted {len(converted_data)} samples to {output_file}")
        return str(output_file)
    
    def create_multi_task_dataset(self, datasets: list, output_name: str = "multi_task") -> str:
        """
        Combine multiple datasets and create multi-task training data
        """
        print("Creating multi-task dataset...")
        
        all_data = {
            "pretrain": [],
            "sft": [],
            "dpo": []
        }
        
        for dataset_info in datasets:
            dataset_path = dataset_info['path']
            dataset_type = dataset_info.get('type', 'pretrain')
            weight = dataset_info.get('weight', 1.0)
            
            with open(dataset_path, 'r') as f:
                data = json.load(f)
            
            # Apply weight by sampling
            if weight != 1.0:
                import random
                random.seed(42)
                sample_size = int(len(data) * weight)
                data = random.sample(data, min(sample_size, len(data)))
            
            all_data[dataset_type].extend(data)
        
        # Save combined datasets
        output_file = self.output_dir / f"{output_name}.json"
        with open(output_file, 'w') as f:
            json.dump(all_data, f, indent=2)
        
        total_samples = sum(len(v) for v in all_data.values())
        print(f"Created multi-task dataset with {total_samples} total samples:")
        for task, data in all_data.items():
            print(f"  {task}: {len(data)} samples")
        
        return str(output_file)


def main():
    parser = argparse.ArgumentParser(description="Convert audio datasets to Qwen2-Audio format")
    parser.add_argument("--dataset", type=str, required=True, 
                        choices=["common_voice", "librispeech", "custom", "multi_task"],
                        help="Dataset type to convert")
    parser.add_argument("--data_dir", type=str, required=True, help="Input data directory")
    parser.add_argument("--output_dir", type=str, default="data/converted", 
                        help="Output directory")
    parser.add_argument("--split", type=str, default="train", help="Dataset split")
    parser.add_argument("--metadata_file", type=str, help="Metadata file for custom dataset")
    parser.add_argument("--config_file", type=str, help="Config file for multi-task dataset")
    
    args = parser.parse_args()
    
    converter = DataConverter(args.output_dir)
    
    if args.dataset == "common_voice":
        output_file = converter.convert_common_voice(args.data_dir, args.split)
    elif args.dataset == "librispeech":
        output_file = converter.convert_librispeech(args.data_dir, args.split)
    elif args.dataset == "custom":
        output_file = converter.convert_custom_dataset(args.data_dir, args.metadata_file)
    elif args.dataset == "multi_task":
        if not args.config_file:
            raise ValueError("Config file required for multi-task dataset")
        with open(args.config_file, 'r') as f:
            config = json.load(f)
        output_file = converter.create_multi_task_dataset(config['datasets'], config.get('output_name', 'multi_task'))
    
    print(f"\nConversion completed successfully!")
    print(f"Output file: {output_file}")
    print(f"Usage: Add this file to your training config under 'datasets'")


if __name__ == "__main__":
    main() 