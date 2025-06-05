#!/usr/bin/env python3

import os
import json
import torchaudio
import librosa
import numpy as np
from pathlib import Path
from tqdm import tqdm

def process_audio(input_path, output_path, target_sr=16000):
    """Process audio file to 16kHz mono format."""
    # Load audio
    audio, sr = librosa.load(input_path, sr=None)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    # Resample if needed
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Save processed audio
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, torch.from_numpy(audio).unsqueeze(0), target_sr)

def process_iemocap(data_dir):
    """Process IEMOCAP dataset."""
    sessions = [f"Session{i}" for i in range(1, 6)]
    emotions = {
        'ang': 'angry',
        'hap': 'happy',
        'neu': 'neutral',
        'sad': 'sad',
        'exc': 'excited',
        'fru': 'frustrated',
        'fea': 'fear',
        'sur': 'surprised',
        'dis': 'disgust'
    }
    
    data = []
    for session in sessions:
        session_dir = os.path.join(data_dir, session, 'sentences', 'wav')
        if not os.path.exists(session_dir):
            continue
            
        for wav_file in tqdm(os.listdir(session_dir), desc=f"Processing {session}"):
            if not wav_file.endswith('.wav'):
                continue
                
            # Get emotion label from filename
            emotion = emotions.get(wav_file[4:7], 'unknown')
            
            # Process audio
            input_path = os.path.join(session_dir, wav_file)
            output_path = os.path.join(data_dir, 'processed', wav_file)
            process_audio(input_path, output_path)
            
            # Add to data list
            data.append({
                "audio_path": os.path.join('processed', wav_file),
                "text": "",  # IEMOCAP has transcriptions, but they're in separate files
                "emotion": emotion,
                "instruction": "What emotion is expressed in this audio?",
                "response": f"The audio expresses a {emotion} emotion."
            })
    
    return data

def process_ravdess(data_dir):
    """Process RAVDESS dataset."""
    emotions = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }
    
    data = []
    for wav_file in tqdm(os.listdir(data_dir), desc="Processing RAVDESS"):
        if not wav_file.endswith('.wav'):
            continue
            
        # Get emotion label from filename
        emotion = emotions.get(wav_file.split('-')[2], 'unknown')
        
        # Process audio
        input_path = os.path.join(data_dir, wav_file)
        output_path = os.path.join(data_dir, 'processed', wav_file)
        process_audio(input_path, output_path)
        
        # Add to data list
        data.append({
            "audio_path": os.path.join('processed', wav_file),
            "text": "",  # RAVDESS doesn't have transcriptions
            "emotion": emotion,
            "instruction": "What emotion is expressed in this audio?",
            "response": f"The audio expresses a {emotion} emotion."
        })
    
    return data

def process_cremad(data_dir):
    """Process CREMA-D dataset."""
    emotions = {
        'ANG': 'angry',
        'DIS': 'disgust',
        'FEA': 'fear',
        'HAP': 'happy',
        'NEU': 'neutral',
        'SAD': 'sad'
    }
    
    data = []
    for wav_file in tqdm(os.listdir(data_dir), desc="Processing CREMA-D"):
        if not wav_file.endswith('.wav'):
            continue
            
        # Get emotion label from filename
        emotion = emotions.get(wav_file.split('_')[2], 'unknown')
        
        # Process audio
        input_path = os.path.join(data_dir, wav_file)
        output_path = os.path.join(data_dir, 'processed', wav_file)
        process_audio(input_path, output_path)
        
        # Add to data list
        data.append({
            "audio_path": os.path.join('processed', wav_file),
            "text": "",  # CREMA-D doesn't have transcriptions
            "emotion": emotion,
            "instruction": "What emotion is expressed in this audio?",
            "response": f"The audio expresses a {emotion} emotion."
        })
    
    return data

def main():
    base_dir = "data/pretrain/emotion"
    
    # Process each dataset
    datasets = {
        'iemocap': process_iemocap,
        'ravdess': process_ravdess,
        'cremad': process_cremad
    }
    
    for dataset_name, process_func in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        data_dir = os.path.join(base_dir, dataset_name)
        
        if not os.path.exists(data_dir):
            print(f"Directory {data_dir} does not exist. Skipping...")
            continue
            
        # Process dataset
        data = process_func(data_dir)
        
        # Save processed data
        output_file = os.path.join(data_dir, 'data.jsonl')
        with open(output_file, 'w') as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
        
        print(f"Processed {len(data)} files for {dataset_name}")

if __name__ == "__main__":
    main() 