#!/usr/bin/env python3

import os
import json
import csv
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

def process_esc50(data_dir):
    """Process ESC-50 dataset."""
    esc50_dir = os.path.join(data_dir, 'ESC-50-master')
    if not os.path.exists(esc50_dir):
        return []
    
    # Load metadata
    metadata_path = os.path.join(esc50_dir, 'meta', 'esc50.csv')
    categories = {}
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            categories[row['filename']] = row['category']
    
    data = []
    audio_dir = os.path.join(esc50_dir, 'audio')
    for wav_file in tqdm(os.listdir(audio_dir), desc="Processing ESC-50"):
        if not wav_file.endswith('.wav'):
            continue
        
        category = categories.get(wav_file, 'unknown')
        
        # Process audio
        input_path = os.path.join(audio_dir, wav_file)
        output_path = os.path.join(data_dir, 'processed', wav_file)
        process_audio(input_path, output_path)
        
        # Add to data list
        data.append({
            "audio_path": os.path.join('processed', wav_file),
            "text": "",
            "category": category,
            "instruction": "What sound is this?",
            "response": f"This is the sound of {category}."
        })
    
    return data

def process_urbansound8k(data_dir):
    """Process UrbanSound8K dataset."""
    if not os.path.exists(data_dir):
        return []
    
    # Load metadata
    metadata_path = os.path.join(data_dir, 'metadata', 'UrbanSound8K.csv')
    if not os.path.exists(metadata_path):
        return []
    
    categories = {
        '0': 'air_conditioner',
        '1': 'car_horn',
        '2': 'children_playing',
        '3': 'dog_bark',
        '4': 'drilling',
        '5': 'engine_idling',
        '6': 'gun_shot',
        '7': 'jackhammer',
        '8': 'siren',
        '9': 'street_music'
    }
    
    data = []
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader, desc="Processing UrbanSound8K"):
            fold = row['fold']
            filename = row['slice_file_name']
            category = categories.get(row['class'], 'unknown')
            
            # Process audio
            input_path = os.path.join(data_dir, 'audio', f'fold{fold}', filename)
            output_path = os.path.join(data_dir, 'processed', filename)
            process_audio(input_path, output_path)
            
            # Add to data list
            data.append({
                "audio_path": os.path.join('processed', filename),
                "text": "",
                "category": category,
                "instruction": "What sound is this?",
                "response": f"This is the sound of {category}."
            })
    
    return data

def process_audioset(data_dir):
    """Process AudioSet dataset."""
    if not os.path.exists(data_dir):
        return []
    
    # Load ontology
    ontology_path = os.path.join(data_dir, 'ontology.json')
    if not os.path.exists(ontology_path):
        return []
    
    with open(ontology_path, 'r') as f:
        ontology = json.load(f)
    
    # Create category mapping
    categories = {item['id']: item['name'] for item in ontology}
    
    data = []
    eval_segments_path = os.path.join(data_dir, 'eval_segments.csv')
    if os.path.exists(eval_segments_path):
        with open(eval_segments_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in tqdm(reader, desc="Processing AudioSet"):
                youtube_id = row['YTID']
                category_ids = row['positive_labels'].split(',')
                
                # Get audio file path
                input_path = os.path.join(data_dir, 'audio', f'{youtube_id}.wav')
                if not os.path.exists(input_path):
                    continue
                
                # Process audio
                output_path = os.path.join(data_dir, 'processed', f'{youtube_id}.wav')
                process_audio(input_path, output_path)
                
                # Get category names
                category_names = [categories.get(cat_id, 'unknown') for cat_id in category_ids]
                category_text = ', '.join(category_names)
                
                # Add to data list
                data.append({
                    "audio_path": os.path.join('processed', f'{youtube_id}.wav'),
                    "text": "",
                    "category": category_text,
                    "instruction": "What sounds can you hear in this audio?",
                    "response": f"This audio contains the following sounds: {category_text}."
                })
    
    return data

def main():
    base_dir = "data/pretrain/classification"
    
    # Process each dataset
    datasets = {
        'esc50': process_esc50,
        'urbansound8k': process_urbansound8k,
        'audioset': process_audioset
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