import os
import json
import glob

def convert_librispeech(libri_dir, output_file):
    data = []
    
    # Find all .trans.txt files
    trans_files = glob.glob(os.path.join(libri_dir, "LibriSpeech", "train-clean-100", "*", "*", "*.trans.txt"))

    
    
    for trans_file in trans_files:
        with open(trans_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ', 1)
                if len(parts) == 2:
                    file_id, text = parts
                    audio_path = os.path.join(os.path.dirname(trans_file), f"{file_id}.flac")
                    if os.path.exists(audio_path):
                        data.append({
                            "audio_path": audio_path,
                            "text": text.lower(),
                            "instruction": "",
                            "response": text.lower()
                        })
    
    # Write data.jsonl
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')
    
    print(f"Converted {len(data)} LibriSpeech samples")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python process_librispeech.py <libri_dir> <output_file>")
        sys.exit(1)
    
    libri_dir = sys.argv[1]
    output_file = sys.argv[2]
    convert_librispeech(libri_dir, output_file) 