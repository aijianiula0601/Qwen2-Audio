import numpy as np
import soundfile as sf
import os
import sys

def create_test_audio(output_dir, duration=3.0, sample_rate=16000):
    """Create simple test audio files"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create sine wave audio
    t = np.linspace(0, duration, int(duration * sample_rate))
    
    # Different frequencies for different samples
    frequencies = [440, 880, 1320]  # A4, A5, E6
    
    for i, freq in enumerate(frequencies):
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        output_path = os.path.join(output_dir, "sample{}.wav".format(i+1))
        sf.write(output_path, audio, sample_rate)
        print("Created {}".format(output_path))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_test_audio.py <output_dir>")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    create_test_audio(output_dir) 