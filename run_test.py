import sys
import os
import json

# Add the project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.noise_detection.detect_noise_events import detect_noise_audio

if __name__ == "__main__":
    wav_file = r"c:\Users\מנהל\Documents\GitHub\Soundpatch-AI-BE\AaronKoblin_2011.wav"
    print(f"Running noise detection on: {wav_file}")
    
    result = detect_noise_audio(wav_file)
    print(json.dumps(result, indent=2))
