"""
Simple Text-to-Speech Implementation
Uses existing vocoder_utils functionality to convert text to speech
"""

import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
print("Hello, world!")

# Import the required modules from the existing codebase
from logic.mel_spectogram_generation.text_to_mel_model_inference import predict_mel_from_text
print("Hello1")
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio, load_hifigan_model
print("Hello2")
print("Hello, world!")

def simple_text_to_speech(text: str = "hello", output_dir: str = "test_audio_outputs"):
    """
    Simple text-to-speech function using existing vocoder utilities
    
    Args:
        text (str): Input text to convert to speech (default: "hello")
        output_dir (str): Directory to save the generated audio files
    """
    print(f"🎵 Converting text to speech: '{text}'")
    
    try:
        # Step 1: Generate mel spectrogram from text
        print("📊 Generating mel spectrogram...")
        mel_spectrogram = predict_mel_from_text(text)
        print(f"✅ Mel spectrogram shape: {mel_spectrogram.shape}")
        
        # Step 2: Create mel_predictions dictionary (like in process_flow.py)
        mel_predictions = {"simple_text": mel_spectrogram}
        
        # Step 3: Load HiFi-GAN model
        print("🎵 Loading HiFi-GAN model...")
        hifigan_model, device = load_hifigan_model()
        
        # Step 4: Convert mel spectrograms to audio using save_mel_predictions_as_audio
        print("🎵 Converting to audio using save_mel_predictions_as_audio...")
        save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)
        
        # Step 5: Create visualization
        output_path = os.path.join(output_dir, "simple_text.wav")
        viz_path = output_path.replace('.wav', '_mel.png')
        plt.figure(figsize=(10, 6))
        plt.imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
        plt.title(f'Mel Spectrogram: "{text}"')
        plt.xlabel('Time Frames')
        plt.ylabel('Mel Frequency Bins')
        plt.colorbar()
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Success! Audio: {output_path}, Visualization: {viz_path}")
        
    except Exception as e:
        print(f"❌ Error: {e}")


simple_text_to_speech("hello")