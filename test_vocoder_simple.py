#!/usr/bin/env python3
"""
Simple test for vocoder functionality
"""

import torch
import numpy as np
import os

print("Testing vocoder functionality...")

try:
    # Test importing vocoder_utils
    print("1. Importing vocoder_utils...")
    from logic.voice_generation.vocoder_utils import load_hifigan_model, mel_to_audio_with_hifigan
    print("   ✓ vocoder_utils imported successfully")
    
    # Test loading HiFi-GAN model
    print("2. Loading HiFi-GAN model...")
    hifigan_model, device = load_hifigan_model()
    print(f"   ✓ HiFi-GAN model loaded on device: {device}")
    
    # Test with simple mel spectrogram
    print("3. Testing with simple mel spectrogram...")
    mel_bins = 80
    time_frames = 30
    mel_spectrogram = torch.randn(time_frames, mel_bins)  # [T, 80] format
    print(f"   ✓ Created mel spectrogram: {mel_spectrogram.shape}")
    
    # Test vocoder conversion
    print("4. Converting mel to audio...")
    audio = mel_to_audio_with_hifigan(mel_spectrogram)
    print(f"   ✓ Generated audio: {audio.shape}, type: {audio.dtype}")
    
    # Test saving
    print("5. Saving audio...")
    import soundfile as sf
    sf.write("test_vocoder_output.wav", audio, 22050)
    print("   ✓ Audio saved as test_vocoder_output.wav")
    
    print("\n🎉 Vocoder test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
