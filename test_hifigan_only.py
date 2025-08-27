#!/usr/bin/env python3
"""
Test HiFi-GAN functionality only, without text-to-mel imports
"""

import torch
import numpy as np
import soundfile as sf
import os

print("Testing HiFi-GAN functionality only...")

try:
    # Test HiFi-GAN import and loading
    print("1. Loading HiFi-GAN model...")
    from nemo.collections.tts.models import HifiGanModel
    hifigan_model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
    device = next(hifigan_model.parameters()).device
    print(f"   ✓ HiFi-GAN loaded on device: {device}")
    
    # Create a synthetic mel spectrogram for testing
    print("2. Creating synthetic mel spectrogram...")
    mel_bins = 80
    time_frames = 50
    mel_spectrogram = torch.randn(time_frames, mel_bins).float()  # [T, 80] format
    print(f"   ✓ Created mel: {mel_spectrogram.shape}")
    
    # Test the exact notebook implementation
    print("3. Testing exact notebook HiFi-GAN implementation...")
    
    def prepare_mel_for_vocoder(mel_2d, mean=None, std=None, hifigan_model=None):
        """Exact implementation from notebook"""
        assert isinstance(mel_2d, torch.Tensor) and mel_2d.ndim == 2, "expected [T, 80] torch tensor"
        x = mel_2d.detach().float()
        
        if (mean is not None) and (std is not None):
            if not torch.is_tensor(mean): mean = torch.tensor(mean, dtype=x.dtype, device=x.device)
            if not torch.is_tensor(std):  std  = torch.tensor(std,  dtype=x.dtype, device=x.device)
            x = x * std + mean
        
        x = x.transpose(0, 1).contiguous().unsqueeze(0)
        
        if hifigan_model is not None:
            device = next(hifigan_model.parameters()).device
            x = x.to(device)
        
        return x

    @torch.no_grad()
    def mel_to_audio_with_hifigan(mel_2d, mean=None, std=None):
        """Exact implementation from notebook"""
        spec = prepare_mel_for_vocoder(mel_2d, mean=mean, std=std, hifigan_model=hifigan_model)
        audio = hifigan_model.convert_spectrogram_to_audio(spec=spec)
        return audio[0].detach().cpu().float().numpy()
    
    # Convert mel to audio
    print("4. Converting mel to audio...")
    audio = mel_to_audio_with_hifigan(mel_spectrogram)
    print(f"   ✓ Audio generated: {audio.shape}, length: {len(audio)/22050:.2f}s")
    
    # Save audio
    print("5. Saving audio...")
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hifigan_test_synthetic.wav")
    sf.write(output_path, audio, 22050, format='WAV', subtype='PCM_16')
    print(f"   ✓ Audio saved: {output_path}")
    
    # Now test with a more realistic mel for "hello"
    print("6. Testing with realistic mel for 'hello'...")
    # Create a more realistic mel that might represent "hello"
    hello_mel = torch.zeros(40, 80).float()
    # Add some frequency content that might represent speech
    for t in range(40):
        for f in range(80):
            # Create some harmonic content
            freq_factor = 1.0 - (f / 80) * 0.8
            time_factor = np.sin(2 * np.pi * t / 40) * 0.3 + 0.7
            noise = np.random.normal(0, 0.1)
            hello_mel[t, f] = freq_factor * time_factor + noise
    
    # Normalize
    hello_mel = (hello_mel - hello_mel.min()) / (hello_mel.max() - hello_mel.min())
    hello_mel = hello_mel * 2 - 1  # Scale to [-1, 1]
    
    print(f"   ✓ Created hello mel: {hello_mel.shape}")
    
    # Convert to audio
    hello_audio = mel_to_audio_with_hifigan(hello_mel)
    hello_output_path = os.path.join(output_dir, "hello_hifigan_test.wav")
    sf.write(hello_output_path, hello_audio, 22050, format='WAV', subtype='PCM_16')
    print(f"   ✓ Hello audio saved: {hello_output_path}")
    print(f"   ✓ Hello audio length: {len(hello_audio)/22050:.2f}s")
    
    print("\n🎉 HiFi-GAN test completed successfully!")
    print(f"Generated files:")
    print(f"  - {output_path}")
    print(f"  - {hello_output_path}")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
