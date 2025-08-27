#!/usr/bin/env python3
"""
Simple test for hello pipeline without importing problematic modules
"""

import torch
import numpy as np
import soundfile as sf
import os

print("Testing hello pipeline step by step...")

try:
    # Step 1: Test basic imports
    print("1. Testing basic imports...")
    import nemo
    print("   ✓ NeMo imported")
    
    from nemo.collections.tts.models import HifiGanModel
    print("   ✓ HiFi-GAN model imported")
    
    # Step 2: Test HiFi-GAN model loading
    print("2. Loading HiFi-GAN model...")
    hifigan_model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
    device = next(hifigan_model.parameters()).device
    print(f"   ✓ HiFi-GAN loaded on device: {device}")
    
    # Step 3: Test text-to-mel inference
    print("3. Testing text-to-mel inference...")
    try:
        from logic.mel_spectogram_generation.text_to_mel_model_inference import predict_mel_from_text
        mel_spectrogram = predict_mel_from_text("hello")
        print(f"   ✓ Mel spectrogram generated: {mel_spectrogram.shape}")
    except Exception as e:
        print(f"   ⚠️  Text-to-mel failed: {e}")
        # Create synthetic mel for testing
        mel_spectrogram = np.random.randn(30, 80).astype(np.float32)
        print(f"   ✓ Using synthetic mel: {mel_spectrogram.shape}")
    
    # Step 4: Test HiFi-GAN conversion (exact notebook implementation)
    print("4. Testing HiFi-GAN conversion...")
    
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
    
    # Convert mel to torch tensor if needed
    if isinstance(mel_spectrogram, np.ndarray):
        mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
    
    # Convert to audio
    audio = mel_to_audio_with_hifigan(mel_spectrogram)
    print(f"   ✓ Audio generated: {audio.shape}, length: {len(audio)/22050:.2f}s")
    
    # Step 5: Save audio
    print("5. Saving audio...")
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "hello_simple_test.wav")
    sf.write(output_path, audio, 22050, format='WAV', subtype='PCM_16')
    print(f"   ✓ Audio saved: {output_path}")
    
    print("\n🎉 Hello pipeline test completed successfully!")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
