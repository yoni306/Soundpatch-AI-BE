#!/usr/bin/env python3
"""
Test file that implements steps 5 and 6 exactly like in process_flow.py
for the word "hello"
"""

import torch
import numpy as np
import soundfile as sf
import os
from typing import Dict, List

# Import the exact functions used in process_flow.py
from logic.mel_spectogram_generation.text_to_mel_model_inference import predict_mel_from_text
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio, load_hifigan_model

def test_hello_pipeline():
    """
    Test the complete pipeline for the word "hello" exactly like in process_flow.py
    """
    print("Testing hello pipeline...")
    
    # Step 5: Run the text to mel spectrogram model (exactly like in process_flow.py)
    print("Step 5: Generating mel spectrograms from text 'hello'...")
    
    # Create a simple list with the word "hello" (simulating restored_text from step 4)
    test_texts = ["hello"]
    mel_spectrograms: List[np.ndarray] = []

    for text in test_texts:
        # Extract the text (simulating event.get("restored_text", ""))
        restored_text_content = text
        if restored_text_content:
            mel_spectrogram: np.ndarray = predict_mel_from_text(restored_text_content)
            mel_spectrograms.append(mel_spectrogram)
    
    print(f"Generated {len(mel_spectrograms)} mel spectrograms")

    # Step 6: Run the vocoder model for each mel spectrogram (exactly like in process_flow.py)
    print("Step 6: Running vocoder model...")
    
    # Create mel_predictions dictionary exactly like in process_flow.py
    mel_predictions: Dict[str, np.ndarray] = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    
    # Create output directory
    output_dir: str = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HiFi-GAN model and get device (exactly like in process_flow.py)
    hifigan_model, device = load_hifigan_model()
    
    # Call save_mel_predictions_as_audio exactly like in process_flow.py
    save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)
    
    print(f"Audio files saved to: {output_dir}")
    
    # List the generated files
    for filename in os.listdir(output_dir):
        if filename.endswith('.wav'):
            filepath = os.path.join(output_dir, filename)
            print(f"Generated: {filepath}")

def test_hello_with_exact_notebook_implementation():
    """
    Test using the exact HiFi-GAN implementation from the notebook
    """
    print("\nTesting with exact notebook HiFi-GAN implementation...")
    
    try:
        # Import the exact HiFi-GAN implementation from the notebook
        from nemo.collections.tts.models import HifiGanModel
        
        # Load HiFi-GAN model exactly like in the notebook
        hifigan_model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
        print("✓ HiFi-GAN model loaded successfully")
        
        # Get the exact functions from the notebook
        def prepare_mel_for_vocoder(
            mel_2d: torch.Tensor,
            *,
            mean: float | torch.Tensor | None = None,
            std: float  | torch.Tensor | None = None,
            hifigan_model=None,
        ) -> torch.Tensor:
            """
            Converts your current mel (as returned by text_to_mel) to the format HiFi-GAN expects.
            - Input:  [T, 80] torch.FloatTensor   (no batch)
            - Output: [1, 80, T] torch.FloatTensor (batched, on hifigan device)
            """
            assert isinstance(mel_2d, torch.Tensor) and mel_2d.ndim == 2, "expected [T, 80] torch tensor"

            x = mel_2d.detach().float()       # [T, 80]

            # 1) (Optional) de-normalize back to the original scale you trained on
            if (mean is not None) and (std is not None):
                # mean/std can be scalars or shape-[80]; broadcast handles both
                if not torch.is_tensor(mean): mean = torch.tensor(mean, dtype=x.dtype, device=x.device)
                if not torch.is_tensor(std):  std  = torch.tensor(std,  dtype=x.dtype, device=x.device)
                x = x * std + mean            # still [T, 80]

            # 2) Layout for HiFi-GAN: [T,80] -> [80,T] -> [1,80,T]
            x = x.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 80, T]

            # 3) Put on same device as vocoder
            if hifigan_model is not None:
                device = next(hifigan_model.parameters()).device
                x = x.to(device)

            return x

        @torch.no_grad()
        def mel_to_audio_with_hifigan(
            mel_2d: torch.Tensor,
            *,
            mean: float | torch.Tensor | None = None,
            std: float  | torch.Tensor | None = None,
        ) -> np.ndarray:
            """
            Convenience wrapper: takes your [T,80] mel, adapts it, and runs the vocoder.
            Returns: 1D np.float32 waveform.
            """
            spec = prepare_mel_for_vocoder(mel_2d, mean=mean, std=std, hifigan_model=hifigan_model)  # [1,80,T]
            audio = hifigan_model.convert_spectrogram_to_audio(spec=spec)  # [1, samples]
            return audio[0].detach().cpu().float().numpy()
        
        # Generate mel spectrogram for "hello"
        print("Generating mel spectrogram for 'hello'...")
        mel_spectrogram = predict_mel_from_text("hello")
        
        # Convert to torch tensor if it's numpy
        if isinstance(mel_spectrogram, np.ndarray):
            mel_spectrogram = torch.from_numpy(mel_spectrogram).float()
        
        print(f"Mel spectrogram shape: {mel_spectrogram.shape}")
        
        # Convert to audio using exact notebook implementation
        print("Converting mel to audio using HiFi-GAN...")
        audio = mel_to_audio_with_hifigan(mel_spectrogram)
        
        # Save audio
        output_dir = "test_output"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "hello_notebook_implementation.wav")
        sf.write(output_path, audio, 22050, format='WAV', subtype='PCM_16')
        
        print(f"✓ Audio saved: {output_path}")
        print(f"✓ Audio length: {len(audio)} samples ({len(audio)/22050:.2f} seconds)")
        
    except Exception as e:
        print(f"❌ Error with notebook implementation: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the exact process_flow.py implementation
    test_hello_pipeline()
    
    # Test the exact notebook implementation
    test_hello_with_exact_notebook_implementation()
    
    print("\n🎉 Hello pipeline test completed!")
