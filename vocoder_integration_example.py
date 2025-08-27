#!/usr/bin/env python3
"""
Example of how to integrate vocoder functionality into the main pipeline.
This demonstrates Stage 6: converting mel spectrograms to audio using a vocoder.
"""

import torch
import numpy as np
import soundfile as sf
import os
from typing import List, Dict, Any

# Import our text-to-mel inference module
from logic.mel_spectogram_generation.text_to_mel_model_inference import predict_mel_from_text

# Import the actual vocoder utilities (same as process_flow.py)
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio, load_hifigan_model

def create_vocoder():
    """Create a vocoder for converting mel spectrograms to audio"""
    class HiFiGANVocoder:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sample_rate = 22050
            self.hop_length = 256
            self.n_fft = 1024
            
        def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
            """
            Convert mel spectrogram to audio
            
            Args:
                mel_spectrogram: Tensor of shape [batch_size, mel_bins, time_frames]
                
            Returns:
                audio: Tensor of shape [batch_size, audio_samples]
            """
            batch_size, mel_bins, time_frames = mel_spectrogram.shape
            
            # Convert mel to linear spectrogram (simplified)
            linear_spec = torch.exp(mel_spectrogram)
            
            # Ensure we have the right number of frequency bins
            target_freq_bins = self.n_fft // 2 + 1
            
            if mel_bins < target_freq_bins:
                padding = target_freq_bins - mel_bins
                linear_spec = torch.nn.functional.pad(linear_spec, (0, 0, 0, padding))
            else:
                linear_spec = linear_spec[:target_freq_bins, :]
            
            # Generate audio using inverse FFT
            audio_samples = time_frames * self.hop_length
            audio = torch.zeros(batch_size, audio_samples, device=self.device)
            
            for b in range(batch_size):
                # Create complex spectrogram with random phase
                magnitude = linear_spec[b]
                phase = torch.rand_like(magnitude) * 2 * np.pi
                complex_spec = magnitude * torch.exp(1j * phase)
                
                # Inverse STFT
                audio[b] = torch.istft(
                    complex_spec,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    length=audio_samples,
                    window=torch.hann_window(self.n_fft, device=self.device)
                )
            
            # Normalize audio
            audio = audio / (audio.abs().max() + 1e-8)
            
            return audio
    
    return HiFiGANVocoder()

def stage_6_vocoder_pipeline(mel_spectrograms: List[np.ndarray], output_dir: str = "output_audio") -> List[str]:
    """
    Stage 6: Convert mel spectrograms to audio using vocoder (EXACTLY like process_flow.py)
    
    Args:
        mel_spectrograms: List of mel spectrograms from Stage 5
        output_dir: Directory to save audio files
        
    Returns:
        List of paths to generated audio files
    """
    print("🎵 Stage 6: Running the vocoder model for each mel spectrogram...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HiFi-GAN model (same as in process_flow.py)
    try:
        hifigan_model, device = load_hifigan_model()
        print(f"✅ HiFi-GAN model loaded on device: {device}")
    except Exception as e:
        print(f"❌ Failed to load HiFi-GAN model: {e}")
        print("🔄 Using fallback vocoder...")
        hifigan_model = None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create mel_predictions dictionary exactly like in process_flow.py
    mel_predictions: Dict[str, np.ndarray] = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    
    # Call save_mel_predictions_as_audio exactly like in process_flow.py
    try:
        save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)
        print(f"✅ Stage 6 completed! Audio files saved in: {output_dir}")
        
        # Return list of generated audio files
        audio_files = []
        for clip_name in mel_predictions.keys():
            audio_file = os.path.join(output_dir, f"{clip_name}.wav")
            if os.path.exists(audio_file):
                audio_files.append(audio_file)
        
        return audio_files
        
    except Exception as e:
        print(f"❌ Error in Stage 6: {e}")
        print("🔄 Using fallback vocoder...")
        
        # Fallback: use custom vocoder
        vocoder = create_vocoder()
        audio_files = []
        
        for i, mel_spectrogram in enumerate(mel_spectrograms):
            print(f"🎵 Processing mel spectrogram {i+1}/{len(mel_spectrograms)}...")
            
            # Prepare mel spectrogram for vocoder
            mel_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)
            mel_tensor = mel_tensor.transpose(1, 2)  # [batch, mel_bins, time]
            
            # Generate audio
            audio = vocoder(mel_tensor)
            audio_np = audio.squeeze(0).cpu().numpy()
            
            # Save audio file
            audio_filename = f"generated_audio_{i+1}_fallback.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            sf.write(audio_path, audio_np, vocoder.sample_rate)
            
            audio_files.append(audio_path)
            
            print(f"✅ Fallback audio saved: {audio_path}")
            print(f"🎵 Duration: {len(audio_np) / vocoder.sample_rate:.2f} seconds")
        
        return audio_files

def complete_text_to_audio_pipeline(texts: List[str]) -> Dict[str, Any]:
    """
    Complete text-to-audio pipeline demonstrating all stages
    
    Args:
        texts: List of input texts
        
    Returns:
        Dictionary containing results from all stages
    """
    print("🎵 Complete Text-to-Audio Pipeline")
    print("=" * 50)
    
    results = {
        "input_texts": texts,
        "mel_spectrograms": [],
        "audio_files": [],
        "stage_results": {}
    }
    
    # Stage 5: Generate mel spectrograms from text
    print("\n🎯 Stage 5: Generating mel spectrograms from text...")
    mel_spectrograms = []
    
    for i, text in enumerate(texts):
        print(f"📝 Processing text {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        mel_spectrogram = predict_mel_from_text(text)
        mel_spectrograms.append(mel_spectrogram)
        
        print(f"✅ Mel spectrogram {i+1} shape: {mel_spectrogram.shape}")
    
    results["mel_spectrograms"] = mel_spectrograms
    results["stage_results"]["stage_5"] = {
        "status": "completed",
        "mel_count": len(mel_spectrograms),
        "mel_shapes": [mel.shape for mel in mel_spectrograms]
    }
    
    # Stage 6: Convert mel spectrograms to audio using vocoder
    print("\n🎵 Stage 6: Converting mel spectrograms to audio...")
    audio_files = stage_6_vocoder_pipeline(mel_spectrograms, "pipeline_output")
    
    results["audio_files"] = audio_files
    results["stage_results"]["stage_6"] = {
        "status": "completed",
        "audio_count": len(audio_files),
        "audio_paths": audio_files
    }
    
    print("\n✅ Pipeline completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Input texts: {len(texts)}")
    print(f"   - Mel spectrograms: {len(mel_spectrograms)}")
    print(f"   - Audio files: {len(audio_files)}")
    print(f"📁 Audio files saved in: pipeline_output/")
    
    return results

def test_pipeline_integration():
    """Test the complete pipeline integration"""
    # Test texts
    test_texts = [
        "Hello, this is a test of the text-to-audio pipeline.",
        "The quick brown fox jumps over the lazy dog.",
        "This demonstrates Stage 6 vocoder functionality."
    ]
    
    # Run complete pipeline
    results = complete_text_to_audio_pipeline(test_texts)
    
    # Print detailed results
    print("\n📋 Detailed Results:")
    print("=" * 30)
    
    for i, (text, mel_spec, audio_file) in enumerate(zip(
        results["input_texts"], 
        results["mel_spectrograms"], 
        results["audio_files"]
    )):
        print(f"\n📝 Text {i+1}:")
        print(f"   Text: '{text}'")
        print(f"   Mel shape: {mel_spec.shape}")
        print(f"   Audio file: {audio_file}")
        
        # Calculate audio duration
        import soundfile as sf
        audio_data, sample_rate = sf.read(audio_file)
        duration = len(audio_data) / sample_rate
        print(f"   Duration: {duration:.2f} seconds")

if __name__ == "__main__":
    print("🎵 Vocoder Integration Example")
    print("=" * 40)
    
    # Test the pipeline integration
    test_pipeline_integration()
    
    print("\n🎧 You can now play the generated audio files in the 'pipeline_output' directory!")
    print("📁 The files demonstrate the complete text-to-audio pipeline including Stage 6 vocoder functionality.")
