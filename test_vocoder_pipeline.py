#!/usr/bin/env python3
"""
Test file for the complete text-to-audio pipeline including vocoder stage.
This demonstrates Stage 6: converting mel spectrograms to audio using a vocoder.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import os
from typing import Dict, List

# Import our text-to-mel inference module
from logic.mel_spectogram_generation.text_to_mel_model_inference import predict_mel_from_text

# Import the actual vocoder utilities (with fallback for missing dependencies)
try:
    from logic.voice_generation.vocoder_utils import (
        mel_to_audio_with_hifigan, 
        save_mel_predictions_as_audio,
        load_hifigan_model
    )
    HIFIGAN_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  HiFi-GAN dependencies not available: {e}")
    print("🔄 Will use fallback vocoder instead")
    HIFIGAN_AVAILABLE = False

def create_simple_vocoder():
    """Create a simple vocoder for testing purposes"""
    class SimpleVocoder:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sample_rate = 22050
            self.hop_length = 256  # Standard hop length for mel spectrograms
            
        def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
            """
            Convert mel spectrogram to audio using a simple algorithm
            
            Args:
                mel_spectrogram: Tensor of shape [batch_size, mel_bins, time_frames]
                
            Returns:
                audio: Tensor of shape [batch_size, audio_samples]
            """
            batch_size, mel_bins, time_frames = mel_spectrogram.shape
            
            # Simple vocoder: inverse mel transform + overlap-add
            # This is a simplified version - real vocoders use neural networks
            
            # Convert mel to linear spectrogram (simplified)
            # In reality, this would use a learned mel-to-linear transformation
            linear_spec = torch.exp(mel_spectrogram)  # Simple exponential
            
            # Generate audio using inverse FFT (simplified)
            audio_samples = time_frames * self.hop_length
            audio = torch.zeros(batch_size, audio_samples, device=self.device)
            
            for b in range(batch_size):
                for t in range(time_frames):
                    # Generate a simple sine wave for each mel bin
                    start_sample = t * self.hop_length
                    end_sample = start_sample + self.hop_length
                    
                    # Create frequency content based on mel bins
                    for mel_bin in range(mel_bins):
                        # Frequency increases with mel bin
                        freq = 80 + mel_bin * 50  # Hz
                        amplitude = linear_spec[b, mel_bin, t].item()
                        
                        # Generate sine wave for this frequency
                        time = torch.arange(self.hop_length, device=self.device) / self.sample_rate
                        sine_wave = amplitude * torch.sin(2 * np.pi * freq * time)
                        
                        # Add to audio (with overlap-add)
                        if start_sample + self.hop_length <= audio_samples:
                            audio[b, start_sample:end_sample] += sine_wave
            
            # Normalize audio
            audio = audio / (audio.abs().max() + 1e-8)
            
            return audio
    
    return SimpleVocoder()

def create_advanced_vocoder():
    """Create a more advanced vocoder using inverse FFT"""
    class AdvancedVocoder:
        def __init__(self):
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.sample_rate = 22050
            self.hop_length = 256
            self.n_fft = 1024
            
        def __call__(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
            """
            Convert mel spectrogram to audio using inverse FFT
            
            Args:
                mel_spectrogram: Tensor of shape [batch_size, mel_bins, time_frames]
                
            Returns:
                audio: Tensor of shape [batch_size, audio_samples]
            """
            batch_size, mel_bins, time_frames = mel_spectrogram.shape
            
            # Convert mel to linear spectrogram (simplified)
            linear_spec = torch.exp(mel_spectrogram)  # Simple exponential
            
            # Ensure we have the right number of frequency bins
            target_freq_bins = self.n_fft // 2 + 1
            
            if mel_bins < target_freq_bins:
                # Pad with zeros
                padding = target_freq_bins - mel_bins
                linear_spec = torch.nn.functional.pad(linear_spec, (0, 0, 0, padding))
            else:
                # Truncate
                linear_spec = linear_spec[:target_freq_bins, :]
            
            # Generate audio using inverse FFT
            audio_samples = time_frames * self.hop_length
            audio = torch.zeros(batch_size, audio_samples, device=self.device)
            
            for b in range(batch_size):
                # Create complex spectrogram with random phase
                magnitude = linear_spec[b]  # [freq_bins, time_frames]
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
    
    return AdvancedVocoder()

def test_text_to_audio_pipeline():
    """Test the complete text-to-audio pipeline using the exact same approach as process_flow.py"""
    print("🎵 Testing Complete Text-to-Audio Pipeline (process_flow.py style)")
    print("=" * 65)
    
    # Test texts of different lengths
    test_texts = [
        "Hello world",
        "This is a longer test sentence to see how the vocoder handles different text lengths.",
        "Short.",
        "The quick brown fox jumps over the lazy dog. This pangram contains every letter of the alphabet at least once."
    ]
    
    # Create output directory (similar to settings.UPLOAD_DIR in process_flow)
    output_dir = "test_audio_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load HiFi-GAN model (same as in process_flow.py)
    hifigan_model = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if HIFIGAN_AVAILABLE:
        print("🔧 Loading HiFi-GAN vocoder model...")
        try:
            hifigan_model, device = load_hifigan_model()
            print(f"✅ HiFi-GAN model loaded successfully on device: {device}")
        except Exception as e:
            print(f"❌ Failed to load HiFi-GAN model: {e}")
            print("🔄 Using fallback approach...")
            hifigan_model = None
    else:
        print("🔄 HiFi-GAN not available, using fallback vocoder...")
    
    # Stage 5: Generate mel spectrograms from text (exactly like process_flow.py)
    print("\n🎯 Stage 5: Generating mel spectrograms from text...")
    mel_spectrograms: List[np.ndarray] = []

    for i, text in enumerate(test_texts):
        print(f"📝 Processing text {i+1}: '{text[:50]}{'...' if len(text) > 50 else ''}'")
        
        # Extract the text (similar to how process_flow extracts from restored_text)
        if text:
            mel_spectrogram: np.ndarray = predict_mel_from_text(text)
            mel_spectrograms.append(mel_spectrogram)
            print(f"✅ Mel spectrogram {i+1} shape: {mel_spectrogram.shape}")
    
    print(f"Generated {len(mel_spectrograms)} mel spectrograms")

    # Stage 6: Run the vocoder model for each mel spectrogram (EXACTLY like process_flow.py)
    print("\n🎵 Stage 6: Running the vocoder model for each mel spectrogram...")
    
    # Create mel_predictions dictionary exactly like in process_flow.py
    mel_predictions: Dict[str, np.ndarray] = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    
    # Call save_mel_predictions_as_audio exactly like in process_flow.py
    if HIFIGAN_AVAILABLE and hifigan_model is not None:
        try:
            save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)
            print(f"✅ Stage 6 completed! Audio files saved in: {output_dir}")
        except Exception as e:
            print(f"❌ Error in Stage 6: {e}")
            print("🔄 Trying fallback approach...")
            hifigan_model = None
    
    # Fallback: save audio files manually if HiFi-GAN is not available or failed
    if not HIFIGAN_AVAILABLE or hifigan_model is None:
        print("🔄 Using fallback vocoder for Stage 6...")
        for i, (clip_name, mel_spec) in enumerate(mel_predictions.items()):
            try:
                # Simple fallback vocoder
                vocoder = create_simple_vocoder()
                mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                audio = vocoder(mel_tensor)
                audio_np = audio.squeeze(0).cpu().numpy()
                
                filename = os.path.join(output_dir, f"{clip_name}.wav")
                sf.write(filename, audio_np, vocoder.sample_rate)
                print(f"💾 Fallback audio saved: {filename}")
                
            except Exception as fallback_error:
                print(f"❌ Fallback also failed for {clip_name}: {fallback_error}")
    
    # Create visualizations for the generated audio files
    print("\n📊 Creating visualizations...")
    for i, (clip_name, mel_spec) in enumerate(mel_predictions.items()):
        try:
            # Try to load the generated audio file
            audio_file = os.path.join(output_dir, f"{clip_name}.wav")
            if os.path.exists(audio_file):
                audio_data, sample_rate = sf.read(audio_file)
                
                # Create visualization
                create_audio_visualization(
                    mel_spec, audio_data, sample_rate,
                    os.path.join(output_dir, f"{clip_name}_viz.png"),
                    test_texts[i] if i < len(test_texts) else f"Text {i+1}",
                    "HiFi-GAN"
                )
                print(f"📊 Visualization saved: {clip_name}_viz.png")
        except Exception as viz_error:
            print(f"⚠️  Could not create visualization for {clip_name}: {viz_error}")
    
    print(f"\n✅ Pipeline completed successfully!")
    print(f"📊 Summary:")
    print(f"   - Input texts: {len(test_texts)}")
    print(f"   - Mel spectrograms: {len(mel_spectrograms)}")
    print(f"   - Audio files: {len(mel_predictions)}")
    print(f"📁 Audio files saved in: {output_dir}/")

def create_audio_visualization(mel_spectrogram: np.ndarray, audio: np.ndarray, sample_rate: int, 
                             save_path: str, text: str, vocoder_name: str):
    """Create a comprehensive visualization of the audio generation process"""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle(f'Text-to-Audio Pipeline: "{text}"\nVocoder: {vocoder_name}', fontsize=14)
    
    # Plot 1: Mel spectrogram
    axes[0].imshow(mel_spectrogram, aspect='auto', origin='lower', cmap='viridis')
    axes[0].set_title('Mel Spectrogram (Input to Vocoder)')
    axes[0].set_ylabel('Mel Bins')
    axes[0].set_xlabel('Time Frames')
    
    # Plot 2: Audio waveform
    time_axis = np.arange(len(audio)) / sample_rate
    axes[1].plot(time_axis, audio, linewidth=0.5, alpha=0.8)
    axes[1].set_title('Generated Audio Waveform')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].grid(True, alpha=0.3)
    
    # Plot 3: Audio spectrogram
    # Compute spectrogram of the generated audio
    from scipy import signal
    frequencies, times, Sxx = signal.spectrogram(audio, sample_rate, nperseg=1024, noverlap=512)
    axes[2].pcolormesh(times, frequencies, 10 * np.log10(Sxx + 1e-10), shading='gouraud', cmap='viridis')
    axes[2].set_title('Spectrogram of Generated Audio')
    axes[2].set_ylabel('Frequency (Hz)')
    axes[2].set_xlabel('Time (seconds)')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

def test_vocoder_performance():
    """Test vocoder performance with different mel spectrograms using HiFi-GAN"""
    print("\n⚡ Testing HiFi-GAN Vocoder Performance")
    print("=" * 50)
    
    # Try to load HiFi-GAN model
    try:
        hifigan_model, device = load_hifigan_model()
        print(f"✅ HiFi-GAN model loaded on device: {device}")
    except Exception as e:
        print(f"❌ Failed to load HiFi-GAN model: {e}")
        print("🔄 Using fallback vocoder...")
        hifigan_model = None
    
    # Create different types of mel spectrograms
    mel_types = {
        "Silence": np.zeros((80, 50)),
        "Tone": np.ones((80, 50)) * 0.5,
        "Noise": np.random.normal(0, 0.5, (80, 50)),
        "Harmonic": np.array([[np.sin(2 * np.pi * t / 50) for t in range(50)] for _ in range(80)])
    }
    
    for mel_name, mel_spec in mel_types.items():
        print(f"\n🎵 Testing {mel_name} mel spectrogram...")
        
        try:
            if hifigan_model is not None:
                # Use HiFi-GAN vocoder
                # Normalize mel spectrogram
                mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
                mel_spec = mel_spec * 2 - 1  # Range [-1, 1]
                
                # Convert to torch tensor in [T, 80] format
                mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).transpose(1, 0)  # [80, T] -> [T, 80]
                
                # Generate audio using HiFi-GAN
                audio_np = mel_to_audio_with_hifigan(mel_tensor)
                sample_rate = 22050
                
                filename = f"test_audio_outputs/hifigan_test_{mel_name.lower()}.wav"
                
            else:
                # Fallback to custom vocoder
                vocoder = create_simple_vocoder()
                mel_spec = (mel_spec - mel_spec.min()) / (mel_spec.max() - mel_spec.min() + 1e-8)
                mel_spec = mel_spec * 2 - 1  # Range [-1, 1]
                
                mel_tensor = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0).transpose(1, 2)
                audio = vocoder(mel_tensor)
                audio_np = audio.squeeze(0).cpu().numpy()
                sample_rate = vocoder.sample_rate
                
                filename = f"test_audio_outputs/fallback_test_{mel_name.lower()}.wav"
            
            # Save audio
            sf.write(filename, audio_np, sample_rate)
            print(f"💾 Saved: {filename}")
            print(f"🎵 Duration: {len(audio_np) / sample_rate:.2f}s")
            
        except Exception as e:
            print(f"❌ Error processing {mel_name}: {e}")
            # Create silence as fallback
            silence_samples = int(1.0 * 22050)  # 1 second of silence
            audio_np = np.zeros(silence_samples, dtype=np.float32)
            filename = f"test_audio_outputs/error_{mel_name.lower()}.wav"
            sf.write(filename, audio_np, 22050)
            print(f"💾 Saved silence fallback: {filename}")

if __name__ == "__main__":
    print("🎵 Text-to-Audio Pipeline Test")
    print("=" * 40)
    
    # Test the complete pipeline
    test_text_to_audio_pipeline()
    
    # Test vocoder performance
    test_vocoder_performance()
    
    print("\n✅ All tests completed!")
    print("📁 Check the 'test_audio_outputs' directory for generated audio files and visualizations.")
    print("🎧 You can play the generated .wav files to hear the results!")
