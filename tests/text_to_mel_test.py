import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from logic.factory.model_factory import ModelFactory
import torch
from typing import List, Dict
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio
from logic.mel_spectogram_generation.text_to_mel_inference import predict_mel_from_text
import numpy as np
import soundfile as sf
from tqdm import tqdm


print(settings.TEXT_TO_MEL_MODEL_WEIGHTS)

# Initialize models
(
    detect_noise_model,
    wav2vec2_processor,
    wav2vec2_model,
    text_to_mel_model,
    hifigan_model,
    device
) = ModelFactory.setup_models()


def test_text_to_mel_pipeline(
    text: str = "Hello, this is a test",
    output_dir: str = "test_outputs",
    sample_rate: int = 22050,
    use_phonemes: bool = True
):
    """
    Complete test pipeline: Text -> Mel Spectrogram -> Audio -> Save WAV file
    
    Args:
        text: Input text to convert to speech
        output_dir: Directory to save output files
        sample_rate: Audio sample rate
        use_phonemes: Whether to use phoneme conversion
    
    Returns:
        dict: Results containing file paths and metadata
    """
    print("🎵 Testing Text-to-Mel Pipeline")
    print("=" * 50)
    print(f"Input text: {text}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Using phonemes: {use_phonemes}")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Step 1: Generate mel spectrogram from text
        print("\n1️⃣ Generating mel spectrogram from text...")
        mel_spectrogram = predict_mel_from_text(text, text_to_mel_model, use_phonemes)
        print(f"✅ Mel spectrogram generated with shape: {mel_spectrogram.shape}")
        
        # Step 2: Save mel spectrogram as numpy array for inspection
        mel_np_path = os.path.join(output_dir, "mel_spectrogram.npy")
        np.save(mel_np_path, mel_spectrogram.cpu().numpy())
        print(f"✅ Mel spectrogram saved to: {mel_np_path}")
        
        # Step 3: Convert mel spectrogram to audio using HiFi-GAN
        print("\n2️⃣ Converting mel spectrogram to audio...")
        
        # Generate audio using HiFi-GAN directly
        # Prepare mel for vocoder (manual implementation)
        mel_for_vocoder = mel_spectrogram.detach().float()
        mel_for_vocoder = mel_for_vocoder.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 80, T]
        mel_for_vocoder = mel_for_vocoder.to(next(hifigan_model.parameters()).device)
        
        # Convert to audio using HiFi-GAN
        with torch.no_grad():
            audio = hifigan_model.convert_spectrogram_to_audio(spec=mel_for_vocoder)
            audio_np = audio[0].detach().cpu().float().numpy()
        
        print(f"✅ Audio generated with length: {len(audio_np)} samples ({len(audio_np)/sample_rate:.2f} seconds)")
        
        # Step 4: Save audio as WAV file
        print("\n3️⃣ Saving audio to WAV file...")
        
        # Create a safe filename from the text
        import re
        safe_text = re.sub(r'[^\w\s-]', '', text.lower())
        safe_text = re.sub(r'[-\s]+', '_', safe_text)
        safe_text = safe_text[:50]  # Limit length
        
        wav_filename = f"test_audio_{safe_text}.wav"
        wav_path = os.path.join(output_dir, wav_filename)
        
        sf.write(wav_path, audio_np, sample_rate, format='WAV', subtype='PCM_16')
        print(f"✅ Audio saved to: {wav_path}")
        
        # Step 5: Generate results summary
        results = {
            "input_text": text,
            "mel_spectrogram_shape": mel_spectrogram.shape,
            "audio_length_samples": len(audio_np),
            "audio_duration_seconds": len(audio_np) / sample_rate,
            "sample_rate": sample_rate,
            "mel_file": mel_np_path,
            "audio_file": wav_path,
            "device_used": str(device),
            "phonemes_used": use_phonemes,
            "success": True
        }
        
        print("\n🎉 Pipeline completed successfully!")
        print("=" * 50)
        print(f"📊 Results Summary:")
        print(f"   • Mel spectrogram shape: {results['mel_spectrogram_shape']}")
        print(f"   • Audio duration: {results['audio_duration_seconds']:.2f} seconds")
        print(f"   • Sample rate: {results['sample_rate']} Hz")
        print(f"   • Output files:")
        print(f"     - Mel spectrogram: {results['mel_file']}")
        print(f"     - Audio file: {results['audio_file']}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return {
            "success": False,
            "error": str(e),
            "input_text": text
        }


def test_multiple_texts():
    """
    Test the pipeline with multiple different texts
    """
    test_texts = [
        "Hello world, this is a test.",
        "The quick brown fox jumps over the lazy dog.",
        "Testing the text to speech pipeline.",
        "Artificial intelligence is transforming the world.",
        "Speech synthesis technology has come a long way."
    ]
    
    results = []
    
    for i, text in enumerate(test_texts):
        print(f"\n🧪 Test {i+1}/{len(test_texts)}")
        print(f"Text: {text}")
        
        result = test_text_to_mel_pipeline(
            text=text,
            output_dir=f"test_outputs/test_{i+1}",
            use_phonemes=True
        )
        
        results.append(result)
        
        if result["success"]:
            print(f"✅ Test {i+1} passed!")
        else:
            print(f"❌ Test {i+1} failed: {result['error']}")
    
    # Summary
    successful_tests = sum(1 for r in results if r["success"])
    print(f"\n📈 Test Summary: {successful_tests}/{len(test_texts)} tests passed")
    
    return results


def test_mel_spectrogram_quality(mel_spectrogram: torch.Tensor):
    """
    Analyze the quality of the generated mel spectrogram
    
    Args:
        mel_spectrogram: Generated mel spectrogram tensor
    
    Returns:
        dict: Quality metrics
    """
    print("\n🔍 Analyzing mel spectrogram quality...")
    
    # Convert to numpy for analysis
    mel_np = mel_spectrogram.cpu().numpy()
    
    # Basic statistics
    mean_val = np.mean(mel_np)
    std_val = np.std(mel_np)
    min_val = np.min(mel_np)
    max_val = np.max(mel_np)
    
    # Check for NaN or infinite values
    has_nan = np.any(np.isnan(mel_np))
    has_inf = np.any(np.isinf(mel_np))
    
    # Check for reasonable range (typical mel spectrogram values)
    reasonable_range = (min_val >= -20) and (max_val <= 20)
    
    quality_metrics = {
        "shape": mel_spectrogram.shape,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "reasonable_range": reasonable_range,
        "time_steps": mel_spectrogram.shape[0],
        "mel_bins": mel_spectrogram.shape[1]
    }
    
    print(f"   • Shape: {quality_metrics['shape']}")
    print(f"   • Mean: {quality_metrics['mean']:.4f}")
    print(f"   • Std: {quality_metrics['std']:.4f}")
    print(f"   • Range: [{quality_metrics['min']:.4f}, {quality_metrics['max']:.4f}]")
    print(f"   • Has NaN: {quality_metrics['has_nan']}")
    print(f"   • Has Inf: {quality_metrics['has_inf']}")
    print(f"   • Reasonable range: {quality_metrics['reasonable_range']}")
    
    return quality_metrics


def quick_test():
    """
    Quick test function that can be called directly to test the pipeline
    """
    print("🚀 Starting Quick Text-to-Mel Test")
    print("=" * 40)
    
    # Test with a simple text
    result = test_text_to_mel_pipeline(
        text="Hello, this is a quick test of the text to speech system.",
        output_dir="quick_test_output",
        use_phonemes=True
    )
    
    if result["success"]:
        print("\n✅ Quick test passed!")
        
        # Analyze the generated mel spectrogram quality
        if "mel_spectrogram" in locals():
            test_mel_spectrogram_quality(mel_spectrogram)
    else:
        print(f"\n❌ Quick test failed: {result['error']}")
    
    return result


if __name__ == "__main__":
    # Run quick test when script is executed directly
    quick_test()

