#!/usr/bin/env python3
"""
Test script to verify model loading functions work correctly
"""

def test_model_loading():
    print("Testing model loading functions...")
    
    try:
        # Test noise detection model loading
        print("\n1. Testing noise detection model loading...")
        from logic.noise_detection.detect_noise_moedl import load_noise_detectoion_model
        from config import settings
        
        noise_model = load_noise_detectoion_model(settings.DETECT_NOISE_MODEL_WEIGHTS)
        if noise_model is not None:
            print("✅ Noise detection model loaded successfully")
        else:
            print("❌ Noise detection model failed to load")
            
    except Exception as e:
        print(f"❌ Error loading noise detection model: {e}")
    
    try:
        # Test text-to-mel model loading
        print("\n2. Testing text-to-mel model loading...")
        from logic.mel_spectogram_generation.text_to_mel_inference import load_text_to_mel_model
        from config import settings
        
        text_mel_model = load_text_to_mel_model(settings.TEXT_TO_MEL_MODEL_WEIGHTS)
        if text_mel_model is not None:
            print("✅ Text-to-mel model loaded successfully")
        else:
            print("❌ Text-to-mel model failed to load")
            
    except Exception as e:
        print(f"❌ Error loading text-to-mel model: {e}")
    
    try:
        # Test wav2vec2 model loading
        print("\n3. Testing wav2vec2 model loading...")
        from logic.speaker_embedding.speaker_embedding import load_wav2vec2_model
        
        processor, model = load_wav2vec2_model()
        if processor is not None and model is not None:
            print("✅ Wav2Vec2 model loaded successfully")
        else:
            print("❌ Wav2Vec2 model failed to load")
            
    except Exception as e:
        print(f"❌ Error loading wav2vec2 model: {e}")
    
    try:
        # Test HiFi-GAN model loading
        print("\n4. Testing HiFi-GAN model loading...")
        from logic.voice_generation.vocoder_utils import load_hifigan_model
        
        model, device = load_hifigan_model()
        if model is not None:
            print("✅ HiFi-GAN model loaded successfully")
        else:
            print("❌ HiFi-GAN model failed to load")
            
    except Exception as e:
        print(f"❌ Error loading HiFi-GAN model: {e}")
    
    print("\n🎉 Model loading tests completed!")

if __name__ == "__main__":
    test_model_loading()
