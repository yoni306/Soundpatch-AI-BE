#!/usr/bin/env python
"""
Complete pipeline test: Noise Detection → Transcription → LLM Restoration
"""
import os
import sys
import json

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.noise_detection.detect_noise_events import detect_noise_audio
from logic.transcription.transcribe_events import clip_and_transcribe_events
from logic.text_generation.restore_missing_text import restore_missing_text_via_rest
from config import settings

def test_complete_pipeline():
    """Test the complete pipeline"""
    # Use generic path to wav folder in project
    wav_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wav")
    
    if not os.path.exists(wav_folder):
        print(f"❌ WAV folder not found: {wav_folder}")
        return
    
    # Find first .wav file in the folder
    wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith('.wav')]
    if not wav_files:
        print(f"❌ No WAV files found in: {wav_folder}")
        return
    
    test_wav = os.path.join(wav_folder, wav_files[0])
    print("🚀 Starting complete pipeline test...")
    print(f"📁 Processing: {wav_files[0]} from wav folder")
    print("=" * 80)
    
    # Step 1: Noise Detection
    print("🔍 Step 1: Detecting noise events...")
    noise_result = detect_noise_audio(test_wav)
    
    if "error" in noise_result:
        print(f"❌ Noise detection failed: {noise_result['error']}")
        return
    
    events = noise_result["events"]
    print(f"✅ Found {len(events)} noise events (3+ seconds)")
    
    if len(events) == 0:
        print("ℹ️ No significant noise events found. Pipeline complete.")
        return
    
    # Print detected events
    print("\n📋 Detected Events:")
    for i, event in enumerate(events):
        print(f"  {i+1}. {event['start_time']:.2f}s-{event['end_time']:.2f}s "
              f"({event['duration']:.2f}s) - {event['noise_type']} "
              f"(confidence: {event['confidence']:.3f})")
    
    print("=" * 80)
    
    # Step 2: Transcription
    print("🎤 Step 2: Transcribing audio segments...")
    print("📝 This may take a few minutes...")
    
    try:
        clip_results = clip_and_transcribe_events(events, test_wav, settings.ASSEMBLYAI_API_KEY)
        print(f"✅ Transcribed {len(clip_results)} clips")
    except Exception as e:
        print(f"❌ Transcription failed: {e}")
        return
    
    print("=" * 80)
    
    # Step 3: LLM Text Restoration
    print("🤖 Step 3: Restoring missing text with Gemini...")
    print("📝 Sending prompts to Gemini API...")
    
    try:
        restored_results = restore_missing_text_via_rest(
            clip_results, 
            api_key=settings.GEMINI_API_KEY,
            log_path="gemini_restore_log.txt"  # Save detailed log
        )
        print(f"✅ Restored text for {len(restored_results)} segments")
        
        # Print detailed Gemini responses
        print("\n🔍 Gemini API Responses:")
        for i, result in enumerate(restored_results):
            print(f"  Segment {i+1}: '{result['restored_text']}'")
            
    except Exception as e:
        print(f"❌ LLM restoration failed: {e}")
        return
    
    print("=" * 80)
    
    # Step 4: Display Results
    print("📊 FINAL RESULTS:")
    print("=" * 80)
    
    for i, result in enumerate(restored_results):
        print(f"\n🎯 Segment {i+1}:")
        print(f"   ⏱️  Time: {result['start_time']:.2f}s - {result['end_time']:.2f}s ({result['duration']:.2f}s)")
        print(f"   🔇 Type: {result['noise_type']} (confidence: {result['confidence']:.3f})")
        print(f"   📝 Before: \"{result['context_before']}\"")
        
        if result['partial_inside']:
            print(f"   🔇 During: \"{result['partial_inside']}\"")
        else:
            print(f"   🔇 During: [no audio detected]")
            
        print(f"   📝 After:  \"{result['context_after']}\"")
        print(f"   🤖 LLM Predicted: \"{result['restored_text']}\"")
        
        if i < len(restored_results) - 1:
            print("   " + "-" * 60)
    
    print("\n" + "=" * 80)
    print("✅ Pipeline test completed successfully!")
    
    # Save detailed results
    with open("pipeline_test_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "audio_file": test_wav,
            "noise_detection": noise_result,
            "transcription": clip_results,
            "restored_text": restored_results
        }, f, indent=2, ensure_ascii=False)
    
    print("💾 Detailed results saved to: pipeline_test_results.json")

if __name__ == "__main__":
    test_complete_pipeline()
