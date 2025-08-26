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
    """Test the complete pipeline on ALL WAV files"""
    # Use generic path to wav folder in project
    wav_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wav")
    
    if not os.path.exists(wav_folder):
        print(f"❌ WAV folder not found: {wav_folder}")
        return
    
    # Find ALL .wav files in the folder
    wav_files = [f for f in os.listdir(wav_folder) if f.lower().endswith('.wav')]
    if not wav_files:
        print(f"❌ No WAV files found in: {wav_folder}")
        return
    
    wav_files.sort()  # Process in alphabetical order
    
    print("🚀 Starting complete pipeline test on ALL files...")
    print(f"📁 Found {len(wav_files)} WAV files: {', '.join(wav_files)}")
    print("=" * 80)
    
    all_results = {}
    
    for file_idx, wav_file in enumerate(wav_files):
        test_wav = os.path.join(wav_folder, wav_file)
        
        print(f"\n🎵 Processing file {file_idx + 1}/{len(wav_files)}: {wav_file}")
        print("=" * 80)
        
        # Step 1: Noise Detection
        print("🔍 Step 1: Detecting noise events...")
        noise_result = detect_noise_audio(test_wav)
        
        if "error" in noise_result:
            print(f"❌ Noise detection failed: {noise_result['error']}")
            all_results[wav_file] = {"error": noise_result['error']}
            continue
        
        events = noise_result["events"]
        print(f"✅ Found {len(events)} noise events (3+ seconds)")
        
        if len(events) == 0:
            print("ℹ️ No significant noise events found. Skipping to next file.")
            all_results[wav_file] = {"status": "no_events"}
            continue
        
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
            all_results[wav_file] = {"error": f"Transcription failed: {e}"}
            continue
        
        print("=" * 80)
        
        # Step 3: LLM Text Restoration
        print("🤖 Step 3: Restoring missing text with Gemini...")
        print("📝 Sending prompts to Gemini API...")
        
        try:
            restored_results = restore_missing_text_via_rest(
                clip_results, 
                api_key=settings.GEMINI_API_KEY,
                log_path=f"gemini_restore_log_{wav_file.replace('.wav', '')}.txt"
            )
            print(f"✅ Restored text for {len(restored_results)} segments")
            
            # Print detailed Gemini responses
            print("\n🔍 Gemini API Responses:")
            for i, result in enumerate(restored_results):
                print(f"  Segment {i+1}: '{result['restored_text']}'")
                
        except Exception as e:
            print(f"❌ LLM restoration failed: {e}")
            all_results[wav_file] = {"error": f"LLM restoration failed: {e}"}
            continue
        
        print("=" * 80)
        
        # Step 4: Display Results
        print("📊 RESULTS FOR THIS FILE:")
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
        
        # Store results
        all_results[wav_file] = {
            "noise_detection": noise_result,
            "transcription": clip_results,
            "restored_text": restored_results,
            "status": "success"
        }
        
        print(f"\n✅ File {wav_file} completed successfully!")
        print("=" * 80)
    
    # Final Summary
    print("\n" + "=" * 80)
    print("📊 FINAL SUMMARY FOR ALL FILES:")
    print("=" * 80)
    
    success_count = 0
    total_events = 0
    
    for wav_file, result in all_results.items():
        print(f"\n📁 {wav_file}:")
        
        if result.get("status") == "success":
            success_count += 1
            events_count = len(result["restored_text"])
            total_events += events_count
            print(f"   ✅ SUCCESS - {events_count} events processed")
            
            for i, segment in enumerate(result["restored_text"]):
                print(f"      {i+1}. {segment['start_time']:.1f}s-{segment['end_time']:.1f}s "
                      f"({segment['noise_type']}) → '{segment['restored_text']}'")
        
        elif result.get("status") == "no_events":
            print(f"   ℹ️ No noise events found")
        
        else:
            error_msg = result.get("error", "Unknown error")
            print(f"   ❌ FAILED - {error_msg}")
    
    print(f"\n📈 STATISTICS:")
    print(f"   🎵 Total files processed: {len(wav_files)}")
    print(f"   ✅ Successful: {success_count}")
    print(f"   ❌ Failed: {len(wav_files) - success_count}")
    print(f"   🔍 Total events found: {total_events}")
    
    # Save detailed results
    with open("pipeline_test_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print("\n💾 Detailed results saved to: pipeline_test_results.json")
    print("=" * 80)

if __name__ == "__main__":
    test_complete_pipeline()
