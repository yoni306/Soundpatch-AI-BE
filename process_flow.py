from logic.utilities.convert_to_wav import convert_to_wav
from logic.noise_detection.detect_noise_events import detect_noise_audio
from logic.transcription.transcribe_events import clip_and_transcribe_events
from logic.text_generation.restore_missing_text import restore_missing_text_via_rest
from logic.speaker_embedding.speaker_embedding import extract_speaker_embedding
from logic.mel_spectogram_generation.text_to_mel_inference_kesem import predict_mel_from_text
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio
from logic.utilities.reconstruct_clean_audio import reconstruct_clean_audio
from logic.utilities.finalize_output import finalize_audio_or_video_output
from config import settings
from logic.utilities.save_to_supabase import save_to_supabase
import os
import json
from typing import Dict, List, Tuple, Any, Union
import numpy as np


def prepare_pred_rows(clip_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert clip_results to format needed for the rest of the pipeline"""
    pred_rows: List[Dict[str, Any]] = []
    
    for tid, data in clip_results.items():
        event = data["ev"]("words", [])

        words = data.get
        # Extract text from words (words now contain {text, start, end})
        transcribed_text = " ".join([word.get("text", "") for word in words if word.get("text", "").strip()])
        
        pred_row = {
            "start_time": event["start_time"],
            "end_time": event["end_time"],
            "filename": event.get("filename", ""),
            "noise_type": event.get("noise_type", "unknown"),
            "confidence": event.get("confidence", 0.0),
            "transcribed_text": transcribed_text,  # Changed from missing_text
            "missing_text": transcribed_text  # Keep for backward compatibility
        }
        pred_rows.append(pred_row)
    
    return pred_rows


def process_file(
    file_path: str,
    detect_noise_model,
    wav2vec2_processor,
    wav2vec2_model,
    text_to_mel_model,
    hifigan_model,
    device
) -> Union[bytes, Dict[str, Any]]:
    """
    Process an audio file through the complete pipeline up to text restoration.
    
    Args:
        file_path (str): Path to the input audio file
        detect_noise_model: Noise detection model (not used - we use detect_noise_audio)
        wav2vec2_processor: Wav2Vec2 processor
        wav2vec2_model: Wav2Vec2 model
        text_to_mel_model: Text to mel spectrogram model
        hifigan_model: HiFi-GAN vocoder model
        device: Device to run models on
    
    Returns:
        Union[bytes, Dict[str, Any]]: Either the processed file content or processing results
    """
    # Step 1: Convert the given file to WAV format
    wav_path: str = convert_to_wav(file_path)

    # Step 2: Detect audio gaps using the noise detection model
    noise_result: Dict[str, Any] = detect_noise_audio(wav_path)
    
    if "error" in noise_result:
        raise Exception(f"Noise detection failed: {noise_result['error']}")
    
    noise_events: List[Dict[str, Any]] = noise_result["events"]
    print(f"Found {len(noise_events)} noise events")
    
    if len(noise_events) == 0:
        print("No noise events found, returning original file")
        # No noise events found, return original file
        with open(file_path, 'rb') as f:
            file_content = f.read()
        
        # Clean up WAV file only if it's different from the original
        if os.path.exists(wav_path) and wav_path != file_path:
            os.remove(wav_path)
        
        return file_content

    # Step 3: Run the STT with AssemblyAI for the noisy events
    clip_results: Dict[str, Any] = clip_and_transcribe_events(
        noise_events, 
        wav_path, 
        settings.ASSEMBLYAI_API_KEY
    )
    print(f"Transcribed {len(clip_results)} clips")

    # Step 4: Run the Gemini LLM to restore the missing text (using REST API)
    print("Step 4: Restoring missing text with Gemini...")
    
    restored_text: List[Dict[str, Any]] = restore_missing_text_via_rest(
        clip_results, 
        api_key=settings.GEMINI_API_KEY,
        log_path="gemini_restore_log.txt"  # Save detailed log
    )
    print(f"Restored text for {len(restored_text)} segments")
    
    print("Pipeline completed up to text restoration!")
    print(f"Summary: {len(noise_events)} events → {len(clip_results)} clips → {len(restored_text)} restorations")
    
    # Create results summary for debugging/testing
    pipeline_results = {
        "input_file": file_path,
        "wav_path": wav_path,
        "noise_events_count": len(noise_events),
        "noise_events": noise_events,
        "transcription_results": clip_results,
        "restoration_results": restored_text,
        "status": "completed_until_text_restoration"
    }
    
    # Save results for debugging (optional)
    results_path = "pipeline_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(pipeline_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"Results saved to: {results_path}")

    # Step 5: Run the text to mel spectrogram model
    mel_spectrograms: List[np.ndarray] = []

    for event in restored_text:
        # Extract the restored text from the event dictionary
        restored_text_content = event.get("restored_text", "")
        if restored_text_content:
            mel_spectrogram: np.ndarray = predict_mel_from_text(restored_text_content, speaker_embedding, text_to_mel_model)
            mel_spectrograms.append(mel_spectrogram)

    # Step 6: Run the vocoder model for each mel spectrogram
    mel_predictions: Dict[str, np.ndarray] = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    output_dir: str = settings.UPLOAD_DIR
    save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)

    # Step 7: Create a new WAV file that replaces the noisy events
    clean_wav_path: str = reconstruct_clean_audio(wav_path, mel_spectrograms, noise_events, output_dir)

    # Step 8: Convert to the original file format
    final_output_path: str = finalize_audio_or_video_output(file_path, clean_wav_path)

    # Step 9: Read the processed file content
    with open(final_output_path, 'rb') as f:
        file_content = f.read()

    # Step 10: Clean up temporary files
    if wav_path != file_path:  # Only remove if it's a converted file
        os.remove(wav_path)
    os.remove(clean_wav_path)
    os.remove(final_output_path)

    return file_content
