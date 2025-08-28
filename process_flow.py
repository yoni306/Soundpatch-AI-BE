from logic.utilities.convert_to_wav import convert_to_wav
from logic.mel_spectogram_generation.text_to_mel_inference import predict_mel_from_text
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio
from logic.utilities.reconstruct_clean_audio import reconstruct_clean_audio
from logic.utilities.finalize_output import finalize_audio_or_video_output
from config import settings
import os
import json
from typing import Dict, List, Any, Union
import numpy as np
import torch
import subprocess
import re


def prepare_pred_rows(clip_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Convert clip_results to format needed for the rest of the pipeline"""
    pred_rows: List[Dict[str, Any]] = []
    
    for tid, data in clip_results.items():
        event = data["ev"]
        words = data.get("words", [])
        
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
    print("🔄 Starting file processing...")
    # Step 1: Convert the given file to WAV format
    wav_path: str = convert_to_wav(file_path)
    print("✅ Step 1 completed: Converted to WAV format")
    
    # Steps 2-5: Run the TF pipeline
    base_tf_process_dir = "/home/deep/Soundpatch-AI-BE"
    print("🔄 Starting TF pipeline...")
    cmd = f"conda run -n tf-minimal python {base_tf_process_dir}/process_flow.py {file_path}"

    result = subprocess.run(
        cmd,
        shell=True,
        capture_output=True,
        text=True
    )
    json_file = re.findall(r"[\w-]+\.json", result.stdout)[0]
    final_results = json.load(open(json_file))
    restored_text = final_results["restored_text"]
    noise_events = final_results["noise_events"]
    print("✅ Step 2-5 completed: TF pipeline completed")
    

    # Step 6: Run the text to mel spectrogram model
    print("🔄 Starting text to mel spectrogram model...")
    mel_spectrograms: List[torch.Tensor] = []

    for event in restored_text:
        # Extract the restored text from the event dictionary
        restored_text_content = event.get("restored_text", "")
        if restored_text_content:
            mel_spectrogram: torch.Tensor = predict_mel_from_text(restored_text_content, text_to_mel_model)
            mel_spectrograms.append(mel_spectrogram)
    print("✅ Step 6 completed: Text to mel spectrogram model completed")


    # Step 7: Run the vocoder model for each mel spectrogram
    print("🔄 Starting vocoder model...")
    mel_predictions: Dict[str, torch.Tensor] = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    output_dir: str = settings.UPLOAD_DIR
    save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)
    print("✅ Step 7 completed: Vocoder model completed")

    # ONLY FOR TESTING
    with open(wav_path, 'rb') as f:
        file_content = f.read()
    
    return file_content



    # Step 8: Create a new WAV file that replaces the noisy events
    print("🔄 Starting clean audio reconstruction...")
    clean_wav_path: str = os.path.join(output_dir, "reconstructed_clean_audio.wav")
    reconstruct_clean_audio(wav_path, noise_events, output_dir, clean_wav_path)
    print("✅ Step 8 completed: Clean audio reconstruction completed")


    # Step 9: Convert to the original file format
    print("🔄 Starting final output...")
    final_output_path: str = finalize_audio_or_video_output(file_path, clean_wav_path)
    print("✅ Step 9 completed: Final output completed")


    # Step 10: Read the processed file content
    print("🔄 Reading the processed file content...")
    with open(final_output_path, 'rb') as f:
        file_content = f.read()
    print("✅ Step 10 completed: Read the processed file content")


    # Step 11: Clean up temporary files
    print("🔄 Cleaning up temporary files...")
    if wav_path != file_path:  # Only remove if it's a converted file
        os.remove(wav_path)
    os.remove(clean_wav_path)
    os.remove(final_output_path)
    print("✅ Step 11 completed: Cleaned up temporary files")


    print("✅ File processing completed successfully")

    return file_content
