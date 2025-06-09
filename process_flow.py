from logic.utilities.convert_to_wav import convert_to_wav
from logic.noise_detection.detect_noise_events import detect_noise_events
from logic.transcription.transcribe_events import clip_and_transcribe_events
from logic.text_generation.restore_missing_text import restore_missing_text
from logic.speaker_embedding.speaker_embedding import extract_speaker_embedding
from logic.mel_spectogram_generation.text_to_mel_inference import predict_mel_from_text
from logic.voice_generation.vocoder_utils import save_mel_predictions_as_audio
from logic.utilities.reconstruct_clean_audio import reconstruct_clean_audio
from logic.utilities.finalize_output import finalize_audio_or_video_output
from config import settings
from logic.utilities.save_to_supabase import save_to_supabase
import os
from typing import Dict, List, Tuple, Any, Union
import numpy as np


def prepare_pred_rows(clip_results: Dict[str, Any]) -> List[Dict[str, Any]]:
    pred_rows: List[Dict[str, Any]] = []
    for tid, data in clip_results.items():
        pred_row = {
            "start_time": data["clip_start"],
            "end_time": data["clip_start"] + (data["ev"]["end"] - data["ev"]["start"]),
            "missing_text": " ".join([word["text"] for word in data["words"]])
        }
        pred_rows.append(pred_row)
    return pred_rows


def process_file(
    file_path: str,
    detect_noise_model,
    gemini_model,
    wav2vec2_processor,
    wav2vec2_model,
    text_to_mel_model,
    hifigan_model,
    device
) -> bytes:
    """
    Process an audio file through the complete pipeline.
    
    Args:
        file_path (str): Path to the input audio file
        detect_noise_model: Noise detection model
        gemini_model: Gemini model for text generation
        wav2vec2_processor: Wav2Vec2 processor
        wav2vec2_model: Wav2Vec2 model
        text_to_mel_model: Text to mel spectrogram model
        hifigan_model: HiFi-GAN vocoder model
        device: Device to run models on
    
    Returns:
        bytes: The processed file content
    """
    # Step 1: Convert the given file to WAV format
    wav_path: str = convert_to_wav(file_path)

    # Step 2: Detect audio gaps using the noise detection model
    noise_events: List[Dict[str, Any]] = detect_noise_events(wav_path, detect_noise_model)

    # Step 3: Run the STT with AssemblyAI for the noisy events
    clip_results: Dict[str, Any] = clip_and_transcribe_events(noise_events, wav_path, settings.ASSEMBLYAI_API_KEY)

    # Transform clip_results into the required format for restore_missing_text
    pred_rows: List[Dict[str, Any]] = prepare_pred_rows(clip_results)

    # Step 4: Run the Gemini LLM to restore the missing text
    restored_text: List[Dict[str, Any]] = restore_missing_text(gemini_model, pred_rows)

    # Step 5: Extract the speaker embedding using the whole WAV file
    speaker_embedding: np.ndarray = extract_speaker_embedding(wav_path, wav2vec2_processor, wav2vec2_model)

    # Step 6: Run the text to mel spectrogram model
    mel_spectrograms: List[np.ndarray] = []

    for event in restored_text:
        mel_spectrogram: np.ndarray = predict_mel_from_text(event, speaker_embedding, text_to_mel_model)
        mel_spectrograms.append(mel_spectrogram)

    # Step 7: Run the vocoder model for each mel spectrogram
    mel_predictions: Dict[str, np.ndarray] = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    output_dir: str = settings.UPLOAD_DIR
    save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)

    # Step 8: Create a new WAV file that replaces the noisy events
    clean_wav_path: str = reconstruct_clean_audio(wav_path, mel_spectrograms, noise_events)

    # Step 9: Convert to the original file format
    final_output_path: str = finalize_audio_or_video_output(file_path, clean_wav_path)

    # Step 10: Read the processed file content
    with open(final_output_path, 'rb') as f:
        file_content = f.read()

    # Step 11: Clean up temporary files
    os.remove(wav_path)
    os.remove(clean_wav_path)
    os.remove(final_output_path)

    return file_content
