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



def prepare_pred_rows(clip_results):
    pred_rows = []
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
    supabase_client,
    detect_noise_model,
    gemini_model,
    wav2vec2_processor,
    wav2vec2_model,
    text_to_mel_model,
    hifigan_model,
    device
):
    # Step 1: Convert the given file to WAV format
    wav_path = convert_to_wav(file_path)

    # Step 2: Detect audio gaps using the noise detection model
    noise_events = detect_noise_events(wav_path, detect_noise_model)

    # Step 3: Run the STT with AssemblyAI for the noisy events
    clip_results = clip_and_transcribe_events(noise_events, wav_path, settings.ASSEMBLYAI_API_KEY)

    # Transform clip_results into the required format for restore_missing_text
    pred_rows = prepare_pred_rows(clip_results)

    # Step 4: Run the Gemini LLM to restore the missing text
    restored_text = restore_missing_text(gemini_model, pred_rows)

    # Step 5: Extract the speaker embedding using the whole WAV file
    speaker_embedding = extract_speaker_embedding(wav_path, wav2vec2_processor, wav2vec2_model)

    # Step 6: Run the text to mel spectrogram model
    mel_spectrograms = []

    for event in restored_text:
        mel_spectrogram = predict_mel_from_text(event, speaker_embedding, text_to_mel_model)
        mel_spectrograms.append(mel_spectrogram)

    # Step 7: Run the vocoder model for each mel spectrogram
    mel_predictions = {f"clip_{i}": mel for i, mel in enumerate(mel_spectrograms)}
    output_dir = "path/to/output"  # Replace with the desired output directory
    save_mel_predictions_as_audio(mel_predictions, output_dir, hifigan_model, device)

    # Step 8: Create a new WAV file that replaces the noisy events
    clean_wav_path = reconstruct_clean_audio(wav_path, mel_spectrograms, noise_events)

    # Step 9: Convert to the original file format
    output_dir = "path/to/final_outputs"  # Replace with the desired output directory
    final_output_path = finalize_audio_or_video_output(file_path, clean_wav_path, output_dir)

    # Step 10: Save the file in Supabase
    file_key = save_to_supabase(final_output_path, supabase_client, settings.SUPABASE_PROCESSED_BUCKET)

    return file_key
