from pydub import AudioSegment
import os

def seconds_to_mmss(seconds):
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"

def reconstruct_clean_audio(original_wav_path, events, vocoder_wav_dir, output_path):
    """
    Reconstructs a clean WAV file by replacing noisy segments with vocoder-generated ones.

    Args:
        original_wav_path (str): Path to the original noisy WAV file.
        events (list): List of events with "start" and "end" keys (in seconds).
        vocoder_wav_dir (str): Directory containing vocoder-generated wav files.
        output_path (str): Where to save the final reconstructed audio.

    Returns:
        str: Path to the reconstructed audio file.
    """
    # Load original full audio
    original_audio = AudioSegment.from_wav(original_wav_path)
    reconstructed = AudioSegment.empty()

    last_end_ms = 0
    original_base = os.path.splitext(os.path.basename(original_wav_path))[0]

    for event in events:
        start_ms = int(event["start"] * 1000)
        end_ms = int(event["end"] * 1000)

        # Append segment before the noise
        reconstructed += original_audio[last_end_ms:start_ms]

        # Construct vocoder-generated filename
        start_tag = seconds_to_mmss(event["start"]).replace(":", "")
        end_tag = seconds_to_mmss(event["end"]).replace(":", "")
        generated_filename = f"{original_base}_{start_tag}_{end_tag}.wav"
        generated_path = os.path.join(vocoder_wav_dir, generated_filename)

        # Load vocoder segment (fallback to original if missing)
        if os.path.exists(generated_path):
            clean_segment = AudioSegment.from_wav(generated_path)
        else:
            print(f"⚠️ Missing: {generated_path}, using original")
            clean_segment = original_audio[start_ms:end_ms]

        reconstructed += clean_segment
        last_end_ms = end_ms

    # Add the remainder of the original file
    reconstructed += original_audio[last_end_ms:]

    # Export final file
    reconstructed.export(output_path, format="wav")
    print(f"✅ Reconstructed audio saved to: {output_path}")
    return output_path
