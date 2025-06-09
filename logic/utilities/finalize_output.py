from pathlib import Path
from pydub import AudioSegment
from moviepy import VideoFileClip, AudioFileClip

AUDIO_EXTENSIONS = {'.wav', '.mp3', '.m4a', '.flac', '.ogg'}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

def finalize_audio_or_video_output(original_file_path: str, new_wav_path: str, output_dir: str = "final_outputs") -> str:
    """
    Converts the new WAV file into the original file's format:
    - If original is audio: converts back to original audio format
    - If original is video: replaces original audio with new audio
    
    Returns:
        str: Path to the final output file
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    original_path = Path(original_file_path)
    extension = original_path.suffix.lower()
    base_name = original_path.stem
    out_path = None

    if extension in AUDIO_EXTENSIONS:
        # Convert .wav back to original audio format
        original_audio = AudioSegment.from_wav(new_wav_path)
        out_path = Path(output_dir) / f"{base_name}_restored{extension}"
        original_audio.export(out_path, format=extension.replace('.', ''))
        print(f"✅ Converted audio saved to {out_path}")

    elif extension in VIDEO_EXTENSIONS:
        # Replace audio track in video with new wav
        video = VideoFileClip(str(original_path))
        new_audio = AudioFileClip(new_wav_path)
        video = video.set_audio(new_audio)

        out_path = Path(output_dir) / f"{base_name}_restored{extension}"
        video.write_videofile(str(out_path), codec='libx264', audio_codec='aac', verbose=False, logger=None)
        print(f"✅ Video with replaced audio saved to {out_path}")

    else:
        raise ValueError(f"Unsupported file type: {extension}")

    return str(out_path)
