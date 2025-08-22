import os
import tempfile
from pathlib import Path
from moviepy import VideoFileClip
from pydub import AudioSegment


SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".avi", ".webm", ".mkv"}

def convert_to_wav(input_path: str, output_dir: str = None, target_sr: int = 16000) -> str:
    """
    Convert an audio or video file to a 16 kHz mono .wav file.
    If already a WAV file with correct format, return the original path.

    Args:
        input_path (str): Path to the input file (audio or video).
        output_dir (str): Optional output directory (defaults to temp).
        target_sr (int): Target sample rate for output WAV.

    Returns:
        str: Path to the WAV file (original or converted).
    """
    input_path = Path(input_path)
    ext = input_path.suffix.lower()

    # Check if already WAV and has correct format
    if ext == ".wav":
        try:
            # Load the existing WAV to check its properties
            audio = AudioSegment.from_wav(input_path)
            
            # Check if it's already mono and 16kHz
            if audio.channels == 1 and audio.frame_rate == target_sr:
                print(f"File is already in correct format: {input_path.name}")
                return str(input_path)  # Return original path
            else:
                print(f"WAV file needs format conversion: {audio.channels} channels, {audio.frame_rate}Hz -> 1 channel, {target_sr}Hz")
        except Exception:
            print(f"Could not read WAV file properties, proceeding with conversion: {input_path.name}")

    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="converted_audio_")
    else:
        os.makedirs(output_dir, exist_ok=True)

    out_path = Path(output_dir) / f"{input_path.stem}_converted.wav"

    try:
        if ext in SUPPORTED_AUDIO:
            audio = AudioSegment.from_file(input_path)
        elif ext in SUPPORTED_VIDEO:
            print(f"Extracting audio from video: {input_path.name}")
            with VideoFileClip(str(input_path)) as clip:
                tmp_audio_path = Path(tempfile.mktemp(suffix=".wav"))
                clip.audio.write_audiofile(str(tmp_audio_path), verbose=False, logger=None)
                audio = AudioSegment.from_wav(tmp_audio_path)
                tmp_audio_path.unlink()  # cleanup
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        # Convert to mono, resample to target_sr, export to .wav
        audio = audio.set_channels(1).set_frame_rate(target_sr)
        audio.export(out_path, format="wav")

        return str(out_path)

    except Exception as e:
        raise RuntimeError(f"Failed to convert {input_path}: {e}")
