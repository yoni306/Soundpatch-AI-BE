import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from nemo.collections.tts.models import HifiGanModel
from typing import Dict, Any

# Load pretrained HiFi-GAN model once
def load_hifigan_model(model_name="nvidia/tts_hifigan"):
    print("🔊 Loading HiFi-GAN vocoder...")
    model = HifiGanModel.from_pretrained(model_name=model_name)
    device = next(model.parameters()).device
    return model, device


# Save mel predictions to .wav files
def save_mel_predictions_as_audio(
    mel_predictions: Dict[str, np.ndarray],
    output_dir: str,
    model,
    device,
    sample_rate: int = 22050
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    for clip_path, mel in tqdm(mel_predictions.items(), total=len(mel_predictions), desc="🎧 Vocoder synthesis"):
        # Convert from NumPy or Tensor to torch.Tensor
        if isinstance(mel, np.ndarray):
            mel_torch = torch.from_numpy(mel).float()
        elif hasattr(mel, 'numpy'):
            mel_torch = torch.from_numpy(mel.numpy()).float()
        else:
            mel_torch = mel  # Already torch.Tensor

        # Add batch dim
        if mel_torch.dim() == 2:
            mel_torch = mel_torch.unsqueeze(0)  # [1, T, 80]

        # [1, T, 80] → [1, 80, T]
        mel_torch = mel_torch.transpose(1, 2).to(device)

        # Run HiFi-GAN
        with torch.no_grad():
            audio = model.convert_spectrogram_to_audio(mel_torch)

        # Detach and convert to NumPy
        audio_np = audio.cpu().numpy().squeeze().astype(np.float32)

        # Extract file name and save
        file_name = os.path.splitext(os.path.basename(clip_path))[0]
        out_path = os.path.join(output_dir, f"{file_name}.wav")
        sf.write(out_path, audio_np, sample_rate, format='WAV', subtype='PCM_16')
