import os
import torch
import numpy as np
import soundfile as sf
from tqdm import tqdm
from nemo.collections.tts.models import HifiGanModel
from typing import Dict, Any, Optional, Union

# Load pretrained HiFi-GAN model once
def load_hifigan_model(model_name="nvidia/tts_hifigan"):
    print("🔊 Loading HiFi-GAN vocoder...")
    model = HifiGanModel.from_pretrained(model_name=model_name)
    device = next(model.parameters()).device
    return model, device


def prepare_mel_for_vocoder(
    mel_2d: torch.Tensor,
    *,
    mean: Optional[Union[float, torch.Tensor]] = None,
    std: Optional[Union[float, torch.Tensor]] = None,
    hifigan_model=None,
) -> torch.Tensor:
    """
    Converts mel spectrogram to the format HiFi-GAN expects (same as notebook).
    - Input:  [T, 80] torch.FloatTensor   (no batch)
    - Output: [1, 80, T] torch.FloatTensor (batched, on hifigan device)

    mean/std are OPTIONAL. Pass the SAME values you used during training normalization
    (if any). If you didn't normalize, leave them None.
    """
    assert isinstance(mel_2d, torch.Tensor) and mel_2d.ndim == 2, "expected [T, 80] torch tensor"

    x = mel_2d.detach().float()       # [T, 80]

    # 1) (Optional) de-normalize back to the original scale you trained on
    if (mean is not None) and (std is not None):
        # mean/std can be scalars or shape-[80]; broadcast handles both
        if not torch.is_tensor(mean): 
            mean = torch.tensor(mean, dtype=x.dtype, device=x.device)
        if not torch.is_tensor(std):  
            std = torch.tensor(std, dtype=x.dtype, device=x.device)
        x = x * std + mean            # still [T, 80]

    # 2) Layout for HiFi-GAN: [T,80] -> [80,T] -> [1,80,T]
    x = x.transpose(0, 1).contiguous().unsqueeze(0)  # [1, 80, T]

    # 3) Put on same device as vocoder
    if hifigan_model is not None:
        device = next(hifigan_model.parameters()).device
        x = x.to(device)

    return x


@torch.no_grad()
def mel_to_audio_with_hifigan(
    mel_2d: torch.Tensor,
    hifigan_model,
    *,
    mean: Optional[Union[float, torch.Tensor]] = None,
    std: Optional[Union[float, torch.Tensor]] = None,
) -> np.ndarray:
    """
    Convenience wrapper: takes your [T,80] mel, adapts it, and runs the vocoder.
    Returns: 1D np.float32 waveform.
    """
    spec = prepare_mel_for_vocoder(mel_2d, mean=mean, std=std, hifigan_model=hifigan_model)  # [1,80,T]
    audio = hifigan_model.convert_spectrogram_to_audio(spec=spec)  # [1, samples]
    return audio[0].detach().cpu().float().numpy()


# Save mel predictions to .wav files
def save_mel_predictions_as_audio(
    mel_predictions: Dict[str, torch.Tensor],
    output_dir: str,
    model,
    device,
    sample_rate: int = 22050,
    mean: Optional[Union[float, torch.Tensor]] = None,
    std: Optional[Union[float, torch.Tensor]] = None
) -> None:
    """
    Save mel spectrograms as audio files using HiFi-GAN vocoder.
    
    Args:
        mel_predictions: Dict mapping clip_path to mel spectrogram (PyTorch tensor)
        output_dir: Directory to save audio files
        model: HiFi-GAN model
        device: Device to run inference on
        sample_rate: Audio sample rate (default: 22050)
        mean: Optional normalization mean (same as used in training)
        std: Optional normalization std (same as used in training)
    """
    os.makedirs(output_dir, exist_ok=True)

    for clip_path, mel_tensor in tqdm(mel_predictions.items(), total=len(mel_predictions), desc="🎧 Vocoder synthesis"):
        # Ensure mel_tensor is a PyTorch tensor
        if not isinstance(mel_tensor, torch.Tensor):
            if isinstance(mel_tensor, np.ndarray):
                mel_tensor = torch.from_numpy(mel_tensor).float()
            else:
                mel_tensor = torch.tensor(mel_tensor, dtype=torch.float32)

        # Ensure correct format: [T, 80] (time first, then mel dimensions)
        if mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.squeeze(0)  # Remove batch dimension if present
        
        # If mel is in [80, T] format, transpose to [T, 80]
        if mel_tensor.size(0) == 80 and mel_tensor.size(1) != 80:
            mel_tensor = mel_tensor.transpose(0, 1)

        # Run HiFi-GAN using the notebook's approach
        audio_np = mel_to_audio_with_hifigan(
            mel_tensor, 
            model, 
            mean=mean, 
            std=std
        )

        # Extract file name and save
        file_name = os.path.splitext(os.path.basename(clip_path))[0]
        out_path = os.path.join(output_dir, f"{file_name}.wav")
        sf.write(out_path, audio_np, sample_rate, format='WAV', subtype='PCM_16')
