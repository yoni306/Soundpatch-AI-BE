import os
import torch
import torchaudio  # pyright: ignore[reportUnusedImport]
import soundfile as sf
import numpy as np
from typing import Dict, List, Any
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

# Load pretrained HiFi-GAN model once
from nemo.collections.tts.models import HifiGanModel
# hifigan_model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
try:
    print("Loading HiFi-GAN model...")
    hifigan_model = HifiGanModel.from_pretrained(model_name="nvidia/tts_hifigan")
    hifigan_model = hifigan_model.to("cpu")  # Move to CPU after loading
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading HiFi-GAN model: {e}")
    hifigan_model = None


def seconds_to_mmss(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def prepare_mel_for_vocoder(
    mel_2d: torch.Tensor,
    *,
    mean: float | torch.Tensor | None = None,
    std: float  | torch.Tensor | None = None,
    hifigan_model=None,
) -> torch.Tensor:
    """
    Converts your current mel (as returned by text_to_mel) to the format HiFi-GAN expects.
    - Input:  [80, T] or [T, 80] torch.FloatTensor   (auto-detects format)
    - Output: [1, 80, T] torch.FloatTensor (batched, on hifigan device)

    mean/std are OPTIONAL. Pass the SAME values you used during training normalization
    (if any). If you didn't normalize, leave them None.
    """
    assert isinstance(mel_2d, torch.Tensor) and mel_2d.ndim == 2, "expected 2D torch tensor"

    x = mel_2d.detach().float()

    # Handle transposed vs original mel spectrograms
    is_transposed_input = x.shape[0] == 80  # [80, T] format indicates transposed
    
    if is_transposed_input:
        print(f"Processing transposed mel spectrogram: {x.shape}")
        # For transposed mels, we want to preserve the frequency-time relationship
        # Instead of transposing back, we'll create a modified version that maintains
        # the transposed characteristics while being compatible with HiFi-GAN
        
        # Method: Apply frequency-domain modifications that preserve the transposed structure
        time_frames = x.shape[1]
        mel_bins = 80
        
        # Create a frequency-shifted version that maintains the transposed characteristics
        freq_shift = max(1, int(mel_bins * 0.1))  # Shift by 10% of mel bins
        x_shifted = torch.roll(x, shifts=freq_shift, dims=0)
        
        # Blend original transposed with shifted version to create unique characteristics
        x = 0.7 * x + 0.3 * x_shifted
        
        print(f"Applied transposed-preserving modifications")
        
        # Now convert to HiFi-GAN format: [80, T] -> [1, 80, T]
        x = x.unsqueeze(0)  # Add batch dimension
        
    else:
        # Standard processing for original [T, 80] format
        print(f"Processing original mel spectrogram: {x.shape}")
        
        # 1) (Optional) de-normalize back to the original scale you trained on
        if (mean is not None) and (std is not None):
            # mean/std can be scalars or shape-[80]; broadcast handles both
            if not torch.is_tensor(mean): mean = torch.tensor(mean, dtype=x.dtype, device=x.device)
            if not torch.is_tensor(std):  std  = torch.tensor(std,  dtype=x.dtype, device=x.device)
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
    *,
    mean: float | torch.Tensor | None = None,
    std: float  | torch.Tensor | None = None,
) -> np.ndarray:
    """
    Convenience wrapper: takes your mel, adapts it, and runs the vocoder.
    Returns: 1D np.float32 waveform.
    """
    spec = prepare_mel_for_vocoder(mel_2d, mean=mean, std=std, hifigan_model=hifigan_model)
    audio = hifigan_model.convert_spectrogram_to_audio(spec=spec)  # [1, samples]
    return audio[0].detach().cpu().float().numpy()


# Save mel predictions to .wav files
def save_mel_predictions_as_audio(
    mel_predictions: Dict[str, np.ndarray],
    output_dir: str,
    model,
    device,
    noise_events_dict: Dict[str, dict],
    original_wav_path: str,
    sample_rate: int = 22050
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    # for (clip_path, mel), event in tqdm(zip(mel_predictions.items(), events), total=len(mel_predictions), desc="Vocoder synthesis"):
    for clip_key in mel_predictions.keys():
        mel = mel_predictions[clip_key]
        event = noise_events_dict[clip_key] 
        try:
            # Convert from NumPy or Tensor to torch.Tensor
            if isinstance(mel, np.ndarray):
                mel_torch = torch.from_numpy(mel).float()
            elif hasattr(mel, 'numpy'):
                mel_torch = torch.from_numpy(mel.numpy()).float()
            else:
                mel_torch = mel  # Already torch.Tensor

            # Ensure mel is in [T, 80] format for mel_to_audio_with_hifigan
            if mel_torch.dim() == 3:
                mel_torch = mel_torch.squeeze(0)  # Remove batch dim if present
            elif mel_torch.dim() == 1:
                # If it's a 1D array, reshape to [1, 80]
                mel_torch = mel_torch.unsqueeze(0)

            # Run HiFi-GAN with proper preprocessing
            audio_np = mel_to_audio_with_hifigan(mel_torch)
                    # Construct vocoder-generated filename

            # original_base = os.path.splitext(os.path.basename(original_wav_path))[0]
            # start_tag = seconds_to_mmss(event["start_time"]).replace(":", "")
            # end_tag = seconds_to_mmss(event["end_time"]).replace(":", "")
            # generated_filename = f"{original_base}_{start_tag}_{end_tag}.wav"
            generated_filename = clip_key
            generated_path = os.path.join(output_dir, generated_filename)

            # Extract file name and save
            # file_name = os.path.splitext(os.path.basename(clip_path))[0]
            # out_path = os.path.join(output_dir, f"{file_name}.wav")
            sf.write(generated_path, audio_np, sample_rate, format='WAV', subtype='PCM_16')
            
            print(f"Generated audio: {out_path} ({len(audio_np)} samples)")
            
        except Exception as e:
            print(f"Error processing {clip_key}: {e}")
            # Create silence as fallback
            silence_samples = int(1.0 * sample_rate)  # 1 second of silence
            audio_np = np.zeros(silence_samples, dtype=np.float32)
            file_name = os.path.splitext(os.path.basename(clip_key))[0]
            out_path = os.path.join(output_dir, f"{file_name}_error.wav")
            sf.write(out_path, audio_np, sample_rate, format='WAV', subtype='PCM_16')
            print(f"Saved silence fallback: {out_path}")


def load_hifigan_model(model_name="nvidia/tts_hifigan"):
    """Load HiFi-GAN model for compatibility with existing code."""
    if hifigan_model is None:
        raise RuntimeError("HiFi-GAN model failed to load during module initialization")
    return hifigan_model, next(hifigan_model.parameters()).device
