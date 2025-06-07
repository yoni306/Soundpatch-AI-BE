import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm
import numpy as np
import gc

# Detect and set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔌 Using device: {DEVICE}")

# ---------------------------------------------------------
# Load Wav2Vec2 model and processor
# ---------------------------------------------------------
def load_wav2vec2_model(
    processor_name: str = "facebook/wav2vec2-base-960h",
    model_name: str = "facebook/wav2vec2-large-960h"
):
    print("🔄 Loading Wav2Vec2 processor and model...")
    processor = Wav2Vec2Processor.from_pretrained(processor_name)
    model = Wav2Vec2Model.from_pretrained(model_name, use_safetensors=True).to(DEVICE)
    return processor, model


# ---------------------------------------------------------
# Extract speaker embedding from a single .wav file
# ---------------------------------------------------------
def extract_speaker_embedding(audio_path: str, processor, model) -> np.ndarray:
    waveform, sr = torchaudio.load(audio_path)

    # Resample to 16 kHz if necessary
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    # Tokenize and move to device
    inputs = processor(waveform.squeeze().numpy(), sampling_rate=16000, return_tensors="pt").input_values.to(DEVICE)

    # Extract features
    with torch.no_grad():
        outputs = model(inputs)

    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()  # shape: (1, hidden_dim)
    return embedding