import numpy as np
import librosa
import os
import json
from typing import List, Dict, Any
from .detect_noise_moedl import load_noise_detectoion_model

# --- Constants ---
LABEL_NAMES = ["signal_loss", "volume_drop", "compression_artifact"]
FRAME_HOP = 160
SR = 16000
N_MELS = 64
THRESHOLDS = [0.8, 0.8, 0.8]
MAX_CHUNK_SIZE = 1000


def detect_noise_audio(wav_file_path: str) -> Dict[str, Any]:
    """
    Single function that loads model, detects noise events and returns JSON.
    
    Args:
        wav_file_path: Path to the WAV file
        
    Returns:
        Dictionary containing detection results in JSON format
    """
    try:
        # Validate input
        if not os.path.exists(wav_file_path):
            raise FileNotFoundError(f"WAV file not found: {wav_file_path}")
        
        if not wav_file_path.lower().endswith('.wav'):
            raise ValueError("Only .wav files are supported")
            
        # Load model - Cross-platform path handling
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        model_path = os.path.join(project_root, "final_model.h5")
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at: {model_path}")
            
        model = load_noise_detectoion_model(model_path, verbose=False)
        
        # Preprocess audio
        y, _ = librosa.load(wav_file_path, sr=SR)
        mel = librosa.feature.melspectrogram(y=y, sr=SR, n_fft=1024, hop_length=FRAME_HOP, n_mels=N_MELS)
        log_mel = librosa.power_to_db(mel).astype(np.float32)
        
        # Run inference
        T = log_mel.shape[1]
        y_pred_prob = np.zeros((T, 3), dtype=float)
        
        if T > MAX_CHUNK_SIZE:
            chunks = [(i, min(i + MAX_CHUNK_SIZE, T)) for i in range(0, T, MAX_CHUNK_SIZE)]
            for start, end in chunks:
                chunk = log_mel[:, start:end]
                chunk = np.expand_dims(chunk, axis=(0, -1))  # shape: (1, 64, chunk_len, 1)
                preds = model.predict(chunk, verbose=0)[0]
                y_pred_prob[start:end, :] = preds
        else:
            X = np.expand_dims(log_mel, axis=(0, -1))  # shape: (1, 64, T, 1)
            y_pred_prob = model.predict(X, verbose=0)[0]
        
        # Extract events
        events = []
        y_pred_bin = np.zeros_like(y_pred_prob)
        for i in range(3):
            y_pred_bin[:, i] = (y_pred_prob[:, i] > THRESHOLDS[i]).astype(int)

        for ch in range(3):
            in_event = False
            start_frame = None
            for t in range(T):
                if y_pred_bin[t, ch] == 1 and not in_event:
                    in_event = True
                    start_frame = t
                elif y_pred_bin[t, ch] == 0 and in_event:
                    end_frame = t
                    in_event = False
                    start_time = round(start_frame * FRAME_HOP / SR, 2)
                    end_time = round(end_frame * FRAME_HOP / SR, 2)
                    duration = round(end_time - start_time, 2)
                    
                    # Filter events: only include events of 3 seconds or longer
                    if duration >= 3.0:
                        avg_confidence = float(np.mean(y_pred_prob[start_frame:end_frame, ch]))
                        events.append({
                            "start_time": start_time,
                            "end_time": end_time,
                            "duration": duration,
                            "noise_type": LABEL_NAMES[ch],
                            "confidence": round(avg_confidence, 3)
                        })
            if in_event:
                end_time = round(T * FRAME_HOP / SR, 2)
                start_time = round(start_frame * FRAME_HOP / SR, 2)
                duration = round(end_time - start_time, 2)
                
                # Filter events: only include events of 3 seconds or longer
                if duration >= 3.0:
                    avg_confidence = float(np.mean(y_pred_prob[start_frame:T, ch]))
                    events.append({
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "noise_type": LABEL_NAMES[ch],
                        "confidence": round(avg_confidence, 3)
                    })
        
        # Sort events by start time
        events.sort(key=lambda x: x['start_time'])
        
        # Create result JSON
        result = {
            "file_path": wav_file_path,
            "total_events": len(events),
            "audio_duration": round(log_mel.shape[1] * FRAME_HOP / SR, 2),
            "events": events,
            "model_info": {
                "noise_types": LABEL_NAMES,
                "thresholds": THRESHOLDS,
                "sample_rate": SR
            }
        }
        
        return result
        
    except Exception as e:
        return {
            "error": str(e),
            "file_path": wav_file_path,
            "events": []
        }