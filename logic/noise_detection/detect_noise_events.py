import numpy as np
import tensorflow as tf
import librosa
import os
import tempfile
from logic.noise_detection.detect_noise_moedl import detect_noisemodel

# --- Constants ---
LABEL_NAMES = ["signal_loss", "volume_drop", "compression_artifact"]
FRAME_HOP = 160
SR = 16000
N_MELS = 64
THRESHOLDS = [0.5, 0.5, 0.8]
MAX_CHUNK_SIZE = 1000


# --- Utility Functions ---
def preprocess_wav(wav_bytes):
    y, _ = librosa.load(wav_bytes, sr=SR)
    mel = librosa.feature.melspectrogram(y, sr=SR, n_fft=1024, hop_length=FRAME_HOP, n_mels=N_MELS)
    log_mel = librosa.power_to_db(mel).astype(np.float32)
    return log_mel  # shape: (64, T)


def predict_chunks(log_mel):
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
    return y_pred_prob


def extract_events(y_pred_prob):
    events = []
    T = y_pred_prob.shape[0]
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
                events.append({
                    "start": round(start_frame * FRAME_HOP / SR, 2),
                    "end": round(end_frame * FRAME_HOP / SR, 2),
                    "label": LABEL_NAMES[ch]
                })
        if in_event:
            events.append({
                "start": round(start_frame * FRAME_HOP / SR, 2),
                "end": round(T * FRAME_HOP / SR, 2),
                "label": LABEL_NAMES[ch]
            })
    return events


def detect_noise_events(file_path: str):
    file = open(file_path, "rb")
    file_name = file_path.split("/")[-1]
    if not file_name.endswith(".wav"):
        raise Exception("Only .wav files are supported.")

    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        # Preprocess + Predict
        log_mel = preprocess_wav(tmp_path)
        y_pred_prob = predict_chunks(log_mel)
        events = extract_events(y_pred_prob)

        # Cleanup
        os.remove(tmp_path)

        return events
    except Exception as e:
        raise Exception(f"Error processing audio file: {e}")