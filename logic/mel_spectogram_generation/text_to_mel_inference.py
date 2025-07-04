import tensorflow as tf
import numpy as np
import os

# Constants
MEL_DIM = 80
MAX_MEL_LENGTH = 1000
VOCAB = "abcdefghijklmnopqrstuvwxyz '"
CHAR2IDX = {c: i + 1 for i, c in enumerate(VOCAB)}
UNSAMPLE_FACTOR = 8

# ----------------------------------------------------------------
# ✅ 1. Simple tokenizer (for Gemini restored text)
# ----------------------------------------------------------------
def simple_tokenizer(text: str) -> np.ndarray:
    text = text.lower()
    tokens = [CHAR2IDX.get(c, 0) for c in text]
    return np.array(tokens, dtype=np.int32)


# ----------------------------------------------------------------
# ✅ 2. Load model and compile for inference
# ----------------------------------------------------------------
def build_text_to_mel_model(vocab_size=30, speaker_embedding_dim=1024, mel_dim=MEL_DIM, max_len=MAX_MEL_LENGTH, upsample_factor=UNSAMPLE_FACTOR):
    text_inputs = tf.keras.Input(shape=(None,), dtype='int32', name="text_inputs")
    speaker_inputs = tf.keras.Input(shape=(speaker_embedding_dim,), name="speaker_inputs")

    # Text token embedding
    x = tf.keras.layers.Embedding(vocab_size, 512)(text_inputs)

    # Sinusoidal positional encoding
    def positional_encoding(seq_len, d_model):
        pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
        angle_rates = 1 / tf.pow(10000., (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        angle_rads = pos * angle_rates
        sines = tf.sin(angle_rads[:, 0::2])
        cosines = tf.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        return pos_encoding

    def add_pos_encoding(x):
        seq_len = tf.shape(x)[1]
        pos_enc = positional_encoding(max_len, tf.shape(x)[-1])[:seq_len]
        return x + pos_enc

    x = tf.keras.layers.Lambda(add_pos_encoding)(x)

    # Transformer Encoder Layers
    for _ in range(4):
        attn_output = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
        x = tf.keras.layers.LayerNormalization()(x + attn_output)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(512)
        ])
        x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    # Tile speaker embedding across time
    def repeat_speaker(inputs):
        speaker, tokens = inputs
        seq_len = tf.shape(tokens)[1]
        speaker = tf.expand_dims(speaker, 1)
        return tf.tile(speaker, [1, seq_len, 1])

    speaker_embed_tiled = tf.keras.layers.Lambda(repeat_speaker)([speaker_inputs, x])
    x = tf.keras.layers.Concatenate(axis=-1)([x, speaker_embed_tiled])  # [B, seq_len, 512 + 192]

    # Upsample sequence length
    x = tf.keras.layers.UpSampling1D(size=upsample_factor)(x)

    # Decoder: add LSTM layers for temporal modeling
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    x = tf.keras.layers.Conv1D(512, kernel_size=5, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(mel_dim, kernel_size=1, padding='same')(x)

    # Resize to fixed mel length
    def resize_to_fixed_length(x):
        pad = tf.zeros([tf.shape(x)[0], MAX_MEL_LENGTH, tf.shape(x)[2]], dtype=x.dtype)
        x_padded = tf.concat([x, pad], axis=1)
        return x_padded[:, :MAX_MEL_LENGTH, :]

    x = tf.keras.layers.Lambda(resize_to_fixed_length, name="fixed_output_length")(x)

    return tf.keras.Model(
        inputs=[text_inputs, speaker_inputs],
        outputs=x,
        name="text_to_mel_model"
    )


    # text_inputs = tf.keras.Input(shape=(None,), dtype='int32', name="text_inputs")
    # speaker_inputs = tf.keras.Input(shape=(speaker_embedding_dim,), name="speaker_inputs")

    # x = tf.keras.layers.Embedding(vocab_size, 512)(text_inputs)

    # # Sinusoidal positional encoding
    # def positional_encoding(seq_len, d_model):
    #     pos = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
    #     i = tf.range(d_model, dtype=tf.float32)[tf.newaxis, :]
    #     angle_rates = 1 / tf.pow(10000., (2 * (i // 2)) / tf.cast(d_model, tf.float32))
    #     angle_rads = pos * angle_rates
    #     sines = tf.sin(angle_rads[:, 0::2])
    #     cosines = tf.cos(angle_rads[:, 1::2])
    #     return tf.concat([sines, cosines], axis=-1)

    # def add_pos_encoding(x):
    #     seq_len = tf.shape(x)[1]
    #     pos_enc = positional_encoding(max_len, tf.shape(x)[-1])[:seq_len]
    #     return x + pos_enc

    # x = tf.keras.layers.Lambda(add_pos_encoding)(x)

    # for _ in range(4):
    #     attn = tf.keras.layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    #     x = tf.keras.layers.LayerNormalization()(x + attn)

    #     ffn = tf.keras.Sequential([
    #         tf.keras.layers.Dense(512, activation='relu'),
    #         tf.keras.layers.Dense(512)
    #     ])
    #     x = tf.keras.layers.LayerNormalization()(x + ffn(x))

    # # Repeat speaker embedding
    # def repeat_speaker(inputs):
    #     speaker, tokens = inputs
    #     seq_len = tf.shape(tokens)[1]
    #     speaker = tf.expand_dims(speaker, 1)
    #     return tf.tile(speaker, [1, seq_len, 1])

    # speaker_embed_tiled = tf.keras.layers.Lambda(repeat_speaker)([speaker_inputs, x])
    # x = tf.keras.layers.Concatenate(axis=-1)([x, speaker_embed_tiled])

    # x = tf.keras.layers.UpSampling1D(size=8)(x)
    # x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(256, return_sequences=True))(x)
    # x = tf.keras.layers.Conv1D(512, kernel_size=5, padding='same', activation='relu')(x)
    # x = tf.keras.layers.Conv1D(mel_dim, kernel_size=1, padding='same')(x)

    # def resize_to_fixed_length(x):
    #     pad = tf.zeros([tf.shape(x)[0], MAX_MEL_LENGTH, tf.shape(x)[2]], dtype=x.dtype)
    #     x_padded = tf.concat([x, pad], axis=1)
    #     return x_padded[:, :MAX_MEL_LENGTH, :]

    # x = tf.keras.layers.Lambda(resize_to_fixed_length)(x)

    # return tf.keras.Model(inputs=[text_inputs, speaker_inputs], outputs=x)


# ----------------------------------------------------------------
# ✅ 3. Load the model and weights
# ----------------------------------------------------------------
def load_text_to_mel_model(weights_path: str, vocab_size: int = 30):
    model = build_text_to_mel_model(vocab_size=vocab_size)
    model.load_weights(weights_path)
    print(f"✅ Text-to-Mel model loaded from: {weights_path}")
    return model


# ----------------------------------------------------------------
# ✅ 4. Predict mel from text + speaker embedding
# ----------------------------------------------------------------
def predict_mel_from_text(text: str, speaker_embedding: np.ndarray, model) -> np.ndarray:
    tokens = simple_tokenizer(text)
    tokens = np.expand_dims(tokens, axis=0)  # [1, L]
    speaker_embedding = np.expand_dims(speaker_embedding, axis=0)  # [1, 1024]

    mel_pred = model.predict([tokens, speaker_embedding], verbose=0)[0]
    return mel_pred  # shape: (MAX_MEL_LENGTH, 80)
