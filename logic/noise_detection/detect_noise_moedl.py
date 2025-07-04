import tensorflow as tf
from tensorflow.keras import layers, models


# --- F1 Metric --------------------------------------------------------------
@tf.function
def f1_metric_05(y_true, y_pred):
    y_pred_ = tf.cast(y_pred > 0.5, tf.float32)

    # משטחים את כל הפריימים והתוויות
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_, [-1])

    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))

    precision = tp / (tp + fp + 1e-7)
    recall = tp / (tp + fn + 1e-7)
    f1 = 2 * precision * recall / (precision + recall + 1e-7)
    return f1


# --- Model Architecture -----------------------------------------------------
def build_model(input_shape=(64, None, 1)):
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.2)(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.25)(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D((2, 1))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((-1, x.shape[2] * x.shape[3]))(x)

    x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Bidirectional(layers.LSTM(64, return_sequences=True))(x)
    x = layers.Dropout(0.4)(x)

    # פלט: 3 סוגי רעש לכל פריים
    outputs = layers.TimeDistributed(layers.Dense(3, activation='sigmoid'))(x)

    model = models.Model(inputs, outputs)
    return model


# --- Load Model -------------------------------------------------------------
def load_noise_detectoion_model(weights_path: str):
    model = build_model()
    model.load_weights(weights_path)
    print(f"✅ Noise-Detection model loaded from: {weights_path}")
    return model
