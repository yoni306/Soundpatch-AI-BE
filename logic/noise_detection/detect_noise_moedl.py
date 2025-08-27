import tensorflow as tf
from tensorflow.keras import layers, models
import os
import numpy as np


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


# --- Focal Loss Function (same as in training) ------------------------------
def focal_loss_dynamic(gamma=2.0, eps=1e-7):
    """Advanced Focal Loss with Dynamic Class Weighting"""
    def loss_fn(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, eps, 1-eps)
        freq_pos = tf.reduce_mean(y_true, axis=[0,1])
        alpha = 1 - freq_pos
        term1 = - alpha * y_true * tf.pow(1-y_pred, gamma) * tf.math.log(y_pred)
        term2 = - (1-alpha) * (1-y_true) * tf.pow(y_pred, gamma) * tf.math.log(1-y_pred)
        return tf.reduce_mean(term1+term2)
    return loss_fn


# --- Load Model -------------------------------------------------------------
def load_noise_detectoion_model(weights_path: str, verbose: bool = True):
    """Load the trained noise detection model with fallback options"""
    
    if verbose:
        print("Loading trained noise detection model...")
    
    # Set function name for proper loading
    f1_metric_05.__name__ = "f1_metric_05"
    
    # Define custom objects exactly as in working notebook
    custom_objects = {
        'focal_loss_dynamic': focal_loss_dynamic,
        'loss_fn': focal_loss_dynamic(),
        'f1_metric_05': f1_metric_05
    }
    
    try:
        # First try: Load with custom objects
        model = tf.keras.models.load_model(weights_path, custom_objects=custom_objects)
        if verbose:
            print(f"Model loaded successfully ({model.count_params():,} parameters)")
        
        # Quick test to verify model is working
        test_input = np.random.random((1, 64, 100, 1)).astype(np.float32)
        test_output = model.predict(test_input, verbose=0)
        if verbose:
            print(f"Model ready for inference")
        
        return model
        
    except Exception as e:
        if verbose:
            print(f"Error loading model: {e}")
        
        # Fallback: Create a new model without loading weights
        if verbose:
            print("Creating new model without pre-trained weights...")
        
        try:
            model = build_model()
            if verbose:
                print(f"New model created successfully ({model.count_params():,} parameters)")
                print("Note: This model will need to be trained or weights need to be loaded manually")
            
            return model
            
        except Exception as e2:
            if verbose:
                print(f"Error creating new model: {e2}")
            return None
