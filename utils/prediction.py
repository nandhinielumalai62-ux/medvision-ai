import tensorflow as tf
import keras
import numpy as np
from PIL import Image
# Import gradcam to generate the heatmap during prediction
from . import gradcam 

# --- HACK Fix for Keras 3 parsing old models with quantization_config ---
original_dense_init = keras.layers.Dense.__init__
def patched_dense_init(self, units, activation=None, use_bias=True, kernel_initializer="glorot_uniform", bias_initializer="zeros", kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None, **kwargs):
    kwargs.pop("quantization_config", None)
    original_dense_init(self, units=units, activation=activation, use_bias=use_bias, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer, kernel_regularizer=kernel_regularizer, bias_regularizer=bias_regularizer, activity_regularizer=activity_regularizer, kernel_constraint=kernel_constraint, bias_constraint=bias_constraint, **kwargs)
keras.layers.Dense.__init__ = patched_dense_init
# -----------------------------------------------

def load_pneumonia_model(path):
    return tf.keras.models.load_model(path, compile=False)

def preprocess(img):
    img = img.resize((224, 224)).convert("RGB")
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

def predict_image(model, img_array):
    # 1. Get the raw prediction
    prediction = model.predict(img_array)[0][0]
    
    # 2. Generate the Grad-CAM heatmap
    # This ensures 'heatmap' is defined before returning
    try:
        heatmap = gradcam.generate_gradcam(model, img_array)
    except Exception as e:
        # Fallback to a blank heatmap if Grad-CAM fails to prevent app crash
        heatmap = np.zeros((224, 224))
        print(f"Grad-CAM Error: {e}")

    # 3. Determine label and confidence
    if prediction > 0.5:
        label, conf = "PNEUMONIA", prediction
    else:
        label, conf = "NORMAL", 1 - prediction

    # 4. RETURN ALL THREE (Fixes the Unpack Error)
    return label, conf, heatmap