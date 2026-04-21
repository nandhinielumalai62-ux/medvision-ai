import tensorflow as tf
import numpy as np
import cv2

def generate_gradcam(model, img_array, last_conv_layer_name=None):
    """
    Computes Grad-CAM heatmaps for a given model and image array.
    """
    _ = model(img_array)
    
    if last_conv_layer_name is None:
        last_conv_layer_name = [l.name for l in model.layers if "conv" in l.name.lower()][-1]

    # Create model_1: From input to the last conv layer
    model_1 = tf.keras.models.Model(model.inputs, model.get_layer(last_conv_layer_name).output)
    
    # Create model_2: From the last conv layer to the final prediction
    last_conv_idx = model.layers.index(model.get_layer(last_conv_layer_name))
    
    conv_input = tf.keras.Input(shape=model.layers[last_conv_idx].output.shape[1:])
    x = conv_input
    for layer in model.layers[last_conv_idx+1:]:
        x = layer(x)
    model_2 = tf.keras.models.Model(conv_input, x)

    # Gradient tracking
    with tf.GradientTape() as tape:
        # Get activations from model_1
        last_conv_layer_output = model_1(img_array)
        tape.watch(last_conv_layer_output)
        
        # Get predictions from model_2
        preds = model_2(last_conv_layer_output)
        
        if preds.shape[-1] > 1:
            top_pred_index = tf.argmax(preds[0])
            class_channel = preds[:, top_pred_index]
        else:
            class_channel = preds[:, 0]

    grads = tape.gradient(class_channel, last_conv_layer_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.4):
    """
    Combines the original image with the colorized thermal heatmap.
    """
    # Convert original PIL image to NumPy BGR (OpenCV format)
    img_np = np.array(img.convert('RGB'))
    
    # Resize heatmap to match the input image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    
    # Apply JET colormap (blue = low intensity, red = high/anomaly)
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Blend images: Result = (Original * 0.6) + (Heatmap * 0.4)
    superimposed_img = cv2.addWeighted(img_np, 1 - alpha, heatmap_color, alpha, 0)
    
    return superimposed_img