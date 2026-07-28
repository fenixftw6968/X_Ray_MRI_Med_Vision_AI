import cv2
import numpy as np
import tensorflow as tf

def find_last_conv_layer(model):
    """
    Dynamically finds the name of the last Conv2D layer in a model or its nested sub-models.
    """
    # Check top-level layers
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
            
    # Check nested base models (like densenet121)
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.Model) or hasattr(layer, "layers"):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name
                    
    raise ValueError("No Conv2D layer found in the model structure.")

def generate_gradcam(img_path, model, target_size=(224, 224), alpha=0.4):
    """
    Generates a Grad-CAM overlay heatmap on the original image.
    
    Args:
        img_path: Path to the input image file.
        model: Trained Keras model.
        target_size: Input size of the model.
        alpha: Opacity of the heatmap overlay.
        
    Returns:
        heatmap_overlay: BGR image with the Grad-CAM heatmap superimposed.
        predicted_class_idx: Integer index of the predicted class.
        confidence: Probability of the predicted class.
    """
    # 1. Load and preprocess the image
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not load image at: {img_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size)
    img_array = img_resized.astype(np.float32)
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_array)
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)

    # 2. Identify the base model and last conv layer name
    base_model = None
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) or "densenet" in layer.name:
            base_model = layer
            break
            
    if base_model is None:
        base_model = model

    last_conv_layer_name = find_last_conv_layer(model)
    
    # 3. Create a model mapping input to last conv layer activation & base model outputs
    try:
        grad_model = tf.keras.Model(
            inputs=[base_model.inputs],
            outputs=[base_model.get_layer(last_conv_layer_name).output, base_model.output]
        )
    except Exception:
        # Fallback for flat model
        grad_model = tf.keras.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(last_conv_layer_name).output, model.output]
        )

    # 4. Extract classification head layers of outer model
    classifier_layers = []
    found_base = False
    for layer in model.layers:
        if found_base:
            classifier_layers.append(layer)
        if layer == base_model or "densenet" in layer.name:
            found_base = True

    # 5. Compute the gradient of the predicted class w.r.t the activations of the last conv layer
    with tf.GradientTape() as tape:
        conv_outputs, base_outputs = grad_model(img_preprocessed)
        
        # Run base output through the rest of the outer classifier layers
        x = base_outputs
        for layer in classifier_layers:
            if isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training=False)
            else:
                x = layer(x)
        preds = x
        
        predicted_class_idx = tf.argmax(preds[0]).numpy()
        confidence = float(preds[0][predicted_class_idx])
        class_channel = preds[:, predicted_class_idx]

    # Gradient of the class channel w.r.t the conv outputs
    grads = tape.gradient(class_channel, conv_outputs)
    
    # Mean intensity of gradients per channel (global average pooled)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map by "how important it is"
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # ReLU activation to retain positive features only, then normalize to [0, 1]
    heatmap = tf.maximum(heatmap, 0.0)
    max_val = tf.math.reduce_max(heatmap)
    if max_val > 0:
        heatmap = heatmap / max_val
        
    heatmap = heatmap.numpy()

    # 6. Overlay heatmap onto the original image
    # Resize heatmap to match the original input image size
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    
    # Convert to JET colormap (RGB)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Superimpose heatmap and original image
    superimposed = heatmap_colored * alpha + img_rgb * (1.0 - alpha)
    superimposed = np.clip(superimposed, 0, 255).astype(np.uint8)
    
    # Return as BGR (OpenCV format) for saving
    superimposed_bgr = cv2.cvtColor(superimposed, cv2.COLOR_RGB2BGR)
    
    return superimposed_bgr, predicted_class_idx, confidence
