import cv2
import numpy as np
import tensorflow as tf

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    """
    Loads an image from disk, converts to RGB, resizes to target_size,
    and applies DenseNet121 preprocessing.
    
    Args:
        image_path: Path to the image file.
        target_size: Tuple (width, height) for resizing.
        
    Returns:
        A preprocessed numpy array with shape (1, height, width, 3) ready for inference.
        A copy of the original RGB image (for visualization/Grad-CAM).
    """
    # Load using OpenCV
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image at: {image_path}")
        
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to target size for visualization
    img_resized = cv2.resize(img_rgb, target_size)
    
    # Convert to float32 and apply DenseNet121 preprocessing
    img_array = img_resized.astype(np.float32)
    # preprocess_input expects inputs in range [0, 255] and normalizes them
    img_preprocessed = tf.keras.applications.densenet.preprocess_input(img_array)
    
    # Add batch dimension
    img_preprocessed = np.expand_dims(img_preprocessed, axis=0)
    
    return img_preprocessed, img_resized
