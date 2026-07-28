import os
import uuid
import tensorflow as tf
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename

import database
from utils.preprocessing import load_and_preprocess_image
from utils.gradcam import generate_gradcam

app = Flask(__name__)

# Configure upload and gradcam folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
GRADCAM_FOLDER = os.path.join('static', 'gradcam')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GRADCAM_FOLDER'] = GRADCAM_FOLDER

# Global dictionaries for models and class mapping
MODELS = {}
CLASSES = {
    'X_RAY': ["COVID19", "NORMAL", "PNEUMONIA"],
    'MRI': ["glioma", "meningioma", "notumor", "pituitary"]
}

def load_models_on_startup():
    """
    Loads both X-ray and MRI models on startup.
    If models do not exist yet (i.e. they are still training), 
    we print a warning but allow the Flask app to start.
    """
    xray_path = "models/xray_model.keras"
    mri_path = "models/mri_model.keras"
    
    if os.path.exists(xray_path):
        print(f"Loading X-Ray model from {xray_path}...")
        try:
            MODELS['X_RAY'] = tf.keras.models.load_model(xray_path)
            print("X-Ray model loaded successfully.")
        except Exception as e:
            print(f"Error loading X-Ray model: {e}")
    else:
        print(f"Warning: X-Ray model file not found at {xray_path}. Train it first.")
        
    if os.path.exists(mri_path):
        print(f"Loading MRI model from {mri_path}...")
        try:
            MODELS['MRI'] = tf.keras.models.load_model(mri_path)
            print("MRI model loaded successfully.")
        except Exception as e:
            print(f"Error loading MRI model: {e}")
    else:
        print(f"Warning: MRI model file not found at {mri_path}. Train it first.")

# Load models when the Flask application starts
load_models_on_startup()

@app.route('/')
def home():
    """Renders the main dashboard."""
    return render_template('index.html')

@app.route('/api/patients', methods=['POST'])
def add_patient_api():
    """Registers a new patient."""
    data = request.get_json() or {}
    name = data.get('name')
    age = data.get('age')
    gender = data.get('gender')
    medical_record_number = data.get('medical_record_number')
    
    if not all([name, age, gender, medical_record_number]):
        return jsonify({"error": "Missing required patient fields"}), 400
        
    try:
        patient = database.add_patient(name, age, gender, medical_record_number)
        return jsonify(patient), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/api/patients', methods=['GET'])
def get_patients_api():
    """Retrieves all patients for selection."""
    try:
        patients = database.get_all_patients()
        return jsonify(patients), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/predict', methods=['POST'])
def predict_api():
    """
    Accepts patientId, scanType, and an uploaded image.
    Performs inference, generates Grad-CAM, saves result in PostgreSQL.
    """
    patient_id = request.form.get('patientId')
    scan_type = request.form.get('scanType')
    image_file = request.files.get('image')
    
    if not all([patient_id, scan_type, image_file]):
        return jsonify({"error": "Missing patientId, scanType, or image file"}), 400
        
    if scan_type not in ['X_RAY', 'MRI']:
        return jsonify({"error": "Invalid scanType. Must be 'X_RAY' or 'MRI'"}), 400

    # Ensure model is loaded
    if scan_type not in MODELS:
        # Retry loading once in case it was trained after startup
        load_models_on_startup()
        if scan_type not in MODELS:
            return jsonify({"error": f"{scan_type} model is not loaded. Train the model first."}), 500

    # Save original uploaded image
    ext = os.path.splitext(image_file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{ext}"
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    image_file.save(image_path)
    
    # Standardize path separator to forward slashes for URLs and storage consistency
    db_image_path = image_path.replace('\\', '/')
    
    try:
        # Run Grad-CAM and inference
        model = MODELS[scan_type]
        gradcam_img, pred_idx, confidence = generate_gradcam(image_path, model, target_size=(224, 224))
        
        # Save Grad-CAM image
        gradcam_filename = f"gradcam_{unique_filename}"
        gradcam_path = os.path.join(app.config['GRADCAM_FOLDER'], gradcam_filename)
        import cv2
        cv2.imwrite(gradcam_path, gradcam_img)
        db_gradcam_path = gradcam_path.replace('\\', '/')
        
        predicted_class = CLASSES[scan_type][pred_idx]
        
        # Save to database
        scan = database.add_scan(patient_id, scan_type, db_image_path)
        prediction = database.add_prediction(scan['id'], predicted_class, confidence, db_gradcam_path)
        
        return jsonify({
            "predictedClass": predicted_class,
            "confidence": confidence,
            "gradCamPath": db_gradcam_path,
            "originalImagePath": db_image_path
        }), 200
        
    except Exception as e:
        # Clean up files if error occurs
        if os.path.exists(image_path):
            os.remove(image_path)
        return jsonify({"error": str(e)}), 500

@app.route('/api/patients/<int:patient_id>/predictions', methods=['GET'])
def get_predictions_api(patient_id):
    """Retrieves previous predictions for a patient."""
    try:
        history = database.get_patient_predictions(patient_id)
        return jsonify(history), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Initialize the database on startup (this runs schema.sql)
    # The database must exist already
    try:
        database.init_db()
    except Exception as e:
        print(f"Skipping DB initialization due to error (check PostgreSQL connection): {e}")
        
    app.run(debug=True, host='0.0.0.0', port=5000)
