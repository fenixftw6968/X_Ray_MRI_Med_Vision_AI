-- Database schema for Medical Image Classification System

-- Drop tables if they exist to allow easy resets
DROP TABLE IF EXISTS predictions CASCADE;
DROP TABLE IF EXISTS scans CASCADE;
DROP TABLE IF EXISTS patients CASCADE;

-- 1. Patients table
CREATE TABLE patients (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    age INT NOT NULL,
    gender VARCHAR(20) NOT NULL,
    medical_record_number VARCHAR(50) UNIQUE NOT NULL
);

-- 2. Scans table
CREATE TABLE scans (
    id SERIAL PRIMARY KEY,
    patient_id INT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    scan_type VARCHAR(20) NOT NULL CHECK (scan_type IN ('X_RAY', 'MRI')),
    image_path TEXT NOT NULL,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Predictions table
CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,
    scan_id INT NOT NULL REFERENCES scans(id) ON DELETE CASCADE,
    predicted_class VARCHAR(50) NOT NULL,
    confidence_score DOUBLE PRECISION NOT NULL,
    gradcam_path TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
