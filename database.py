import os
import sqlite3

DB_FILE = os.path.join(os.path.dirname(__file__), "medical.db")

def get_connection():
    """
    Returns a connection to the SQLite database.
    Enables dictionary-like row factories.
    """
    conn = sqlite3.connect(DB_FILE)
    # Enable foreign keys
    conn.execute("PRAGMA foreign_keys = ON;")
    # Set row factory to return dict-like rows
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Initializes the database schema if tables do not exist.
    """
    conn = get_connection()
    try:
        with conn:
            # Create patients table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS patients (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    age INTEGER NOT NULL,
                    gender TEXT NOT NULL,
                    medical_record_number TEXT UNIQUE NOT NULL
                );
                """
            )
            # Create scans table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS scans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id INTEGER NOT NULL,
                    scan_type TEXT NOT NULL CHECK (scan_type IN ('X_RAY', 'MRI')),
                    image_path TEXT NOT NULL,
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                );
                """
            )
            # Create predictions table
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    scan_id INTEGER NOT NULL,
                    predicted_class TEXT NOT NULL,
                    confidence_score REAL NOT NULL,
                    gradcam_path TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
                );
                """
            )
        print("SQLite Database initialized successfully.")
    except Exception as e:
        print(f"Error initializing SQLite database: {e}")
        raise e
    finally:
        conn.close()

def add_patient(name, age, gender, medical_record_number):
    """
    Inserts a new patient and returns the registered patient row.
    """
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO patients (name, age, gender, medical_record_number)
                VALUES (?, ?, ?, ?);
                """,
                (name, age, gender, medical_record_number)
            )
            patient_id = cur.lastrowid
            
        # Fetch and return the newly inserted patient
        return get_patient_by_id(patient_id)
    finally:
        conn.close()

def get_all_patients():
    """
    Retrieves all registered patients.
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM patients ORDER BY name ASC;")
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()

def get_patient_by_id(patient_id):
    """
    Retrieves a single patient by ID.
    """
    conn = get_connection()
    try:
        cur = conn.execute("SELECT * FROM patients WHERE id = ?;", (patient_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def add_scan(patient_id, scan_type, image_path):
    """
    Records a new scan upload for a patient.
    """
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO scans (patient_id, scan_type, image_path)
                VALUES (?, ?, ?);
                """,
                (patient_id, scan_type, image_path)
            )
            scan_id = cur.lastrowid
            
        # Fetch and return the scan
        cur = conn.execute("SELECT * FROM scans WHERE id = ?;", (scan_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def add_prediction(scan_id, predicted_class, confidence_score, gradcam_path):
    """
    Saves a classification prediction with confidence and Grad-CAM path.
    """
    conn = get_connection()
    try:
        with conn:
            cur = conn.execute(
                """
                INSERT INTO predictions (scan_id, predicted_class, confidence_score, gradcam_path)
                VALUES (?, ?, ?, ?);
                """,
                (scan_id, predicted_class, confidence_score, gradcam_path)
            )
            pred_id = cur.lastrowid
            
        # Fetch and return the prediction
        cur = conn.execute("SELECT * FROM predictions WHERE id = ?;", (pred_id,))
        row = cur.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()

def get_patient_predictions(patient_id):
    """
    Retrieves the prediction history of a specific patient by joining
    patients, scans, and predictions.
    """
    conn = get_connection()
    try:
        cur = conn.execute(
            """
            SELECT 
                s.scan_type,
                s.image_path,
                p.predicted_class,
                p.confidence_score,
                p.gradcam_path,
                p.created_at
            FROM scans s
            JOIN predictions p ON s.id = p.scan_id
            WHERE s.patient_id = ?
            ORDER BY p.created_at DESC;
            """,
            (patient_id,)
        )
        rows = cur.fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
