import os
import sqlite3

# Try importing psycopg2 for PostgreSQL support
try:
    import psycopg2
    import psycopg2.extras
    HAS_POSTGRES = True
except ImportError:
    HAS_POSTGRES = False

DATABASE_URL = os.environ.get("DATABASE_URL")
IS_POSTGRES = HAS_POSTGRES and (DATABASE_URL is not None and DATABASE_URL.startswith("postgres"))

DB_FILE = os.path.join(os.path.dirname(__file__), "medical.db")

def get_connection():
    """
    Returns a connection to the database (either PostgreSQL or SQLite).
    """
    if IS_POSTGRES:
        conn = psycopg2.connect(DATABASE_URL)
        return conn
    else:
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
        if IS_POSTGRES:
            with conn:
                with conn.cursor() as cur:
                    # Create patients table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS patients (
                            id SERIAL PRIMARY KEY,
                            name TEXT NOT NULL,
                            age INTEGER NOT NULL,
                            gender TEXT NOT NULL,
                            medical_record_number TEXT UNIQUE NOT NULL
                        );
                        """
                    )
                    # Create scans table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS scans (
                            id SERIAL PRIMARY KEY,
                            patient_id INTEGER NOT NULL,
                            scan_type TEXT NOT NULL CHECK (scan_type IN ('X_RAY', 'MRI')),
                            image_path TEXT NOT NULL,
                            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (patient_id) REFERENCES patients(id) ON DELETE CASCADE
                        );
                        """
                    )
                    # Create predictions table
                    cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS predictions (
                            id SERIAL PRIMARY KEY,
                            scan_id INTEGER NOT NULL,
                            predicted_class TEXT NOT NULL,
                            confidence_score REAL NOT NULL,
                            gradcam_path TEXT NOT NULL,
                            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                            FOREIGN KEY (scan_id) REFERENCES scans(id) ON DELETE CASCADE
                        );
                        """
                    )
            print("PostgreSQL Database initialized successfully.")
        else:
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
        print(f"Error initializing database: {e}")
        raise e
    finally:
        conn.close()

def add_patient(name, age, gender, medical_record_number):
    """
    Inserts a new patient and returns the registered patient row.
    """
    conn = get_connection()
    try:
        if IS_POSTGRES:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO patients (name, age, gender, medical_record_number)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, name, age, gender, medical_record_number;
                        """,
                        (name, age, gender, medical_record_number)
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
            with conn:
                cur = conn.execute(
                    """
                    INSERT INTO patients (name, age, gender, medical_record_number)
                    VALUES (?, ?, ?, ?);
                    """,
                    (name, age, gender, medical_record_number)
                )
                patient_id = cur.lastrowid
            return get_patient_by_id(patient_id)
    finally:
        conn.close()

def get_all_patients():
    """
    Retrieves all registered patients.
    """
    conn = get_connection()
    try:
        if IS_POSTGRES:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM patients ORDER BY name ASC;")
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        else:
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
        if IS_POSTGRES:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute("SELECT * FROM patients WHERE id = %s;", (patient_id,))
                row = cur.fetchone()
                return dict(row) if row else None
        else:
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
        if IS_POSTGRES:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO scans (patient_id, scan_type, image_path)
                        VALUES (%s, %s, %s)
                        RETURNING id, patient_id, scan_type, image_path, uploaded_at;
                        """,
                        (patient_id, scan_type, image_path)
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
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
        if IS_POSTGRES:
            with conn:
                with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                    cur.execute(
                        """
                        INSERT INTO predictions (scan_id, predicted_class, confidence_score, gradcam_path)
                        VALUES (%s, %s, %s, %s)
                        RETURNING id, scan_id, predicted_class, confidence_score, gradcam_path, created_at;
                        """,
                        (scan_id, predicted_class, confidence_score, gradcam_path)
                    )
                    row = cur.fetchone()
                    return dict(row) if row else None
        else:
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
        if IS_POSTGRES:
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
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
                    WHERE s.patient_id = %s
                    ORDER BY p.created_at DESC;
                    """,
                    (patient_id,)
                )
                rows = cur.fetchall()
                return [dict(row) for row in rows]
        else:
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
