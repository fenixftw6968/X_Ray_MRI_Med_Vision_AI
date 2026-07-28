import os
import sqlite3
from datetime import datetime
from bson import ObjectId

# Try importing pymongo for MongoDB support
try:
    from pymongo import MongoClient
    HAS_MONGO = True
except ImportError:
    HAS_MONGO = False

# Manually load .env file if it exists to retrieve database credentials
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("=", 1)
                if len(parts) == 2:
                    key, val = parts[0].strip(), parts[1].strip()
                    # Strip quotes if present
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    os.environ[key] = val

MONGO_URI = os.environ.get("MONGO_URI")
USE_MONGO = HAS_MONGO and (MONGO_URI is not None)

DB_FILE = os.path.join(os.path.dirname(__file__), "medical.db")

# Setup Mongo Database Client
if USE_MONGO:
    try:
        mongo_client = MongoClient(MONGO_URI)
        # Access database (default from URI or fallback to 'medical_db' to avoid ConfigurationError)
        try:
            db = mongo_client.get_default_database()
            if db is None:
                db = mongo_client["medical_db"]
        except Exception:
            db = mongo_client["medical_db"]
    except Exception as e:
        print(f"MongoDB connection configuration error: {e}")
        db = None
        USE_MONGO = False
else:
    db = None


def serialize_doc(doc):
    """Converts MongoDB BSON types like ObjectId and datetime to serializable format."""
    if not doc:
        return None
    doc = dict(doc)
    if "_id" in doc:
        doc["id"] = str(doc["_id"])
        del doc["_id"]
    if "patient_id" in doc:
        doc["patient_id"] = str(doc["patient_id"])
    if "scan_id" in doc:
        doc["scan_id"] = str(doc["scan_id"])
    if "uploaded_at" in doc and isinstance(doc["uploaded_at"], datetime):
        doc["uploaded_at"] = doc["uploaded_at"].isoformat()
    if "created_at" in doc and isinstance(doc["created_at"], datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


def init_db():
    """Initializes the database index constraints (MongoDB) or schemas (SQLite fallback)."""
    if USE_MONGO:
        try:
            # Create a unique index on medical_record_number for patients
            db.patients.create_index("medical_record_number", unique=True)
            print("MongoDB initialized successfully with unique indexes.")
        except Exception as e:
            print(f"Error initializing MongoDB: {e}")
            raise e
    else:
        conn = sqlite3.connect(DB_FILE)
        try:
            with conn:
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
            print("SQLite Fallback Database initialized successfully.")
        except Exception as e:
            print(f"Error initializing SQLite database: {e}")
            raise e
        finally:
            conn.close()


def add_patient(name, age, gender, medical_record_number):
    """Inserts a new patient and returns the registered patient."""
    if USE_MONGO:
        patient_doc = {
            "name": name,
            "age": int(age),
            "gender": gender,
            "medical_record_number": medical_record_number
        }
        res = db.patients.insert_one(patient_doc)
        patient_doc["_id"] = res.inserted_id
        return serialize_doc(patient_doc)
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
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
            return get_patient_by_id(patient_id)
        finally:
            conn.close()


def get_all_patients():
    """Retrieves all registered patients."""
    if USE_MONGO:
        patients = list(db.patients.find().sort("name", 1))
        return [serialize_doc(p) for p in patients]
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT * FROM patients ORDER BY name ASC;")
            rows = cur.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()


def get_patient_by_id(patient_id):
    """Retrieves a single patient by ID."""
    if USE_MONGO:
        try:
            patient = db.patients.find_one({"_id": ObjectId(patient_id)})
            return serialize_doc(patient)
        except Exception:
            return None
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        try:
            cur = conn.execute("SELECT * FROM patients WHERE id = ?;", (patient_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def add_scan(patient_id, scan_type, image_path):
    """Records a new scan upload for a patient."""
    if USE_MONGO:
        scan_doc = {
            "patient_id": ObjectId(patient_id) if ObjectId.is_valid(patient_id) else patient_id,
            "scan_type": scan_type,
            "image_path": image_path,
            "uploaded_at": datetime.utcnow()
        }
        res = db.scans.insert_one(scan_doc)
        scan_doc["_id"] = res.inserted_id
        return serialize_doc(scan_doc)
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
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
            
            cur = conn.execute("SELECT * FROM scans WHERE id = ?;", (scan_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def add_prediction(scan_id, predicted_class, confidence_score, gradcam_path):
    """Saves a classification prediction with confidence and Grad-CAM path."""
    if USE_MONGO:
        prediction_doc = {
            "scan_id": ObjectId(scan_id) if ObjectId.is_valid(scan_id) else scan_id,
            "predicted_class": predicted_class,
            "confidence_score": float(confidence_score),
            "gradcam_path": gradcam_path,
            "created_at": datetime.utcnow()
        }
        res = db.predictions.insert_one(prediction_doc)
        prediction_doc["_id"] = res.inserted_id
        return serialize_doc(prediction_doc)
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
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
            
            cur = conn.execute("SELECT * FROM predictions WHERE id = ?;", (pred_id,))
            row = cur.fetchone()
            return dict(row) if row else None
        finally:
            conn.close()


def get_patient_predictions(patient_id):
    """Retrieves prediction history for a specific patient."""
    if USE_MONGO:
        try:
            pid = ObjectId(patient_id) if ObjectId.is_valid(patient_id) else patient_id
            
            # Use MongoDB aggregation framework to join scans and predictions collections
            pipeline = [
                { "$match": { "patient_id": pid } },
                {
                    "$lookup": {
                        "from": "predictions",
                        "localField": "_id",
                        "foreignField": "scan_id",
                        "as": "preds"
                    }
                },
                { "$unwind": "$preds" },
                {
                    "$project": {
                        "scan_type": 1,
                        "image_path": 1,
                        "predicted_class": "$preds.predicted_class",
                        "confidence_score": "$preds.confidence_score",
                        "gradcam_path": "$preds.gradcam_path",
                        "created_at": "$preds.created_at"
                    }
                },
                { "$sort": { "created_at": -1 } }
            ]
            
            results = list(db.scans.aggregate(pipeline))
            serialized_results = []
            for r in results:
                serialized_results.append({
                    "scan_type": r.get("scan_type"),
                    "image_path": r.get("image_path"),
                    "predicted_class": r.get("predicted_class"),
                    "confidence_score": r.get("confidence_score"),
                    "gradcam_path": r.get("gradcam_path"),
                    "created_at": r.get("created_at").isoformat() if isinstance(r.get("created_at"), datetime) else r.get("created_at")
                })
            return serialized_results
        except Exception as e:
            print(f"Error in MongoDB get_patient_predictions: {e}")
            return []
    else:
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
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
