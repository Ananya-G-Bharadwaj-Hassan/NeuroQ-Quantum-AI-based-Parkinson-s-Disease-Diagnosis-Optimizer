import json
import os
import re
import secrets
import tempfile
import traceback
from datetime import datetime, timezone
from functools import wraps

from bson import ObjectId
from bson.errors import InvalidId
from flask import Flask, jsonify, request, send_from_directory, session
from pymongo import MongoClient, ASCENDING
from pymongo.errors import DuplicateKeyError, ConnectionFailure, ServerSelectionTimeoutError
from werkzeug.security import check_password_hash, generate_password_hash

from src import config
import predict_patient as pp

# ------------------------------------------------------------------ #
#  App bootstrap & security                                            #
# ------------------------------------------------------------------ #
KEY_PATH = os.path.join(config.INSTANCE_DIR, "secret.key")

app = Flask(__name__, static_folder="static", static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 30 * 1024 * 1024  # 30 MB upload cap for large DICOM/MRI
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"


def _secret_key():
    """Persistent random key (generated on first run, never hardcoded)."""
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    if not os.path.exists(KEY_PATH):
        os.makedirs(config.INSTANCE_DIR, exist_ok=True)
        with open(KEY_PATH, "w") as f:
            f.write(secrets.token_hex(32))
    try:
        return open(KEY_PATH).read().strip()
    except Exception:
        return secrets.token_hex(32)


app.secret_key = _secret_key()

# ------------------------------------------------------------------ #
#  MongoDB Atlas connection                                            #
# ------------------------------------------------------------------ #
MONGO_URI = os.environ.get("MONGODB_URI", "mongodb://localhost:27017/neuroq")

mongo_client = None
mongo_db = None
users_col = None
assessments_col = None

try:
    mongo_client = MongoClient(
        MONGO_URI,
        serverSelectionTimeoutMS=4000,
        connectTimeoutMS=4000,
        retryWrites=True,
    )
    # Determine DB name from URI or default to 'neuroq'
    try:
        mongo_db = mongo_client.get_default_database()
    except Exception:
        mongo_db = mongo_client["neuroq"]
    if mongo_db is None or mongo_db.name is None:
        mongo_db = mongo_client["neuroq"]

    users_col = mongo_db["users"]
    assessments_col = mongo_db["assessments"]

    # Ensure unique email index for login accounts
    users_col.create_index([("email", ASCENDING)], unique=True)
    assessments_col.create_index([("timestamp", -1)])
    assessments_col.create_index([("user_id", ASCENDING)])
    print(f"[MongoDB] Connected successfully to database '{mongo_db.name}'")
except Exception as exc:
    print(f"[MongoDB Warning] Connection init: {exc}")
    # Will attempt reconnect on request if disconnected


def get_mongo_db():
    """Ensure database connection is active."""
    global mongo_client, mongo_db, users_col, assessments_col
    if mongo_db is not None:
        return mongo_db
    try:
        mongo_client = MongoClient(
            MONGO_URI,
            serverSelectionTimeoutMS=4000,
            connectTimeoutMS=4000,
            retryWrites=True,
        )
        try:
            mongo_db = mongo_client.get_default_database()
        except Exception:
            mongo_db = mongo_client["neuroq"]
        if mongo_db is None or mongo_db.name is None:
            mongo_db = mongo_client["neuroq"]
        users_col = mongo_db["users"]
        assessments_col = mongo_db["assessments"]
        users_col.create_index([("email", ASCENDING)], unique=True)
        return mongo_db
    except Exception as exc:
        print(f"[MongoDB Error] Reconnection failed: {exc}")
        return None


# ------------------------------------------------------------------ #
#  Auth helpers                                                        #
# ------------------------------------------------------------------ #
def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            return jsonify(error="Please sign in first."), 401
        return fn(*a, **kw)
    return wrapper


def doctor_required(fn):
    """Allows only clinical staff (doctors) to access specific restricted actions."""
    @wraps(fn)
    def wrapper(*a, **kw):
        if "user_id" not in session:
            return jsonify(error="Please sign in first."), 401
        if session.get("role") != "doctor":
            return jsonify(error="This action requires a doctor account."), 403
        return fn(*a, **kw)
    return wrapper


# ------------------------------------------------------------------ #
#  Frontend Static Serving                                             #
# ------------------------------------------------------------------ #
@app.route("/")
def index():
    return send_from_directory("static", "index.html")


# ------------------------------------------------------------------ #
#  Health & Telemetry Endpoint (Deployment check)                     #
# ------------------------------------------------------------------ #
@app.route("/api/health")
def health():
    db_ok = False
    db_name = "unknown"
    try:
        db = get_mongo_db()
        if db is not None:
            mongo_client.admin.command("ping")
            db_ok = True
            db_name = db.name
    except Exception as e:
        db_ok = False

    clinical_model_ready = os.path.exists(os.path.join(config.MODELS_DIR, "clinical_diagnosis.joblib"))
    imaging_model_ready = os.path.exists(os.path.join(config.MODELS_DIR, "imaging_diagnosis.joblib"))

    return jsonify({
        "status": "healthy" if db_ok and clinical_model_ready else "degraded",
        "mongodb": {
            "connected": db_ok,
            "database": db_name,
            "provider": "MongoDB Atlas" if "mongodb+srv" in MONGO_URI or "mongodb.net" in MONGO_URI else "Local/Standalone MongoDB"
        },
        "models": {
            "clinical_diagnosis": clinical_model_ready,
            "imaging_diagnosis": imaging_model_ready,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


# ------------------------------------------------------------------ #
#  Authentication Endpoints (Doctors & Patients)                       #
# ------------------------------------------------------------------ #
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


@app.post("/api/register")
def register():
    """Register a new Doctor or Patient account stored in MongoDB Atlas.

    JSON parameters:
        name     - Full name
        email    - Valid unique email address
        password - Minimum 8 characters
        role     - "doctor" (default) or "patient"
        hospital - Optional hospital or clinic affiliation
    """
    d = request.get_json(silent=True) or {}
    name = (d.get("name") or "").strip()
    email = (d.get("email") or "").strip().lower()
    hospital = (d.get("hospital") or "").strip()
    password = d.get("password") or ""
    role = (d.get("role") or "doctor").strip().lower()

    if role not in ("doctor", "patient"):
        return jsonify(error="Role must be 'doctor' or 'patient'."), 400
    if not name:
        return jsonify(error="Name is required."), 400
    if not EMAIL_RE.match(email):
        return jsonify(error="Enter a valid email address."), 400
    if len(password) < 8:
        return jsonify(error="Password must be at least 8 characters."), 400

    db = get_mongo_db()
    if db is None:
        return jsonify(error="Database connection unavailable. Please check MongoDB Atlas connection."), 503

    doc = {
        "name": name,
        "email": email,
        "hospital": hospital,
        "role": role,
        "password_hash": generate_password_hash(password),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        result = users_col.insert_one(doc)
    except DuplicateKeyError:
        return jsonify(error="An account with this email already exists. Please sign in."), 409
    except Exception as e:
        return jsonify(error=f"Database registration error: {str(e)}"), 500

    session.clear()
    session["user_id"] = str(result.inserted_id)
    session["role"] = role

    user_info = {
        "name": name,
        "email": email,
        "hospital": hospital,
        "role": role
    }

    # Return both user and doctor keys for full frontend backward/forward compatibility
    return jsonify(ok=True, user=user_info, doctor=user_info), 201


@app.post("/api/login")
def login():
    """Sign in using email and password stored in MongoDB Atlas.
    Works for both doctors and registered patients.
    """
    d = request.get_json(silent=True) or {}
    email = (d.get("email") or "").strip().lower()
    password = d.get("password") or ""

    if not email or not password:
        return jsonify(error="Email and password are required."), 400

    db = get_mongo_db()
    if db is None:
        return jsonify(error="Database connection unavailable. Please verify MongoDB Atlas connection."), 503

    try:
        row = users_col.find_one({"email": email})
    except Exception as e:
        return jsonify(error=f"Database query error: {str(e)}"), 500

    if not row or not check_password_hash(row["password_hash"], password):
        return jsonify(error="Incorrect email or password."), 401

    session.clear()
    session["user_id"] = str(row["_id"])
    session["role"] = row.get("role", "doctor")

    user_info = {
        "name": row["name"],
        "email": row["email"],
        "hospital": row.get("hospital", ""),
        "role": row.get("role", "doctor"),
    }

    # Returns both user and doctor for seamless backward compatibility
    return jsonify(ok=True, user=user_info, doctor=user_info)


@app.post("/api/logout")
def logout():
    session.clear()
    return jsonify(ok=True)


@app.get("/api/me")
def me():
    """Returns current active authenticated session user."""
    if "user_id" not in session:
        return jsonify(user=None, doctor=None)

    db = get_mongo_db()
    if db is None:
        return jsonify(user=None, doctor=None)

    try:
        oid = ObjectId(session["user_id"])
        row = users_col.find_one({"_id": oid})
    except (InvalidId, Exception):
        session.clear()
        return jsonify(user=None, doctor=None)

    if not row:
        session.clear()
        return jsonify(user=None, doctor=None)

    user_info = {
        "name": row["name"],
        "email": row["email"],
        "hospital": row.get("hospital", ""),
        "role": row.get("role", "doctor"),
    }
    return jsonify(user=user_info, doctor=user_info)


# ------------------------------------------------------------------ #
#  Prediction, Benchmarks & Diagnostic Telemetry                      #
# ------------------------------------------------------------------ #
@app.get("/api/features")
@login_required
def features():
    """Returns clinical feature definitions, bounds, and trained model metadata."""
    try:
        art = pp.load_artifact("clinical_diagnosis.joblib", "train_diagnosis.py")
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    sc = art["scaler_classical"]
    return jsonify(
        features=[
            {"name": n, "min": float(sc.data_min_[i]), "max": float(sc.data_max_[i])}
            for i, n in enumerate(art["feature_names"])
        ],
        classical_model=art.get("classical_model_name", "xgboost"),
        quantum_model=art.get("quantum_model_name", "quantum_kernel_svm"),
        quantum_features=art.get("quantum_feature_names", []),
        classical_accuracy=art.get("classical_accuracy"),
        quantum_accuracy=art.get("quantum_accuracy"),
        hybrid_accuracy=art.get("hybrid_accuracy"),
    )


@app.get("/api/benchmarks")
@login_required
def benchmarks():
    """Returns validation metrics comparing Classical, Quantum, and Hybrid Stacking."""
    out = {}
    diag_path = os.path.join(config.RESULTS_DIR, "diagnosis_metrics.json")
    if os.path.exists(diag_path):
        try:
            out["diagnosis"] = json.load(open(diag_path))
        except Exception:
            pass

    img_path = os.path.join(config.RESULTS_DIR, "imaging_diagnosis_metrics.json")
    if os.path.exists(img_path):
        try:
            out["imaging"] = json.load(open(img_path))
        except Exception:
            pass

    return jsonify(out)


@app.post("/api/predict")
@login_required
def predict():
    """Multipart form submission: clinical feature fields + optional 'mri' scan.

    Executes multimodal diagnostic synthesis:
      1. Clinical Hybrid Model: Classical Gradient Boosted Trees (XGBoost) +
         Angle-Embedded Quantum Kernel SVM stacked via Logistic Meta-Learner.
      2. Brain MRI Radiomics Hybrid (Optional): Texture & intensity radiomic
         feature extraction fused with Quantum Hilbert-space Kernel.
      3. Result is stored in MongoDB Atlas under the 'assessments' collection.
    """
    try:
        clin = pp.load_artifact("clinical_diagnosis.joblib", "train_diagnosis.py")
    except FileNotFoundError as e:
        return jsonify(error=str(e)), 503

    raw_qw = request.form.get("quantum_weight")
    qweight = None
    if raw_qw is not None and raw_qw.strip() != "":
        try:
            qweight = min(1.0, max(0.0, float(raw_qw)))
        except ValueError:
            qweight = None

    try:
        clinical, defaulted = pp.clinical_hybrid_from_values(request.form, clin, qweight)
    except ValueError:
        return jsonify(error="All clinical fields must be valid numeric values."), 400

    out = {
        "clinical": clinical,
        "defaulted_features": defaulted,
        "mri_used": False,
        "p_final": clinical["p_hybrid"],
        "confidence": clinical.get("confidence", 0.0),
        "confidence_band": clinical.get("confidence_band", "Moderate"),
    }

    f = request.files.get("mri")
    if f and f.filename and f.filename.strip():
        try:
            mri_weight = min(1.0, max(0.0, float(request.form.get("mri_weight", 0.5))))
        except ValueError:
            mri_weight = 0.5

        try:
            img = pp.load_artifact("imaging_diagnosis.joblib", "train_imaging.py")
        except FileNotFoundError as e:
            return jsonify(error="MRI model not trained yet: " + str(e)), 503

        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".dcm"):
            return jsonify(error="Unsupported format. Please upload a brain MRI scan in .png, .jpg, .tiff, or .dcm (DICOM) format."), 400

        # Safe tempfile handling on Windows: close OS handle before PIL/pydicom opens it
        tmp_fd, path = tempfile.mkstemp(suffix=ext)
        try:
            os.close(tmp_fd)
            f.save(path)
            try:
                imaging = pp.imaging_hybrid(path, img, qweight)
            except Exception as exc:
                print(f"[MRI Processing Error]: {exc}")
                traceback.print_exc()
                return jsonify(
                    error=f"MRI feature extraction failed: {str(exc)}. Ensure the image is a valid brain MRI scan."
                ), 400
        finally:
            if os.path.exists(path):
                try:
                    os.unlink(path)
                except Exception:
                    pass

        p_final = mri_weight * imaging["p_hybrid"] + (1.0 - mri_weight) * clinical["p_hybrid"]
        out.update(
            mri_used=True,
            imaging=imaging,
            mri_weight=mri_weight,
            p_final=p_final,
            confidence=abs(p_final - 0.5) * 2.0,
            confidence_band="High" if abs(p_final - 0.5) >= 0.3 else "Moderate",
        )

    out["label"] = "Parkinson's likely" if out["p_final"] >= 0.5 else "Parkinson's unlikely"

    # Persist assessment record in MongoDB Atlas
    db = get_mongo_db()
    if db is not None:
        try:
            patient_name = (request.form.get("patient_name") or "").strip() or "Unnamed Patient"
            record = {
                "user_id": session.get("user_id"),
                "user_role": session.get("role", "doctor"),
                "patient_name": patient_name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "p_final": float(out["p_final"]),
                "label": out["label"],
                "confidence": float(out["confidence"]),
                "confidence_band": out["confidence_band"],
                "mri_used": bool(out["mri_used"]),
                "clinical_hybrid_p": float(out["clinical"]["p_hybrid"]),
                "imaging_hybrid_p": float(out["imaging"]["p_hybrid"]) if out["mri_used"] else None,
            }
            assessments_col.insert_one(record)
        except Exception as e:
            print(f"[MongoDB Warning] Could not store assessment record: {e}")

    return jsonify(out)


@app.get("/api/history")
@login_required
def history():
    """Retrieve past patient assessment records stored in MongoDB Atlas."""
    db = get_mongo_db()
    if db is None:
        return jsonify(records=[])

    try:
        user_id = session.get("user_id")
        query = {"user_id": user_id} if session.get("role") == "patient" else {}
        cursor = assessments_col.find(query).sort("timestamp", -1).limit(20)
        items = []
        for doc in cursor:
            items.append({
                "id": str(doc["_id"]),
                "patient_name": doc.get("patient_name", "Unknown"),
                "timestamp": doc.get("timestamp"),
                "p_final": doc.get("p_final"),
                "label": doc.get("label"),
                "confidence_band": doc.get("confidence_band"),
                "mri_used": doc.get("mri_used", False)
            })
        return jsonify(records=items)
    except Exception as e:
        return jsonify(error=str(e), records=[]), 500


# ------------------------------------------------------------------ #
#  Main entrypoint                                                     #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug)
