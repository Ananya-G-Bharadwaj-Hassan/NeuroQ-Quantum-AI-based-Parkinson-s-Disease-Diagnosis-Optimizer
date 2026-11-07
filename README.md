# 🧠 NeuroQ — Quantum-AI Based Parkinson's Disease Diagnosis Optimizer

> **A clinical decision support system combining classical ML + quantum computing for early Parkinson's Disease diagnosis.**

🌐 **Live Website:** [https://neuroq-quantum-ai-based-parkinson-s.onrender.com](https://neuroq-quantum-ai-based-parkinson-s.onrender.com)
📦 **GitHub:** [NeuroQ Repository](https://github.com/Ananya-G-Bharadwaj-Hassan/NeuroQ-Quantum-AI-based-Parkinson-s-Disease-Diagnosis-Optimizer)

---

## Table of Contents

1. [Features](#1-features)
2. [Tech Stack](#2-tech-stack)
3. [How It Works](#3-how-it-works)
4. [Results](#4-results)
5. [Project Structure](#5-project-structure)
6. [How to Run](#6-how-to-run)
7. [Usage](#7-usage)
8. [API Reference](#8-api-reference)
9. [Deployment](#9-deployment-render--railway--heroku)
10. [Known Limitations](#10-known-limitations)

---

## 1. Features

- ⚛️ **Hybrid Quantum-Classical Architecture** — Stacking meta-learner combining XGBoost + Quantum Kernel SVM
- 🏥 **Dual Input Modalities** — Accepts clinical patient symptoms AND brain MRI scans
- 🔬 **DICOM Support** — Accepts hospital-standard `.dcm` files alongside `.png` / `.jpg`
- 🩺 **Role-Based Portal** — Separate Doctor and Patient login flows
- 📊 **Confidence Scoring** — Returns `High`, `Moderate`, or `Borderline` confidence with every prediction
- 🗄️ **MongoDB Atlas Cloud** — All credentials and assessment audit logs stored securely in the cloud
- 🔐 **Encrypted Authentication** — Passwords hashed with scrypt (Python `werkzeug`)
- 📈 **Benchmark Dashboard** — Live model performance metrics visible in portal
- 🌐 **REST API** — All features accessible via documented JSON endpoints

---

## 2. Tech Stack

| Layer | Technology |
|---|---|
| **Quantum ML** | PennyLane 0.45.1 (`default.qubit` simulator), 4 qubits |
| **Classical ML** | XGBoost, Random Forest, Scikit-learn |
| **Meta-Learner** | Logistic Regression (Stacking) |
| **Backend** | Flask (Python 3.10+) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Database** | MongoDB Atlas (Cloud) via PyMongo |
| **Image Processing** | Pillow (PIL), scikit-image (GLCM), pydicom |
| **Deployment** | Render.com (Gunicorn WSGI) |
| **Auth** | Werkzeug scrypt password hashing |
| **Environment** | python-dotenv, .env config |

---

## 3. How It Works

### Clinical Prediction Pipeline

```
Patient Symptoms (33 features)
        │
        ▼
  Feature Scaling (StandardScaler)
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  XGBoost Classifier            Quantum Kernel SVM
  (Classical tree ensemble)     (PennyLane, 4 qubits)
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
        Logistic Regression Meta-Learner
        (Stacking: learns weights for each model)
                   │
                   ▼
        Final Diagnosis + Confidence Score
```

### Quantum Kernel (How It Works)

- **Encoding:** 4 clinical features → AngleEmbedding → qubit rotation angles R_y(θ)
  - Qubit 0 → `UPDRS` score
  - Qubit 1 → `FunctionalAssessment`
  - Qubit 2 → `Tremor`
  - Qubit 3 → `Rigidity`
- **Kernel Formula:**

$$K(x_1, x_2) = \prod_{k=0}^{3} \cos^2\!\left(\frac{x_{1,k} - x_{2,k}}{2}\right)$$

### MRI Prediction Pipeline

```
Brain MRI Upload (.png / .jpg / .dcm)
        │
        ▼
  Image Preprocessing (grayscale, 128×128 resize)
        │
        ▼
  Radiomic Feature Extraction (10 features)
  ├── GLCM: contrast, homogeneity, energy, correlation
  └── Intensity: mean, std, skewness, kurtosis, p10, p90
        │
        ├──────────────────────────────────┐
        ▼                                  ▼
  Random Forest Classifier      Quantum Kernel SVM
        │                                  │
        └──────────┬───────────────────────┘
                   ▼
        Logistic Regression Meta-Learner
                   │
                   ▼
        MRI Diagnosis + Confidence Score
```

---

## 4. Results

### Clinical Dataset (2,105 patient records)

| Model | Accuracy | ROC-AUC |
|---|:---:|:---:|
| XGBoost (Classical Baseline) | 93.17% | 0.9660 |
| **Hybrid Quantum-Classical (Ours)** | **92.98%** | **0.9658** |
| Quantum Kernel SVM (Standalone) | 79.89% | 0.8529 |
| VQC (Variational Quantum Classifier) | 70.00% | 0.5700 |

### MRI Dataset (831 labeled brain scans)

| Model | Accuracy | ROC-AUC |
|---|:---:|:---:|
| Random Forest (Classical Baseline) | 97.60% | 0.9951 |
| **Hybrid Quantum-Classical (Ours)** | **96.63%** | **0.9943** |
| Quantum Kernel SVM (Standalone) | 73.56% | 0.7892 |

> The Hybrid model provides **orthogonal confidence validation** — the quantum kernel adds an independent check in Hilbert space for borderline patients where classical tree cuts alone are ambiguous.

---

## 5. Project Structure

```
quantum_parkinsons/
│
├── app.py                      # Flask backend — all API routes + MongoDB integration
├── Procfile                    # Gunicorn start command for Render deployment
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variable template
├── .gitignore
│
├── src/                        # Core ML source modules
│   ├── quantum_models.py       # Quantum Kernel SVM + VQC (PennyLane)
│   ├── classical_models.py     # XGBoost + Random Forest training
│   ├── hybrid_models.py        # Stacking meta-learner
│   ├── data_tabular.py         # Clinical dataset loader + preprocessor
│   ├── data_imaging.py         # MRI image loader + GLCM feature extractor
│   ├── data_imaging_labeled.py # Labeled MRI dataset handler
│   ├── fusion.py               # Multi-modal fusion logic
│   ├── config.py               # Paths + hyperparameters config
│   └── utils.py                # Shared utility functions
│
├── static/                     # Frontend (served by Flask)
│   ├── index.html              # Main portal UI
│   ├── app.js                  # Frontend logic + API calls
│   └── style.css               # Styling
│
├── models/                     # Saved trained model files (.pkl / .joblib)
├── data/                       # Datasets (clinical CSV + MRI folders)
│   ├── parkinsons_data.csv
│   └── images/
│       ├── PD/
│       └── Control/
├── results/                    # Training result logs + plots
│
├── train_diagnosis.py          # Train clinical hybrid model
├── train_imaging.py            # Train MRI hybrid model
├── train_progression.py        # Train disease progression model
├── run_all.py                  # Run all training pipelines
├── predict_patient.py          # CLI patient prediction script
└── predict_mri.py              # CLI MRI prediction script
```

---

## 6. How to Run

### Prerequisites

- Python 3.10+
- MongoDB Atlas account
- Git

### Step 1 — Clone the Repository

```bash
git clone https://github.com/Ananya-G-Bharadwaj-Hassan/NeuroQ-Quantum-AI-based-Parkinson-s-Disease-Diagnosis-Optimizer.git
cd NeuroQ-Quantum-AI-based-Parkinson-s-Disease-Diagnosis-Optimizer
```

### Step 2 — Create Virtual Environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### Step 3 — Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4 — Configure Environment Variables

```bash
# Copy the template
cp .env.example .env
```

Edit `.env` and fill in your values:

```env
MONGODB_URI=mongodb+srv://<username>:<password>@cluster0.xxxxx.mongodb.net/neuroq?retryWrites=true&w=majority
SECRET_KEY=your-random-secret-key-here
FLASK_ENV=development
```

### Step 5 — Train the Models (First Time Only)

```bash
# Train all pipelines
python run_all.py

# Or train individually:
python train_diagnosis.py     # Clinical hybrid model
python train_imaging.py       # MRI hybrid model
```

> ⚠️ Training the quantum kernel is slow (~10–30 min) on first run due to vectorized kernel computation. Trained models are saved to `models/` and reloaded on subsequent runs.

### Step 6 — Run the Flask Server

```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 7. Usage

### As a Doctor

1. Click **Doctor** tab on the login page
2. Register with name, email, and password
3. Log in → access the **Doctor Dashboard**
4. Use **Clinical Assessment** tab → enter 33 patient symptom values → get diagnosis
5. Use **MRI Analysis** tab → upload brain scan → get radiomic-based diagnosis
6. View full **Prediction History** (audit log from MongoDB)

### As a Patient

1. Click **Patient** tab on the login page
2. Register with name, email, and password
3. Log in → access **Patient Dashboard**
4. Submit self-reported symptoms → receive preliminary screening result
5. View personal assessment history

### CLI Prediction (Without Web UI)

```bash
# Clinical prediction
python predict_patient.py

# MRI prediction
python predict_mri.py --image path/to/scan.png
```

---

## 8. API Reference

All endpoints return JSON. Auth is session-cookie based.

### Authentication

| Method | Endpoint | Description | Body |
|---|---|---|---|
| `POST` | `/api/register` | Register new user | `{name, email, password, role}` |
| `POST` | `/api/login` | Login | `{email, password, role}` |
| `POST` | `/api/logout` | Logout | — |
| `GET` | `/api/me` | Get current user info | — |

### Prediction

| Method | Endpoint | Description | Body |
|---|---|---|---|
| `POST` | `/api/predict` | Clinical prediction | `{features: {...}}` |
| `POST` | `/api/predict` | MRI prediction | `multipart/form-data` with `mri_file` |

### Data

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/features` | List of 33 clinical feature names |
| `GET` | `/api/benchmarks` | Model benchmark accuracy table |
| `GET` | `/api/history` | Assessment history for logged-in user |
| `GET` | `/api/health` | Live MongoDB + model status check |

### Example — Clinical Predict Request

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "Age": 65,
      "UPDRS": 28.5,
      "MoCA": 24,
      "Tremor": 1,
      "Rigidity": 1,
      "Bradykinesia": 1,
      "FunctionalAssessment": 72.3
    }
  }'
```

### Example — Health Check Response

```json
{
  "status": "ok",
  "mongodb": "connected",
  "clinical_model": "loaded",
  "mri_model": "loaded"
}
```

---

## 9. Deployment (Render / Railway / Heroku)

### Deploy on Render

1. Push code to GitHub (ensure `.env` is in `.gitignore`)
2. Go to [render.com](https://render.com) → **New Web Service**
3. Connect your GitHub repository
4. Set the following:

| Setting | Value |
|---|---|
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `gunicorn app:app --workers 2 --timeout 120` |
| **Python Version** | 3.10+ |

5. Add **Environment Variables** in Render Dashboard:

```
MONGODB_URI = mongodb+srv://<user>:<pass>@cluster0.xxxxx.mongodb.net/neuroq?retryWrites=true&w=majority
SECRET_KEY  = <your-secret-key>
```

6. In **MongoDB Atlas** → Network Access → Add `0.0.0.0/0` (Allow from anywhere)
7. Click **Deploy** ✅

## 10. Known Limitations

| Limitation | Detail |
|---|---|
| **Quantum Simulator Only** | Currently uses PennyLane `default.qubit` simulator, not real QPU hardware |
| **Small Dataset** | Clinical dataset: 2,105 records; MRI dataset: 831 scans — results may not generalize to full clinical populations |
| **2D MRI Only** | MRI pipeline processes 2D slice images, not full 3D volumetric DICOM brain scans |
| **Quantum Kernel Speed** | First-time training is slow (~10–30 min) due to pairwise kernel computation; vectorized NumPy broadcasting mitigates but doesn't eliminate this |
| **4 Qubits** | Only 4 of 33 clinical features are encoded into qubits — chosen by clinical significance (UPDRS, FunctionalAssessment, Tremor, Rigidity) |
| **Not FDA Approved** | NeuroQ is a research prototype and **not a replacement for a licensed neurologist** |
---
