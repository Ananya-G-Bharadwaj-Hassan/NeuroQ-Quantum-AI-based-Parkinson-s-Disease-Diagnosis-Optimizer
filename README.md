# NeuroQ: Hybrid Quantum-Classical AI for Parkinson's Disease Diagnosis

A clinical decision-support web portal that combines **classical machine learning** (XGBoost, Random Forest) with **quantum machine learning** (PennyLane Quantum Kernel SVM and Variational Quantum Classifier). A stacking meta-learner fuses the two into one hybrid diagnosis from **clinical records** and, optionally, a **brain MRI scan**. Doctors and patients sign in, run diagnoses and review their assessment history, all backed by MongoDB Atlas.

---

## Table of Contents

1. [Features](#1-features)
2. [Tech Stack](#2-tech-stack)
3. [How It Works](#3-how-it-works)
4. [Results](#4-results)
5. [Project Structure](#5-project-structure)
6. [Getting Started](#6-getting-started)
7. [Usage](#7-usage)
8. [API Reference](#8-api-reference)
9. [Deployment](#9-deployment-render--railway--heroku)
10. [Known Limitations](#10-known-limitations)

---

## 1. Features

- **Hybrid diagnosis**: classical and quantum models stacked by a logistic-regression meta-learner.
- **Multimodal input**: 33 clinical features plus an optional brain MRI (`.jpg`, `.png`, `.dcm`).
- **Adjustable quantum weight**: a slider in the UI lets the clinician blend classical and quantum scores.
- **Doctor and patient accounts**: role-based login, with password hashing and cookie sessions.
- **Assessment history**: every diagnosis is stored in MongoDB. Doctors see all records, patients see their own.
- **Benchmarks dashboard**: live accuracy, F1 and ROC-AUC tables for every model.
- **CLI tools** for batch prediction and for retraining every model.
- **Health endpoint** (`/api/health`) that reports database and model status.

## 2. Tech Stack

| Layer | Technology |
| :--- | :--- |
| Backend | Python, Flask, Gunicorn |
| Database | MongoDB Atlas (PyMongo) |
| Classical ML | scikit-learn, XGBoost |
| Quantum ML | PennyLane (`default.qubit` simulator) |
| Imaging | scikit-image, Pillow, pydicom (radiomic feature extraction) |
| Frontend | HTML, CSS, vanilla JavaScript |
| Deployment | Render / Railway / Heroku (`Procfile`) |

## 3. How It Works

```
 Clinical record (33 features)            Brain MRI (optional)
            │                                      │
   ┌────────┴────────┐                    radiomic feature extraction
   │                 │                             │
Scaled features   Top-4 biomarkers          ┌──────┴──────┐
   │              (supervised selection)    │             │
   │                 │                    Classical     Quantum
Classical models  Quantum Kernel SVM      models        kernel
(XGBoost / RF)    (4 qubits)                └──────┬──────┘
   │                 │                        Hybrid stack
   └────────┬────────┘                             │
   Logistic-regression meta-learner                │
            │                                      │
            └──────────────┬───────────────────────┘
                     Final probability + confidence
```

### 3.1 Supervised biomarker selection
Four qubits can only take four inputs. Instead of unsupervised PCA, which let demographic variance such as `Gender` drown out the neurological signals, the pipeline uses `SelectKBest` (F-score) to choose the four most informative markers:

| Qubit | Feature |
| :---: | :--- |
| 0 | `UPDRS` (Unified Parkinson's Disease Rating Scale) |
| 1 | `FunctionalAssessment` |
| 2 | `Tremor` |
| 3 | `Rigidity` |

The features are scaled into `[0, π]` and angle-encoded:

$$|\psi(x)\rangle = \bigotimes_{k=0}^{3}\left(\cos\tfrac{x_k}{2}|0\rangle + \sin\tfrac{x_k}{2}|1\rangle\right)$$

### 3.2 Vectorized quantum kernel
The kernel is the state overlap between two encoded inputs:

$$K(x_1,x_2)=|\langle\psi(x_1)|\psi(x_2)\rangle|^2=\prod_{k}\cos^2\!\tfrac{x_{1,k}-x_{2,k}}{2}$$

`QuantumKernelSVM` computes this in closed form with NumPy. The result is mathematically identical to PennyLane's `AngleEmbedding` followed by its adjoint, but it trains on hundreds of samples in a fraction of a second. The old pair-by-pair circuit loop forced a 120-sample cap.

### 3.3 Hybrid stacking
`HybridQuantumClassicalClassifier` (`src/hybrid_models.py`):

1. Generates **5-fold stratified out-of-fold** probabilities from the classical model and the quantum kernel SVM.
2. Trains an L2-regularized **logistic-regression meta-learner** on those probabilities.
3. Exposes an optimal quantum weight, which the UI slider can override at inference time.

Early versions scored about 52.5% on the quantum side. The causes were unsupervised PCA, the 120-sample cap, and an ad-hoc 50/50 blend with no trained hybrid. The changes above fixed all three.

## 4. Results

### 4.1 Clinical diagnosis (2,105 patient records, 527 test samples)

| Model | Type | Accuracy | F1 |
| :--- | :--- | :---: | :---: |
| XGBoost | Classical | **93.17%** | 93.13% |
| **Hybrid Stacking** | **Hybrid** | **92.98%** | 92.93% |
| Random Forest | Classical | 92.41% | 92.36% |
| Logistic Regression | Classical | 81.59% | 81.42% |
| Quantum Kernel SVM | Quantum | 79.89% | 79.13% |
| Variational Quantum Classifier | Quantum | 70.00% | 62.12% |

### 4.2 MRI diagnosis (831 labeled scans: 221 PD, 610 control; 208 test samples)

| Model | Type | Accuracy | F1 |
| :--- | :--- | :---: | :---: |
| Random Forest | Classical | **97.60%** | 97.57% |
| **Hybrid Stacking** | **Hybrid** | **96.63%** | 96.58% |
| XGBoost | Classical | 96.15% | 96.15% |
| Logistic Regression | Classical | 78.85% | 73.12% |
| Variational Quantum Classifier | Quantum | 74.00% | 66.09% |
| Quantum Kernel SVM | Quantum | 73.56% | 62.35% |

The hybrid model stays within about one point of the best classical model. The quantum models on their own are clearly weaker. Full metrics, confusion matrices and ROC curves are in `results/*.json`.

## 5. Project Structure

```
quantum_parkinsons/
├── app.py                       # Flask backend: auth, predict, history, benchmarks, health
├── predict_patient.py           # Hybrid clinical (+ MRI) prediction CLI and helpers
├── predict_mri.py               # Standalone MRI prediction CLI
├── train_diagnosis.py           # Train clinical models (classical, quantum, hybrid)
├── train_imaging.py             # Train MRI models
├── train_progression.py         # Train disease-progression severity models
├── export_backend_summary.py    # Consolidate metrics into results/backend_summary.json
├── explore_imaging.py           # MRI exploration / anomaly detection
├── run_all.py                   # Pipeline runner
├── requirements.txt
├── Procfile                     # Gunicorn start command for cloud hosts
├── .env.example                 # Environment variable template
├── src/
│   ├── config.py                # Paths, qubit count, seeds
│   ├── data_tabular.py          # Tabular loaders and supervised feature selection
│   ├── data_imaging.py          # Radiomic featurization
│   ├── data_imaging_labeled.py  # Labeled MRI loader
│   ├── classical_models.py      # Logistic Regression, Random Forest, XGBoost
│   ├── quantum_models.py        # QuantumKernelSVM, VariationalQuantumClassifier
│   ├── hybrid_models.py         # HybridQuantumClassicalClassifier
│   ├── fusion.py                # Multimodal probability fusion
│   └── utils.py                 # Metrics, serialization, helpers
├── static/                      # index.html, app.js, style.css (doctor portal UI)
├── data/
│   ├── 1784793241712_parkinsons_disease_data.csv   # clinical diagnosis data (2,105 rows)
│   ├── parkinsons_disease_progression_500.csv      # progression data (500 rows)
│   └── mri_labeled/{PD,control}/                   # labeled MRI scans
├── models/                      # Saved model bundles (.joblib)
├── results/                     # Metrics JSON files
└── instance/                    # Runtime files (generated secret key)
```

## 6. Getting Started

### Prerequisites
- Python 3.11 or newer
- A MongoDB Atlas account (free M0 tier), or a local MongoDB instance

### Installation

```bash
git clone https://github.com/AnanyaG-Bharadwaj-Hassan/<repo-name>.git
cd quantum_parkinsons

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Configuration

Copy the template and fill in your own values:

```bash
cp .env.example .env
```

| Variable | Required | Description |
| :--- | :---: | :--- |
| `MONGODB_URI` | Yes | Atlas connection string. Defaults to `mongodb://localhost:27017/neuroq` |
| `SECRET_KEY` | No | Flask session key. A random one is generated into `instance/secret.key` if omitted |
| `PORT` | No | Server port (default `5000`) |
| `FLASK_DEBUG` | No | Set to `1` for debug mode |

**Setting up MongoDB Atlas**
1. Create a free **M0** cluster at [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas).
2. Under **Database Access**, create a database user.
3. Under **Network Access**, allow your IP (use `0.0.0.0/0` only for cloud deployment).
4. Click **Connect → Drivers → Python** and copy the connection string into `MONGODB_URI`.

The app creates two collections, `users` and `assessments`, with indexes on first run.

> The `.env` file is not loaded automatically. Export the variables in your shell, or load them with your hosting platform's settings.

## 7. Usage

### 7.1 Web portal

```bash
python app.py
```

Open <http://127.0.0.1:5000>, then:

1. Register as a **doctor** or **patient** and sign in.
2. Open **Model Benchmarks** to view live metrics.
3. Enter patient values and set the **Quantum Model Weight** slider.
4. Optionally attach a brain MRI.
5. Click **Run diagnosis** to see the probability gauge, the classical vs quantum scores, the active qubit features and the confidence level.

### 7.2 Command-line prediction

```bash
# Clinical CSV
python predict_patient.py --clinical path/to/patient.csv

# Clinical + MRI, with MRI weighted at 40%
python predict_patient.py --clinical path/to/patient.csv \
    --mri data/mri_labeled/PD/<scan>.png --mri_weight 0.4

# MRI only
python predict_mri.py data/mri_labeled/PD/<scan>.png
```

### 7.3 Retraining

```bash
python train_diagnosis.py        # clinical models  -> models/clinical_diagnosis.joblib
python train_imaging.py          # MRI models       -> models/imaging_diagnosis.joblib
python train_progression.py      # progression models
python export_backend_summary.py # rebuild results/backend_summary.json
```

## 8. API Reference

| Method | Endpoint | Auth | Description |
| :---: | :--- | :---: | :--- |
| GET | `/` | No | Serves the portal UI |
| GET | `/api/health` | No | Database connectivity and model availability |
| POST | `/api/register` | No | Create a `doctor` (default) or `patient` account |
| POST | `/api/login` | No | Sign in and start a session |
| POST | `/api/logout` | Yes | End the session |
| GET | `/api/me` | Yes | Current user profile |
| GET | `/api/features` | Yes | Clinical feature definitions, bounds and model metadata |
| GET | `/api/benchmarks` | Yes | Model comparison metrics |
| POST | `/api/predict` | Yes | Multipart form: clinical fields, optional `mri` file, optional `quantum_weight` (0 to 1) |
| GET | `/api/history` | Yes | Stored assessments (all for doctors, own for patients) |

## 9. Known Limitations

- **Research use only.** The models are trained on a single public-style dataset and have had no external clinical validation.
- **Quantum models are simulated** on classical hardware with 4 qubits. On their own they do not beat the classical models. The value here is the hybrid architecture and the research comparison.

