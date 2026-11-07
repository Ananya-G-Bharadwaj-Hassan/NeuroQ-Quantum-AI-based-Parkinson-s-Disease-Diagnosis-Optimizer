import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

TABULAR_DIAGNOSIS_CSV = os.path.join(DATA_DIR, "1784793241712_parkinsons_disease_data.csv")
TABULAR_PROGRESSION_CSV = os.path.join(DATA_DIR, "parkinsons_disease_progression_500.csv")
VOICE_DIR = os.path.join(DATA_DIR, "voice_raw")       # extracted voice.zip
IMAGING_DIR = os.path.join(DATA_DIR, "photo_raw")     # extracted photo.zip (NO diagnosis labels)

LABELED_IMAGING_DIR = os.path.join(DATA_DIR, "mri_labeled")
MODELS_DIR = os.path.join(BASE_DIR, "models")
INSTANCE_DIR = os.path.join(BASE_DIR, "instance")      # doctor accounts DB + secret key (created at runtime)

IMAGING_CLASS_ALIASES = {
    "pd": 1, "parkinsons": 1, "parkinson": 1, "positive": 1, "pp": 1,
    "control": 0, "hc": 0, "healthy": 0, "normal": 0, "negative": 0, "np": 0,
}

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(INSTANCE_DIR, exist_ok=True)

# ---- Shared settings ---------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.25

# Quantum settings (PennyLane, default.qubit simulator)
N_QUBITS = 4          # = number of top clinical features fed into the quantum circuit
QUANTUM_LAYERS = 2     # for the variational classifier
QUANTUM_SUBSET_LIMIT = 500  # sample cap for quantum simulation (now vectorized & fast)
