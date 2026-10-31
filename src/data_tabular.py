import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif

from . import config


def load_diagnosis_dataset(n_qubits=config.N_QUBITS):
    if not os.path.exists(config.TABULAR_DIAGNOSIS_CSV):
        raise FileNotFoundError(
            f"Can't find {config.TABULAR_DIAGNOSIS_CSV}\n"
            f"Copy 1784793241712_parkinsons_disease_data.csv into the 'data' folder "
            f"next to this project (see README step 2)."
        )
    df = pd.read_csv(config.TABULAR_DIAGNOSIS_CSV)
    df = df.drop(columns=["DoctorInCharge"], errors="ignore")  # non-informative
    y = df["Diagnosis"].astype(int).values

    X = df.drop(columns=["PatientID", "Diagnosis"], errors="ignore")
    X = X.select_dtypes(include=[np.number])
    feature_names = X.columns.tolist()
    X_vals = X.values

    # Classical feature scaler (0 to 1)
    scaler_classical = MinMaxScaler().fit(X_vals)
    X_scaled_01 = scaler_classical.transform(X_vals)

    # Supervised feature selection for the quantum state embedding:
    # Identifies the top clinically discriminative biomarkers (e.g. UPDRS, MoCA, Tremor)
    # so the n_qubits quantum state encodes genuine pathological signal rather than demographic noise.
    quantum_selector = SelectKBest(f_classif, k=n_qubits).fit(X_vals, y)
    selected_indices = quantum_selector.get_support(indices=True)
    quantum_feature_names = [feature_names[i] for i in selected_indices]

    X_selected = quantum_selector.transform(X_vals)
    scaler_quantum = MinMaxScaler((0.0, np.pi)).fit(X_selected)
    X_quantum = scaler_quantum.transform(X_selected)

    # Legacy PCA & angle scaler retained for full backwards compatibility
    scaler_angle = MinMaxScaler((0.0, np.pi)).fit(X_vals)
    pca_quantum = PCA(n_components=n_qubits, random_state=config.RANDOM_STATE).fit(
        scaler_angle.transform(X_vals)
    )

    return {
        "X_classical": X_scaled_01,
        "X_quantum": X_quantum,
        "y": y,
        "feature_names": feature_names,
        "quantum_feature_names": quantum_feature_names,
        "scaler_classical": scaler_classical,
        "quantum_selector": quantum_selector,
        "scaler_quantum": scaler_quantum,
        "scaler_angle": scaler_angle,
        "pca_quantum": pca_quantum,
    }


def load_progression_dataset(n_qubits=config.N_QUBITS):
    """
    parkinsons_disease_progression_500.csv
    Target: Disease_Progression (1/2/3 severity class), 500 rows.
    """
    if not os.path.exists(config.TABULAR_PROGRESSION_CSV):
        raise FileNotFoundError(
            f"Can't find {config.TABULAR_PROGRESSION_CSV}\n"
            f"Copy parkinsons_disease_progression_500.csv into the 'data' folder "
            f"next to this project (see README step 2)."
        )
    df = pd.read_csv(config.TABULAR_PROGRESSION_CSV)
    y = df["Disease_Progression"].astype(int).values

    cat_cols = ["Gender", "Medications", "Exercise_Level"]
    df_enc = df.drop(columns=["Patient_ID", "Disease_Progression"], errors="ignore").copy()
    for c in cat_cols:
        if c in df_enc.columns:
            df_enc[c] = LabelEncoder().fit_transform(df_enc[c].astype(str))

    feature_names = df_enc.columns.tolist()
    X = df_enc.values.astype(float)

    scaler_classical = MinMaxScaler().fit(X)
    X_scaled_01 = scaler_classical.transform(X)

    quantum_selector = SelectKBest(f_classif, k=n_qubits).fit(X, y)
    selected_indices = quantum_selector.get_support(indices=True)
    quantum_feature_names = [feature_names[i] for i in selected_indices]

    scaler_quantum = MinMaxScaler((0.0, np.pi)).fit(quantum_selector.transform(X))
    X_quantum = scaler_quantum.transform(quantum_selector.transform(X))

    scaler_angle = MinMaxScaler((0.0, np.pi)).fit(X)
    pca_quantum = PCA(n_components=n_qubits, random_state=config.RANDOM_STATE).fit(
        scaler_angle.transform(X)
    )

    return {
        "X_classical": X_scaled_01,
        "X_quantum": X_quantum,
        "y": y,
        "feature_names": feature_names,
        "quantum_feature_names": quantum_feature_names,
        "scaler_classical": scaler_classical,
        "quantum_selector": quantum_selector,
        "scaler_quantum": scaler_quantum,
        "scaler_angle": scaler_angle,
        "pca_quantum": pca_quantum,
    }
