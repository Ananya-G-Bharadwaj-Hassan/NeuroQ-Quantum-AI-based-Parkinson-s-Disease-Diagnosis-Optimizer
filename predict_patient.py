import argparse
import os
import sys

import joblib
import pandas as pd
import numpy as np

from src import config
from src.data_imaging_labeled import extract_features_for_prediction


def load_artifact(name, train_script):
    path = os.path.join(config.MODELS_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained model at {path}.\nRun `python {train_script}` first.")
    return joblib.load(path)


def _hybrid_score(raw_row, artifact, quantum_weight=None):
    raw_row = np.asarray(raw_row, dtype=np.float64)
    if raw_row.ndim == 1:
        raw_row = raw_row.reshape(1, -1)

    X_classical = artifact["scaler_classical"].transform(raw_row)

    if "quantum_selector" in artifact and "scaler_quantum" in artifact:
        X_quantum = artifact["scaler_quantum"].transform(artifact["quantum_selector"].transform(raw_row))
    elif "pca_quantum" in artifact and "scaler_angle" in artifact:
        X_quantum = artifact["pca_quantum"].transform(artifact["scaler_angle"].transform(raw_row))
    else:
        X_quantum = X_classical[:, :config.N_QUBITS]

    if "hybrid_model" in artifact and artifact["hybrid_model"] is not None:
        detailed = artifact["hybrid_model"].predict_detailed(
            X_classical, X_quantum, custom_quantum_weight=quantum_weight
        )
        return detailed

    # Fallback if an older artifact is loaded
    p_classical = float(artifact["classical_model"].predict_proba(X_classical)[0, 1])
    p_quantum = float(artifact["quantum_model"].predict_proba(X_quantum)[0, 1])
    qw = 0.20 if quantum_weight is None else float(quantum_weight)
    p_hybrid = (1.0 - qw) * p_classical + qw * p_quantum

    return {
        "classical_model_name": artifact.get("classical_model_name", "classical"),
        "p_classical": p_classical,
        "quantum_model_name": artifact.get("quantum_model_name", "quantum"),
        "p_quantum": p_quantum,
        "quantum_weight": qw,
        "p_hybrid": p_hybrid,
        "confidence": abs(p_hybrid - 0.5) * 2.0,
        "confidence_band": "High" if abs(p_hybrid - 0.5) >= 0.3 else "Moderate",
        "predicted_label": int(p_hybrid >= 0.5),
        "predicted_class": "Parkinson's Disease" if p_hybrid >= 0.5 else "Control / Healthy",
        "strategy": "soft_voting_fallback",
        "weights": {"classical": 1.0 - qw, "quantum": qw},
        "quantum_features": artifact.get("quantum_feature_names", []),
    }


def clinical_hybrid(clinical_csv, artifact, quantum_weight=None):
    row = pd.read_csv(clinical_csv)
    if len(row) != 1:
        raise ValueError(f"{clinical_csv} must contain exactly one patient row, found {len(row)}.")
    missing = [c for c in artifact["feature_names"] if c not in row.columns]
    if missing:
        raise ValueError(
            f"{clinical_csv} is missing column(s) {missing} that the model was trained on. "
            f"Copy the header row from the original diagnosis CSV."
        )
    raw_row = row[artifact["feature_names"]].values
    return _hybrid_score(raw_row, artifact, quantum_weight)


def clinical_hybrid_from_values(values, artifact, quantum_weight=None):
    """values: {feature_name: number}. Blank/missing features are filled with
    the midpoint of the training range so clinicians can submit partial charts.
    Returns (breakdown_dict, list_of_defaulted_features)."""
    scaler = artifact["scaler_classical"]
    row, defaulted = [], []
    for i, name in enumerate(artifact["feature_names"]):
        v = values.get(name)
        if v is None or v == "":
            v = (scaler.data_min_[i] + scaler.data_max_[i]) / 2.0
            defaulted.append(name)
        row.append(float(v))
    return _hybrid_score([row], artifact, quantum_weight), defaulted


def imaging_hybrid(mri_path, artifact, quantum_weight=None):
    raw_row = extract_features_for_prediction(mri_path).reshape(1, -1)
    return _hybrid_score(raw_row, artifact, quantum_weight)


def predict(clinical_csv, mri_path=None, mri_weight=0.5, quantum_weight=None):
    clinical_artifact = load_artifact("clinical_diagnosis.joblib", "train_diagnosis.py")
    clinical = clinical_hybrid(clinical_csv, clinical_artifact, quantum_weight)

    result = {"clinical": clinical, "mri_used": mri_path is not None}

    if mri_path is None:
        result["p_pd_final"] = clinical["p_hybrid"]
        result["predicted_label"] = int(clinical["p_hybrid"] >= 0.5)
        result["predicted_class"] = "Parkinson's Disease" if result["predicted_label"] == 1 else "Control / Healthy"
        return result

    imaging_artifact = load_artifact("imaging_diagnosis.joblib", "train_imaging.py")
    imaging = imaging_hybrid(mri_path, imaging_artifact, quantum_weight)
    p_final = mri_weight * imaging["p_hybrid"] + (1.0 - mri_weight) * clinical["p_hybrid"]

    result.update({
        "imaging": imaging,
        "mri_weight": mri_weight,
        "p_pd_final": p_final,
        "predicted_label": int(p_final >= 0.5),
        "predicted_class": "Parkinson's Disease" if p_final >= 0.5 else "Control / Healthy",
    })
    return result


def _print_breakdown(label, b):
    print(f"\n{label} Hybrid Diagnostic Assessment:")
    print(f"  * Classical Model ({b['classical_model_name']}): P(PD) = {b['p_classical']:.4f}")
    print(f"  * Quantum Model   ({b['quantum_model_name']}): P(PD) = {b['p_quantum']:.4f}")
    print(f"  * Hybrid Model    ({b.get('strategy', 'hybrid')}): P(PD) = {b['p_hybrid']:.4f}")
    print(f"  * Confidence:     {b.get('confidence_band', 'Normal')} (Score: {b.get('confidence', 0.0):.2f})")
    if b.get("quantum_features"):
        print(f"  * Quantum Qubits: {b['quantum_features']}")


def main():
    parser = argparse.ArgumentParser(description="NeuroQ Clinical Diagnostic Predictor")
    parser.add_argument("--clinical", required=True, help="one-row CSV of clinical patient features")
    parser.add_argument("--mri", default=None, help="optional brain MRI image (.jpg/.png/.dcm)")
    parser.add_argument("--quantum_weight", type=float, default=None,
                        help="optional 0-1 weight overriding the meta-learner's learned quantum balance")
    parser.add_argument("--mri_weight", type=float, default=0.5,
                        help="0-1 weight blending MRI with clinical score (default 0.5)")
    args = parser.parse_args()

    if not os.path.exists(args.clinical):
        print(f"File not found: {args.clinical}")
        sys.exit(1)
    if args.mri and not os.path.exists(args.mri):
        print(f"File not found: {args.mri}")
        sys.exit(1)

    try:
        r = predict(args.clinical, args.mri, args.mri_weight, args.quantum_weight)
    except (FileNotFoundError, ValueError) as e:
        print(str(e))
        sys.exit(1)

    print("=" * 65)
    print("NEUROQ HYBRID QUANTUM-CLASSICAL DIAGNOSTIC REPORT")
    print("=" * 65)
    _print_breakdown("Clinical", r["clinical"])
    if r["mri_used"]:
        _print_breakdown("MRI Imaging", r["imaging"])
        print(f"\nMultimodal Combined Score (MRI Weight={r['mri_weight']:.2f}): P(PD) = {r['p_pd_final']:.4f}")
    else:
        print("\nNote: Clinical data assessment only (no MRI attached).")

    print("-" * 65)
    print(f"FINAL PREDICTED DIAGNOSIS: {r['predicted_class'].upper()}")
    print(f"Final Parkinson's Probability: {r['p_pd_final']*100:.1f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
