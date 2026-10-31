import argparse
import os
import sys

import joblib

from src import config
from predict_patient import _hybrid_score
from src.data_imaging_labeled import extract_features_for_prediction


def load_artifact():
    artifact_path = os.path.join(config.MODELS_DIR, "imaging_diagnosis.joblib")
    if not os.path.exists(artifact_path):
        raise FileNotFoundError(
            f"No trained imaging model at {artifact_path}.\n"
            f"Run `python train_imaging.py` first."
        )
    return joblib.load(artifact_path)


def predict(image_path, artifact=None, quantum_weight=None):
    artifact = artifact or load_artifact()
    raw_row = extract_features_for_prediction(image_path).reshape(1, -1)
    breakdown = _hybrid_score(raw_row, artifact, quantum_weight)

    label_to_name = {v: k for k, v in artifact.get("class_mapping", {}).items()}
    pred_label = int(breakdown["p_hybrid"] >= 0.5)

    class_str = label_to_name.get(pred_label, "Parkinson's Disease" if pred_label == 1 else "Control")

    return {
        "file": image_path,
        "predicted_class": class_str,
        "predicted_label": pred_label,
        **breakdown,
    }


def main():
    parser = argparse.ArgumentParser(description="NeuroQ MRI Hybrid Diagnostic Predictor")
    parser.add_argument("image", help="path to a brain MRI image (.jpg/.jpeg/.png/.dcm)")
    parser.add_argument("--quantum_weight", type=float, default=None,
                        help="0-1 weight overriding the meta-learner's learned quantum balance")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"File not found: {args.image}")
        sys.exit(1)

    try:
        r = predict(args.image, quantum_weight=args.quantum_weight)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    print("=" * 65)
    print("NEUROQ MRI SCAN HYBRID QUANTUM-CLASSICAL DIAGNOSIS")
    print("=" * 65)
    print(f"File:               {r['file']}")
    print(f"Predicted Diagnosis:{r['predicted_class'].upper()} (P = {r['p_hybrid']*100:.1f}%)")
    print(f"Confidence Level:   {r.get('confidence_band', 'Normal')} (Score: {r.get('confidence', 0.0):.2f})")
    print("-" * 65)
    print(f"Classical Model ({r['classical_model_name']}): P(PD) = {r['p_classical']:.4f}")
    print(f"Quantum Model   ({r['quantum_model_name']}): P(PD) = {r['p_quantum']:.4f}")
    print(f"Hybrid Fusion   ({r.get('strategy', 'hybrid')}): P(PD) = {r['p_hybrid']:.4f}")
    print("=" * 65)


if __name__ == "__main__":
    main()
