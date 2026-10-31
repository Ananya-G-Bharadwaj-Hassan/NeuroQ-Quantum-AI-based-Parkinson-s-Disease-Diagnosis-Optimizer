import argparse
import sys
import time

import joblib
from sklearn.model_selection import train_test_split

from src import config
from src.data_imaging_labeled import build_labeled_imaging_dataset
from src.classical_models import get_classical_models
from src.quantum_models import QuantumKernelSVM, VariationalQuantumClassifier
from src.hybrid_models import HybridQuantumClassicalClassifier
from src.utils import compute_classification_report, save_json, cap_for_quantum, print_confusion_matrix


def main(subset=None):
    print("=" * 70, flush=True)
    print("TRAINING MRI IMAGING DIAGNOSIS MODELS (CLASSICAL + QUANTUM + HYBRID)", flush=True)
    print("=" * 70, flush=True)

    print("\nLoading and extracting radiomic features from MRI dataset...", flush=True)
    data = build_labeled_imaging_dataset()
    Xc, Xq, y = data["X_classical"], data["X_quantum"], data["y"]

    if subset:
        Xc, Xq, y = Xc[:subset], Xq[:subset], y[:subset]

    Xc_tr, Xc_te, Xq_tr, Xq_te, y_tr, y_te = train_test_split(
        Xc, Xq, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )
    print(f"Dataset ready: {len(y)} total samples ({len(y_tr)} train, {len(y_te)} test).", flush=True)

    results = {
        "dataset": "imaging_diagnosis",
        "n_samples": len(y),
        "class_mapping": data["class_mapping"],
        "models": {},
    }

    best_c_name, best_c_model, best_c_acc = None, None, -1.0

    # ---------------- 1. Classical Baselines ----------------
    print("\n--- 1. Training Classical Baselines on MRI Radiomics ---", flush=True)
    for name, model in get_classical_models().items():
        t0 = time.time()
        model.fit(Xc_tr, y_tr)
        preds = model.predict(Xc_te)
        proba = model.predict_proba(Xc_te)[:, 1]
        report = compute_classification_report(y_te, preds, proba)
        report["train_time_sec"] = time.time() - t0
        results["models"][name] = report
        print(f"\n{name.upper()}:", flush=True)
        print_confusion_matrix(report, name)

        if report["accuracy"] > best_c_acc:
            best_c_name, best_c_model, best_c_acc = name, model, report["accuracy"]

    # ---------------- 2. Quantum Models ----------------
    print("\n--- 2. Training Quantum Models on MRI Radiomics ---", flush=True)
    Xq_tr_q, y_tr_q = cap_for_quantum(Xq_tr, y_tr, config.QUANTUM_SUBSET_LIMIT)
    Xq_te_q, y_te_q = Xq_te, y_te

    # 2a. Quantum Kernel SVM
    t0 = time.time()
    qk = QuantumKernelSVM(n_qubits=config.N_QUBITS).fit(Xq_tr_q, y_tr_q)
    preds = qk.predict(Xq_te_q)
    proba = qk.predict_proba(Xq_te_q)[:, 1]
    report_qk = compute_classification_report(y_te_q, preds, proba)
    report_qk["train_time_sec"] = time.time() - t0
    report_qk["n_samples_used"] = len(y_tr_q) + len(y_te_q)
    results["models"]["quantum_kernel_svm"] = report_qk
    print("\nQUANTUM_KERNEL_SVM (MRI):", flush=True)
    print_confusion_matrix(report_qk, "quantum_kernel_svm")

    # 2b. Variational Quantum Classifier (on small benchmark subset for speed)
    Xq_tr_vqc, y_tr_vqc = cap_for_quantum(Xq_tr, y_tr, min(80, len(Xq_tr)))
    Xq_te_vqc, y_te_vqc = Xq_te[:50], y_te[:50]
    t0 = time.time()
    vqc = VariationalQuantumClassifier(epochs=15).fit(Xq_tr_vqc, y_tr_vqc)
    preds = vqc.predict(Xq_te_vqc)
    proba = vqc.predict_proba(Xq_te_vqc)[:, 1]
    report_vqc = compute_classification_report(y_te_vqc, preds, proba)
    report_vqc["train_time_sec"] = time.time() - t0
    report_vqc["n_samples_used"] = len(y_tr_vqc) + len(y_te_vqc)
    results["models"]["variational_quantum_classifier"] = report_vqc
    print("\nVARIATIONAL_QUANTUM_CLASSIFIER (MRI):", flush=True)
    print_confusion_matrix(report_vqc, "variational_quantum_classifier")

    best_q_name = "quantum_kernel_svm"
    best_q_model = qk
    best_q_acc = report_qk["accuracy"]

    # ---------------- 3. Hybrid Model ----------------
    print("\n--- 3. Training Unified Hybrid Quantum-Classical MRI Model ---", flush=True)
    t0 = time.time()
    hybrid_model = HybridQuantumClassicalClassifier(
        classical_model=best_c_model,
        quantum_model=best_q_model,
        strategy="stacking",
        classical_model_name=best_c_name,
        quantum_model_name=best_q_name,
    )
    hybrid_model.fit(Xc_tr, Xq_tr, y_tr)

    preds_hyb = hybrid_model.predict(Xc_te, Xq_te)
    proba_hyb = hybrid_model.predict_proba(Xc_te, Xq_te)[:, 1]
    report_hyb = compute_classification_report(y_te, preds_hyb, proba_hyb)
    report_hyb["train_time_sec"] = time.time() - t0
    report_hyb["strategy"] = "stacking_meta_learner"
    report_hyb["meta_weights"] = hybrid_model.meta_weights
    report_hyb["optimal_quantum_weight"] = hybrid_model.optimal_quantum_weight
    results["models"]["hybrid_model"] = report_hyb

    print(f"\nHYBRID_MODEL (Combining {best_c_name.upper()} + {best_q_name.upper()}):", flush=True)
    print_confusion_matrix(report_hyb, "hybrid_model")

    # Leaderboard summary
    print("\n" + "=" * 70, flush=True)
    print("MRI MODEL PERFORMANCE LEADERBOARD (TEST SET ACCURACY & ROC-AUC)", flush=True)
    print("=" * 70, flush=True)
    print(f"{'Model Name':<32} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<10}", flush=True)
    print("-" * 90, flush=True)
    for m_name, m_rep in sorted(results["models"].items(), key=lambda kv: kv[1]["accuracy"], reverse=True):
        auc_val = f"{m_rep.get('roc', {}).get('auc', 0.0):.4f}" if "roc" in m_rep else "N/A"
        print(f"{m_name:<32} {m_rep['accuracy']:<12.4f} {m_rep['precision']:<12.4f} {m_rep['recall']:<12.4f} {m_rep['f1']:<12.4f} {auc_val:<10}", flush=True)
    print("=" * 70, flush=True)

    save_json(results, "imaging_diagnosis_metrics.json", config.RESULTS_DIR)

    artifact_path = f"{config.MODELS_DIR}/imaging_diagnosis.joblib"
    joblib.dump(
        {
            "hybrid_model": hybrid_model,
            "classical_model_name": best_c_name,
            "classical_model": best_c_model,
            "classical_accuracy": best_c_acc,
            "quantum_model_name": best_q_name,
            "quantum_model": best_q_model,
            "quantum_accuracy": best_q_acc,
            "hybrid_accuracy": report_hyb["accuracy"],
            "scaler_classical": data["scaler_classical"],
            "quantum_selector": data["quantum_selector"],
            "scaler_quantum": data["scaler_quantum"],
            "scaler_angle": data["scaler_angle"],
            "pca_quantum": data["pca_quantum"],
            "class_mapping": data["class_mapping"],
        },
        artifact_path,
    )
    print(f"\n[Saved Artifact] {artifact_path}", flush=True)
    print(f"  * Classical Baseline: {best_c_name} (Accuracy: {best_c_acc:.4f})", flush=True)
    print(f"  * Quantum Model:      {best_q_name} (Accuracy: {best_q_acc:.4f})", flush=True)
    print(f"  * Hybrid Model:       Stacking Ensemble (Accuracy: {report_hyb['accuracy']:.4f})", flush=True)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", type=int, default=None,
                        help="cap sample count for faster simulation")
    args = parser.parse_args()
    main(subset=args.subset)
