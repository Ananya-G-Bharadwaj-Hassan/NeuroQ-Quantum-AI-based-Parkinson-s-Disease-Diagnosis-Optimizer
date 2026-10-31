import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src import config
from src.data_tabular import load_progression_dataset
from src.classical_models import get_classical_models
from src.quantum_models import QuantumKernelSVM
from src.utils import compute_classification_report, save_json, cap_for_quantum, print_confusion_matrix


def main():
    print("=" * 70)
    print("TRAINING PARKINSON'S DISEASE PROGRESSION SEVERITY MODELS")
    print("=" * 70)

    data = load_progression_dataset()
    Xc, Xq, y_raw = data["X_classical"], data["X_quantum"], data["y"]
    y = y_raw - y_raw.min()  # zero-index classes (0, 1, 2)

    Xc_tr, Xc_te, Xq_tr, Xq_te, y_tr, y_te = train_test_split(
        Xc, Xq, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE, stratify=y
    )

    results = {"dataset": "progression", "n_samples": len(y), "models": {}}
    best_c_name, best_c_model, best_c_acc = None, None, -1.0

    print("\n--- 1. Classical Baselines ---")
    for name, model in get_classical_models().items():
        t0 = time.time()
        model.fit(Xc_tr, y_tr)
        preds = model.predict(Xc_te)
        report = compute_classification_report(y_te, preds, labels=[0, 1, 2])
        report["train_time_sec"] = time.time() - t0
        results["models"][name] = report
        print(f"\n{name.upper()}:")
        print_confusion_matrix(report, name)

        if report["accuracy"] > best_c_acc:
            best_c_name, best_c_model, best_c_acc = name, model, report["accuracy"]

    print("\n--- 2. Quantum Kernel SVM (Multiclass) ---")
    Xq_tr_q, y_tr_q = cap_for_quantum(Xq_tr, y_tr, config.QUANTUM_SUBSET_LIMIT)
    Xq_te_q, y_te_q = Xq_te, y_te

    t0 = time.time()
    qk = QuantumKernelSVM(n_qubits=config.N_QUBITS).fit(Xq_tr_q, y_tr_q)
    preds_qk = qk.predict(Xq_te_q)
    report_qk = compute_classification_report(y_te_q, preds_qk, labels=[0, 1, 2])
    report_qk["train_time_sec"] = time.time() - t0
    report_qk["n_samples_used"] = len(y_tr_q) + len(y_te_q)
    results["models"]["quantum_kernel_svm"] = report_qk
    print("\nQUANTUM_KERNEL_SVM:")
    print_confusion_matrix(report_qk, "quantum_kernel_svm")

    print("\n--- 3. Hybrid Progression Model (Ensemble Fusion) ---")
    t0 = time.time()
    p_c = best_c_model.predict_proba(Xc_te)
    p_q = qk.predict_proba(Xq_te)
    
    # Blended progression probabilities
    p_hyb = 0.75 * p_c + 0.25 * p_q
    preds_hyb = np.argmax(p_hyb, axis=1)
    report_hyb = compute_classification_report(y_te, preds_hyb, labels=[0, 1, 2])
    report_hyb["train_time_sec"] = time.time() - t0
    report_hyb["strategy"] = "soft_voting_ensemble"
    results["models"]["hybrid_model"] = report_hyb
    print("\nHYBRID_MODEL (Classical + Quantum Soft Voting):")
    print_confusion_matrix(report_hyb, "hybrid_model")

    save_json(results, "progression_metrics.json", config.RESULTS_DIR)
    return results


if __name__ == "__main__":
    main()
