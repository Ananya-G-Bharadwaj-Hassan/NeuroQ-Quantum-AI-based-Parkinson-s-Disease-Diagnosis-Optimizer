import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, roc_curve, auc
)


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def compute_classification_report(y_true, y_pred, y_proba=None, labels=None):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    report = {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
        "labels": labels if labels is not None else sorted(set(list(y_true) + list(y_pred))),
    }

    # ROC only meaningful for binary classification with probabilities
    if y_proba is not None and len(set(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        report["roc"] = {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "auc": auc(fpr, tpr)}

    return {k: to_serializable(v) for k, v in report.items()}


def cap_for_quantum(X, y, limit, random_state=42):
    """
    Quantum kernel methods are O(n^2) simulated circuit evaluations —
    on a laptop that becomes impractically slow past a few hundred
    samples. This subsamples (stratified) down to `limit` before the
    quantum step only; classical models upstream still see the full
    dataset. Returns X, y unchanged if already under the limit.
    """
    import numpy as np
    if limit is None or len(X) <= limit:
        return X, y
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(X), size=limit, replace=False)
    print(f"[quantum] capping {len(X)} -> {limit} samples for the quantum kernel step "
          f"(O(n^2) circuit evals; edit config.QUANTUM_SUBSET_LIMIT to change this)")
    return X[idx], y[idx]


def print_confusion_matrix(report, model_name):
    """
    Prints accuracy + a readable confusion matrix to the console right
    after each model trains, so it's visible in a live run/demo instead
    of only living in the saved JSON.
    """
    labels = report["labels"]
    cm = report["confusion_matrix"]
    print(f"  accuracy={report['accuracy']:.3f}  precision={report['precision']:.3f}  "
          f"recall={report['recall']:.3f}  f1={report['f1']:.3f}")
    header = "        " + "  ".join(f"pred={l}" for l in labels)
    print(header)
    for true_label, row in zip(labels, cm):
        row_str = "  ".join(f"{v:7d}" for v in row)
        print(f"true={true_label}  {row_str}")


def save_json(payload, filename, results_dir):
    path = os.path.join(results_dir, filename)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, default=to_serializable)
    print(f"[saved] {path}")
    return path
