import json
import os

from src import config


def best_model(models_dict):
    """Pick the model with the highest accuracy - useful for a
    dashboard headline number."""
    return max(models_dict.items(), key=lambda kv: kv[1].get("accuracy", 0))


def main():
    summary = {"datasets": {}}

    diag_path = os.path.join(config.RESULTS_DIR, "diagnosis_metrics.json")
    if os.path.exists(diag_path):
        d = json.load(open(diag_path))
        best_name, best = best_model(d["models"])
        summary["datasets"]["diagnosis"] = {
            "label": "Parkinson's Diagnosis (clinical data)",
            "task_type": "binary_classification",
            "n_samples": d["n_samples"],
            "best_model": best_name,
            "best_accuracy": best["accuracy"],
            "hybrid_available": "hybrid_model" in d["models"],
            "hybrid_accuracy": d["models"].get("hybrid_model", {}).get("accuracy"),
            "models": d["models"],
        }

    prog_path = os.path.join(config.RESULTS_DIR, "progression_metrics.json")
    if os.path.exists(prog_path):
        d = json.load(open(prog_path))
        best_name, best = best_model(d["models"])
        summary["datasets"]["progression"] = {
            "label": "Disease Progression Severity (1-3)",
            "task_type": "multiclass_classification",
            "n_samples": d["n_samples"],
            "best_model": best_name,
            "best_accuracy": best["accuracy"],
            "hybrid_available": "hybrid_model" in d["models"],
            "hybrid_accuracy": d["models"].get("hybrid_model", {}).get("accuracy"),
            "models": d["models"],
        }

    img_diag_path = os.path.join(config.RESULTS_DIR, "imaging_diagnosis_metrics.json")
    if os.path.exists(img_diag_path):
        d = json.load(open(img_diag_path))
        best_name, best = best_model(d["models"])
        summary["datasets"]["imaging_diagnosis"] = {
            "label": "Parkinson's Diagnosis (labeled MRI images)",
            "task_type": "binary_classification",
            "n_samples": d["n_samples"],
            "best_model": best_name,
            "best_accuracy": best["accuracy"],
            "hybrid_available": "hybrid_model" in d["models"],
            "hybrid_accuracy": d["models"].get("hybrid_model", {}).get("accuracy"),
            "models": d["models"],
        }

    img_path = os.path.join(config.RESULTS_DIR, "imaging_summary.json")
    if os.path.exists(img_path):
        d = json.load(open(img_path))
        summary["datasets"]["imaging"] = {
            "label": "MRI Radiomics (unsupervised anomaly detection)",
            "task_type": "unsupervised_anomaly_detection",
            "n_samples": d["n_samples"],
            "note": d.get("note", ""),
            "top_anomalous_scans": d.get("top_anomalous_scans", []),
        }

    out_path = os.path.join(config.RESULTS_DIR, "backend_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
