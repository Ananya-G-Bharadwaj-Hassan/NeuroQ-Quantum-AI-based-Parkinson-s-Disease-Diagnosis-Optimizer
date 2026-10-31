import os
from src import config
from src.data_imaging import build_imaging_dataset, unsupervised_anomaly_scores, FEATURE_NAMES
from src.utils import save_json


def main():
    Xc, Xq, patient_ids, paths = build_imaging_dataset()
    scores, _ = unsupervised_anomaly_scores(Xc)

    # store paths relative to the project root so the JSON is portable
    # across machines (frontend dev's laptop won't share your file layout)
    rel_paths = [os.path.relpath(p, config.BASE_DIR) for p in paths]

    ranked = sorted(
        zip(patient_ids, rel_paths, scores.tolist()),
        key=lambda r: r[2], reverse=True
    )

    payload = {
        "dataset": "imaging",
        "n_samples": len(patient_ids),
        "note": "No diagnosis label available in these DICOMs; scores are unsupervised anomaly scores, not predictions.",
        "feature_names": FEATURE_NAMES,
        "top_anomalous_scans": [
            {"patient_id": pid, "file": path, "anomaly_score": score}
            for pid, path, score in ranked[:20]
        ],
    }
    save_json(payload, "imaging_summary.json", config.RESULTS_DIR)
    print(f"Processed {len(patient_ids)} scans.")


if __name__ == "__main__":
    main()
