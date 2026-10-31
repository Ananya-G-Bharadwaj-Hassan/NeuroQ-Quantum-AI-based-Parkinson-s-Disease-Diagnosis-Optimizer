import glob
import os
import numpy as np
import pydicom
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from . import config

FEATURE_NAMES = [
    "mean_intensity", "std_intensity", "skew_intensity", "kurtosis_intensity",
    "p10", "p90", "contrast", "homogeneity", "energy", "correlation",
]


def radiomic_features(pixel_array):
    img = pixel_array.astype(np.float64)
    if img.ndim == 3:
        img = img[img.shape[0] // 2]  # multi-frame volume -> take middle slice
    elif img.ndim > 3:
        img = img.reshape(-1, img.shape[-2], img.shape[-1])[0]
    img = (img - img.min()) / (img.max() - img.min() + 1e-9)
    img8 = (img * 255).astype(np.uint8)

    mean_i, std_i = img.mean(), img.std()
    skew = np.mean((img - mean_i) ** 3) / (std_i ** 3 + 1e-9)
    kurt = np.mean((img - mean_i) ** 4) / (std_i ** 4 + 1e-9)
    p10, p90 = np.percentile(img, [10, 90])

    glcm = graycomatrix(img8, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, "contrast")[0, 0]
    homogeneity = graycoprops(glcm, "homogeneity")[0, 0]
    energy = graycoprops(glcm, "energy")[0, 0]
    correlation = graycoprops(glcm, "correlation")[0, 0]

    return np.array([mean_i, std_i, skew, kurt, p10, p90, contrast, homogeneity, energy, correlation])


# Old private name kept as an alias so nothing else in the codebase breaks.
_radiomic_features = radiomic_features


def build_imaging_dataset(imaging_dir=config.IMAGING_DIR, limit=None):
    """
    Returns X_classical (scaled features), X_quantum (PCA-reduced),
    patient_ids (from DICOM PatientID tag), file paths.
    There is no `y` here — see module docstring.
    """
    files = glob.glob(os.path.join(imaging_dir, "**", "*.dcm"), recursive=True)
    if limit:
        files = files[:limit]

    X, patient_ids, paths = [], [], []
    for f in files:
        try:
            ds = pydicom.dcmread(f)
            feats = _radiomic_features(ds.pixel_array)
            if not np.isnan(feats).any():
                X.append(feats)
                patient_ids.append(str(getattr(ds, "PatientID", "unknown")))
                paths.append(f)
        except Exception as e:
            print(f"[skip] {f}: {e}")

    X = np.array(X)
    X_classical = MinMaxScaler().fit_transform(X)
    X_angle = MinMaxScaler((0, np.pi)).fit_transform(X)
    X_quantum = PCA(n_components=config.N_QUBITS, random_state=config.RANDOM_STATE).fit_transform(X_angle)

    return X_classical, X_quantum, patient_ids, paths


def unsupervised_anomaly_scores(X_classical, contamination=0.25):
    clf = IsolationForest(contamination=contamination, random_state=config.RANDOM_STATE)
    clf.fit(X_classical)
    scores = -clf.score_samples(X_classical)  # higher = more anomalous
    return scores, clf
