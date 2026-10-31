import glob
import os

import numpy as np
import pydicom
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import MinMaxScaler

from . import config
from .data_imaging import radiomic_features

IMAGE_EXTENSIONS = (".dcm", ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def _read_pixel_array(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dcm":
        return pydicom.dcmread(path).pixel_array
    return np.array(Image.open(path).convert("L"))  # grayscale


def _resolve_class_folders(labeled_dir):
    subfolders = [
        d for d in sorted(os.listdir(labeled_dir))
        if os.path.isdir(os.path.join(labeled_dir, d))
    ]
    if len(subfolders) < 2:
        raise FileNotFoundError(
            f"Expected at least 2 class subfolders under {labeled_dir} "
            f"(e.g. PD/ and Control/), found: {subfolders}"
        )

    mapping = {}
    unrecognized = []
    for folder in subfolders:
        key = folder.strip().lower().replace("-", "").replace("_", "").replace(" ", "")
        if key in config.IMAGING_CLASS_ALIASES:
            mapping[folder] = config.IMAGING_CLASS_ALIASES[key]
        else:
            unrecognized.append(folder)

    if unrecognized:
        raise ValueError(
            f"Don't recognize folder name(s) {unrecognized} under {labeled_dir}. "
            f"Rename them to something like PD/Control or Parkinsons/Healthy, "
            f"or add the name to config.IMAGING_CLASS_ALIASES."
        )
    if len(set(mapping.values())) < 2:
        raise ValueError(
            f"All recognized folders under {labeled_dir} mapped to the same "
            f"label {set(mapping.values())} - need both a PD-side and a "
            f"Control-side folder to train a classifier."
        )
    return mapping


def build_labeled_imaging_dataset(labeled_dir=config.LABELED_IMAGING_DIR, limit_per_class=None):
    if not os.path.isdir(labeled_dir):
        raise FileNotFoundError(
            f"Can't find {labeled_dir}\n"
            f"Create it with one subfolder per class (e.g. PD/, Control/) "
            f"containing MRI images - see README.md 'Adding a labeled MRI dataset'."
        )

    folder_to_label = _resolve_class_folders(labeled_dir)

    X, y, paths = [], [], []
    for folder, label in folder_to_label.items():
        files = []
        for ext in IMAGE_EXTENSIONS:
            files.extend(glob.glob(os.path.join(labeled_dir, folder, "**", f"*{ext}"), recursive=True))
        if limit_per_class:
            files = files[:limit_per_class]

        for f in files:
            try:
                feats = radiomic_features(_read_pixel_array(f))
                if not np.isnan(feats).any():
                    X.append(feats)
                    y.append(label)
                    paths.append(f)
            except Exception as e:
                print(f"[skip] {f}: {e}")

    if not X:
        raise RuntimeError(f"Found the class folders but no readable images under {labeled_dir}.")

    X = np.array(X)
    y = np.array(y)

    scaler_classical = MinMaxScaler().fit(X)
    X_classical = scaler_classical.transform(X)

    # Supervised feature selection for radiomic features into quantum state space
    quantum_selector = SelectKBest(f_classif, k=config.N_QUBITS).fit(X, y)
    scaler_quantum = MinMaxScaler((0.0, np.pi)).fit(quantum_selector.transform(X))
    X_quantum = scaler_quantum.transform(quantum_selector.transform(X))

    # Retain PCA transformers for legacy backwards compatibility
    scaler_angle = MinMaxScaler((0.0, np.pi)).fit(X)
    pca_quantum = PCA(n_components=config.N_QUBITS, random_state=config.RANDOM_STATE).fit(
        scaler_angle.transform(X)
    )

    return {
        "X_classical": X_classical,
        "X_quantum": X_quantum,
        "y": y,
        "paths": paths,
        "class_mapping": folder_to_label,
        "scaler_classical": scaler_classical,
        "quantum_selector": quantum_selector,
        "scaler_quantum": scaler_quantum,
        "scaler_angle": scaler_angle,
        "pca_quantum": pca_quantum,
    }


def extract_features_for_prediction(image_path):
    return radiomic_features(_read_pixel_array(image_path))
