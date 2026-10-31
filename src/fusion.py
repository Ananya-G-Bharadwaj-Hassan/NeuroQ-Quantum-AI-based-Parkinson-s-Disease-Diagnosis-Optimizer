import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from . import config


def build_meta_features(proba_dict):
    """
    proba_dict: {"tabular": proba_array_Nx2, "voice": proba_array_Nx2, ...}
    All arrays must have the same N (matched patients, same order).
    Returns an (N, 2*n_modalities) feature matrix of positive-class probs.
    """
    cols = [p[:, 1].reshape(-1, 1) for p in proba_dict.values()]
    return np.hstack(cols)


def fit_fusion(proba_dict, y, test_size=config.TEST_SIZE):
    X_meta = build_meta_features(proba_dict)
    Xtr, Xte, ytr, yte = train_test_split(
        X_meta, y, test_size=test_size, random_state=config.RANDOM_STATE, stratify=y
    )
    meta_clf = LogisticRegression(max_iter=1000)
    meta_clf.fit(Xtr, ytr)
    acc = meta_clf.score(Xte, yte)
    return meta_clf, acc
