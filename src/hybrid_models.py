import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

from . import config
from .quantum_models import QuantumKernelSVM


class HybridQuantumClassicalClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        classical_model,
        quantum_model,
        strategy="stacking",
        classical_model_name="xgboost",
        quantum_model_name="quantum_kernel_svm",
        feature_names=None,
        quantum_feature_names=None,
        n_splits=5,
        random_state=config.RANDOM_STATE,
    ):
        self.classical_model = classical_model
        self.quantum_model = quantum_model
        self.strategy = strategy
        self.classical_model_name = classical_model_name
        self.quantum_model_name = quantum_model_name
        self.feature_names = feature_names or []
        self.quantum_feature_names = quantum_feature_names or []
        self.n_splits = n_splits
        self.random_state = random_state

        self.meta_learner = None
        self.optimal_quantum_weight = 0.20
        self.meta_weights = None
        self.meta_intercept = None
        self.is_fitted_ = False

    def fit(self, X_classical, X_quantum, y):
        X_classical = np.asarray(X_classical, dtype=np.float64)
        X_quantum = np.asarray(X_quantum, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)

        n_samples = len(y)
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)

        oof_c = np.zeros(n_samples, dtype=np.float64)
        oof_q = np.zeros(n_samples, dtype=np.float64)

        for train_idx, val_idx in skf.split(X_classical, y):
            # Efficient fold classical model
            fold_c = clone(self.classical_model)
            if hasattr(fold_c, "n_estimators") and fold_c.n_estimators > 100:
                fold_c.n_estimators = 100
            fold_c.fit(X_classical[train_idx], y[train_idx])
            oof_c[val_idx] = fold_c.predict_proba(X_classical[val_idx])[:, 1]

            # Fit quantum model on fold
            if isinstance(self.quantum_model, QuantumKernelSVM):
                fold_q = QuantumKernelSVM(
                    n_qubits=self.quantum_model.n_qubits,
                    c=self.quantum_model.c,
                    random_state=self.random_state,
                )
                fold_q.fit(X_quantum[train_idx], y[train_idx])
                oof_q[val_idx] = fold_q.predict_proba(X_quantum[val_idx])[:, 1]
            else:
                fold_q = clone(self.quantum_model) if hasattr(self.quantum_model, "get_params") else self.quantum_model.__class__()
                sub_idx = train_idx[:min(100, len(train_idx))]
                fold_q.fit(X_quantum[sub_idx], y[sub_idx])
                oof_q[val_idx] = fold_q.predict_proba(X_quantum[val_idx])[:, 1]

        # Train meta-learner on out-of-fold predictions
        meta_X = np.column_stack([oof_c, oof_q])
        self.meta_learner = LogisticRegression(C=1.0, random_state=self.random_state, max_iter=1000)
        self.meta_learner.fit(meta_X, y)
        self.meta_weights = self.meta_learner.coef_[0].tolist()
        self.meta_intercept = float(self.meta_learner.intercept_[0])

        # Optimize convex blend weight on OOF predictions
        best_auc = -1.0
        best_w = 0.15
        for w in np.linspace(0.0, 0.5, 51):
            p_blend = (1.0 - w) * oof_c + w * oof_q
            try:
                score = roc_auc_score(y, p_blend)
                if score > best_auc:
                    best_auc = score
                    best_w = float(w)
            except Exception:
                pass
        self.optimal_quantum_weight = best_w

        # Fit full base models on entire dataset
        self.classical_model.fit(X_classical, y)
        self.quantum_model.fit(X_quantum, y)
        self.is_fitted_ = True
        return self

    def predict_proba(self, X_classical, X_quantum, custom_quantum_weight=None):
        X_classical = np.asarray(X_classical, dtype=np.float64)
        X_quantum = np.asarray(X_quantum, dtype=np.float64)

        p_c = self.classical_model.predict_proba(X_classical)
        p_q = self.quantum_model.predict_proba(X_quantum)

        p_c_pos = p_c[:, 1]
        p_q_pos = p_q[:, 1]

        if custom_quantum_weight is not None:
            w_q = np.clip(float(custom_quantum_weight), 0.0, 1.0)
            p_pos = (1.0 - w_q) * p_c_pos + w_q * p_q_pos
        elif self.strategy == "stacking" and self.meta_learner is not None:
            meta_features = np.column_stack([p_c_pos, p_q_pos])
            p_pos = self.meta_learner.predict_proba(meta_features)[:, 1]
        else:
            w_q = self.optimal_quantum_weight
            p_pos = (1.0 - w_q) * p_c_pos + w_q * p_q_pos

        p_pos = np.clip(p_pos, 0.0, 1.0)
        return np.column_stack([1.0 - p_pos, p_pos])

    def predict(self, X_classical, X_quantum, custom_quantum_weight=None, threshold=0.5):
        proba = self.predict_proba(X_classical, X_quantum, custom_quantum_weight=custom_quantum_weight)
        return (proba[:, 1] >= threshold).astype(int)

    def predict_detailed(self, X_classical, X_quantum, custom_quantum_weight=None):
        proba = self.predict_proba(X_classical, X_quantum, custom_quantum_weight=custom_quantum_weight)
        p_hybrid = float(proba[0, 1])

        p_c = float(self.classical_model.predict_proba(X_classical)[0, 1])
        p_q = float(self.quantum_model.predict_proba(X_quantum)[0, 1])

        pred_label = int(p_hybrid >= 0.5)
        pred_class = "Parkinson's Disease" if pred_label == 1 else "Control / Healthy"

        conf_margin = abs(p_hybrid - 0.5) * 2.0
        if conf_margin >= 0.6:
            conf_band = "High"
        elif conf_margin >= 0.3:
            conf_band = "Moderate"
        else:
            conf_band = "Borderline / Low"

        used_qw = custom_quantum_weight if custom_quantum_weight is not None else self.optimal_quantum_weight

        return {
            "p_hybrid": p_hybrid,
            "p_classical": p_c,
            "p_quantum": p_q,
            "predicted_label": pred_label,
            "predicted_class": pred_class,
            "confidence": conf_margin,
            "confidence_band": conf_band,
            "classical_model_name": self.classical_model_name,
            "quantum_model_name": self.quantum_model_name,
            "strategy": self.strategy if custom_quantum_weight is None else "custom_interactive_blend",
            "weights": {
                "classical": 1.0 - used_qw,
                "quantum": used_qw,
                "meta_weights": self.meta_weights,
            },
            "quantum_features": self.quantum_feature_names,
        }

    def __getstate__(self):
        return {
            "classical_model": self.classical_model,
            "quantum_model": self.quantum_model,
            "strategy": self.strategy,
            "classical_model_name": self.classical_model_name,
            "quantum_model_name": self.quantum_model_name,
            "feature_names": self.feature_names,
            "quantum_feature_names": self.quantum_feature_names,
            "n_splits": self.n_splits,
            "random_state": self.random_state,
            "meta_learner": self.meta_learner,
            "optimal_quantum_weight": self.optimal_quantum_weight,
            "meta_weights": self.meta_weights,
            "meta_intercept": self.meta_intercept,
            "is_fitted_": self.is_fitted_,
        }

    def __setstate__(self, state):
        self.__dict__.update(state)
