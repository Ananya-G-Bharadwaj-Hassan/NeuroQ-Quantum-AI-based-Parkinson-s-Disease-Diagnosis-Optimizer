from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from . import config


def get_classical_models():
    return {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=config.RANDOM_STATE),
        "random_forest": RandomForestClassifier(n_estimators=300, random_state=config.RANDOM_STATE),
        "xgboost": XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            eval_metric="logloss", random_state=config.RANDOM_STATE
        ),
    }
