"""Model factories. Production model: Random Forest + one-hot Type + engineered features."""
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.compose import ColumnTransformer

from .config import RANDOM_STATE

NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "temp_diff",
    "power_w",
    "wear_torque",
    "strain_ratio",
]


def make_preprocessor(feature_columns):
    numeric = [c for c in feature_columns if c != "Type"]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Type"]),
            ("num", "passthrough", numeric),
        ]
    )


def make_pipeline(model, feature_columns):
    return Pipeline([("prep", make_preprocessor(feature_columns)), ("model", model)])


def make_rf(feature_columns=None):
    if feature_columns is None:
        feature_columns = NUMERIC_FEATURES
    return make_pipeline(
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced_subsample",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        feature_columns,
    )


def make_models(feature_columns=None):
    """Benchmark model factories (for comparison, not production core)."""
    fc = feature_columns or NUMERIC_FEATURES
    models = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogReg": make_pipeline(
            LogisticRegression(max_iter=3000, class_weight="balanced"), fc
        ),
        "RandomForest": make_rf(fc),
        "HistGB": make_pipeline(
            HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          random_state=RANDOM_STATE), fc
        ),
    }
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = make_pipeline(
            XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=4,
                          subsample=0.9, eval_metric="logloss",
                          random_state=RANDOM_STATE, n_jobs=-1), fc
        )
    except Exception:
        pass
    return models
