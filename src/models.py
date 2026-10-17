"""Model definitions: logistic regression, random forest and KNN.

All three are trained with class imbalance in mind. Logistic regression and
the random forest use balanced class weights; KNN has no weighting option, so
it trains on a stratified subsample where the minority class is preserved and
the majority class is downsampled. This keeps KNN prediction times practical
while leaving the evaluation set untouched.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier

KNN_MAJORITY_RATIO = 10  # majority samples kept per fraud case for KNN


def _knn_subsample(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Downsample the majority class for KNN.

    KNN degrades badly at 1:600 imbalance because fraud cases rarely appear in
    any neighbourhood. Keeping every fraud case plus a small majority sample
    gives it a workable class balance; the test set is left untouched.
    """
    rng = np.random.default_rng(seed)
    fraud_idx = np.flatnonzero(y_train == 1)
    legit_idx = np.flatnonzero(y_train == 0)
    keep_legit = rng.choice(
        legit_idx,
        size=min(len(legit_idx), len(fraud_idx) * KNN_MAJORITY_RATIO),
        replace=False,
    )
    keep = np.concatenate([fraud_idx, keep_legit])
    rng.shuffle(keep)
    return X_train[keep], y_train[keep]


def train_all(
    X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> dict[str, object]:
    """Train the three classifiers and return them keyed by name."""
    models: dict[str, object] = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced", max_iter=1_000, random_state=seed
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
        "knn": KNeighborsClassifier(n_neighbors=5, weights="distance"),
    }

    models["logistic_regression"].fit(X_train, y_train)
    models["random_forest"].fit(X_train, y_train)

    X_knn, y_knn = _knn_subsample(X_train, y_train, seed=seed)
    models["knn"].fit(X_knn, y_knn)
    return models


def tune_threshold(
    model, X_train: np.ndarray, y_train: np.ndarray, seed: int = 42
) -> float:
    """Pick the decision threshold that maximises F1 via 3-fold cross
    validation on the training data. With 1:600 imbalance the default 0.5
    threshold is meaningless, so tuning it is part of the modelling, not an
    evaluation trick: the test set is never used here."""
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    if isinstance(model, KNeighborsClassifier):
        X_sub, y_sub = _knn_subsample(X_train, y_train, seed=seed)
        scores = cross_val_predict(model, X_sub, y_sub, cv=cv,
                                   method="predict_proba")[:, 1]
        y_true = y_sub
    else:
        scores = cross_val_predict(model, X_train, y_train, cv=cv,
                                   method="predict_proba")[:, 1]
        y_true = y_train

    best_threshold, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 19):
        f1 = f1_score(y_true, scores >= threshold, zero_division=0)
        if f1 > best_f1:
            best_threshold, best_f1 = threshold, f1
    return float(best_threshold)
