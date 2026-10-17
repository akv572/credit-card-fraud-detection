"""Preprocessing: stratified split and feature scaling."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "Class"


def split_and_scale(
    data: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Stratified train/test split with ``Time`` and ``Amount`` standardised.

    The PCA features are already centred, so only ``Time`` and ``Amount`` are
    scaled (fit on the training fold to avoid leakage). Returns numpy arrays
    plus the ordered feature names.
    """
    feature_cols = [c for c in data.columns if c != TARGET]
    X = data[feature_cols].copy()
    y = data[TARGET].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Fit once on train, apply to both folds.
    scaler = StandardScaler().fit(X_train[["Time", "Amount"]])
    X_train[["Time", "Amount"]] = scaler.transform(X_train[["Time", "Amount"]])
    X_test[["Time", "Amount"]] = scaler.transform(X_test[["Time", "Amount"]])

    return (
        X_train.to_numpy(),
        X_test.to_numpy(),
        y_train,
        y_test,
        feature_cols,
    )
