"""Synthetic transaction data generator.

Produces a dataset with the same shape as the well known credit card fraud
detection dataset: anonymised PCA features ``V1``..``V28`` plus ``Time`` and
``Amount``, with a heavily imbalanced ``Class`` label. The generator exists so
the pipeline runs end to end without shipping a large data file; swap
``load_transactions`` for a CSV reader to run against real data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

N_PCA_FEATURES = 28
FRAUD_RATIO = 0.0017  # ~0.17%, mirroring real world card fraud rates


def generate(
    n_samples: int = 60_000,
    fraud_ratio: float = FRAUD_RATIO,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic labelled transaction dataset.

    Legitimate transactions are drawn from standard normals. Fraudulent ones
    shift a subset of the PCA components and skew towards larger amounts, with
    enough overlap that the classification problem stays realistic.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(n_samples * fraud_ratio))
    n_legit = n_samples - n_fraud

    def _rows(n: int, fraudulent: bool) -> pd.DataFrame:
        frame = pd.DataFrame(
            rng.normal(0.0, 1.0, size=(n, N_PCA_FEATURES)),
            columns=[f"V{i}" for i in range(1, N_PCA_FEATURES + 1)],
        )
        if fraudulent:
            # Fraud shifts a handful of components and favours larger amounts.
            for col, shift in (("V3", 2.4), ("V4", -2.0), ("V10", 1.8),
                               ("V12", -1.6), ("V14", 2.2), ("V17", -1.9)):
                frame[col] += rng.normal(shift, 0.7, size=n)
            frame["Amount"] = rng.lognormal(mean=4.6, sigma=0.9, size=n)
        else:
            frame["Amount"] = rng.lognormal(mean=3.4, sigma=1.1, size=n)
        frame["Time"] = rng.uniform(0, 172_800, size=n)  # seconds over 48h
        frame["Class"] = 1 if fraudulent else 0
        return frame

    data = pd.concat(
        [_rows(n_legit, False), _rows(n_fraud, True)], ignore_index=True
    )
    return data.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_transactions(csv_path: str | None = None, **kwargs) -> pd.DataFrame:
    """Load transactions from CSV, or generate synthetic data when no path
    is given. ``kwargs`` are forwarded to :func:`generate`."""
    if csv_path:
        return pd.read_csv(csv_path)
    return generate(**kwargs)
