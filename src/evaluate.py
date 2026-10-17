"""Evaluation: metrics, comparison table and precision-recall curves."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def score_model(
    model, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5
) -> dict:
    """Compute the core metrics for one trained model at a given decision
    threshold."""
    y_score = (
        model.predict_proba(X_test)[:, 1]
        if hasattr(model, "predict_proba")
        else model.decision_function(X_test)
    )
    y_pred = (y_score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    return {
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_score)),
        "avg_precision": float(average_precision_score(y_test, y_score)),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
    }


def print_comparison(results: dict[str, dict]) -> None:
    header = f"{'model':<22}{'precision':>10}{'recall':>10}{'f1':>10}{'roc_auc':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(
            f"{name:<22}{m['precision']:>10.4f}{m['recall']:>10.4f}"
            f"{m['f1']:>10.4f}{m['roc_auc']:>10.4f}"
        )


def save_reports(
    models: dict, results: dict[str, dict],
    X_test: np.ndarray, y_test: np.ndarray, out_dir: str | Path,
) -> None:
    """Write metrics.json and a precision-recall curve plot."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(results, indent=2))

    plt.figure(figsize=(7, 5))
    for name, model in models.items():
        y_score = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else model.decision_function(X_test)
        )
        precision, recall, _ = precision_recall_curve(y_test, y_score)
        plt.plot(recall, precision, label=f"{name} (AP={results[name]['avg_precision']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curves: fraud detection models")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "pr_curves.png", dpi=120)
    plt.close()
